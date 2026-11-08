"""Inference adapter for the local interactive anonymization demo.

The research pipeline evaluates S2 linkage between multiple resumes belonging to
the same person. Arbitrary user text has no such ground truth, so this module uses
the deployment-time S1 proxy: the original input is appended to the fixed WorkUA
background and treated as the target that an attacker tries to retrieve.
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .adaptive_text import (
    TextRiskController,
    adaptive_release,
    apply_actions,
    make_free_text_document,
    mandatory_actions,
    unit_features,
)
from .attack import BM25Index, TfidfIndex, ranking_result
from .pipeline import MASK
from .research import detector_set, resolve_config


@dataclass
class DemoResources:
    config: dict
    background_texts: list[str]
    codec: object
    reference: object
    model: object
    retention_vectorizer: object
    detector: Callable
    artifact_dir: Path
    feature_index: object | None = None


@dataclass
class TextDemoResult:
    original_text: str
    anonymized_text: str
    strategy: str
    pool_size: int
    original_proxy_rr: float
    initial_proxy_rr: float
    final_proxy_rr: float
    retention: float
    threshold_met: bool
    stop_reason: str
    units: list[dict]
    trace: list[dict]
    attackers: list[dict]


@dataclass
class AnonymizationOption:
    key: str
    title: str
    description: str
    anonymized_text: str
    final_proxy_rr: float
    risk_reduction: float
    final_relative_margin: float
    margin_reduction: float
    retention: float
    edits: list[dict]
    output_spans: list[dict]
    attackers: list[dict]


@dataclass
class ProposalBundle:
    original_text: str
    original_proxy_rr: float
    original_relative_margin: float
    pool_size: int
    detected_units: list[dict]
    options: list[AnonymizationOption]
    llm_status: str


def load_demo_resources(config_path: str | Path) -> DemoResources:
    """Load locally generated artifacts and the calibrated detector ensemble."""
    config = resolve_config(config_path)
    artifact_dir = Path(config["out"])
    required = {
        "WorkUA bundle": artifact_dir / "workua" / "bundle.pkl",
        "risk model": artifact_dir / "pipeline" / "unit_risk_model.pkl",
        "retention vectorizer": artifact_dir / "pipeline" / "retention_vectorizer.pkl",
        "detector selection": artifact_dir / "detection" / "selected_detector.json",
    }
    missing = [f"{label}: {path}" for label, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Run the pipeline for this profile first:\n" + "\n".join(missing))

    selected = json.loads(required["detector selection"].read_text(encoding="utf-8"))
    names = config.get("release_detectors", ["regex", "ner_uk", "gliner"])
    if selected.get("status") != "ready" or not set(names) <= set(selected.get("detectors", [])):
        raise ValueError("The calibrated detector ensemble is not ready for the UI")
    # A completed research run has already downloaded and calibrated these
    # models. The UI should not perform surprise network requests on startup.
    config = {**config, "gliner_threshold": selected["gliner_threshold"], "local_files_only": True}

    with required["WorkUA bundle"].open("rb") as handle:
        bundle = pickle.load(handle)
    with required["risk model"].open("rb") as handle:
        model = pickle.load(handle)
    with required["retention vectorizer"].open("rb") as handle:
        retention_vectorizer = pickle.load(handle)
    detector, _objects = detector_set(config, names)
    feature_index = BM25Index(max_features=config.get("bm25_max_features", 150_000)).fit(bundle["texts"])
    return DemoResources(
        config=config,
        background_texts=bundle["texts"],
        codec=bundle["codec"],
        reference=bundle["reference"],
        model=model,
        retention_vectorizer=retention_vectorizer,
        detector=detector,
        artifact_dir=artifact_dir,
        feature_index=feature_index,
    )


def _attacker_measure(index, text: str, target: int) -> dict:
    result, margin, target_score = ranking_result(index, [text], [[target]], batch_size=1)
    scale = max(abs(float(target_score[0])), 1e-12)
    return {
        "expected_rank": float(result.ranks[0]),
        "reciprocal_rank": float(result.reciprocal_ranks[0]),
        "recall_at_1": float(result.recalls[1][0]),
        "margin": float(margin[0]),
        "target_score": float(target_score[0]),
        "relative_margin": float(margin[0]) / scale,
    }


def _unit_rows(doc, actions, scores) -> list[dict]:
    finite_scores = [max(float(score), 0.0) for score in scores if np.isfinite(score)]
    maximum = max(finite_scores, default=0.0)
    rows = []
    for index, unit in enumerate(doc.units):
        action = actions.get(index, "keep")
        if action == "generalize":
            replacement = unit.generalized
        elif action == "mask":
            replacement = MASK.get(unit.label, "[REDACTED]")
        else:
            replacement = doc.text[unit.start : unit.end]
        score = float(scores[index]) if index < len(scores) else 0.0
        relative = max(score, 0.0) / maximum if maximum > 0 else 0.0
        if unit.role in {"DIRECT", "SENSITIVE"}:
            level = "mandatory"
        elif relative >= 0.67:
            level = "high"
        elif relative >= 0.34:
            level = "medium"
        else:
            level = "low"
        rows.append(
            {
                "unit": index,
                "text": doc.text[unit.start : unit.end],
                "label": unit.label,
                "role": unit.role,
                "source": unit.source,
                "start": unit.start,
                "end": unit.end,
                "risk_priority": score,
                "relative_priority": relative,
                "priority_level": level,
                "action": action,
                "replacement": replacement,
            }
        )
    return rows


def _apply_replacements(
    doc, replacements: dict[int, str], reasons: dict[int, str] | None = None, origins: dict[int, str] | None = None
) -> tuple[str, list[dict], list[dict]]:
    """Apply validated replacements by local offsets and preserve output spans for highlighting."""
    reasons, origins = reasons or {}, origins or {}
    out, edits, spans, last, output_length = [], [], [], 0, 0
    for index, unit in enumerate(doc.units):
        if index not in replacements:
            continue
        replacement = replacements[index]
        if not isinstance(replacement, str) or not replacement.strip():
            raise ValueError("A replacement must be a non-empty string")
        untouched = doc.text[last : unit.start]
        out.extend([untouched, replacement])
        output_length += len(untouched)
        output_start, output_end = output_length, output_length + len(replacement)
        output_length = output_end
        original = doc.text[unit.start : unit.end]
        edits.append(
            {
                "unit": index,
                "original": original,
                "replacement": replacement,
                "label": unit.label,
                "role": unit.role,
                "start": unit.start,
                "end": unit.end,
                "reason": reasons.get(index, "Reduced identifying specificity"),
                "origin": origins.get(index, "local_policy"),
            }
        )
        spans.append({"start": output_start, "end": output_end, "label": unit.label})
        last = unit.end
    tail = doc.text[last:]
    out.append(tail)
    return "".join(out), edits, spans


def _action_replacements(doc, actions: dict[int, str]) -> tuple[dict[int, str], dict[int, str]]:
    replacements, reasons = {}, {}
    for index, action in actions.items():
        unit = doc.units[index]
        if action == "mask":
            replacements[index] = MASK.get(unit.label, "[REDACTED]")
            reasons[index] = "Direct or sensitive identifier redacted"
        elif action == "generalize" and unit.generalized:
            replacements[index] = unit.generalized
            reasons[index] = "Specific value replaced with a broader category"
    return replacements, reasons


def _risk_band(value: float) -> str:
    if value >= 0.75:
        return "high"
    if value >= 0.34:
        return "medium"
    return "lower"


def propose_anonymizations(resources: DemoResources, text: str, *, llm_suggester=None) -> ProposalBundle:
    """Generate user-facing alternatives and evaluate every one with local attackers.

    The LLM may only suggest replacement strings for known QUASI unit IDs. It never
    supplies offsets, privacy scores, or the final decision.
    """
    text = text.strip()
    if not text:
        raise ValueError("Enter non-empty text")
    limit = int(resources.config.get("max_document_chars") or 20_000)
    if len(text) > limit:
        raise ValueError(f"Text contains {len(text)} characters; the profile limit is {limit}")
    doc = make_free_text_document(
        text,
        resources.detector,
        include_residual=resources.config.get("include_residual", True),
    )
    if not doc.units:
        raise ValueError("The text contains no transformable units")
    original_vector = resources.retention_vectorizer.transform([doc.text])
    if original_vector.nnz == 0:
        raise ValueError("The text has no overlap with the model vocabulary; enter resume-domain text")

    pool = [*resources.background_texts, doc.text]
    target = len(pool) - 1
    bm25 = BM25Index(max_features=resources.config.get("bm25_max_features", 150_000)).fit(pool)
    indexes = {"BM25": bm25}
    if resources.config.get("use_char_tfidf", False):
        indexes["char-TF-IDF"] = TfidfIndex(max_features=100_000).fit(pool)
    control_names = resources.config.get("controller_attackers") or list(indexes)
    controller = TextRiskController(
        {name: indexes[name] for name in control_names}, require_margin=resources.config.get("require_margin", False)
    )
    features = unit_features(doc, resources.reference, bm25)
    scores = np.maximum(resources.model.predict(features), 0) if len(features) else np.array([])
    unit_rows = _unit_rows(doc, {}, scores)

    original_attackers = [
        {"attacker": name, **_attacker_measure(index, doc.text, target)} for name, index in indexes.items()
    ]
    original_risk = max(row["reciprocal_rank"] for row in original_attackers)
    original_margin = max(row["relative_margin"] for row in original_attackers)

    mandatory = mandatory_actions(doc)
    minimal = dict(mandatory)
    for index, unit in enumerate(doc.units):
        if unit.role == "QUASI" and unit.source != "residual_text" and unit.generalized:
            minimal[index] = "generalize"

    balanced_text, balanced_actions, _trace = adaptive_release(
        doc,
        target,
        controller,
        resources.model,
        resources.reference,
        bm25,
        resources.retention_vectorizer,
        tau=0.5,
        strategy="model",
        utility_floor=0.5,
        max_steps=None,
        candidate_limit=resources.config.get("candidate_limit", 8),
        max_candidates_per_step=resources.config.get("max_candidates_per_step"),
        expand_on_plateau=resources.config.get("expand_on_plateau", True),
    )
    del balanced_text
    strong = {
        index: "mask"
        for index, unit in enumerate(doc.units)
        if unit.role != "CONTEXT" and unit.source != "residual_text"
    }
    strong.update(mandatory)

    policies = [
        (
            "minimal",
            "Minimal changes",
            "Redacts direct identifiers and generalizes detected quasi-identifiers",
            minimal,
        ),
        (
            "balanced",
            "Balanced",
            "The adaptive model selects transformations using its internal threshold",
            balanced_actions,
        ),
        (
            "strong",
            "Stronger protection",
            "Redacts every explicitly detected identifying fragment",
            strong,
        ),
    ]
    candidates: list[tuple[str, str, str, dict[int, str], dict[int, str], dict[int, str]]] = []
    for key, title, description, actions in policies:
        replacements, reasons = _action_replacements(doc, actions)
        candidates.append((key, title, description, replacements, reasons, {}))

    llm_status = "OpenAI paraphrasing is not configured; local alternatives remain available."
    if llm_suggester is not None:
        try:
            suggestions = llm_suggester.suggest(doc.text, unit_rows)
            llm_replacements, llm_reasons = _action_replacements(doc, mandatory)
            origins = {}
            for suggestion in suggestions:
                llm_replacements[suggestion.unit_id] = suggestion.replacement
                llm_reasons[suggestion.unit_id] = suggestion.reason
                origins[suggestion.unit_id] = "openai"
            if suggestions:
                candidates.append(
                    (
                        "contextual",
                        "Contextual paraphrasing",
                        "OpenAI suggests less specific wording; risk is reassessed locally",
                        llm_replacements,
                        llm_reasons,
                        origins,
                    )
                )
                llm_status = "OpenAI contextual replacements were added and reassessed locally."
            else:
                llm_status = "OpenAI did not suggest additional safe paraphrases."
        except Exception as exc:  # noqa: BLE001 - API failure must preserve deterministic alternatives
            llm_status = f"OpenAI is unavailable: {exc}. Local alternatives were still generated."

    options, seen = [], set()
    for key, title, description, replacements, reasons, origins in candidates:
        released, edits, output_spans = _apply_replacements(doc, replacements, reasons, origins)
        by_unit = {row["unit"]: row for row in unit_rows}
        for edit, span in zip(edits, output_spans, strict=True):
            unit = by_unit[edit["unit"]]
            for key in ("risk_priority", "relative_priority", "priority_level", "role"):
                edit[key] = unit[key]
                span[key] = unit[key]
        if released in seen:
            continue
        seen.add(released)
        released_vector = resources.retention_vectorizer.transform([released])
        retention = float(released_vector.multiply(original_vector).sum())
        attacker_rows = [
            {"attacker": name, **_attacker_measure(index, released, target)} for name, index in indexes.items()
        ]
        final_risk = max(row["reciprocal_rank"] for row in attacker_rows)
        final_margin = max(row["relative_margin"] for row in attacker_rows)
        options.append(
            AnonymizationOption(
                key=key,
                title=title,
                description=description,
                anonymized_text=released,
                final_proxy_rr=float(final_risk),
                risk_reduction=float(original_risk - final_risk),
                final_relative_margin=float(final_margin),
                margin_reduction=float(original_margin - final_margin),
                retention=retention,
                edits=edits,
                output_spans=output_spans,
                attackers=attacker_rows,
            )
        )
    if not options:
        raise RuntimeError("Could not generate distinct anonymization alternatives")
    for option in options:
        option.description += f". Estimated risk: {_risk_band(option.final_proxy_rr)}."
    return ProposalBundle(
        doc.text,
        float(original_risk),
        float(original_margin),
        len(pool),
        unit_rows,
        options,
        llm_status,
    )


def anonymize_text(
    resources: DemoResources,
    text: str,
    *,
    strategy: str = "model",
    tau: float = 0.5,
    utility_floor: float = 0.5,
    max_steps: int | None = None,
) -> TextDemoResult:
    """Run the real learned release controller on one arbitrary free-text input.

    ``strategy`` may be ``model``, ``category`` or ``mask_detected``. Privacy
    values are S1 reciprocal-rank proxies against the configured background,
    not population-level S2 Recall@1 estimates or anonymity guarantees.
    """
    text = text.strip()
    if not text:
        raise ValueError("Enter non-empty text")
    limit = int(resources.config.get("max_document_chars") or 20_000)
    if len(text) > limit:
        raise ValueError(f"Text contains {len(text)} characters; the profile limit is {limit}")
    if strategy not in {"model", "category", "mask_detected"}:
        raise ValueError(f"Unknown strategy: {strategy}")

    doc = make_free_text_document(
        text,
        resources.detector,
        include_residual=resources.config.get("include_residual", True),
    )
    if not doc.units:
        raise ValueError("The text contains no transformable words or units")

    original_vector = resources.retention_vectorizer.transform([doc.text])
    if original_vector.nnz == 0:
        raise ValueError("The text has no overlap with the model vocabulary; enter resume-domain text")

    pool = [*resources.background_texts, doc.text]
    target = len(pool) - 1
    bm25 = BM25Index(max_features=resources.config.get("bm25_max_features", 150_000)).fit(pool)
    indexes = {"BM25": bm25}
    if resources.config.get("use_char_tfidf", False):
        indexes["char-TF-IDF"] = TfidfIndex(max_features=100_000).fit(pool)
    control_names = resources.config.get("controller_attackers") or list(indexes)
    if not set(control_names) <= set(indexes):
        raise ValueError("The UI configuration contains an unknown controller attacker")
    controller = TextRiskController(
        {name: indexes[name] for name in control_names},
        require_margin=resources.config.get("require_margin", False),
    )

    features = unit_features(doc, resources.reference, bm25)
    if strategy == "category":
        scores = np.array([resources.model.class_prior.get(unit.label, 0.0) for unit in doc.units])
    else:
        scores = np.maximum(resources.model.predict(features), 0) if len(features) else np.array([])

    mandatory_text = apply_actions(doc, mandatory_actions(doc))
    initial_risk = float(controller.assess(target, [mandatory_text])[0][0])
    if strategy == "mask_detected":
        actions = {
            i: "mask" for i, unit in enumerate(doc.units) if unit.source != "residual_text" and unit.role != "CONTEXT"
        }
        actions.update(mandatory_actions(doc))
        released = apply_actions(doc, actions)
        final_risk = float(controller.assess(target, [released])[0][0])
        trace = [
            {
                "step": 0,
                "proxy_rr": final_risk,
                "retention": float(
                    resources.retention_vectorizer.transform([released]).multiply(original_vector).sum()
                ),
                "threshold_met": final_risk <= tau,
                "stop_reason": "fixed_policy",
            }
        ]
    else:
        released, actions, trace = adaptive_release(
            doc,
            target,
            controller,
            resources.model,
            resources.reference,
            bm25,
            resources.retention_vectorizer,
            tau=tau,
            strategy=strategy,
            utility_floor=utility_floor,
            max_steps=max_steps,
            candidate_limit=resources.config.get("candidate_limit", 8),
            max_candidates_per_step=resources.config.get("max_candidates_per_step"),
            expand_on_plateau=resources.config.get("expand_on_plateau", True),
        )

    released_vector = resources.retention_vectorizer.transform([released])
    retention = float(released_vector.multiply(original_vector).sum())
    attacker_rows = []
    for name, index in indexes.items():
        for phase, value in [("original", doc.text), ("anonymized", released)]:
            attacker_rows.append({"attacker": name, "phase": phase, **_attacker_measure(index, value, target)})
    original_proxy = max(row["reciprocal_rank"] for row in attacker_rows if row["phase"] == "original")
    final_proxy = max(row["reciprocal_rank"] for row in attacker_rows if row["phase"] == "anonymized")
    last = trace[-1]
    return TextDemoResult(
        original_text=doc.text,
        anonymized_text=released,
        strategy=strategy,
        pool_size=len(pool),
        original_proxy_rr=float(original_proxy),
        initial_proxy_rr=initial_risk,
        final_proxy_rr=float(final_proxy),
        retention=retention,
        threshold_met=bool(last.get("threshold_met", final_proxy <= tau)),
        stop_reason=str(last.get("stop_reason", "fixed_policy")),
        units=_unit_rows(doc, actions, scores),
        trace=trace,
        attackers=attacker_rows,
    )
