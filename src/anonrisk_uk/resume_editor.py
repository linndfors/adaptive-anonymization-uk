"""Model-based resume anonymization workspace for interactive deployment."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import numpy as np

from .adaptive_text import ReleaseDocument, make_free_text_document, unit_features
from .attack import BM25Index
from .pipeline import MASK

ACTION_FACTORS = {
    "keep": 1.0,
    "technical": 0.50,
    "ai": 0.40,
    "mask": 0.05,
}


@dataclass
class ResumeProfile:
    key: str
    title: str
    description: str
    actions: dict[int, str]
    predicted_exposure: float
    utility: float


@dataclass
class ResumeWorkspace:
    workspace_id: str
    original_text: str
    document: ReleaseDocument
    units: list[dict]
    profiles: list[ResumeProfile]
    llm_status: str


@dataclass
class EditedResume:
    text: str
    predicted_exposure: float
    utility: float
    spans: list[dict]
    edits: list[dict]


def _priority_level(role: str, share: float) -> str:
    if role in {"DIRECT", "SENSITIVE"}:
        return "mandatory"
    if share >= 0.20:
        return "high"
    if share >= 0.08:
        return "medium"
    return "low"


def _fallback_generalization(label: str, value: str) -> str | None:
    low = value.casefold()
    if label == "ORGANIZATION":
        if "банк" in low:
            return "банківська установа"
        if any(part in low for part in ("університет", "академ", "школ", "коледж")):
            return "освітній заклад"
        if any(part in low for part in ("лікарн", "клінік")):
            return "медичний заклад"
        if any(part in low for part in ("компан", "тов", "фірм")):
            return "компанія"
        return "організація"
    if label == "LOCATION":
        return "населений пункт"
    if label == "OCCUPATION_EDU":
        return "фахова роль"
    if label == "DATE_TIME":
        return "зазначений період"
    if label == "FINANCIAL":
        return "узагальнена сума"
    if label == "DEMOGRAPHIC":
        return "вікова група"
    if label == "OTHER_QUASI":
        return "узагальнена інформація"
    return None


def _score_units(resources, document: ReleaseDocument) -> list[dict]:
    index = resources.feature_index
    if index is None:
        index = BM25Index(max_features=resources.config.get("bm25_max_features", 150_000)).fit(
            resources.background_texts,
        )
    features = unit_features(document, resources.reference, index)
    raw_scores = np.maximum(resources.model.predict(features), 0) if len(features) else np.array([])
    positive_quasi = [
        float(raw_scores[i]) for i, unit in enumerate(document.units) if unit.role == "QUASI" and raw_scores[i] > 0
    ]
    direct_floor = max(positive_quasi, default=1.0)
    weights = []
    for index, unit in enumerate(document.units):
        raw = float(raw_scores[index]) if index < len(raw_scores) else 0.0
        weight = max(raw, direct_floor) if unit.role in {"DIRECT", "SENSITIVE"} else max(raw, 1e-6)
        weights.append(weight)
    visible_total = (
        sum(weight for unit, weight in zip(document.units, weights, strict=True) if unit.source != "residual_text")
        or 1.0
    )
    rows = []
    for index, (unit, weight) in enumerate(zip(document.units, weights, strict=True)):
        share = weight / visible_total if unit.source != "residual_text" else 0.0
        rows.append(
            {
                "unit": index,
                "text": document.text[unit.start : unit.end],
                "label": unit.label,
                "role": unit.role,
                "source": unit.source,
                "start": unit.start,
                "end": unit.end,
                "model_score": float(raw_scores[index]) if index < len(raw_scores) else 0.0,
                "risk_weight": weight,
                "risk_share": share,
                "priority_level": _priority_level(unit.role, share),
                "technical_replacement": _fallback_generalization(
                    unit.label,
                    document.text[unit.start : unit.end],
                )
                or unit.generalized,
                "placeholder": MASK.get(unit.label, "[REDACTED]"),
                "ai_replacement": None,
                "ai_reason": None,
            }
        )
    return rows


def _replacement(unit: dict, action: str) -> str:
    if action == "keep":
        return unit["text"]
    if action == "technical":
        return unit.get("technical_replacement") or unit["placeholder"]
    if action == "ai":
        return unit.get("ai_replacement") or unit.get("technical_replacement") or unit["placeholder"]
    if action == "mask":
        return unit["placeholder"]
    raise ValueError(f"Unknown editor action: {action}")


def _preserve_edge_whitespace(original: str, replacement: str) -> str:
    match = re.fullmatch(r"(\s*).*?(\s*)", original, flags=re.DOTALL)
    if not match:
        return replacement
    return match.group(1) + replacement.strip() + match.group(2)


def _mergeable_mask_gap(text: str) -> bool:
    words = re.findall(r"[\w'’]+", text.casefold())
    generic = {
        "у",
        "в",
        "та",
        "і",
        "й",
        "компанія",
        "компанії",
        "компанію",
        "компанією",
        "організація",
        "організації",
        "банк",
        "банку",
        "заклад",
        "закладі",
    }
    return len(text) <= 32 and len(words) <= 3 and all(word in generic for word in words)


def render_resume(workspace: ResumeWorkspace, resources, actions: dict[int, str]) -> EditedResume:
    """Apply interactive choices and recompute relative exposure and lexical utility."""
    by_id = {unit["unit"]: unit for unit in workspace.units}
    remaining = 0.0
    for index, document_unit in enumerate(workspace.document.units):
        unit = by_id[index]
        action = actions.get(index, "keep")
        factor = ACTION_FACTORS[action]
        if unit["source"] != "residual_text":
            remaining += unit["risk_weight"] * factor

    operations = []
    for unit in workspace.units:
        if unit["source"] == "residual_text":
            continue
        action = actions.get(unit["unit"], "keep")
        replacement = _preserve_edge_whitespace(unit["text"], _replacement(unit, action))
        operation = {
            **unit,
            "start": unit["start"],
            "end": unit["end"],
            "action": action,
            "replacement": replacement,
            "unit_ids": [unit["unit"]],
        }
        previous = operations[-1] if operations else None
        if (
            previous
            and action == previous["action"] == "mask"
            and unit["label"] == previous["label"]
            and _mergeable_mask_gap(workspace.original_text[previous["end"] : unit["start"]])
        ):
            previous["end"] = unit["end"]
            previous["replacement"] = unit["placeholder"]
            previous["unit_ids"].append(unit["unit"])
            previous["risk_share"] += unit["risk_share"]
            previous["risk_weight"] += unit["risk_weight"]
        else:
            operations.append(operation)

    output, spans, edits, cursor, output_length = [], [], [], 0, 0
    for operation in operations:
        untouched = workspace.original_text[cursor : operation["start"]]
        output.append(untouched)
        output_length += len(untouched)
        replacement = operation["replacement"]
        output_start = output_length
        output.append(replacement)
        output_length += len(replacement)
        spans.append({**operation, "start": output_start, "end": output_length})
        edits.append(operation)
        cursor = operation["end"]
    output.append(workspace.original_text[cursor:])
    text = "".join(output)
    total = sum(unit["risk_weight"] for unit in workspace.units if unit["source"] != "residual_text") or 1.0
    original_vector = resources.retention_vectorizer.transform([workspace.original_text])
    result_vector = resources.retention_vectorizer.transform([text])
    utility = float(result_vector.multiply(original_vector).sum())
    return EditedResume(text, 100.0 * remaining / total, utility, spans, edits)


def _candidate_actions(units: list[dict]) -> list[dict[int, str]]:
    visible = [unit for unit in units if unit["source"] != "residual_text"]
    mandatory = {unit["unit"]: "mask" for unit in visible if unit["role"] in {"DIRECT", "SENSITIVE"}}
    quasi = sorted(
        (unit for unit in visible if unit["role"] == "QUASI"), key=lambda item: (-item["risk_weight"], item["unit"])
    )
    candidates = [dict(mandatory)]
    current = dict(mandatory)
    for unit in quasi:
        current = {**current, unit["unit"]: "technical"}
        candidates.append(current)
    for unit in quasi:
        current = {**current, unit["unit"]: "mask"}
        candidates.append(current)
    return candidates


def _select_profiles(workspace: ResumeWorkspace, resources) -> list[ResumeProfile]:
    evaluated = []
    seen = set()
    for actions in _candidate_actions(workspace.units):
        result = render_resume(workspace, resources, actions)
        if result.text in seen:
            continue
        seen.add(result.text)
        evaluated.append((result.predicted_exposure, result.utility, dict(actions)))
    frontier = []
    for candidate in evaluated:
        risk, utility, _actions = candidate
        dominated = any(
            other_risk <= risk and other_utility >= utility and (other_risk < risk or other_utility > utility)
            for other_risk, other_utility, _ in evaluated
        )
        if not dominated:
            frontier.append(candidate)
    frontier.sort(key=lambda item: (-item[0], -item[1]))
    source = frontier if len(frontier) >= 3 else sorted(evaluated, key=lambda item: (-item[0], -item[1]))
    if not source:
        raise RuntimeError("Could not generate anonymization profiles")
    positions = [0, (len(source) - 1) // 2, len(source) - 1]
    chosen = []
    for position in positions:
        candidate = source[position]
        if candidate not in chosen:
            chosen.append(candidate)
    for candidate in source:
        if len(chosen) == 3:
            break
        if candidate not in chosen:
            chosen.append(candidate)
    names = [
        (
            "gentle",
            "Gentle",
            "Preserves as much content as possible while removing the clearest identifiers.",
        ),
        (
            "balanced",
            "Balanced",
            "A Pareto-frontier compromise between predicted exposure and utility.",
        ),
        (
            "protective",
            "Protective",
            "The lowest predicted exposure among the generated candidates.",
        ),
    ]
    profiles = []
    for (key, title, description), (risk, utility, actions) in zip(names, chosen, strict=False):
        profiles.append(ResumeProfile(key, title, description, actions, risk, utility))
    return profiles


def analyze_resume(resources, text: str, *, llm_suggester=None) -> ResumeWorkspace:
    text = text.strip()
    if not text:
        raise ValueError("Enter non-empty resume text")
    limit = int(resources.config.get("max_document_chars") or 20_000)
    if len(text) > limit:
        raise ValueError(f"Text contains {len(text)} characters; the limit is {limit}")
    document = make_free_text_document(
        text,
        resources.detector,
        include_residual=resources.config.get("include_residual", True),
    )
    if not document.units:
        raise ValueError("The model found no units to evaluate")
    units = _score_units(resources, document)
    workspace = ResumeWorkspace(
        workspace_id=hashlib.sha256(text.encode()).hexdigest()[:12],
        original_text=text,
        document=document,
        units=units,
        profiles=[],
        llm_status=("AI generalization is disabled; rule-based replacements and placeholders are available."),
    )
    if llm_suggester is not None:
        try:
            suggestions = llm_suggester.suggest(text, units)
            by_id = {unit["unit"]: unit for unit in units}
            for suggestion in suggestions:
                by_id[suggestion.unit_id]["ai_replacement"] = suggestion.replacement
                by_id[suggestion.unit_id]["ai_reason"] = suggestion.reason
            workspace.llm_status = (
                f"AI prepared contextual generalizations for {len(suggestions)} fragments."
                if suggestions
                else "AI did not suggest additional safe generalizations."
            )
        except Exception as exc:  # noqa: BLE001 - local profiles must survive an API failure
            workspace.llm_status = (
                f"AI is unavailable: {exc}. Rule-based replacements and placeholders remain available."
            )
    workspace.profiles = _select_profiles(workspace, resources)
    return workspace
