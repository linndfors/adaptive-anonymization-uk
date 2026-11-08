"""Free-text inference using the trained entity policy and its real executor.

No S2 estimate is possible for an arbitrary text with no linked identity. The
controller uses its usual neighbour proxy; only textual retention is displayed.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .action_learning import lexical_retention
from .action_space import typed_document
from .adaptive_text import ReleaseDocument
from .attack import BM25Index
from .entity_policy import EntityPolicyModel, release_sequential, release_with_policy
from .policy_execution import ToolRouter, execute_fixed
from .resume_fields import ContentGuardRouter, FieldToolRouter, field_document


@dataclass
class DemoRelease:
    text: str
    edits: list[dict]
    retention: float | None
    trace: list[dict]


@dataclass
class EntityDemoResult:
    original: str
    entities: list[dict]
    releases: dict[str, DemoRelease]
    mode: str = "legacy"
    diagnostics: dict = field(default_factory=dict)


def text_document(text, codec, detector, *, include_residual=True):
    """Reuse research typing/skill detection while preserving user-text offsets."""
    wrapped = typed_document(
        {"additional_info": [text]}, codec, detector, include_residual=include_residual
    )
    offset = wrapped.text.index(text)
    units = [replace(u, start=u.start - offset, end=u.end - offset) for u in wrapped.units]
    return ReleaseDocument(text, units)


def anonymize_entity_text(resources, text: str, *, mode="field_aware") -> EntityDemoResult:
    """Run the configured policy plus fixed baselines without any paid tools."""
    if not isinstance(resources.model, EntityPolicyModel):
        raise TypeError("Виберіть завершений профіль entity_decision_v2.")
    if not text.strip():
        raise ValueError("Введіть текст резюме.")
    if mode not in {"legacy", "field_aware"}:
        raise ValueError(f"Невідомий режим розбору: {mode}")
    config = resources.config
    limit = int(config.get("max_document_chars") or 20_000)
    if len(text) > limit:
        raise ValueError(f"Ліміт цього профілю — {limit} символів; введено {len(text)}.")
    doc = text_document(
        text, resources.codec, resources.detector,
        include_residual=config.get("include_residual", True) if mode == "legacy" else False,
    )
    if mode == "field_aware":
        doc = field_document(text, doc)
    # Append the input so the original controller's self/proxy features are
    # defined. Exact duplicates are excluded from the neighbour pseudo-targets.
    pool = [*resources.background_texts, text]
    pos = len(pool) - 1
    index = BM25Index(max_features=config.get("bm25_max_features", 20000)).fit(pool)
    index.exact_text_targets = {pos: [i for i, value in enumerate(pool) if value == text]}
    router_class = ContentGuardRouter if mode == "field_aware" else ToolRouter
    router = router_class(resources.codec, config, llm=None)
    settings = {
        "max_steps": config.get("max_steps"),
        "utility_floor": config.get("action_lexical_floor", 0.45),
        "content_floor": config.get("action_content_floor", 0.8),
        "min_self_gain": config.get("plan_min_proxy_gain", 0.01),
        "self_rr_target": config.get("action_self_rr_target", 0.5),
        "skill_floor": config.get("action_skill_floor", 0.5),
        "plan_check": config.get("plan_check", "neighbor_proxy"),
        "neighbor_k": config.get("neighbor_k", 3),
    }
    release = release_with_policy
    if config.get("policy_mode") == "sequential":
        release = release_sequential
        settings["continue_on_keep"] = config.get("sequential_continue_on_keep", False) if mode == "legacy" else False
    output, actions, trace, audit = release(
        doc, router, resources.model, index, resources.retention_vectorizer, pos, **settings
    )
    rows = [("adaptive_model", output, actions, trace, audit)]
    # Baselines keep their original action space; only the adaptive variant is
    # constrained. Avoid claiming this is the original v4c protocol.
    fixed_router_class = FieldToolRouter if mode == "field_aware" else ToolRouter
    fixed_router = fixed_router_class(resources.codec, config, llm=None)
    for name in ("L1_scrub_text", "L2_generalize", "L3_strong", "mask_detected"):
        fixed_text, fixed_actions, fixed_audit = execute_fixed(doc, fixed_router, name)
        rows.append((name, fixed_text, fixed_actions, [], fixed_audit))

    entities = [
        {
            "unit": i, "text": text[u.start:u.end], "label": u.label,
            "role": u.role, "group": u.group, "source": u.source, "start": u.start, "end": u.end,
        }
        for i, u in enumerate(doc.units)
    ]
    in_vocabulary = resources.retention_vectorizer.transform([text]).nnz > 0
    releases = {}
    for name, output, actions, trace, audit in rows:
        by_unit = {row["unit"]: row for row in audit}
        edits = []
        for i, unit in enumerate(doc.units):
            row = by_unit.get(i, {})
            action = actions.get(i)
            applied = action is not None and action.replacement != text[unit.start:unit.end]
            edits.append({
                **entities[i],
                "decision": row.get("decision", row.get("action", "keep")),
                "status": row.get("status", "unchanged"),
                "applied": applied,
                "replacement": action.replacement if action else text[unit.start:unit.end],
                "tool": action.tool if action else "not_called",
                "risk_score": row.get("risk_score"),
                "confidence": row.get("confidence"),
            })
        retention = (
            float(lexical_retention(resources.retention_vectorizer, text, [output])[0])
            if in_vocabulary else None
        )
        releases[name] = DemoRelease(output, edits, retention, trace)
    return EntityDemoResult(text, entities, releases, mode)
