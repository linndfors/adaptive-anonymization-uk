"""The learned controller: candidate choice is driven by risk regression."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .action_learning import lexical_retention
from .entity_demo import DemoRelease, EntityDemoResult
from .learned_resume_risk import VERSION
from .resume_adaptive import DirectIdentifierError, check_direct_identifiers, options, planned_edits, render
from .resume_parser import parse_resume


def learned_search(doc, estimator, *, utility_floor=.55, target=None, min_gain=.001):
    target = estimator.target if target is None else target
    if not 0 <= target <= 1 or not 0 <= utility_floor <= 1 or min_gain < 0:
        raise ValueError("Invalid learned-controller settings")
    policy, trace = {}, []
    risk = float(estimator.predict(doc, [policy])[0])
    output = render(doc, policy)
    utility = float(lexical_retention(estimator.retention_vectorizer, doc.text, [output])[0])
    groups = sorted({u.group for u in doc.units if len(options(doc, u)) > 1})
    while risk > target:
        candidates, texts, actions = [], [], []
        for group in groups:
            seen = {output}
            for level in range(policy.get(group, 0) + 1, 4):
                candidate = {**policy, group: level}
                text = render(doc, candidate)
                if text not in seen:
                    candidates.append(candidate)
                    texts.append(text)
                    actions.append((group, level))
                    seen.add(text)
        if not candidates:
            break
        predictions = estimator.predict(doc, candidates)
        utilities = lexical_retention(estimator.retention_vectorizer, doc.text, texts)
        gain = risk - predictions
        valid = (gain > min_gain) & (utilities >= utility_floor)
        if not valid.any():
            break
        merit = gain / (np.maximum(0., utility - utilities) + .01)
        j = int(np.argmax(np.where(valid, merit, -np.inf)))
        group, level = actions[j]
        policy, output, risk, utility = candidates[j], texts[j], float(predictions[j]), float(utilities[j])
        trace.append({"group": group, "level": level,
                      "operation": ("keep", "generalize", "generalize_broad", "mask")[level],
                      "predicted_risk": risk, "retention": utility, "decision_source": "risk_regression"})
    trace.append({"stop_reason": "predicted_target_met" if risk <= target else "predicted_target_unmet",
                  "predicted_risk": risk, "retention": utility})
    return policy, trace


def anonymize_learned(resources, text, *, extractor=None):
    estimator = getattr(resources, "risk_estimator", None)
    if estimator is None:
        raise FileNotFoundError("Trained risk regressor is required; run scripts/train_resume_risk.py")
    doc = parse_resume(text, estimator.codec, resources.detector, extractor=extractor)
    floor = float(resources.config["adaptive"]["utility_floor"])
    policy, trace = learned_search(doc, estimator, utility_floor=floor)
    output = render(doc, policy)
    direct = check_direct_identifiers(output)
    if direct["issues"]:
        raise DirectIdentifierError("Known direct-identifier patterns remain in the proposed release")
    after_direct, after = estimator.predict(doc, [{}, policy])
    # Mandatory masking is built into render(); no unmasked risk is invented.
    entities = [{"unit": i, "text": text[u.start:u.end], **asdict(u)} for i, u in enumerate(doc.units)]
    edits = [{**entity, **edit, "decision_source": "mandatory_rule" if entity["role"] in {"DIRECT", "SENSITIVE"}
              else "risk_regression"} for entity, edit in zip(entities, planned_edits(doc, policy))]
    retention = float(lexical_retention(estimator.retention_vectorizer, text, [output])[0])
    metrics = {"risk_before": None, "risk_after_direct_masking": float(after_direct), "risk_after": float(after),
               "risk_target": estimator.target, "lexical_retention": retention,
               "risk_kind": "predicted_worst_same_person_reciprocal_rank", "risk_probability": None}
    return EntityDemoResult(text, entities, {"adaptive": DemoRelease(output, edits, retention, trace)}, "adaptive", {
        "method_version": VERSION, "controller": "regression", "stop_reason": trace[-1]["stop_reason"],
        "metrics": metrics, "policy": policy, "direct_check": direct,
        "pool_size": 0, "live_reference_pool_used": False, "proxy_assessable": False,
        "training_reference_documents": estimator.statistics.documents,
        "risk_model_schema": estimator.schema, "risk_target": estimator.target, "utility_floor": floor,
        "model_validation": estimator.metadata.get("validation", {})})
