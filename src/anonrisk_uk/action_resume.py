"""Autonomous counterfactual action-value regression for resume anonymization.

Only training uses linked people and attacker corpora. Deployment predicts the
benefit of an edit from the current document and frozen training statistics.
There is deliberately no absolute-risk stopping threshold or live-search gate.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np

from .action_learning import lexical_retention
from .adaptive_text import ReleaseDocument
from .entity_demo import DemoRelease, EntityDemoResult
from .learned_resume_risk import TOKENIZER, fingerprint
from .resume_adaptive import DirectIdentifierError, check_direct_identifiers, options, planned_edits, render
from .resume_parser import parse_resume
from .workua import norm

VERSION = "adaptive_resume_v5_action_regression"
SCHEMA = "counterfactual_action_value_v1"
PROFESSIONAL = {"skills", "title", "exp_position", "exp_resp", "edu_faculty", "add_education"}
DEFAULT_SETTINGS = {"utility_floor": .55, "detail_floor": .35, "cost_weight": .3,
                    "min_gain": .005, "max_steps": 12, "fragment_limit": 8}


def level_for(doc, policy, index):
    unit = doc.units[index]
    return int(policy.get(f"@{index}", policy.get(unit.group, 0)))


@dataclass
class EditCandidate:
    policy: dict
    group: str
    level: int
    units: tuple
    scope: str


def detail_retention(doc, policy):
    """Typed detail proxy; not semantic similarity or factual preservation.

    Generalization receives partial credit and masking none. This soft cost does
    not exempt professional information from privacy transformations.
    """
    edits = planned_edits(doc, policy)
    numerator, denominator = 0., 0.
    for i, (unit, edit) in enumerate(zip(doc.units, edits)):
        if unit.group not in PROFESSIONAL or unit.role != "QUASI":
            continue
        weight = math.sqrt(max(1, min(50, len(TOKENIZER(doc.text[unit.start:unit.end])))))
        denominator += weight
        if not edit["applied"]:
            credit = 1.
        elif edit.get("decision", "").startswith("mask") or re.fullmatch(r"\[[^]]*\]", edit["replacement"]):
            credit = 0.
        else:
            credit = .7 if level_for(doc, policy, i) == 1 else .4
        numerator += weight * credit
    return numerator / denominator if denominator else 1.


@dataclass
class ActionResumeRegressor:
    model: object
    statistics: object
    codec: object
    retention_vectorizer: object
    settings: dict
    metadata: dict
    schema: str = SCHEMA
    version: str = VERSION

    def predict_records(self, records):
        if self.schema != SCHEMA:
            raise ValueError("Unsupported action feature schema")
        values = np.asarray(self.model.predict(records), dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Action model returned non-finite gains")
        return values


class ActionFeatures:
    """Per-request cache, containing no population documents or search index."""

    def __init__(self, doc, estimator):
        self.source_doc, self.estimator = doc, estimator
        # Opt-in rendering caches live on a private immutable view. Never mutate
        # a caller's parsed document or keep submitted text in a global cache.
        self.doc = ReleaseDocument(doc.text, tuple(doc.units))
        self.doc._anonrisk_options = {}
        self.doc._anonrisk_edits = {}
        self.states, self.feature_cache = {}, {}
        self.outputs = {}
        self.original_vector = estimator.retention_vectorizer.transform([doc.text])
        self.record_cache, self.context_cache, self.prediction_cache = {}, {}, {}
        self.groups = defaultdict(list)
        self.values = {}
        for i, unit in enumerate(self.doc.units):
            if unit.role == "QUASI" and len(options(self.doc, unit)) > 1:
                self.groups[unit.group].append(i)
                self.values[i] = self.value_features([i])

    def value_features(self, indices):
        doc, estimator = self.doc, self.estimator
        values = [doc.text[doc.units[i].start:doc.units[i].end] for i in indices]
        tokens = [t for value in values for t in TOKENIZER(value)]
        unseen = math.log1p(2 * estimator.statistics.documents)
        # Put singletons and unseen words in the same rare-support regime;
        # avoid letting a training person's own pair masquerade as population
        # evidence that will be absent for an unseen person at deployment.
        idf_cap = math.log1p(estimator.statistics.documents / 3)
        idfs = [min(idf_cap, estimator.statistics.token_idf.get(fingerprint(t), unseen)) for t in tokens] or [0.]
        counts = [estimator.codec.population_frequency(doc.units[i].group, value)
                  for i, value in zip(indices, values)]
        frequencies = [count if count >= 5 else 0 for count in counts]
        return {"value_tokens": len(tokens), "value_chars": sum(map(len, values)),
                "value_idf_mean": float(np.mean(idfs)), "value_idf_max": float(np.max(idfs)),
                "value_oov": sum(fingerprint(t) not in estimator.statistics.token_idf for t in tokens)
                             / max(1, len(tokens)),
                "value_frequency_min": math.log1p(min(frequencies, default=0)),
                "value_frequency_mean": math.log1p(float(np.mean(frequencies)) if frequencies else 0.),
                "value_numbers": sum(t.isdigit() for t in tokens),
                "value_years": len(re.findall(r"\b(?:19|20)\d{2}\b", " ".join(values)))}

    def output(self, policy):
        key = tuple(sorted(policy.items()))
        if key not in self.outputs:
            self.outputs[key] = render(self.doc, policy)
        return self.outputs[key]

    def prefetch(self, policies):
        missing = {tuple(sorted(policy.items())): policy for policy in policies
                   if tuple(sorted(policy.items())) not in self.states}
        if not missing:
            return
        items = list(missing.items())
        texts = [self.output(policy) for _, policy in items]
        utilities = np.asarray(self.estimator.retention_vectorizer.transform(texts).multiply(
            self.original_vector).sum(axis=1)).ravel()
        statistics = self.estimator.statistics
        latents = np.zeros((len(items), 32))
        if statistics.projection is not None:
            visible = [re.sub(r"\[[^\]]*\]", " ", text) for text in texts]
            values = statistics.projection.transform(statistics.vectorizer.transform(visible))
            latents[:, :values.shape[1]] = values
        for i, ((key, policy), text) in enumerate(zip(items, texts)):
            self.states[key] = (text, float(utilities[i]), detail_retention(self.doc, policy))
            self.feature_cache[key] = statistics.features(self.doc, policy, _output=text, _latent=latents[i])

    def state(self, policy):
        key = tuple(sorted(policy.items()))
        self.prefetch([policy])
        return self.states[key]

    def state_features(self, policy):
        key = tuple(sorted(policy.items()))
        self.prefetch([policy])
        return self.feature_cache[key]

    def candidates(self, policy, *, fragment_limit=None):
        fragment_limit = (self.estimator.settings.get("fragment_limit", 8)
                          if fragment_limit is None else fragment_limit)
        before = self.output(policy)
        seen = {before}
        for group, indices in sorted(self.groups.items()):
            for level in range(int(policy.get(group, 0)) + 1, 4):
                affected = tuple(i for i in indices if level_for(self.doc, policy, i) < level)
                if not affected:
                    continue
                overridden = {f"@{i}" for i in indices if policy.get(f"@{i}", 0) <= level}
                changed = {k: v for k, v in policy.items() if k not in overridden}
                changed[group] = level
                output = self.output(changed)
                if output not in seen:
                    seen.add(output)
                    yield EditCandidate(changed, group, level, affected, "group")
        linked = defaultdict(list)
        for group, indices in self.groups.items():
            if len(indices) < 2:
                continue
            for i in indices:
                u = self.doc.units[i]
                linked[(group, norm(self.doc.text[u.start:u.end]))].append(i)
        # An inference-only deterministic bound, independent of outcomes. Cover
        # rare fragments first, retaining whole-group candidates for every group.
        fragments = sorted(linked.values(), key=lambda ii: (-max(self.values[i]["value_idf_mean"] for i in ii), ii[0]))
        for indices in fragments[:fragment_limit]:
            group = self.doc.units[indices[0]].group
            if len(indices) == len(self.groups[group]):
                continue
            current = max(level_for(self.doc, policy, i) for i in indices)
            for level in range(current + 1, 4):
                changed = {**policy, **{f"@{i}": level for i in indices}}
                output = self.output(changed)
                if output not in seen:
                    seen.add(output)
                    yield EditCandidate(changed, group, level, tuple(indices), "fragment")

    def records(self, policy, candidates):
        self.prefetch([policy, *[c.policy for c in candidates]])
        _, utility, detail = self.state(policy)
        base = self.state_features(policy)
        records = []
        for candidate in candidates:
            cache_key = (tuple(sorted(policy.items())), tuple(sorted(candidate.policy.items())), candidate.scope)
            if cache_key in self.record_cache:
                records.append(self.record_cache[cache_key])
                continue
            _, after_utility, after_detail = self.state(candidate.policy)
            after = self.state_features(candidate.policy)
            indices = candidate.units
            record = {**{f"state_{key}": value for key, value in base.items()},
                      **{f"delta_{key}": after[key] - value for key, value in base.items()},
                      **self.value_features(indices),
                      "group": candidate.group, "scope": candidate.scope,
                      "label": self.doc.units[indices[0]].label,
                      "level": candidate.level,
                      "current_level": max(level_for(self.doc, policy, i) for i in indices),
                      "affected_units": len(indices), "group_units": len(self.groups[candidate.group]),
                      "affected_share": len(indices) / max(1, len(self.doc.units)),
                      "utility_before": utility, "utility_after": after_utility,
                      "lexical_loss": utility - after_utility,
                      "detail_before": detail, "detail_after": after_detail,
                      "detail_loss": detail - after_detail,
                      "professional_action": float(candidate.group in PROFESSIONAL)}
            records.append(record)
            self.record_cache[cache_key] = record
        return records

    def context(self, policy, fragment_limit=None):
        fragment_limit = self.estimator.settings.get("fragment_limit", 8) if fragment_limit is None else fragment_limit
        key = (tuple(sorted(policy.items())), fragment_limit)
        if key not in self.context_cache:
            candidates = list(self.candidates(policy, fragment_limit=fragment_limit))
            self.context_cache[key] = (candidates, self.records(policy, candidates))
        return self.context_cache[key]

    def predictions(self, policy, candidates, records):
        key = (id(self.estimator.model), tuple(sorted(policy.items())),
               tuple(tuple(sorted(c.policy.items())) for c in candidates))
        if key not in self.prediction_cache:
            self.prediction_cache[key] = self.estimator.predict_records(records)
        return self.prediction_cache[key]


def action_search(doc, estimator, *, _features=None, **overrides):
    settings = {**DEFAULT_SETTINGS, **estimator.settings, **overrides}
    if (not 0 <= settings["utility_floor"] <= 1 or not 0 <= settings["detail_floor"] <= 1
            or settings["max_steps"] < 0 or settings["cost_weight"] < 0
            or settings["min_gain"] < 0 or settings["fragment_limit"] < 0):
        raise ValueError("Invalid action controller settings")
    features = _features if _features is not None else ActionFeatures(doc, estimator)
    if features.source_doc is not doc or features.estimator is not estimator:
        raise ValueError("Cached features belong to another inference request")
    policy, trace = {}, []
    reason = "max_steps"
    for _ in range(settings["max_steps"]):
        candidates, records = features.context(policy, settings["fragment_limit"])
        feasible = np.array([features.state(c.policy)[1] >= settings["utility_floor"]
                             and features.state(c.policy)[2] >= settings["detail_floor"] for c in candidates])
        if not feasible.any():
            reason = "no_feasible_action"
            break
        gains = features.predictions(policy, candidates, records)
        costs = np.array([.5 * max(0., r["lexical_loss"]) + .5 * max(0., r["detail_loss"]) for r in records])
        net = gains - settings["cost_weight"] * costs
        j = int(np.argmax(np.where(feasible, net, -np.inf)))
        if net[j] <= settings["min_gain"]:
            reason = "no_predicted_beneficial_action"
            break
        chosen = candidates[j]
        policy = chosen.policy
        trace.append({"group": chosen.group, "level": chosen.level, "scope": chosen.scope,
                      "units": list(chosen.units), "predicted_gain": float(gains[j]),
                      "net_value": float(net[j]), "retention": features.state(policy)[1],
                      "detail_retention": features.state(policy)[2],
                      "operation": ("keep", "generalize", "generalize_broad", "mask")[chosen.level],
                      "decision_source": "counterfactual_action_regression"})
    trace.append({"stop_reason": reason, "retention": features.state(policy)[1],
                  "detail_retention": features.state(policy)[2], "absolute_risk_assessed": False})
    return policy, trace


def anonymize_action(resources, text, *, extractor=None):
    estimator = getattr(resources, "risk_estimator", None)
    if not isinstance(estimator, ActionResumeRegressor):
        raise FileNotFoundError("A trained action-value regressor is required")
    doc = parse_resume(text, estimator.codec, resources.detector, extractor=extractor)
    overrides = resources.config.get("adaptive", {})
    policy, trace = action_search(doc, estimator, **overrides)
    output = render(doc, policy)
    direct = check_direct_identifiers(output)
    if direct["issues"]:
        raise DirectIdentifierError("Known direct-identifier patterns remain in the proposed release")
    entities = [{"unit": i, "text": text[u.start:u.end], **asdict(u)} for i, u in enumerate(doc.units)]
    edits = [{**entity, **edit, "decision_source": "mandatory_rule" if entity["role"] in {"DIRECT", "SENSITIVE"}
              else "counterfactual_action_regression"} for entity, edit in zip(entities, planned_edits(doc, policy))]
    retention = float(lexical_retention(estimator.retention_vectorizer, text, [output])[0])
    return EntityDemoResult(text, entities, {"adaptive": DemoRelease(output, edits, retention, trace)}, "adaptive", {
        "method_version": VERSION, "controller": "action_regression", "stop_reason": trace[-1]["stop_reason"],
        "policy": policy, "metrics": {"risk_before": None, "risk_after": None, "risk_target": None,
            "risk_probability": None, "risk_kind": "not_assessed_by_action_value_model",
            "lexical_retention": retention, "professional_detail_proxy": detail_retention(doc, policy)},
        "direct_check": direct, "pool_size": 0, "live_reference_pool_used": False,
        "proxy_assessable": False, "utility_floor": overrides.get("utility_floor", estimator.settings.get("utility_floor", .55)),
        "model_validation": estimator.metadata.get("validation", {})})
