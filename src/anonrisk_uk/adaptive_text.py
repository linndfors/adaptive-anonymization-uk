"""Shared controller for structured fields and detected or residual text units.

The regressor learns train-only S2 unit contributions and ranks candidate actions.
An independent self-linkage ensemble verifies release decisions. Test S2 outcomes
never control transformations.
"""

from __future__ import annotations

import math
import re
from collections import OrderedDict
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .attack import ranking_result
from .field_typing import COARSE, LOCATION_GROUPS, employer_parent, type_location
from .pipeline import MASK, generalize_span, select_spans, span_role
from .taxonomy import ROLE_OF_CLASS
from .workua import FREE_TEXT_GROUPS, GROUP_LABELS, GROUPS, SENSITIVE_GROUPS, norm

GROUP_CLASS = {
    "title": "OCCUPATION_EDU",
    "exp_position": "OCCUPATION_EDU",
    "skills": "OCCUPATION_EDU",
    "edu_faculty": "OCCUPATION_EDU",
    "exp_industry": "OCCUPATION_EDU",
    "age": "DEMOGRAPHIC",
    "salary": "FINANCIAL",
    "city": "LOCATION",
    "exp_city": "LOCATION",
    "edu_city": "LOCATION",
    "exp_company": "ORGANIZATION",
    "edu_institution": "ORGANIZATION",
    "add_education": "ORGANIZATION",
    "exp_dates": "DATE_TIME",
    "edu_years": "DATE_TIME",
    "languages": "OTHER_QUASI",
    "driver_license": "OTHER_QUASI",
    "veteran": "HEALTH",
    "disability": "HEALTH",
    "employment": "CONTEXT",
}


@dataclass(frozen=True)
class TextUnit:
    start: int
    end: int
    label: str
    role: str
    group: str
    source: str
    generalized: str | None = None


@dataclass
class ReleaseDocument:
    text: str
    units: list[TextUnit]


def _residual_units(text, begin, end, group):
    # Residual text may also identify a person. Split it into sentences or
    # paragraphs, then split very long sentences without dropping words.
    out = []
    for match in re.finditer(r"[^.!?\n]+(?:[.!?]+|$)", text[begin:end]):
        a, b = begin + match.start(), begin + match.end()
        words = list(re.finditer(r"\S+", text[a:b]))
        if not words:
            continue
        first = 0
        for i in range(1, len(words) + 1):
            if i == len(words) or words[i].end() - words[first].start() > 600:
                start, stop = a + words[first].start(), a + words[i - 1].end()
                if re.search(r"\w", text[start:stop]):
                    out.append(TextUnit(start, stop, "OTHER_QUASI", "QUASI", group, "residual_text"))
                first = i
    return out


def make_document(ex, codec, detector, view="hybrid", exclude=(), include_residual=True):
    if view not in {"hybrid", "structured", "free_text"}:
        raise ValueError(view)
    parts, units, offset = [], [], 0
    for group in GROUPS:
        if (
            group in exclude
            or (view == "structured" and group in FREE_TEXT_GROUPS)
            or (view == "free_text" and group not in FREE_TEXT_GROUPS)
        ):
            continue
        values = ex.get(group, [])
        if not values:
            continue
        prefix = (" " if parts else "") + GROUP_LABELS[group] + ": "
        parts.append(prefix)
        offset += len(prefix)
        for j, value in enumerate(values):
            if j:
                parts.append("; ")
                offset += 2
            start, end = offset, offset + len(value)
            parts.append(value)
            if group in FREE_TEXT_GROUPS:
                selected = select_spans(value, detector(value))
                previous = 0
                for s in selected:
                    if include_residual:
                        units.extend(_residual_units("".join(parts), start + previous, start + s.start, group))
                    units.append(
                        TextUnit(
                            start + s.start,
                            start + s.end,
                            s.label,
                            span_role(s),
                            group,
                            s.source,
                            generalize_span(s.label, value[s.start : s.end]),
                        )
                    )
                    previous = s.end
                if include_residual:
                    units.extend(_residual_units("".join(parts), start + previous, end, group))
            else:
                label = GROUP_CLASS.get(group, "OTHER_QUASI")
                role = "SENSITIVE" if group in SENSITIVE_GROUPS else ROLE_OF_CLASS.get(label, "QUASI")
                source, kind = "structured", None
                if group in LOCATION_GROUPS:
                    # Type the value, not the column: "ТОВ" or "Дистанційно" is not a place.
                    kind, provenance = type_location(value, getattr(codec, "settlements", ()))
                    (label, role), source = COARSE[kind], f"structured:{provenance}"
                if group in {"age", "salary", "exp_dates", "edu_years", "exp_company", "edu_institution"}:
                    generalized = codec.generalize(group, [value])
                    generalized = "; ".join(generalized) if generalized else None
                elif kind in {"EMPLOYER_TYPE", "EMPLOYER"}:
                    generalized = employer_parent(value)
                else:
                    generalized = generalize_span(label, value)
                if generalized == value:
                    generalized = None
                units.append(TextUnit(start, end, label, role, group, source, generalized))
            offset = end
        parts.append(".")
        offset += 1
    text = "".join(parts)
    units.sort(key=lambda u: (u.start, u.end))
    if any(a.end > b.start for a, b in pairwise(units)):
        raise ValueError("Transformation units overlap")
    return ReleaseDocument(text, units)


def make_free_text_document(text, detector, group="additional_info", include_residual=True):
    """Create a release document without adding synthetic field labels to user text.

    This is the inference counterpart of ``make_document`` for a single free-text
    input. Offsets always refer to the exact string supplied by the user.
    """
    if group not in FREE_TEXT_GROUPS:
        raise ValueError(f"Expected a free-text group, received: {group}")
    selected = select_spans(text, detector(text))
    units, previous = [], 0
    for span in selected:
        if include_residual:
            units.extend(_residual_units(text, previous, span.start, group))
        units.append(
            TextUnit(
                span.start,
                span.end,
                span.label,
                span_role(span),
                group,
                span.source,
                generalize_span(span.label, text[span.start : span.end]),
            )
        )
        previous = span.end
    if include_residual:
        units.extend(_residual_units(text, previous, len(text), group))
    units.sort(key=lambda unit: (unit.start, unit.end))
    if any(left.end > right.start for left, right in pairwise(units)):
        raise ValueError("Transformation units overlap")
    return ReleaseDocument(text, units)


def mandatory_actions(doc):
    return {i: "mask" for i, u in enumerate(doc.units) if u.role in {"DIRECT", "SENSITIVE"}}


def linked_units(doc, index):
    """Link repeated units with the same class and exact normalised value."""
    unit = doc.units[index]
    key = norm(doc.text[unit.start : unit.end])
    return [
        i
        for i, u in enumerate(doc.units)
        if u.label == unit.label and u.role == unit.role and norm(doc.text[u.start : u.end]) == key
    ]


def apply_actions(doc, actions):
    out, last = [], 0
    for i, u in enumerate(doc.units):
        action = actions.get(i, "keep")
        if action == "keep":
            continue
        if action not in {"generalize", "mask"}:
            raise ValueError(action)
        replacement = u.generalized if action == "generalize" else MASK.get(u.label, "[REDACTED]")
        if replacement is None:
            raise ValueError("No generalization is available for this unit")
        out.extend([doc.text[last : u.start], replacement])
        last = u.end
    out.append(doc.text[last:])
    return "".join(out)


def unit_features(doc, reference, bm25):
    from collections import Counter

    analyzer, vocab = bm25.vec.build_analyzer(), bm25.vec.vocabulary_
    n_tokens = max(1, len(analyzer(doc.text)))
    mentions = Counter((u.label, u.role, norm(doc.text[u.start : u.end])) for u in doc.units)
    rows = []
    for u in doc.units:
        value = doc.text[u.start : u.end]
        ids = [vocab[t] for t in analyzer(value) if t in vocab]
        idf = bm25.idf[ids] if ids else np.array([0.0])
        rows.append(
            {
                "n_chars": len(value),
                "n_tokens": len(analyzer(value)),
                "n_mentions": mentions[(u.label, u.role, norm(value))],
                "share_tokens": len(analyzer(value)) / n_tokens,
                "log_frequency": math.log1p(reference.frequency(u.label, value)),
                "max_idf": float(idf.max()),
                "mean_idf": float(idf.mean()),
                "doc_tokens": n_tokens,
                "n_units": len(doc.units),
                "structured": float(u.source.startswith("structured")),
                "residual": float(u.source == "residual_text"),
                "can_generalize": float(u.generalized is not None),
                **{f"class_{label}": float(label == u.label) for label in [*ROLE_OF_CLASS, "OTHER_QUASI"]},
                **{f"role_{role}": float(role == u.role) for role in ["DIRECT", "SENSITIVE", "QUASI", "CONTEXT"]},
                **{f"group_{g}": float(g == u.group) for g in GROUPS},
            }
        )
    return pd.DataFrame(rows)


class UnitRiskModel:
    """Estimate a unit's attack contribution; this is not a calibrated probability.

    The class-level prior is deliberately separated from the contextual residual.
    Setting ``context_weight`` to zero therefore gives the exact category baseline,
    which makes validation-time fallback explicit and reproducible.
    """

    def __init__(
        self, *, max_iter=150, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=10, l2_regularization=0.0
    ):
        self.model_params = {
            "max_iter": int(max_iter),
            "learning_rate": float(learning_rate),
            "max_leaf_nodes": int(max_leaf_nodes),
            "min_samples_leaf": int(min_samples_leaf),
            "l2_regularization": float(l2_regularization),
        }
        if self.model_params["max_iter"] < 1 or self.model_params["max_leaf_nodes"] < 2:
            raise ValueError("Invalid gradient-boosting configuration")
        if self.model_params["min_samples_leaf"] < 1 or self.model_params["learning_rate"] <= 0:
            raise ValueError("Invalid gradient-boosting configuration")

    def fit(self, features, labels, seed=42, sample_weight=None):
        if len(features) < 10 or not np.isfinite(labels).all():
            raise ValueError("Training contributions are missing or invalid")
        self.columns = list(features)
        categories = features.filter(regex="^class_").idxmax(axis=1).str.removeprefix("class_")
        weights = np.ones(len(labels)) if sample_weight is None else np.asarray(sample_weight)
        prior = (
            pd.DataFrame({"label": categories.to_numpy(), "weight": weights, "weighted_target": weights * labels})
            .groupby("label")
            .sum()
        )
        self.class_prior = (prior.weighted_target / prior.weight).clip(lower=0).to_dict()
        baseline = categories.map(self.class_prior).fillna(0).to_numpy()
        self.context_weight = 1.0
        self.model = HistGradientBoostingRegressor(
            **self.model_params,
            random_state=seed,
        ).fit(features, labels - baseline, sample_weight=sample_weight)
        return self

    def predict(self, features):
        categories = features.filter(regex="^class_").idxmax(axis=1).str.removeprefix("class_")
        baseline = categories.map(self.class_prior).fillna(0).to_numpy()
        return baseline + self.context_weight * self.model.predict(features[self.columns])


def train_unit_contributions(pop, positions, documents, reference, progress=lambda x: x):
    """Create S2 contribution labels in two counterfactual contexts.

    ``forward_delta`` measures the risk reduction caused by masking the unit in
    a lightly masked context. For direct or sensitive units, that context masks
    every other mandatory unit but deliberately keeps the evaluated unit.
    ``addback_delta`` measures the increase after restoring the unit into an
    otherwise strongly masked document. Their mean is the regression target.
    Context-only units are excluded, and no test observation is used.
    """
    X, rows = [], []
    for k in progress(positions):
        doc = documents[int(k)]
        base = mandatory_actions(doc)
        strong = {i: "mask" for i, u in enumerate(doc.units) if u.role != "CONTEXT"}
        candidates = [
            i for i, unit in enumerate(doc.units) if unit.role != "CONTEXT" and i == min(linked_units(doc, i))
        ]
        queries = [apply_actions(doc, base), apply_actions(doc, strong)]
        mandatory = set(base)
        for i in candidates:
            linked = linked_units(doc, i)
            if i in mandatory:
                forward_query = apply_actions(doc, {j: action for j, action in base.items() if j not in linked})
            else:
                forward_query = apply_actions(doc, {**base, **{j: "mask" for j in linked}})
            addback_query = apply_actions(doc, {j: action for j, action in strong.items() if j not in linked})
            queries += [forward_query, addback_query]
        targets, excludes = pop.targets_and_excludes([int(k)] * len(queries), "S2")
        rr = ranking_result(pop.index, queries, targets, excludes, batch_size=16)[0].reciprocal_ranks
        features = unit_features(doc, reference, pop.index)
        for j, i in enumerate(candidates):
            forward = (rr[2 + 2 * j] - rr[0]) if i in mandatory else (rr[0] - rr[2 + 2 * j])
            addback = rr[3 + 2 * j] - rr[1]
            rows.append(
                {
                    "pos": int(k),
                    "person": str(pop.clusters[k]),
                    "unit": i,
                    "label": doc.units[i].label,
                    "role": doc.units[i].role,
                    "source": doc.units[i].source,
                    "group": doc.units[i].group,
                    "forward_delta": forward,
                    "addback_delta": addback,
                    "target_delta": 0.5 * (forward + addback),
                }
            )
            X.append(features.iloc[i])
    return pd.DataFrame(X).reset_index(drop=True), pd.DataFrame(rows)


class TextRiskController:
    """Evaluate self-linkage under several attackers without exposing S2 targets."""

    def __init__(self, indexes, require_margin=False, cache_size=4096):
        self.indexes, self.require_margin = indexes, require_margin
        self.cache_size = cache_size
        self._cache = OrderedDict()

    def assess(self, pos, queries):
        keys = [(int(pos), text) for text in queries]
        missing = list(dict.fromkeys(key for key in keys if key not in self._cache))
        if not missing:
            values = [self._cache[key] for key in keys]
            return np.array([v[0] for v in values]), np.array([v[1] for v in values])
        rr, margins = [], []
        for index in self.indexes.values():
            result, margin, _ = ranking_result(
                index, [key[1] for key in missing], [[int(pos)]] * len(missing), batch_size=16
            )
            rr.append(result.reciprocal_ranks)
            margins.append(margin)
        fresh = dict(zip(missing, zip(np.max(rr, axis=0), np.max(margins, axis=0))))
        values = [fresh[key] if key in fresh else self._cache[key] for key in keys]
        self._cache.update(fresh)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return np.array([v[0] for v in values]), np.array([v[1] for v in values])

    def meets(self, risk, margin, tau):
        return (risk <= tau) & ((margin <= 0) | (not self.require_margin) | (tau >= 1))


def adaptive_release(
    doc,
    pos,
    controller,
    model,
    reference,
    bm25,
    retention_vec,
    *,
    tau=0.2,
    strategy="model",
    utility_floor=0.2,
    max_steps=None,
    candidate_limit=8,
    expand_on_plateau=True,
    max_candidates_per_step=None,
    progress_steps=False,
):
    """Rank actions by the model and choose measured gain divided by utility loss.

    Context and undetected sentences remain represented as residual units. The
    function reports utility-floor and unreachable-threshold outcomes explicitly;
    neither outcome is presented as a privacy guarantee.
    """
    if not 0 < tau <= 1 or not 0 <= utility_floor <= 1 or candidate_limit < 1:
        raise ValueError("Invalid release parameters")
    if max_steps is not None and (isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 0):
        raise ValueError("max_steps must be a non-negative integer or None")
    if max_candidates_per_step is not None and max_candidates_per_step < 1:
        raise ValueError("max_candidates_per_step must be positive or None")
    # Every quasi unit follows keep -> generalize -> mask at most once. This is a
    # finite action bound, not an arbitrary limit on resume length.
    action_bound = sum(1 + int(u.generalized is not None) for u in doc.units if u.role == "QUASI")
    step_limit = action_bound if max_steps is None else min(max_steps, action_bound)
    actions = mandatory_actions(doc)
    if strategy == "category":
        prediction = np.array([model.class_prior.get(u.label, 0) for u in doc.units])
    elif strategy == "model":
        features = unit_features(doc, reference, bm25)
        prediction = np.maximum(model.predict(features), 0) if len(features) else np.array([])
    elif strategy == "greedy":
        prediction = np.zeros(len(doc.units))
    else:
        raise ValueError(strategy)
    original = retention_vec.transform([doc.text])

    def retention(texts):
        return np.asarray(retention_vec.transform(texts).multiply(original).sum(axis=1)).ravel()

    trace = []
    reason = "max_steps"
    iterator = range(step_limit + 1)
    if progress_steps:
        from tqdm.auto import tqdm

        iterator = tqdm(iterator, desc=f"Resume {pos}: {strategy}, {len(doc.units)} units", leave=False)
    # Resolve repeated mentions once rather than in every candidate loop.
    links = {}
    for i, u in enumerate(doc.units):
        links.setdefault((u.label, u.role, norm(doc.text[u.start : u.end])), []).append(i)
    by_unit = {i: tuple(indices) for indices in links.values() for i in indices}
    for step in iterator:
        text = apply_actions(doc, actions)
        risk, margin = controller.assess(pos, [text])
        current_retention = float(retention([text])[0])
        met = bool(controller.meets(risk, margin, tau)[0])
        trace.append(
            {
                "step": step,
                "proxy_rr": float(risk[0]),
                "margin": float(margin[0]),
                "retention": current_retention,
                "threshold_met": met,
            }
        )
        if progress_steps:
            iterator.set_postfix(risk=round(float(risk[0]), 3), retention=round(current_retention, 3))
        if met:
            reason = "threshold_met"
            break
        if step == step_limit:
            if step_limit == action_bound:
                reason = "actions_exhausted"
            break
        candidates = []
        seen_links = set()
        for i, u in enumerate(doc.units):
            if u.role != "QUASI" or actions.get(i) == "mask":
                continue
            operation = "generalize" if i not in actions and u.generalized else "mask"
            linked = by_unit[i]
            if linked in seen_links:
                continue
            seen_links.add(linked)
            if operation == "generalize" and any(doc.units[j].generalized is None for j in linked):
                operation = "mask"
            candidates.append((i, operation, {**actions, **{j: operation for j in linked}}))
        if not candidates:
            reason = "actions_exhausted"
            break
        if strategy != "greedy":
            candidates.sort(key=lambda c: (-prediction[c[0]], c[0]))
        total_candidates = len(candidates)
        if max_candidates_per_step is not None:
            candidates = candidates[:max_candidates_per_step]
        # candidate_limit is a batch size, not a reason to report a false
        # utility-floor stop. On plateaus, inspect later batches as well.
        assessed = []
        checked = 0
        for begin in range(0, len(candidates), candidate_limit):
            batch = candidates[begin : begin + candidate_limit]
            texts = [apply_actions(doc, candidate[2]) for candidate in batch]
            keeps = retention(texts)
            allowed = np.flatnonzero(keeps >= utility_floor - 1e-12)
            checked += len(batch)
            if not len(allowed):
                continue
            next_risk, next_margin = controller.assess(pos, [texts[j] for j in allowed])
            for j, nr, nm in zip(allowed, next_risk, next_margin):
                cost = max(current_retention - keeps[j], 0) + 0.01
                assessed.append((begin + int(j), float(risk[0] - nr), float(margin[0] - nm), cost))
            if strategy != "greedy" and (
                not expand_on_plateau
                or any(a[1] > 1e-10 for a in assessed)
                or (begin > 0 and any(a[2] > 1e-10 for a in assessed))
            ):
                break
        trace[-1].update(
            candidate_count=total_candidates, candidates_checked=checked, feasible_candidates=len(assessed)
        )
        if not assessed:
            reason = "search_budget" if len(candidates) < total_candidates else "utility_floor"
            break
        indices, gain, mg, cost = np.asarray(assessed).T
        scores = gain / cost
        if np.max(gain) <= 1e-10:
            # On an RR plateau, prefer a lower margin; otherwise choose the
            # least expensive escalation.
            scores = mg / cost if np.max(mg) > 1e-10 else -cost
        chosen = int(indices[int(np.argmax(scores))])
        unit, operation, actions = candidates[chosen]
        trace[-1].update(
            unit=unit,
            operation=operation,
            label=doc.units[unit].label,
            group=doc.units[unit].group,
            source=doc.units[unit].source,
        )
    trace[-1]["stop_reason"] = reason
    trace[-1]["effective_step_limit"] = step_limit
    trace[-1]["utility_floor_met"] = trace[-1]["retention"] >= utility_floor - 1e-12
    if progress_steps:
        iterator.close()
    return text, actions, trace
