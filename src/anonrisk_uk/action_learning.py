"""Learn action values, then choose entity/action pairs under deployment checks.

The learner sees train-only S2 counterfactual outcomes for every available action.
At inference it has no same-person target links. The self-linkage gate may veto an
action, but never selects actions using test S2 results or rewards RR plateaus.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline

from .action_space import fixed_actions, mandatory, render, semantic_retention
from .adaptive_text import linked_units
from .attack import ranking_result


def self_targets(index, pos):
    """Exact-text equivalence uses public text, never same-person annotations."""
    return getattr(index, "exact_text_targets", {}).get(int(pos), [int(pos)])


def self_exposure(index, pos, texts):
    """Bounded continuous self-score separation, supplementary to exact S1 RR.

    Use the same index on every candidate; never compare raw BM25 and TF-IDF
    margins. This is an attack surrogate, not an identification probability.
    """
    scores = index.scores(texts).astype(float)
    own = scores[:, pos].copy()
    scores[:, self_targets(index, pos)] = -np.inf
    other = scores.max(axis=1)
    other = np.where(np.isfinite(other), other, 0.0)
    scale = np.abs(own) + np.abs(other) + 1e-9
    return np.clip(0.5 + 0.5 * (own - other) / scale, 0, 1)


def features(doc, unit_id, action, current, index, lexical_loss):
    unit = doc.units[unit_id]
    tokens = index.vec.build_analyzer()(doc.text[unit.start : unit.end])
    idfs = [index.token_idf(token) for token in tokens] or [0]
    return {
        "entity_type": unit.label,
        "role": unit.role,
        "group": unit.group,
        "source": unit.source,
        "operation": action.operation,
        "tool": action.tool,
        "severity": action.severity,
        "level": action.level,
        "n_tokens": len(tokens),
        "n_chars": unit.end - unit.start,
        "max_idf": max(idfs),
        "mean_idf": float(np.mean(idfs)),
        "current_severity": current.get(unit_id).severity if unit_id in current else 0.0,
        "context_edits": len(current),
        "semantic_before": semantic_retention(doc, current),
        "lexical_loss": float(lexical_loss),
        "replacement_chars": len(action.replacement),
        "doc_chars": len(doc.text),
        "n_units": len(doc.units),
    }


def linked_actions(doc, bank, actions, unit_id, action):
    """Transform identical repeated mentions together, only with compatible tools."""
    candidate = dict(actions)
    for other in linked_units(doc, unit_id):
        match = next((a for a in bank[other] if a.key == action.key and a.replacement == action.replacement), None)
        if match is None:
            return None
        candidate[other] = match
    return candidate


def candidates(doc, bank, actions):
    seen = set()
    for unit_id, unit in enumerate(doc.units):
        if unit.role != "QUASI":
            continue
        linked = tuple(linked_units(doc, unit_id))
        if linked in seen:
            continue
        seen.add(linked)
        severity = actions[unit_id].severity if unit_id in actions else 0
        for action in bank[unit_id]:
            if action.severity <= severity:
                continue
            changed = linked_actions(doc, bank, actions, unit_id, action)
            if changed is not None:
                yield unit_id, action, changed


def lexical_retention(vectorizer, original, texts):
    source = vectorizer.transform([original])
    return np.asarray(vectorizer.transform(texts).multiply(source).sum(axis=1)).ravel()


def action_training_rows(pop, positions, documents, banks, vectorizer, max_rows=5000):
    """Measure each action in mandatory-only and generalized contexts on train/val.

    Validation outcomes are reported separately and never passed to fit().
    """
    rows = []
    for pos in positions:
        doc, bank = documents[int(pos)], banks[int(pos)]
        contexts = [mandatory(doc, bank), fixed_actions(doc, bank, "L2_generalize")]
        for context_id, current in enumerate(contexts):
            base = render(doc, current)
            options = list(candidates(doc, bank, current))
            per_context = max(1, max_rows // len(contexts))
            if len(options) > per_context:
                # Sample across the complete action inventory, not only the
                # first fields/actions of a long resume. Bound attacker calls too.
                selected = np.random.default_rng(int(pos) * 2 + context_id).choice(
                    len(options), per_context, replace=False
                )
                options = [options[i] for i in sorted(selected)]
            texts = [base, *[render(doc, changed) for _, _, changed in options]]
            targets, excludes = pop.targets_and_excludes([int(pos)] * len(texts), "S2")
            rr = ranking_result(pop.index, texts, targets, excludes, batch_size=16)[0].reciprocal_ranks
            exposure = self_exposure(pop.index, int(pos), texts)
            retention = lexical_retention(vectorizer, doc.text, texts)
            for j, (unit, action, changed) in enumerate(options, 1):
                row = features(doc, unit, action, current, pop.index, retention[0] - retention[j])
                row.update(
                    pos=int(pos),
                    person=str(pop.clusters[pos]),
                    unit=unit,
                    context=context_id,
                    action=action.key,
                    s2_gain=float(rr[0] - rr[j]),
                    self_gain=float(exposure[0] - exposure[j]),
                    semantic_loss=semantic_retention(doc, current) - semantic_retention(doc, changed),
                )
                row["target_gain"] = 0.5 * row["s2_gain"] + 0.5 * row["self_gain"]
                rows.append(row)
                if len(rows) >= max_rows:
                    return pd.DataFrame(rows)
    return pd.DataFrame(rows)


METADATA = {"pos", "person", "unit", "context", "action", "s2_gain", "self_gain", "semantic_loss", "target_gain"}


def feature_records(table):
    return table.drop(columns=list(METADATA & set(table)), errors="ignore").to_dict("records")


def train_action_model(train, *, seed=42, max_iter=60):
    if len(train) < 10 or not np.isfinite(train.target_gain).all():
        raise ValueError("Too few finite training action outcomes")
    weights = 1 / train.groupby("person").person.transform("size").to_numpy()
    weights *= len(weights) / weights.sum()
    model = make_pipeline(
        DictVectorizer(sparse=False),
        HistGradientBoostingRegressor(max_iter=max_iter, max_leaf_nodes=7, min_samples_leaf=5, random_state=seed),
    )
    model.fit(feature_records(train), train.target_gain, histgradientboostingregressor__sample_weight=weights)
    return model


def choose_actions(
    doc,
    bank,
    model,
    index,
    vectorizer,
    pos,
    *,
    utility_floor=0.7,
    semantic_floor=0.8,
    max_steps=6,
    candidate_limit=8,
    min_self_gain=0.01,
    cost_weight=0.2,
    self_rr_target=0.5,
):
    actions = mandatory(doc, bank)
    trace = []
    reason = "max_steps"
    for step in range(max_steps + 1):
        text = render(doc, actions)
        current_rr = ranking_result(index, [text], [self_targets(index, pos)])[0].reciprocal_ranks[0]
        exposure = float(self_exposure(index, int(pos), [text])[0])
        lexical = float(lexical_retention(vectorizer, doc.text, [text])[0])
        meaning = semantic_retention(doc, actions)
        trace.append(
            {
                "step": step,
                "proxy_rr": float(current_rr),
                "self_exposure": exposure,
                "retention": lexical,
                "content_retention": meaning,
            }
        )
        if current_rr <= self_rr_target:
            reason = "threshold_met"
            break
        if step == max_steps:
            break
        feasible = []
        for unit, action, changed in candidates(doc, bank, actions):
            new_text = render(doc, changed)
            keep = float(lexical_retention(vectorizer, doc.text, [new_text])[0])
            semantic = semantic_retention(doc, changed)
            if keep < utility_floor or semantic < semantic_floor:
                continue
            loss = max(0, 0.5 * (lexical - keep) + 0.5 * (meaning - semantic))
            record = features(doc, unit, action, actions, index, lexical - keep)
            feasible.append((unit, action, changed, new_text, loss, record))
        if not feasible:
            reason = "utility_floor"
            break
        prediction = model.predict([candidate[-1] for candidate in feasible])
        # The model ranks entity/action pairs. A deployment-only S1 gate verifies
        # the shortlist; no test S2 target or outcome is used in this decision.
        order = np.argsort(
            -np.asarray([float(p) - cost_weight * c[4] for p, c in zip(prediction, feasible)]), kind="stable"
        )
        shortlist = order[:candidate_limit]
        texts = [feasible[int(j)][3] for j in shortlist]
        rr = ranking_result(index, texts, [self_targets(index, pos)] * len(texts))[0].reciprocal_ranks
        exposures = self_exposure(index, int(pos), texts)
        chosen = None
        for rank, j in enumerate(shortlist):
            gain = exposure - exposures[rank]
            if rr[rank] > current_rr + 1e-12:
                continue
            if rr[rank] < current_rr - 1e-12 or gain >= min_self_gain:
                chosen = int(j), float(gain)
                break
        trace[-1].update(feasible_candidates=len(feasible), candidates_checked=len(shortlist))
        if chosen is None:
            reason = "no_verified_improvement"
            break
        j, gain = chosen
        unit, action, actions, _, loss, _ = feasible[j]
        trace[-1].update(
            unit=unit,
            group=doc.units[unit].group,
            label=doc.units[unit].label,
            action=action.key,
            operation=action.operation,
            tool=action.tool,
            predicted_gain=float(prediction[j]),
            measured_self_gain=gain,
            utility_loss=loss,
        )
    trace[-1]["stop_reason"] = reason
    trace[-1]["threshold_met"] = bool(trace[-1]["proxy_rr"] <= self_rr_target)
    return render(doc, actions), actions, trace
