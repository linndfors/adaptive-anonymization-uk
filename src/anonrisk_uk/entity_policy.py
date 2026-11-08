"""A supervised per-entity ACTION classifier with a separate risk head.

Inputs contain entity/context/capability features only. Neither a generated
replacement, a tool identity nor a counterfactual outcome enters prediction.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupKFold

from .action_learning import lexical_retention, self_exposure, self_targets
from .action_space import render, semantic_retention
from .adaptive_text import linked_units
from .attack import ranking_result
from .policy_execution import CLASSES, Decision, UnsupportedDecision, execute_fixed, protected_actions


def state_features(doc, unit_id, actions, index, codec, available, *, rr, exposure, neighbor=None, steps=None):
    """``neighbor`` (proxy RR, margin) and ``steps`` (policy edits so far) are added only when given."""
    unit = doc.units[unit_id]
    value = doc.text[unit.start : unit.end]
    tokens = index.vec.build_analyzer()(value)
    idfs = [index.token_idf(token) for token in tokens] or [0]
    record = {
        "entity_type": unit.label,
        "group": unit.group,
        "role": unit.role,
        "source": unit.source,
        "n_chars": len(value),
        "n_tokens": len(tokens),
        "max_idf": max(idfs),
        "mean_idf": float(np.mean(idfs)),
        "frequency": float(np.log1p(codec.population_frequency(unit.group, value))),
        "mentions": len(linked_units(doc, unit_id)),
        "doc_chars": len(doc.text),
        "n_units": len(doc.units),
        "current_level": actions[unit_id].severity if unit_id in actions else 0,
        "state_edits": len(actions),
        "state_content": semantic_retention(doc, actions),
        "state_skills_retention": skills_retention(doc, actions),
        "state_self_rr": float(rr),
        "state_self_exposure": float(exposure),
        **{f"available_{key}": float(key in available) for key in CLASSES},
    }
    if neighbor is not None:
        record.update(state_neighbor_rr=float(neighbor[0]), state_neighbor_margin=float(neighbor[1]))
    if steps is not None:
        record["state_policy_steps"] = int(steps)
    return record


def state_risk(doc, actions, index, pos):
    text = render(doc, actions)
    rr = float(ranking_result(index, [text], [self_targets(index, pos)])[0].reciprocal_ranks[0])
    exposure = float(self_exposure(index, pos, [text])[0])
    return rr, exposure


class ScoreRows:
    """Index stand-in: queries are row numbers of one precomputed score matrix."""

    def __init__(self, matrix):
        self.matrix = np.asarray(matrix)

    def scores(self, rows):
        return self.matrix[rows]


def normalized_margin(margins, best):
    """(best target - best other) / best target, clipped to [-1, 1]; the sign alone when best <= 0."""
    margins, best = np.asarray(margins, dtype=float), np.asarray(best, dtype=float)
    return np.clip(np.where(best > 0, margins / np.maximum(best, 1e-12), np.sign(margins)), -1, 1)


def neighbor_targets(index, pos, text, k=3):
    """Top-k neighbours of the ORIGINAL text in the population index: release-time pseudo-targets.

    The document and its exact copies are excluded via ``self_targets``; person labels are never used.
    """
    scores = np.asarray(index.scores([text])[0], dtype=float)
    scores[self_targets(index, pos)] = -np.inf
    order = np.argsort(-scores, kind="stable")[:k]
    return [int(j) for j in order if np.isfinite(scores[j])]


def neighbor_proxy(index, pos, texts, pseudo, scores=None):
    """Per text: best pseudo-target RR and normalised pseudo-target margin, self excluded."""
    if not pseudo:
        return np.zeros(len(texts)), np.zeros(len(texts))
    scores = index.scores(texts) if scores is None else scores
    n, excluded = len(texts), list(self_targets(index, pos))
    result, margins, best = ranking_result(ScoreRows(scores), list(range(n)), [pseudo] * n, [excluded] * n)
    return result.reciprocal_ranks, normalized_margin(margins, best)


PLAN_CHECKS = ("self_exposure", "neighbor_proxy")


def plan_proxy(doc, actions, index, pos, plan_check, pseudo):
    """(rank proxy, continuous proxy) the plan verifier compares before and after a plan."""
    if plan_check == "self_exposure":
        return state_risk(doc, actions, index, pos)
    rr, margin = neighbor_proxy(index, pos, [render(doc, actions)], pseudo)
    return float(rr[0]), float(margin[0])


def skills_retention(doc, actions):
    values = [1 - actions[i].severity if i in actions else 1.0 for i, u in enumerate(doc.units) if u.group == "skills"]
    return float(np.mean(values)) if values else 1.0


LABEL_OBJECTIVES = ("v2_mixed", "v3_s2_least_destructive")


def select_least_destructive(gains, severities, feasible, *, noise_margin=0.01, relative_tolerance=0.8):
    """Index of the v3 label among options whose index 0 is KEEP.

    Candidates are feasible changes whose S2 gain clears the noise margin; none
    means KEEP. Among candidates within ``relative_tolerance`` of the best gain,
    the lowest severity wins, then the higher gain.
    """
    candidates = [j for j in range(1, len(gains)) if feasible[j] and gains[j] > noise_margin]
    if not candidates:
        return 0
    best = max(gains[j] for j in candidates)
    near = [j for j in candidates if gains[j] >= relative_tolerance * best]
    return min(near, key=lambda j: (severities[j], -gains[j], j))


def eligible_units(doc, pos, max_units):
    """First mention of each quasi-identifier, sampled round-robin by field when over ``max_units``."""
    eligible = [i for i, unit in enumerate(doc.units) if unit.role == "QUASI" and i == min(linked_units(doc, i))]
    if len(eligible) > max_units:
        # Round-robin fields before seeded sampling so long skills lists do not
        # displace salary/title/age from the supervised dataset.
        groups = {}
        for unit in np.random.default_rng(int(pos)).permutation(eligible):
            groups.setdefault(doc.units[unit].group, []).append(int(unit))
        eligible = []
        while len(eligible) < max_units and any(groups.values()):
            for group in sorted(groups):
                if groups[group] and len(eligible) < max_units:
                    eligible.append(groups[group].pop())
    return eligible


def oracle_rows(
    pop,
    pos,
    doc,
    router,
    vectorizer,
    *,
    max_units=24,
    utility_floor=0.65,
    content_floor=0.75,
    cost_weight=0.2,
    skill_floor=0.5,
    objective="v2_mixed",
    noise_margin=0.01,
    relative_tolerance=0.8,
    neighbor_k=None,
):
    """Generate TRAIN/VAL supervision by executing actions offline, including KEEP.

    Exhaustively evaluate all permitted decisions for each sampled entity. Select
    the best feasible action against a declared objective, not hand-assigned
    classes based on entity type. Outcomes never become model inputs.

    ``v2_mixed``: argmax of 0.5*S2 + 0.5*S1 effect - cost_weight*retention loss.
    ``v3_s2_least_destructive``: S2 effect only, priced by the action's own
    severity, label chosen by ``select_least_destructive``.

    With ``neighbor_k`` set, outcomes also carry current-context release-time
    proxy gains and the normalised S2 margin gain; labels are unchanged.
    """
    if objective not in LABEL_OBJECTIVES:
        raise ValueError(f"Unknown label_objective: {objective!r}")
    contexts = [protected_actions(doc, router), execute_fixed(doc, router, "L2_generalize")[1]]
    eligible = eligible_units(doc, pos, max_units)
    pseudo = neighbor_targets(pop.index, pos, doc.text, neighbor_k) if neighbor_k else []
    examples, outcomes = [], []
    for context_id, current in enumerate(contexts):
        rr, exposure = state_risk(doc, current, pop.index, pos)
        before = render(doc, current)
        before_lexical = float(lexical_retention(vectorizer, doc.text, [before])[0])
        before_content = semantic_retention(doc, current)
        for unit in eligible:
            available = router.available(doc, unit)
            # KEEP means retain the entity's current state in this context.
            options = [("keep", current, before, "identity")]
            for key in available:
                if key == "keep":
                    continue
                try:
                    changed = router.apply(doc, current, Decision.from_key(unit, key, reason="offline_supervision"))
                except UnsupportedDecision:
                    continue
                options.append((key, changed, render(doc, changed), changed[unit].tool))
            texts = [option[2] for option in options]
            linked = set(linked_units(doc, unit))
            isolated_base = protected_actions(doc, router)
            for other, item in enumerate(doc.units):
                if item.role == "QUASI" and other not in linked:
                    isolated_base = router.apply(
                        doc, isolated_base, Decision(other, "mask", reason="counterfactual_isolation")
                    )
            isolated_texts = []
            for _, changed, _, _ in options:
                isolated = dict(isolated_base)
                for member in linked:
                    if member in changed:
                        isolated[member] = changed[member]
                    else:
                        isolated.pop(member, None)
                isolated_texts.append(render(doc, isolated))
            all_texts = texts + isolated_texts
            targets, excludes = pop.targets_and_excludes([int(pos)] * len(all_texts), "S2")
            s2_result, s2_margins, s2_best = ranking_result(pop.index, all_texts, targets, excludes, batch_size=16)
            all_s2 = s2_result.reciprocal_ranks
            all_self = self_exposure(pop.index, pos, all_texts)
            s2, isolated_s2 = all_s2[: len(texts)], all_s2[len(texts) :]
            self_score, isolated_self = all_self[: len(texts)], all_self[len(texts) :]
            lexical = lexical_retention(vectorizer, doc.text, texts)
            s2_margin = normalized_margin(s2_margins[: len(texts)], s2_best[: len(texts)])
            proxy_rr, proxy_margin = neighbor_proxy(pop.index, pos, texts, pseudo)  # zeros without neighbor_k
            scores, gains, severities, feasibility = [], [], [], []
            for j, (key, changed, text, tool) in enumerate(options):
                content = semantic_retention(doc, changed)
                loss = max(0, 0.5 * (before_lexical - lexical[j]) + 0.5 * (before_content - content))
                s2_effect = 0.5 * ((s2[0] - s2[j]) + (isolated_s2[0] - isolated_s2[j]))
                self_effect = 0.5 * ((self_score[0] - self_score[j]) + (isolated_self[0] - isolated_self[j]))
                severity = changed[unit].severity if key != "keep" else 0.0
                feasible = key == "keep" or (
                    lexical[j] >= utility_floor
                    and content >= content_floor
                    and skills_retention(doc, changed) >= skill_floor
                )
                if objective == "v2_mixed":
                    gain, cost = 0.5 * s2_effect + 0.5 * self_effect, loss
                else:
                    gain, cost = s2_effect, severity
                score = float(gain - cost_weight * cost) if feasible else -1e6
                if key == "keep":
                    score = 0.0
                scores.append(score)
                gains.append(float(s2_effect))
                severities.append(severity)
                feasibility.append(feasible)
                outcomes.append(
                    {
                        "pos": int(pos),
                        "person": str(pop.clusters[pos]),
                        "unit": unit,
                        "context": context_id,
                        "decision": key,
                        "tool": tool,
                        "s2_gain": float(s2_effect),
                        "self_gain": float(self_effect),
                        "current_context_s2_gain": float(s2[0] - s2[j]),
                        "isolated_context_s2_gain": float(isolated_s2[0] - isolated_s2[j]),
                        "utility_loss": loss,
                        "severity": severity,
                        "retention": float(lexical[j]),
                        "content_retention": content,
                        "feasible": feasible,
                        "objective": score,
                        **(
                            {
                                "current_context_s2_margin_gain": float(s2_margin[0] - s2_margin[j]),
                                "neighbor_rr_gain": float(proxy_rr[0] - proxy_rr[j]),
                                "neighbor_margin_gain": float(proxy_margin[0] - proxy_margin[j]),
                            }
                            if neighbor_k
                            else {}
                        ),
                    }
                )
            if objective == "v2_mixed":
                # Keep wins ties. Do not train on arbitrary destructive tie breakers.
                best = int(np.argmax(scores))
                if scores[best] <= 1e-5:
                    best = 0
            else:
                best = select_least_destructive(
                    gains, severities, feasibility, noise_margin=noise_margin, relative_tolerance=relative_tolerance
                )
            risk = max(0, max(0.5 * ((s2[0] - s2) + (isolated_s2[0] - isolated_s2))))
            available = [key for key, _, _, _ in options]
            record = state_features(doc, unit, current, pop.index, router.codec, available, rr=rr, exposure=exposure)
            examples.append(
                {
                    "pos": int(pos),
                    "person": str(pop.clusters[pos]),
                    "unit": unit,
                    "context": context_id,
                    "entity_type": doc.units[unit].label,
                    "features_json": json.dumps(record, sort_keys=True),
                    "target_decision": options[best][0],
                    "risk_target": risk,
                    "risk_effect": bool(risk > noise_margin),
                    "oracle_objective": float(scores[best]),
                }
            )
    return pd.DataFrame(examples), pd.DataFrame(outcomes)


ORACLE_STOPS = {
    "target_met": "true S2 RR at or below the oracle target",
    "no_gain": "no remaining entity-action clears the noise margin",
    "utility_floor": "only changes violating a utility floor clear the noise margin",
    "max_steps": "oracle_max_steps changes applied",
}


def sequential_oracle_rows(
    pop,
    pos,
    doc,
    router,
    vectorizer,
    *,
    max_units=24,
    utility_floor=0.65,
    content_floor=0.75,
    cost_weight=0.2,
    skill_floor=0.5,
    noise_margin=0.01,
    relative_tolerance=0.8,
    rr_target=0.5,
    max_steps=8,
    neighbor_k=3,
):
    """Greedy S2 oracle trajectory from mandatory protection; one example per entity and step.

    Each step scores every unchanged eligible entity's non-keep actions from the
    CURRENT state in one ranking batch. Gain is the drop of the normalised S2
    margin (true other resume vs best other person); ``select_least_destructive``
    labels each entity (``context`` = step), then the labelled change with the
    best gain per severity is applied. Stops are listed in ``ORACLE_STOPS``.
    """
    eligible = eligible_units(doc, pos, max_units)
    pseudo = neighbor_targets(pop.index, pos, doc.text, neighbor_k)
    targets, excludes = pop.targets_and_excludes([int(pos)], "S2")
    current, changed_units, person = protected_actions(doc, router), set(), str(pop.clusters[pos])
    examples, outcomes, trajectory = [], [], []
    for step in range(max_steps + 1):
        options: list[tuple] = [(None, "keep", current, render(doc, current))]
        for unit in eligible:
            if unit in changed_units:
                continue
            for key in router.available(doc, unit):
                if key == "keep":
                    continue
                try:
                    changed = router.apply(doc, current, Decision.from_key(unit, key, reason="offline_supervision"))
                except UnsupportedDecision:
                    continue
                options.append((unit, key, changed, render(doc, changed)))
        n, texts = len(options), [option[3] for option in options]
        scores = pop.index.scores(texts)
        result, margins, best = ranking_result(ScoreRows(scores), list(range(n)), targets * n, excludes * n)
        s2_rr, s2_margin = result.reciprocal_ranks, normalized_margin(margins, best)
        proxy_rr, proxy_margin = neighbor_proxy(pop.index, pos, texts, pseudo, scores=scores)
        lexical = lexical_retention(vectorizer, doc.text, texts)
        state = {
            "pos": int(pos),
            "person": person,
            "step": step,
            "s2_rr": float(s2_rr[0]),
            "s2_margin": float(s2_margin[0]),
            "neighbor_rr": float(proxy_rr[0]),
            "neighbor_margin": float(proxy_margin[0]),
            "retention": float(lexical[0]),
            "content_retention": semantic_retention(doc, current),
            "candidates": n - 1,
            "unit": None,
            "decision": None,
        }
        if s2_rr[0] <= rr_target:
            trajectory.append({**state, "stop_reason": "target_met"})
            break
        if step == max_steps:
            trajectory.append({**state, "stop_reason": "max_steps"})
            break
        rr, exposure = state_risk(doc, current, pop.index, pos)
        content = [semantic_retention(doc, option[2]) for option in options]
        gains = s2_margin[0] - s2_margin
        feasible = [
            j == 0
            or (
                lexical[j] >= utility_floor
                and content[j] >= content_floor
                and skills_retention(doc, options[j][2]) >= skill_floor
            )
            for j in range(n)
        ]
        severity = [0.0] + [option[2][option[0]].severity for option in options[1:]]
        picks, blocked = [], False
        for unit in eligible:
            if unit in changed_units:
                continue
            rows = [0] + [j for j in range(1, n) if options[j][0] == unit]
            unit_gains = [0.0] + [float(gains[j]) for j in rows[1:]]
            choice = rows[
                select_least_destructive(
                    unit_gains,
                    [severity[j] for j in rows],
                    [feasible[j] for j in rows],
                    noise_margin=noise_margin,
                    relative_tolerance=relative_tolerance,
                )
            ]
            blocked |= any(gains[j] > noise_margin and not feasible[j] for j in rows[1:])
            objective = {}
            for j in rows:
                key = options[j][1]
                objective[j] = 0.0 if j == 0 else float(gains[j] - cost_weight * severity[j]) if feasible[j] else -1e6
                outcomes.append(
                    {
                        "pos": int(pos),
                        "person": person,
                        "unit": unit,
                        "context": step,
                        "decision": key,
                        "tool": "identity" if j == 0 else options[j][2][unit].tool,
                        "s2_gain": float(gains[j]),
                        "current_context_s2_gain": float(s2_rr[0] - s2_rr[j]),
                        "current_context_s2_margin_gain": float(gains[j]),
                        "neighbor_rr_gain": float(proxy_rr[0] - proxy_rr[j]),
                        "neighbor_margin_gain": float(proxy_margin[0] - proxy_margin[j]),
                        "severity": severity[j],
                        "retention": float(lexical[j]),
                        "content_retention": content[j],
                        "feasible": feasible[j],
                        "objective": objective[j],
                    }
                )
            record = state_features(
                doc,
                unit,
                current,
                pop.index,
                router.codec,
                [options[j][1] for j in rows],
                rr=rr,
                exposure=exposure,
                neighbor=(proxy_rr[0], proxy_margin[0]),
                steps=step,
            )
            risk = max(0.0, max(unit_gains))
            examples.append(
                {
                    "pos": int(pos),
                    "person": person,
                    "unit": unit,
                    "context": step,
                    "entity_type": doc.units[unit].label,
                    "features_json": json.dumps(record, sort_keys=True),
                    "target_decision": options[choice][1],
                    "risk_target": risk,
                    "risk_effect": bool(risk > noise_margin),
                    "oracle_objective": objective[choice],
                }
            )
            if choice:
                picks.append(choice)
        if not picks:
            trajectory.append({**state, "stop_reason": "utility_floor" if blocked else "no_gain"})
            break
        chosen = max(picks, key=lambda j: (gains[j] / max(severity[j], 1e-6), -j))
        unit, key, current, _ = options[chosen]
        changed_units.add(unit)
        trajectory.append({**state, "unit": unit, "decision": key, "gain": float(gains[chosen])})
    return pd.DataFrame(examples), pd.DataFrame(outcomes), pd.DataFrame(trajectory)


class EntityPolicyModel:
    """Predict an action/level before a tool is called. Risk is not a probability."""

    def __init__(self, *, max_iter=60, balanced=False, seed=42):
        self.parameters = {"max_iter": max_iter, "balanced": balanced, "seed": seed}
        self.encoder = DictVectorizer(sparse=False)
        self.risk_encoder = DictVectorizer(sparse=False)
        self.keep_threshold = 0.0
        self.classifier = HistGradientBoostingClassifier(
            max_iter=max_iter, max_leaf_nodes=7, min_samples_leaf=5, random_state=seed
        )
        self.risk_head = HistGradientBoostingRegressor(
            max_iter=40, max_leaf_nodes=7, min_samples_leaf=5, random_state=seed
        )

    def fit(self, table):
        if len(table) < 20 or table.target_decision.nunique() < 2:
            raise ValueError(
                "Insufficient action supervision: need at least two observed decisions, not an all-KEEP fit"
            )
        records = [json.loads(value) for value in table.features_json]
        forbidden = {
            "replacement",
            "tool",
            "decision",
            "s2_gain",
            "self_gain",
            "utility_loss",
            "target_decision",
            "risk_target",
            "risk_effect",
            "oracle_objective",
            "estimated_entity_risk",
        }
        if any(forbidden & set(row) for row in records):
            raise ValueError("Outcome/tool leakage in policy features")
        risk_features = self.risk_encoder.fit_transform(records)
        weight = 1 / table.groupby("person").person.transform("size").to_numpy()
        weight *= len(weight) / weight.sum()
        risk_weight = weight.copy()
        people = table.person.astype(str).to_numpy()
        if len(set(people)) < 2:
            raise ValueError("Risk-conditioned policy training requires multiple people for cross-fitting")
        oof_risk = np.zeros(len(table))
        self.risk_crossfit_groups = []
        for fit_indices, held_indices in GroupKFold(n_splits=min(3, len(set(people)))).split(
            risk_features, groups=people
        ):
            fold = HistGradientBoostingRegressor(
                max_iter=40, max_leaf_nodes=7, min_samples_leaf=5, random_state=self.parameters["seed"]
            )
            fold.fit(
                risk_features[fit_indices],
                table.risk_target.to_numpy()[fit_indices],
                sample_weight=risk_weight[fit_indices],
            )
            oof_risk[held_indices] = np.maximum(fold.predict(risk_features[held_indices]), 0)
            self.risk_crossfit_groups.append(
                {"train": sorted(set(people[fit_indices])), "held_out": sorted(set(people[held_indices]))}
            )
        policy_records = [{**row, "estimated_entity_risk": float(risk)} for row, risk in zip(records, oof_risk)]
        features = self.encoder.fit_transform(policy_records)
        if self.parameters["balanced"]:
            frequency = table.target_decision.value_counts()
            balance = table.target_decision.map(lambda x: min(4.0, len(table) / (len(frequency) * frequency[x])))
            weight *= balance.to_numpy()
        self.classifier.fit(features, table.target_decision, sample_weight=weight)
        self.risk_head.fit(risk_features, table.risk_target, sample_weight=risk_weight)
        self.oof_risk_mae = mean_absolute_error(table.risk_target, oof_risk)
        self.majority_class = table.target_decision.mode().iloc[0]
        return self

    def decide(self, records, units, available, *, exclude_keep=False):
        """``exclude_keep`` picks the most probable non-keep action wherever one is available."""
        risks = np.maximum(self.risk_head.predict(self.risk_encoder.transform(records)), 0)
        matrix = self.encoder.transform(
            [{**row, "estimated_entity_risk": float(risk)} for row, risk in zip(records, risks)]
        )
        probabilities = self.classifier.predict_proba(matrix)
        decisions = []
        for i, unit in enumerate(units):
            choices = [
                (float(probabilities[i, j]), str(key))
                for j, key in enumerate(self.classifier.classes_)
                if key in available[i] and not (exclude_keep and key == "keep" and len(available[i]) > 1)
            ]
            if not choices:
                key, confidence = "keep", 0.0
            else:
                confidence, key = max(choices, key=lambda item: (item[0], item[1] == "keep"))
                if not exclude_keep and key != "keep" and confidence < self.keep_threshold and "keep" in available[i]:
                    key = "keep"
            decisions.append(
                Decision.from_key(
                    unit,
                    key,
                    risk_score=float(risks[i]),
                    confidence=confidence,
                    reason="learned_entity_action_classifier",
                )
            )
        return decisions


def validate_policy(model, examples, outcomes):
    records = [json.loads(value) for value in examples.features_json]
    available = [[key for key in CLASSES if row.get(f"available_{key}", 0)] for row in records]
    decisions = model.decide(records, examples.unit.tolist(), available)
    predicted = [d.key for d in decisions]
    majority = model.majority_class
    result = {
        "accuracy": accuracy_score(examples.target_decision, predicted),
        "macro_f1": f1_score(examples.target_decision, predicted, average="macro", zero_division=0),
        "keep_accuracy": float(examples.target_decision.eq("keep").mean()),
        "majority_accuracy": float(examples.target_decision.eq(majority).mean()),
        "risk_mae": mean_absolute_error(examples.risk_target, [d.risk_score for d in decisions]),
        "risk_zero_mae": mean_absolute_error(examples.risk_target, np.zeros(len(examples))),
        "n_people": examples.person.nunique(),
        "n_entities_contexts": len(examples),
    }
    lookup = outcomes.set_index(["pos", "unit", "context", "decision"]).objective
    losses, predictions = [], []
    for (_, row), decision in zip(examples.iterrows(), decisions):
        achieved = float(lookup.loc[(row.pos, row.unit, row.context, decision.key)])
        would_reject = achieved <= -1e5
        if would_reject:
            achieved = 0.0  # Deployment verifier rejects it and preserves current text.
        regret = max(0, float(row.oracle_objective) - achieved)
        losses.append(regret)
        predictions.append(
            {
                **row.to_dict(),
                "predicted_decision": decision.key,
                "predicted_risk": decision.risk_score,
                "confidence": decision.confidence,
                "regret": regret,
                "would_reject_utility": would_reject,
            }
        )
    result["person_balanced_regret"] = (
        pd.DataFrame({"person": examples.person, "regret": losses}).groupby("person").regret.mean().mean()
    )
    result["keep_regret"] = examples.groupby("person").oracle_objective.mean().mean()
    return result, pd.DataFrame(predictions)


BASELINE_RANKERS = ("constant_zero", "value_idf", "entity_type_mean")


def ranker_scores(ranker, records, risks):
    """Release priority before confidence: the risk head or a trivial baseline."""
    name = (ranker or {}).get("name", "risk_head")
    if name == "risk_head":
        return np.asarray(risks, dtype=float)
    if name == "constant_zero":
        return np.zeros(len(records))
    if name == "value_idf":
        return np.array([row["max_idf"] for row in records], dtype=float)
    if name == "entity_type_mean":
        return np.array([ranker["type_means"].get(row["entity_type"], ranker["default"]) for row in records])
    raise ValueError(f"Unknown release ranker: {name}")


def risk_ranking_metrics(table, scores):
    """AUC for ``risk_effect`` and mean within-document Spearman against realised max S2 gain.

    Documents whose realised gains are all equal are skipped; a constant score in
    an informative document counts as zero correlation.
    """
    frame = table.assign(score=np.asarray(scores, dtype=float))
    effect = frame.risk_effect.astype(bool)
    auc = float(roc_auc_score(effect, frame.score)) if effect.nunique() == 2 else float("nan")
    rhos = [
        0.0 if group.score.nunique() < 2 else float(group.score.corr(group.risk_target, method="spearman"))
        for _, group in frame.groupby("pos")
        if group.risk_target.nunique() > 1
    ]
    return {
        "auc_risk_effect": auc,
        "spearman_within_document": float(np.mean(rhos)) if rhos else float("nan"),
        "n_rows": len(frame),
        "n_documents_ranked": len(rhos),
        "positive_rate": float(effect.mean()),
    }


PROXY_COLUMNS = ("neighbor_rr_gain", "neighbor_margin_gain")
TRUE_S2_COLUMNS = ("current_context_s2_gain", "current_context_s2_margin_gain")


def proxy_tracking(outcomes, *, noise_margin=0.01, n_boot=200, seed=0):
    """How well release-time proxy gains track true S2 gains on non-keep options.

    Truth: current-context S2 RR gain and normalised S2 margin gain. Spearman over
    all rows; AUC for ``truth > noise_margin``. 95% intervals resample documents.
    A proxy tracks better than chance only if both lower bounds clear chance.
    """
    rows = outcomes[outcomes.decision.ne("keep")].reset_index(drop=True)
    docs = rows.groupby("pos").indices
    keys = list(docs)
    rng = np.random.default_rng(seed)
    draws = [np.concatenate([docs[keys[i]] for i in rng.integers(0, len(keys), len(keys))]) for _ in range(n_boot)]

    def metrics(frame, proxy, truth):
        positive = frame[truth] > noise_margin
        auc = float(roc_auc_score(positive, frame[proxy])) if positive.nunique() == 2 else float("nan")
        rho = frame[proxy].corr(frame[truth], method="spearman") if frame[proxy].nunique() > 1 else 0.0
        return auc, float(rho)

    report = {"rows": len(rows), "documents": len(keys), "noise_margin": noise_margin, "results": {}}
    for truth in TRUE_S2_COLUMNS:
        for proxy in PROXY_COLUMNS:
            auc, rho = metrics(rows, proxy, truth)
            boot = np.array([metrics(rows.iloc[d], proxy, truth) for d in draws])
            low, high = np.nanquantile(boot, [0.025, 0.975], axis=0)
            report["results"][f"{proxy}~{truth}"] = {
                "auc": auc,
                "auc_ci95": [float(low[0]), float(high[0])],
                "spearman": rho,
                "spearman_ci95": [float(low[1]), float(high[1])],
                "positive_rate": float((rows[truth] > noise_margin).mean()),
                "better_than_chance": bool(low[0] > 0.5 and low[1] > 0),
            }
    return report


def evaluate_rankers(model, train, val, *, noise_margin=0.01):
    """Compare the risk head with trivial rankers on validation; pick the release ranker.

    Rows are context 0 (mandatory protection only), the state release ranks in.
    The risk head is kept only if it beats every baseline on both metrics.
    """

    def prepare(table):
        table = table[table.context.eq(0)] if "context" in table else table
        if "risk_effect" not in table:
            table = table.assign(risk_effect=table.risk_target > noise_margin)
        return table.reset_index(drop=True)

    train, val = prepare(train), prepare(val)
    records = [json.loads(value) for value in val.features_json]
    risks = np.maximum(model.risk_head.predict(model.risk_encoder.transform(records)), 0)
    means = train.groupby("entity_type").risk_target.mean()
    rankers = {
        "risk_head": {"name": "risk_head"},
        "constant_zero": {"name": "constant_zero"},
        "value_idf": {"name": "value_idf", "feature": "max_idf"},
        "entity_type_mean": {
            "name": "entity_type_mean",
            "type_means": {str(k): float(v) for k, v in means.items()},
            "default": float(train.risk_target.mean()),
        },
    }
    metrics = {name: risk_ranking_metrics(val, ranker_scores(spec, records, risks)) for name, spec in rankers.items()}

    def value(name, metric):
        result = metrics[name][metric]
        return (0.5 if metric.startswith("auc") else 0.0) if np.isnan(result) else result

    keys = ("auc_risk_effect", "spearman_within_document")
    best_baseline = max(BASELINE_RANKERS, key=lambda name: (sum(value(name, k) for k in keys), name == "value_idf"))
    beats = {k: value("risk_head", k) > max(value(name, k) for name in BASELINE_RANKERS) for k in keys}
    chosen = "risk_head" if all(beats.values()) else best_baseline
    reason = (
        "risk head beats every baseline on both validation metrics"
        if chosen == "risk_head"
        else f"risk head does not beat every baseline on both metrics ({beats}); using best baseline by AUC+Spearman"
    )
    return {
        "metrics": metrics,
        "noise_margin": noise_margin,
        "evaluation_rows": "validation, context 0 (mandatory protection only, as at release)",
        "best_baseline": best_baseline,
        "risk_head_beats_best": beats,
        "chosen_ranker": chosen,
        "reason": reason,
        "ranker": rankers[chosen],
    }


# Every audit row of release_with_policy carries exactly one of these statuses.
DECISION_STATUSES = {
    "model_keep": "model chose keep; no tool called",
    "not_executed_step_budget": "change ranked beyond the max_steps cap (None = no cap); tool not called",
    "tool_unavailable": "change proposed but no tool supports it",
    "rejected_utility": "tool ran; its replacement was discarded by the utility verifier",
    "rejected_no_improvement": "tool ran; the whole plan was discarded because proxy risk did not drop",
    "not_executed_target_met": "change proposed but the sequential release stopped at the proxy target; tool not called",
    "executed": "tool ran; its replacement is in the release",
    "final_state": "not a model decision (mandatory, linked or context unit); shows the final action",
}
NO_TOOL = "not_called"


def release_with_policy(
    doc,
    router,
    model,
    index,
    vec,
    pos,
    *,
    max_steps=None,
    utility_floor=0.65,
    content_floor=0.75,
    min_self_gain=0.01,
    self_rr_target=0.5,
    skill_floor=0.5,
    plan_check="self_exposure",
    neighbor_k=3,
):
    """Predict per-entity decisions first, execute, and verify their combined effect.

    The verifier can reject model proposals but cannot invent another action.
    This handles redundant identifiers without arbitrary plateau escalation.
    All non-keep decisions run unless ``max_steps`` caps them; utility floors
    reject decisions one at a time, lowest priority first. Priority is the
    model's release ranker score (risk head unless validation chose a
    baseline) times confidence.

    ``plan_check`` picks the plan verifier's proxy: S1 self RR + self-exposure,
    or RR + normalised margin of the original's top-``neighbor_k`` population
    neighbours (``neighbor_proxy``). ``min_self_gain`` and ``self_rr_target``
    apply to the chosen proxy.
    """
    if plan_check not in PLAN_CHECKS:
        raise ValueError(f"Unknown plan_check: {plan_check!r}")
    baseline = protected_actions(doc, router)
    rr, exposure = state_risk(doc, baseline, index, pos)
    pseudo = neighbor_targets(index, pos, doc.text, neighbor_k) if plan_check == "neighbor_proxy" else []
    proxy_rr, proxy = plan_proxy(doc, baseline, index, pos, plan_check, pseudo)
    units = [i for i, u in enumerate(doc.units) if u.role == "QUASI" and i == min(linked_units(doc, i))]
    available = [router.available(doc, i) for i in units]
    records = [
        state_features(doc, i, baseline, index, router.codec, keys, rr=rr, exposure=exposure)
        for i, keys in zip(units, available)
    ]
    proposals = model.decide(records, units, available) if units else []
    scores = ranker_scores(getattr(model, "ranker", None), records, [d.risk_score for d in proposals])
    priority = {d.unit: float(score) * d.confidence for d, score in zip(proposals, scores)}
    audit = [
        {
            **asdict(d),
            "decision": d.key,
            "entity_type": doc.units[d.unit].label,
            "status": "model_keep" if d.action == "keep" else "proposed",
            "tool": NO_TOOL,
            "applied": False,
            "priority": priority[d.unit],
        }
        for d in proposals
    ]
    by_unit = {row["unit"]: row for row in audit}
    ranked = sorted([d for d in proposals if d.action != "keep"], key=lambda d: (-priority[d.unit], d.unit))
    cap = len(ranked) if max_steps is None else int(max_steps)
    staged = []
    for decision in ranked[:cap]:
        row = by_unit[decision.unit]
        try:
            changed = router.apply(doc, baseline, decision)
        except UnsupportedDecision as exc:
            row.update(status="tool_unavailable", tool="unavailable", explanation=str(exc))
            continue
        selected = {i: a for i, a in changed.items() if i in linked_units(doc, decision.unit)}
        row.update(tool=selected[decision.unit].tool, replacement=selected[decision.unit].replacement)
        staged.append((decision, selected))
    for decision in ranked[cap:]:
        by_unit[decision.unit]["status"] = "not_executed_step_budget"

    def assemble(items):
        result = dict(baseline)
        for _, additions in items:
            result.update(additions)
        return result

    while staged:
        actions = assemble(staged)
        text = render(doc, actions)
        lexical = float(lexical_retention(vec, doc.text, [text])[0])
        if (
            lexical >= utility_floor
            and semantic_retention(doc, actions) >= content_floor
            and skills_retention(doc, actions) >= skill_floor
        ):
            break
        rejected, _ = staged.pop()
        by_unit[rejected.unit]["status"] = "rejected_utility"
    actions = assemble(staged)
    next_rr, next_proxy = plan_proxy(doc, actions, index, pos, plan_check, pseudo)
    improved = next_rr < proxy_rr - 1e-12 or (next_rr <= proxy_rr + 1e-12 and proxy - next_proxy >= min_self_gain)
    if staged and not improved:
        for decision, _ in staged:
            by_unit[decision.unit]["status"] = "rejected_no_improvement"
        actions = baseline
        reason = "plan_rejected_no_improvement"
    elif staged:
        for decision, _ in staged:
            by_unit[decision.unit].update(status="executed", applied=True)
        reason = "plan_applied"
    else:
        reason = "model_keep" if not ranked else "plan_rejected_utility_or_tool"
    for i, unit in enumerate(doc.units):
        if i not in by_unit:
            audit.append(
                {
                    "unit": i,
                    "action": actions[i].operation if i in actions else "keep",
                    "level": actions[i].level if i in actions else 0,
                    "tool": actions[i].tool if i in actions else NO_TOOL,
                    "applied": i in actions,
                    "reason": "mandatory_protection" if unit.role in {"DIRECT", "SENSITIVE"} else "linked_or_context",
                    "status": "final_state",
                }
            )
    final_rr, final_exposure = state_risk(doc, actions, index, pos)
    final_proxy_rr, final_proxy = plan_proxy(doc, actions, index, pos, plan_check, pseudo)
    if final_proxy_rr <= self_rr_target:
        reason = "threshold_met"
    if unknown := {row["status"] for row in audit} - set(DECISION_STATUSES):
        raise RuntimeError(f"Undocumented audit status: {unknown}")
    trace = [{**row, "step": j} for j, row in enumerate(audit) if row.get("decision")]
    trace.append(
        {
            "step": sum(row["status"] == "executed" for row in audit),
            "stop_reason": reason,
            "proxy_rr": final_rr,
            "self_exposure": final_exposure,
            "threshold_met": final_proxy_rr <= self_rr_target,
            "retention": float(lexical_retention(vec, doc.text, [render(doc, actions)])[0]),
            "content_retention": semantic_retention(doc, actions),
            "skills_retention": skills_retention(doc, actions),
            "plan_self_rr_before": rr,
            "plan_self_gain": exposure - final_exposure,
            "plan_check": plan_check,
            "plan_proxy_rr_before": proxy_rr,
            "plan_proxy_rr": final_proxy_rr,
            "plan_proxy_gain": proxy - final_proxy,
        }
    )
    return render(doc, actions), actions, trace, audit


RELEASE_STOPS = {
    "model_keep_all": "the model keeps every remaining entity",
    "proxy_target_met": "neighbour proxy RR at or below the target",
    "step_cap": "max_steps changes applied; remaining proposals not executed",
    "no_feasible_change": "every remaining proposal was rejected for utility or had no tool",
}


def release_sequential(
    doc,
    router,
    model,
    index,
    vec,
    pos,
    *,
    max_steps=None,
    utility_floor=0.65,
    content_floor=0.75,
    min_self_gain=0.01,
    self_rr_target=0.5,
    skill_floor=0.5,
    plan_check="self_exposure",
    neighbor_k=3,
    continue_on_keep=False,
):
    """Document-level loop: re-decide remaining entities from the current state, apply one change per step.

    A step applies the highest-priority non-keep decision (ranker x confidence)
    that passes the utility floors; failures are statused and dropped. Loop stops
    are listed in ``RELEASE_STOPS`` (``max_steps`` None = number of entities).
    Loop stops use the neighbour proxy; the ``plan_check`` verifier then
    compares the result with mandatory protection. With ``continue_on_keep``,
    a step where the model keeps everything takes each entity's most probable
    non-keep action instead, so only the proxy target, utility floors and the
    step cap end the loop.
    """
    if plan_check not in PLAN_CHECKS:
        raise ValueError(f"Unknown plan_check: {plan_check!r}")
    baseline = protected_actions(doc, router)
    pseudo = neighbor_targets(index, pos, doc.text, neighbor_k)
    remaining = [i for i, u in enumerate(doc.units) if u.role == "QUASI" and i == min(linked_units(doc, i))]
    decidable = set(remaining)
    available = {i: router.available(doc, i) for i in remaining}
    cap = len(remaining) if max_steps is None else int(max_steps)
    current, by_unit, executed, steps = dict(baseline), {}, [], []
    while True:
        rr, exposure = state_risk(doc, current, index, pos)
        proxy_rr, proxy_margin = (float(v[0]) for v in neighbor_proxy(index, pos, [render(doc, current)], pseudo))
        state = {"step": len(executed), "neighbor_rr": proxy_rr, "neighbor_margin": proxy_margin, "self_rr": rr}
        if proxy_rr <= self_rr_target:
            stop = "proxy_target_met"
            break
        records = [
            state_features(
                doc,
                i,
                current,
                index,
                router.codec,
                available[i],
                rr=rr,
                exposure=exposure,
                neighbor=(proxy_rr, proxy_margin),
                steps=len(executed),
            )
            for i in remaining
        ]
        proposals = model.decide(records, remaining, [available[i] for i in remaining]) if remaining else []
        if continue_on_keep and remaining and all(d.action == "keep" for d in proposals):
            proposals = model.decide(records, remaining, [available[i] for i in remaining], exclude_keep=True)
        scores = ranker_scores(getattr(model, "ranker", None), records, [d.risk_score for d in proposals])
        for d, score in zip(proposals, scores):
            by_unit[d.unit] = {
                **asdict(d),
                "decision": d.key,
                "entity_type": doc.units[d.unit].label,
                "status": "model_keep" if d.action == "keep" else "proposed",
                "tool": NO_TOOL,
                "applied": False,
                "priority": float(score) * d.confidence,
                "decided_at_step": len(executed),
            }
        ranked = sorted(
            [d for d in proposals if d.action != "keep"], key=lambda d: (-by_unit[d.unit]["priority"], d.unit)
        )
        if not ranked:
            stop = "model_keep_all"
            break
        if len(executed) >= cap:
            stop = "step_cap"
            break
        for decision in ranked:
            row = by_unit[decision.unit]
            remaining.remove(decision.unit)
            try:
                changed = router.apply(doc, current, decision)
            except UnsupportedDecision as exc:
                row.update(status="tool_unavailable", tool="unavailable", explanation=str(exc))
                continue
            row.update(tool=changed[decision.unit].tool, replacement=changed[decision.unit].replacement)
            lexical = float(lexical_retention(vec, doc.text, [render(doc, changed)])[0])
            if (
                lexical < utility_floor
                or semantic_retention(doc, changed) < content_floor
                or skills_retention(doc, changed) < skill_floor
            ):
                row["status"] = "rejected_utility"
                continue
            row["status"] = "executed"
            steps.append({**state, "unit": decision.unit, "decision": decision.key, "priority": row["priority"]})
            current = changed
            executed.append(decision.unit)
            break
        else:
            stop = "no_feasible_change"
            break
    pending = {"step_cap": "not_executed_step_budget", "proxy_target_met": "not_executed_target_met"}
    for row in by_unit.values():
        if row["status"] == "proposed":
            row["status"] = pending[stop]
    before_rr, before = plan_proxy(doc, baseline, index, pos, plan_check, pseudo)
    after_rr, after = plan_proxy(doc, current, index, pos, plan_check, pseudo)
    improved = after_rr < before_rr - 1e-12 or (after_rr <= before_rr + 1e-12 and before - after >= min_self_gain)
    reason = stop
    if executed and not improved:
        for unit in executed:
            by_unit[unit]["status"] = "rejected_no_improvement"
        current, reason = baseline, "plan_rejected_no_improvement"
        after_rr, after = before_rr, before
    for unit in executed if reason != "plan_rejected_no_improvement" else []:
        by_unit[unit]["applied"] = True
    audit = list(by_unit.values())
    for i, unit in enumerate(doc.units):
        if i not in by_unit:
            audit.append(
                {
                    "unit": i,
                    "action": current[i].operation if i in current else "keep",
                    "level": current[i].level if i in current else 0,
                    "tool": current[i].tool if i in current else NO_TOOL,
                    "applied": i in current,
                    "reason": "mandatory_protection"
                    if unit.role in {"DIRECT", "SENSITIVE"}
                    else "stopped_before_decision"
                    if i in decidable
                    else "linked_or_context",
                    "status": "final_state",
                }
            )
    if unknown := {row["status"] for row in audit} - set(DECISION_STATUSES):
        raise RuntimeError(f"Undocumented audit status: {unknown}")
    final_rr, final_exposure = state_risk(doc, current, index, pos)
    steps.append(
        {
            "step": len(executed),
            "stop_reason": reason,
            "loop_stop": stop,
            "proxy_rr": final_rr,
            "self_exposure": final_exposure,
            "threshold_met": after_rr <= self_rr_target,
            "retention": float(lexical_retention(vec, doc.text, [render(doc, current)])[0]),
            "content_retention": semantic_retention(doc, current),
            "skills_retention": skills_retention(doc, current),
            "plan_check": plan_check,
            "plan_proxy_rr_before": before_rr,
            "plan_proxy_rr": after_rr,
            "plan_proxy_gain": before - after,
        }
    )
    return render(doc, current), current, steps, audit
