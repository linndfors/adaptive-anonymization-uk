"""One development-only round of autonomous action-policy data aggregation.

Roll out a frozen parent on training and validation people. Only training
actions fit the refit; validation actions select configurations. The reserved
test is neither rendered nor scored. No runtime source is modified.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from action_data import load_action_data
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits
from train_action_resume import MODEL_GRID, attack_views, register_policy

from anonrisk_uk.action_resume import ActionFeatures, ActionResumeRegressor, action_search, level_for
from anonrisk_uk.attack import ranking_result

ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def replay_trace(doc, trace):
    """Reconstruct visited policies, preserving stronger fragment overrides."""
    policy, visited = {}, []
    for action in trace:
        if "group" not in action:
            continue
        group, level = action["group"], int(action["level"])
        before = [level_for(doc, policy, i) for i in range(len(doc.units))]
        if action["scope"] == "fragment":
            changed = {**policy, **{f"@{int(i)}": level for i in action["units"]}}
        elif action["scope"] == "group":
            removable = {f"@{i}" for i, unit in enumerate(doc.units)
                         if unit.group == group and policy.get(f"@{i}", 0) <= level}
            changed = {key: value for key, value in policy.items() if key not in removable}
            changed[group] = level
        else:
            raise ValueError("Unsupported parent trace scope")
        if any(level_for(doc, changed, i) < value for i, value in enumerate(before)):
            raise ValueError("Parent trace would weaken a previous edit")
        policy = changed
        visited.append(dict(policy))
    return visited


def choose_contexts(doc, features, trace, *, seed, maximum=2):
    """Select unique noninitial parent-visited states without attack outcomes."""
    groups = features.groups
    excluded = {features.state({})[0], features.state({g: 1 for g in groups})[0]}
    unique = {}
    for policy in replay_trace(doc, trace):
        text = features.state(policy)[0]
        if text not in excluded:
            unique.setdefault(text, policy)
    policies = list(unique.values())
    if len(policies) > maximum:
        indices = sorted(np.random.default_rng(seed).choice(len(policies), size=maximum, replace=False))
        policies = [policies[i] for i in indices]
    return policies


def fast_teacher_scores(view, person, pos, outputs):
    """Same teacher formula as the parent, without repeated retrieval scoring."""
    excluded = [view["local"][pos]] if pos in view["local"] else []
    targets = [j for j in view["people"][str(person)] if j not in excluded]
    if not targets:
        raise ValueError("An on-policy query needs another same-person target")
    values = []
    for index in view["indexes"].values():
        actual, margins, best = ranking_result(index, outputs, [targets] * len(outputs),
                                               [excluded] * len(outputs), batch_size=64, ks=(1,))
        competitor = np.maximum(0., best - margins)
        separation = .5 + .5 * (best - competitor) / (np.abs(best) + np.abs(competitor) + 1e-9)
        values.append(.65 * actual.reciprocal_ranks + .35 * separation)
    return np.maximum.reduce(values)


def balanced_person_source_weights(frame):
    """Equal person mass; equal offline/on-policy mass when both are present."""
    source_counts = frame.groupby(["person", "source"])["person"].transform("size").to_numpy()
    available = frame.groupby("person")["source"].transform("nunique").to_numpy()
    weights = 1. / (source_counts * available)
    return weights * len(weights) / weights.sum()


def decision_metrics(frame, predictions, *, cost_weight=.3, min_gain=.005):
    """Balanced person/source regret over feasible validation decisions."""
    predictions = np.asarray(predictions)
    if len(predictions) != len(frame):
        raise ValueError("Prediction and validation row counts differ")
    decisions = []
    for (person, _, _, source), indices in frame.groupby(
            ["person", "id", "context", "source"], sort=False).indices.items():
        indices = np.asarray(indices)
        indices = indices[frame.iloc[indices].feasible.to_numpy(dtype=bool)]
        if not len(indices):
            continue
        target = frame.iloc[indices].target_gain.to_numpy()
        cost = frame.iloc[indices].cost.to_numpy()
        truth, estimate = target - cost_weight * cost, predictions[indices] - cost_weight * cost
        j = int(np.argmax(estimate))
        selected = float(truth[j]) if estimate[j] > min_gain else 0.
        oracle = max(0., float(truth.max()))
        correlation = (float(spearmanr(target, predictions[indices]).statistic)
                       if np.std(target) > 1e-9 and np.std(predictions[indices]) > 1e-9 else np.nan)
        decisions.append({"person": person, "source": source, "regret": oracle - selected,
                          "selected": selected, "oracle": oracle, "correlation": correlation})
    if not decisions:
        return {"mean_decision_regret": None, "contexts": 0, "persons": 0}
    context = pd.DataFrame(decisions)
    balanced = context.groupby(["person", "source"])[["regret", "selected", "oracle", "correlation"]].mean()
    averages = balanced.groupby("person").mean().mean()
    weights = balanced_person_source_weights(frame)
    return {"mean_decision_regret": float(averages.regret),
            "mean_selected_net_benefit": float(averages.selected),
            "mean_oracle_net_benefit": float(averages.oracle),
            "within_context_spearman": float(averages.correlation) if np.isfinite(averages.correlation) else None,
            "person_weighted_gain_mae": float(np.average(np.abs(frame.target_gain - predictions), weights=weights)),
            "contexts": len(context), "persons": int(context.person.nunique()),
            "weighting": "equal people; equal offline/on-policy source mass when both available"}


def add_costs_and_feasibility(records, labels, settings):
    labels = labels.copy()
    labels["cost"] = [.5 * max(0., r["lexical_loss"]) + .5 * max(0., r["detail_loss"]) for r in records]
    labels["feasible"] = [r["utility_after"] >= settings.get("utility_floor", .55)
                          and r["detail_after"] >= settings.get("detail_floor", .35) for r in records]
    return labels


def aggregate_split(data, split, estimator, documents, views, out, *, seed, maximum_contexts=2):
    """Label counterfactual edits only in states reached by the frozen parent."""
    records, labels, diagnostics = [], [], []
    started = time.monotonic()
    cache_dir = out / "checkpoints" / split
    cache_dir.mkdir(parents=True)
    for number, (raw_pos, doc) in enumerate(sorted(documents.items()), 1):
        pos = int(raw_pos)
        if str(data["splits"][pos]) != split:
            raise ValueError("Saved parent documents cross the supervised split boundary")
        features = ActionFeatures(doc, estimator)
        final, trace = action_search(doc, estimator, _features=features)
        replayed = replay_trace(doc, trace)
        if features.state(final)[0] != features.state(replayed[-1] if replayed else {})[0]:
            raise ValueError("Replayed parent trace differs from its released policy")
        identity = str(data["ids"][pos])
        row_seed = seed + int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16)
        contexts = choose_contexts(doc, features, trace, seed=row_seed, maximum=maximum_contexts)
        policy_index, policies, transitions = {}, [], []

        for context_id, current in enumerate(contexts, 100):
            base = register_policy(features, current, policy_index, policies)
            candidates = list(features.candidates(current))
            for candidate, record in zip(candidates, features.records(current, candidates)):
                after = register_policy(features, candidate.policy, policy_index, policies)
                transitions.append((base, after, context_id, record, candidate))
        outputs = [features.state(policy)[0] for policy in policies]
        teacher = (np.mean([fast_teacher_scores(view, data["people"][pos], pos, outputs) for view in views], axis=0)
                   if outputs else [])
        row_features, row_labels = [], []
        for base, after, context_id, record, candidate in transitions:
            row_features.append(record)
            row_labels.append({"person": str(data["people"][pos]), "id": identity,
                               "context": context_id, "source": "onpolicy",
                               "target_gain": float(teacher[base] - teacher[after]),
                               "base_teacher": float(teacher[base]), "after_teacher": float(teacher[after]),
                               "group": candidate.group, "level": candidate.level, "scope": candidate.scope})
        diagnostic = {"id": identity, "person": str(data["people"][pos]),
                      "visited_states": len(replayed), "selected_contexts": len(contexts),
                      "actions": len(row_labels), "distinct_states_scored": len(outputs)}
        with (cache_dir / f"{hashlib.sha256(identity.encode()).hexdigest()[:20]}.pkl").open("xb") as handle:
            pickle.dump({"features": row_features, "labels": row_labels, "diagnostic": diagnostic}, handle)
        records.extend(row_features)
        labels.extend(row_labels)
        diagnostics.append(diagnostic)
        if number % 10 == 0 or number == len(documents):
            print(f"{split} on-policy {number}/{len(documents)} resumes; {len(labels)} actions; "
                  f"{time.monotonic() - started:.0f}s", flush=True)
    frame = pd.DataFrame(labels)
    with (out / f"{split}_onpolicy_features.pkl").open("xb") as handle:
        pickle.dump(records, handle)
    frame.to_csv(out / f"{split}_onpolicy_labels.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(out / f"{split}_rollouts.csv", index=False)
    return records, frame


def run(args):
    if not args.out.resolve().is_relative_to((ROOT / "artifacts").resolve()):
        raise ValueError("Private action labels must remain under ignored artifacts/")
    args.out.mkdir(parents=True, exist_ok=False)
    with args.model.open("rb") as handle:
        parent = pickle.load(handle)
    if not isinstance(parent, ActionResumeRegressor):
        raise TypeError("Parent must be a fitted action-value artifact")
    data = load_action_data(args.data_dir)
    parent_dir = args.model.parent
    parent_protocol = json.loads((parent_dir / "protocol.json").read_text())
    if file_hash(args.data_dir / "manifest.json") != parent_protocol["data_manifest_sha256"]:
        raise ValueError("On-policy data differ from the frozen parent cohort")
    for relative, digest in parent_protocol.get("source_sha256", {}).items():
        if file_hash(ROOT / relative) != digest:
            raise ValueError(f"Parent training/runtime source has changed: {relative}")
    parent_inputs = [parent_dir / f"{split}_{kind}.{extension}"
                     for split in ("train", "validation")
                     for kind, extension in (("features", "pkl"), ("labels", "csv"), ("documents", "pkl"))]
    sources = [Path(__file__), ROOT / "scripts/train_action_resume.py", ROOT / "scripts/action_data.py",
               *sorted((ROOT / "src/anonrisk_uk").glob("*.py"))]
    protocol = {"method": parent.version, "purpose": "development_only", "round": 1,
                "parent_model_sha256": file_hash(args.model), "data_manifest_sha256": file_hash(args.data_dir / "manifest.json"),
                "cohort_sha256": file_hash(args.data_dir / "cohort.pkl"),
                "parent_inputs_sha256": {str(path): file_hash(path) for path in parent_inputs},
                "source_sha256": {str(path.relative_to(ROOT)): file_hash(path) for path in sources},
                "seed": args.seed, "maximum_contexts_per_resume": args.contexts,
                "test_used": False, "inference_access": "one resume and frozen artifact only",
                "reference_size": parent_protocol["reference_size"],
                "reference_views": parent_protocol["reference_views"],
                "teacher": "unchanged parent .65RR+.35bounded-separation, averaged reference views",
                "weighting": "equal people; 50/50 offline/on-policy when both sources available",
                "selection": "balanced combined validation feasible-action regret; no operating-point retuning",
                "decision_cost_weight": .3, "minimum_net_gain": parent.settings.get("min_gain", .005),
                "settings_preserved": dict(parent.settings),
                "context_deduplication": "exclude mandatory-only/all-level1 and duplicate visited rendered states; "
                    "original random-context identity unavailable, possible overlap disclosed"}
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    test_people = set(data["people"][data["splits"] == "test"])
    texts = [parent.codec.render(extracted, {}) if person not in test_people else ""
             for extracted, person in zip(data["extracted"], data["people"])]
    X, y, split_reports = {}, {}, {}
    for split in ("train", "validation"):
        with (parent_dir / f"{split}_features.pkl").open("rb") as handle:
            offline_X = pickle.load(handle)
        offline_y = pd.read_csv(parent_dir / f"{split}_labels.csv", dtype={"id": str, "person": str})
        offline_y["source"] = "offline"
        with (parent_dir / f"{split}_documents.pkl").open("rb") as handle:
            documents = pickle.load(handle)
        if len(offline_X) != len(offline_y):
            raise ValueError("Parent feature/label counts differ")
        if {str(data["ids"][int(k)]) for k in documents} != set(offline_y.id):
            raise ValueError("Saved parent documents do not match supervised query IDs")
        views = attack_views(data, split, parent_protocol["reference_size"],
                             parent_protocol["reference_views"], parent_protocol["seed"], texts)
        new_X, new_y = aggregate_split(data, split, parent, documents, views, args.out,
                                       seed=args.seed, maximum_contexts=args.contexts)
        X[split] = offline_X + new_X
        y[split] = add_costs_and_feasibility(
            X[split], pd.concat([offline_y, new_y], ignore_index=True), parent.settings)
        split_reports[split] = {"offline_actions": len(offline_y), "onpolicy_actions": len(new_y),
                                "people": int(y[split].person.nunique()), "queries": len(documents)}
    weights = balanced_person_source_weights(y["train"])
    models, scores = [], []
    min_gain = float(parent.settings.get("min_gain", .005))
    for i, parameters in enumerate(MODEL_GRID):
        model = make_pipeline(DictVectorizer(sparse=False, dtype=np.float32),
                              HistGradientBoostingRegressor(**parameters, learning_rate=.05,
                                  early_stopping=False, random_state=args.seed))
        model.fit(X["train"], y["train"].target_gain, histgradientboostingregressor__sample_weight=weights)
        prediction = model.predict(X["validation"])
        scores.append({"candidate": i, **parameters,
                       **decision_metrics(y["validation"], prediction, min_gain=min_gain)})
        models.append(model)
        print(f"on-policy fitted model {i}: {scores[-1]}", flush=True)
    best = min(scores, key=lambda r: (r["mean_decision_regret"], r["person_weighted_gain_mae"]))
    parent_prediction = parent.predict_records(X["validation"])
    parent_metrics = decision_metrics(y["validation"], parent_prediction, min_gain=min_gain)
    selected_prediction = models[best["candidate"]].predict(X["validation"])
    source_metrics = {}
    for source in ("offline", "onpolicy"):
        mask = y["validation"].source.eq(source).to_numpy()
        if mask.any():
            subset = y["validation"].loc[mask].reset_index(drop=True)
            source_metrics[source] = {"parent": decision_metrics(subset, parent_prediction[mask], min_gain=min_gain),
                                      "refit": decision_metrics(subset, selected_prediction[mask], min_gain=min_gain)}
    report = {**protocol, "splits": split_reports, "model_search": scores, "selected_model": best,
              "parent_validation": parent_metrics, "validation_by_source": source_metrics,
              "refit_has_lower_combined_regret": best["mean_decision_regret"] < parent_metrics["mean_decision_regret"],
              "absolute_risk_prediction": None,
              "limitations": "one parent-policy aggregation round; no guarantees, no final-test outcomes, "
                  "validation rollout diagnostics do not fit model weights"}
    estimator = copy.copy(parent)
    estimator.model = models[best["candidate"]]
    estimator.settings = dict(parent.settings)
    estimator.metadata = {"protocol": protocol, "validation": report,
                          "parent_metadata": parent.metadata, "risk_calibration": "none; relative action gains only"}
    if file_hash(args.model) != protocol["parent_model_sha256"]:
        raise RuntimeError("Parent model changed during aggregation")
    if file_hash(args.data_dir / "manifest.json") != protocol["data_manifest_sha256"]:
        raise RuntimeError("Frozen cohort manifest changed during aggregation")
    if file_hash(args.data_dir / "cohort.pkl") != protocol["cohort_sha256"]:
        raise RuntimeError("Frozen cohort records changed during aggregation")
    if any(file_hash(path) != digest for path, digest in protocol["parent_inputs_sha256"].items()):
        raise RuntimeError("Parent training inputs changed during aggregation")
    if any(file_hash(ROOT / relative) != digest for relative, digest in protocol["source_sha256"].items()):
        raise RuntimeError("Training/runtime sources changed during aggregation")
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame(scores).to_csv(args.out / "model_search.csv", index=False)
    with (args.out / "model.pkl").open("xb") as handle:
        pickle.dump(estimator, handle)
    print(json.dumps({"selected": best, "parent": parent_metrics, "splits": split_reports}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--contexts", type=int, choices=(1, 2), default=2)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args)


if __name__ == "__main__":
    main()
