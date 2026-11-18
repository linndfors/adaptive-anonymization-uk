"""Train a pool-free document-state risk regressor on development people only.

Private reference documents and person links create S2 labels at training time.
The deployable artifact contains weights and frozen statistics, not that pool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from evaluate_adaptive import replay_pool, used_ids
from group_policy_eval import holdout_pool
from scipy.stats import spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from threadpoolctl import threadpool_limits

from anonrisk_uk.attack import BM25Index, TfidfIndex, ranking_result
from anonrisk_uk.experiment import Population
from anonrisk_uk.learned_resume import learned_search
from anonrisk_uk.learned_resume_risk import (
    SCHEMA,
    VERSION,
    FeatureSubsetRegressor,
    FrozenCodec,
    ResumeRiskRegressor,
    TrainingStatistics,
    state_policies,
)
from anonrisk_uk.policy import fixed_policies
from anonrisk_uk.resume_adaptive import NeighbourProxy, render, search
from anonrisk_uk.resume_parser import parse_resume
from anonrisk_uk.resume_resources import load_resume_resources
from anonrisk_uk.workua import ResumeCodec

ROOT = Path(__file__).resolve().parents[1]
GRID = [{"max_iter": 100, "learning_rate": .05, "max_leaf_nodes": 15, "min_samples_leaf": 10,
         "l2_regularization": .1},
        {"max_iter": 150, "learning_rate": .05, "max_leaf_nodes": 15, "min_samples_leaf": 20,
         "l2_regularization": .1},
        {"max_iter": 200, "learning_rate": .03, "max_leaf_nodes": 31, "min_samples_leaf": 20,
         "l2_regularization": 1.}]


def prediction_metrics(frame, predictions):
    weights = 1 / frame.groupby("person").person.transform("size").to_numpy()
    correlations = []
    for indices in frame.groupby("id").indices.values():
        a, b = frame.iloc[indices].target.to_numpy(), np.asarray(predictions)[indices]
        if np.std(a) > 1e-8 and np.std(b) > 1e-8:
            correlations.append(float(spearmanr(a, b).statistic))
    return {"person_weighted_mae": float(np.average(np.abs(frame.target.to_numpy() - predictions), weights=weights)),
            "within_document_spearman": float(np.mean(correlations)) if correlations else None,
            "ranked_documents": len(correlations)}


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    resources = load_resume_resources(args.resources_config)
    if args.pool_ids:
        ids, extracted, people = replay_pool(args.pool_ids)
    else:
        used, _ = used_ids(ROOT)
        ids, extracted, people, _ = holdout_pool(ROOT / "data/raw/all_resumes.csv", used, 8000, args.seed)
    people = np.asarray(people)
    texts = [resources.codec.render(ex, {}) for ex in extracted]
    indexes = {"bm25": BM25Index(max_features=20000).fit(texts),
               "char": TfidfIndex(max_features=60000).fit(texts)}
    population = Population(ids, extracted, people, resources.codec, indexes["bm25"], texts)
    linked = population.linked_positions()
    eligible = [p for p in sorted(set(people[linked]))
                if all(0 < len(texts[k]) <= 20000 for k in np.flatnonzero(people == p))]
    rng = np.random.default_rng(args.seed)
    selected = list(rng.permutation(eligible)[:args.max_persons])
    if len(selected) < 20:
        raise ValueError("Need at least twenty development people with linked resumes")
    n_val = max(5, int(np.ceil(len(selected) * .2)))
    validation_people = set(selected[:n_val])
    training_people = set(selected[n_val:])
    assert not training_people & validation_people
    queries = [int(k) for k in linked if people[k] in set(selected)]
    reference_rows = [k for k in range(len(ids)) if people[k] not in validation_people]
    codec = FrozenCodec.from_codec(ResumeCodec(min_freq=20).fit([extracted[k] for k in reference_rows]))
    statistics = TrainingStatistics.fit([texts[k] for k in reference_rows])
    retention = TfidfVectorizer(sublinear_tf=True).fit([texts[k] for k in reference_rows])
    protocol = {"method": VERSION, "feature_schema": SCHEMA, "train_only_lsa_dimensions": 32,
                "purpose": "development training and validation only", "seed": args.seed,
                "pool_size": len(ids), "queries": len(queries), "train_people": len(training_people),
                "validation_people": len(validation_people), "reference_documents": len(reference_rows),
                "target": "max(BM25 same-person reciprocal rank, character same-person reciprocal rank)",
                "inference_features": "text/type/transformation features and train-only frozen statistics; no live search",
                "state_design": "mandatory-only, all-group levels, single-group levels and eight seeded mixed states",
                "model_grid": GRID, "policy_targets": [.1, .2, .3, .4], "utility_floor": .55,
                "test_used": False, "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [Path(__file__), ROOT / "src/anonrisk_uk/learned_resume_risk.py",
                              ROOT / "src/anonrisk_uk/learned_resume.py", ROOT / "src/anonrisk_uk/resume_adaptive.py",
                              ROOT / "src/anonrisk_uk/resume_parser.py", ROOT / "src/anonrisk_uk/resume_context.py"]}}
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    pd.DataFrame({"id": ids}).to_csv(args.out / "pool_ids.csv", index=False)
    features, labels, documents = [], [], {}
    for number, k in enumerate(queries, 1):
        doc = parse_resume(texts[k], codec, resources.detector)
        documents[k] = doc
        policies = state_policies(doc, rng)
        outputs = [render(doc, policy) for policy in policies]
        targets, excludes = population.targets_and_excludes([k] * len(outputs), "S2")
        risks = np.maximum.reduce([ranking_result(index, outputs, targets, excludes)[0].reciprocal_ranks
                                  for index in indexes.values()])
        for policy, risk in zip(policies, risks):
            features.append(statistics.features(doc, policy))
            labels.append({"id": str(ids[k]), "person": str(people[k]), "target": float(risk),
                           "split": "validation" if people[k] in validation_people else "train"})
        print(f"labelled {number}/{len(queries)}; states {len(labels)}", flush=True)
    X, y = pd.DataFrame(features), pd.DataFrame(labels)
    X.to_csv(args.out / "features.csv", index=False)
    y.to_csv(args.out / "labels.csv", index=False)
    train, val = y.split.eq("train"), y.split.eq("validation")
    weight = 1 / y.loc[train].groupby("person").person.transform("size").to_numpy()
    weight *= len(weight) / weight.sum()
    validation = y.loc[val].reset_index(drop=True)
    results, models = [], []
    for i, parameters in enumerate(GRID):
        model = HistGradientBoostingRegressor(**parameters, early_stopping=False, random_state=args.seed)
        model.fit(X.loc[train], y.loc[train].target, sample_weight=weight)
        results.append({"candidate": i, **parameters, **prediction_metrics(validation, model.predict(X.loc[val]))})
        models.append(model)
    best = min(results, key=lambda row: (row["person_weighted_mae"], row["max_iter"]))
    model = models[best["candidate"]]
    dummy = DummyRegressor().fit(X.loc[train], y.loc[train].target, sample_weight=weight)
    constant = prediction_metrics(validation, dummy.predict(X.loc[val]))
    structural_columns = [c for c in X if c.endswith(("_original", "_visible", "_changed", "_level"))]
    structural = HistGradientBoostingRegressor(**GRID[0], early_stopping=False, random_state=args.seed)
    structural.fit(X.loc[train, structural_columns], y.loc[train].target, sample_weight=weight)
    structural_metrics = prediction_metrics(validation, structural.predict(X.loc[val, structural_columns]))
    regression_report = {"selected": best, "constant_mean": constant, "structure_only": structural_metrics,
                         "training_states": int(train.sum()), "validation_states": int(val.sum()),
                         "benefit_over_constant_mae": best["person_weighted_mae"] < constant["person_weighted_mae"]}
    (args.out / "regression_validation.json").write_text(json.dumps(regression_report, indent=2))
    pd.DataFrame(results).to_csv(args.out / "model_search.csv", index=False)
    estimator = ResumeRiskRegressor(model, statistics, list(X), codec, retention, .2,
                                   {"validation": regression_report, "protocol": protocol})
    validation_queries = [k for k in queries if people[k] in validation_people]
    methods = {"L2": [], "search_v3_3": [], **{f"regression_{t}": [] for t in protocol["policy_targets"]}}
    scores = []
    for number, k in enumerate(validation_queries, 1):
        doc = documents[k]
        methods["L2"].append(codec.render(extracted[k], fixed_policies()["L2_generalize"]))
        proxy = NeighbourProxy(indexes, texts, texts[k])
        actions, _ = search(doc, proxy, retention, target=0., utility_floor=.55)
        methods["search_v3_3"].append(render(doc, actions))
        for target in protocol["policy_targets"]:
            actions, _ = learned_search(doc, estimator, target=target, utility_floor=.55)
            methods[f"regression_{target}"].append(render(doc, actions))
        print(f"validation policy {number}/{len(validation_queries)}", flush=True)
    targets, excludes = population.targets_and_excludes(validation_queries, "S2")
    original_matrix = retention.transform([texts[k] for k in validation_queries])
    for name, outputs in methods.items():
        row = {"method": name}
        for attacker, index in indexes.items():
            result = ranking_result(index, outputs, targets, excludes)[0]
            row[f"{attacker}_recall_at_1"] = float(result.recalls[1].mean())
            row[f"{attacker}_mrr"] = float(result.reciprocal_ranks.mean())
        row["lexical_retention"] = float(retention.transform(outputs).multiply(original_matrix).sum(axis=1).mean())
        scores.append(row)
    l2 = next(row for row in scores if row["method"] == "L2")
    candidates = [row for row in scores if row["method"].startswith("regression_")]
    eligible = [row for row in candidates if all(row[f"{a}_recall_at_1"] <= l2[f"{a}_recall_at_1"] + .05
                                               for a in indexes)]
    selected_policy = (max(eligible, key=lambda row: row["lexical_retention"]) if eligible else
                       min(candidates, key=lambda row: max(row[f"{a}_recall_at_1"] for a in indexes)))
    estimator.target = float(selected_policy["method"].split("_")[1])
    estimator.metadata["policy_validation"] = {"selected": selected_policy, "privacy_constraint_met": bool(eligible)}
    report = {**protocol, "regression": regression_report, "policy_summary": scores,
              "selected_target": estimator.target, "validation_privacy_constraint_met": bool(eligible),
              "ready_for_testing": bool(regression_report["benefit_over_constant_mae"] and eligible),
              "note": "development result; not a final test, not a calibrated probability"}
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame(scores).to_csv(args.out / "policy_validation.csv", index=False)
    with (args.out / "model.pkl").open("xb") as handle:
        pickle.dump(estimator, handle)
    ablation = ResumeRiskRegressor(FeatureSubsetRegressor(structural, structural_columns), statistics, list(X),
                                  codec, retention, estimator.target, {"ablation": "structure-only risk regression"})
    with (args.out / "structure_model.pkl").open("xb") as handle:
        pickle.dump(ablation, handle)
    print(json.dumps({"regression": regression_report, "policy": selected_policy, "ready_for_testing": report["ready_for_testing"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resources-config", default="configs/baselines/adaptive_resume_v3_3.json")
    parser.add_argument("--pool-ids", type=Path)
    parser.add_argument("--max-persons", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args)


if __name__ == "__main__":
    main()
