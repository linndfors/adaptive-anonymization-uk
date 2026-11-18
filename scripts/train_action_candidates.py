"""Development-only learner comparison using frozen offline/on-policy labels.

This experiment changes the regression learner, not the people, features,
teacher targets or reserved test. Selection still minimizes validation action
regret. The resulting candidate requires validation operating-point selection.
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
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits
from train_action_onpolicy import add_costs_and_feasibility, balanced_person_source_weights, decision_metrics

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def run(args):
    started = time.monotonic()
    if not args.out.resolve().is_relative_to((ROOT / "artifacts").resolve()):
        raise ValueError("Private fitted candidates belong under artifacts/")
    args.out.mkdir(parents=True, exist_ok=False)
    with (args.onpolicy / "model.pkl").open("rb") as handle:
        parent = pickle.load(handle)
    source_files = [args.onpolicy / "model.pkl", args.development / "model.pkl",
                    *[directory / f"{split}_{suffix}.{extension}"
                      for split in ("train", "validation")
                      for directory, suffix, extension in (
                          (args.development, "features", "pkl"), (args.development, "labels", "csv"),
                          (args.onpolicy, "onpolicy_features", "pkl"), (args.onpolicy, "onpolicy_labels", "csv"))]]
    source_hashes = {str(path): sha(path) for path in source_files}
    protocol = {"method": parent.version, "purpose": "development_only", "test_used": False,
                "feature_contract": "unchanged frozen representation and labelled counterfactual gains",
                "source_files_sha256": source_hashes, "seed": args.seed,
                "selection": "lowest balanced feasible-action validation regret, then gain MAE",
                "comparison_reason": "residual action regret after first on-policy round",
                "scope": "adaptive development iteration; not preregistered confirmatory evidence"}
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    features, labels = {}, {}
    for split in ("train", "validation"):
        records, tables = [], []
        for source, directory, suffix in (("offline", args.development, ""), ("onpolicy", args.onpolicy, "onpolicy_")):
            with (directory / f"{split}_{suffix}features.pkl").open("rb") as handle:
                part = pickle.load(handle)
            table = pd.read_csv(directory / f"{split}_{suffix}labels.csv", dtype={"person": str, "id": str})
            if len(part) != len(table):
                raise ValueError("Cached feature/label counts differ")
            table["source"] = source
            records.extend(part)
            tables.append(table)
        features[split] = records
        labels[split] = add_costs_and_feasibility(records, pd.concat(tables, ignore_index=True), parent.settings)
    if set(labels["train"].person) & set(labels["validation"].person):
        raise ValueError("Training and validation people overlap")
    vectorizer = DictVectorizer(sparse=False, dtype=np.float32)
    train_matrix = vectorizer.fit_transform(features["train"])
    val_matrix = vectorizer.transform(features["validation"])
    parent_predictions = parent.predict_records(features["validation"])
    weights = balanced_person_source_weights(labels["train"])
    parent_metrics = decision_metrics(labels["validation"], parent_predictions)
    learners = [
        ("extra_trees_sqrt", ExtraTreesRegressor(n_estimators=120, max_depth=18, min_samples_leaf=6,
            max_features="sqrt", n_jobs=1, random_state=args.seed), weights),
        ("extra_trees_leaf20", ExtraTreesRegressor(n_estimators=100, max_depth=16, min_samples_leaf=20,
            max_features=.15, n_jobs=1, random_state=args.seed), weights),
    ]
    # Preserve equal person/source mass while giving rare consequential edits
    # more weight within each source. This is a candidate, never a claim that
    # the weighted predictor is a calibrated expected risk decrease.
    emphasized = weights * (1 + 4 * (np.abs(labels["train"].target_gain.to_numpy()) > .05))
    totals = pd.Series(emphasized).groupby([labels["train"].person, labels["train"].source]).transform("sum")
    source_count = labels["train"].groupby("person").source.transform("nunique").to_numpy()
    emphasized = emphasized / totals.to_numpy() / source_count
    emphasized *= len(emphasized) / emphasized.sum()
    learners.append(("boosting_consequential_edits", HistGradientBoostingRegressor(max_iter=160,
        max_leaf_nodes=15, min_samples_leaf=30, l2_regularization=2., learning_rate=.05,
        early_stopping=False, random_state=args.seed), emphasized))
    results, best_model, best = [], None, None
    for name, learner, fitting_weights in learners:
        learner.fit(train_matrix, labels["train"].target_gain, sample_weight=fitting_weights)
        row = {"learner": name, "parameters": learner.get_params(),
               **decision_metrics(labels["validation"], learner.predict(val_matrix))}
        results.append(row)
        if best is None or (row["mean_decision_regret"], row["person_weighted_gain_mae"]) < (
                best["mean_decision_regret"], best["person_weighted_gain_mae"]):
            best, best_model = row, learner
        print(json.dumps(row), flush=True)
    estimator = copy.copy(parent)
    estimator.model = make_pipeline(vectorizer, best_model)
    estimator.settings = dict(parent.settings)
    report = {**protocol, "parent_validation": parent_metrics, "model_search": results, "selected_model": best,
              "lower_regret_than_parent": best["mean_decision_regret"] < parent_metrics["mean_decision_regret"],
              "elapsed_seconds": time.monotonic() - started, "absolute_risk_prediction": None}
    estimator.metadata = {**parent.metadata, "learner_development": report}
    if any(sha(Path(path)) != digest for path, digest in source_hashes.items()):
        raise RuntimeError("Frozen inputs changed during learner comparison")
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    with (args.out / "model.pkl").open("xb") as handle:
        pickle.dump(estimator, handle)
    print(json.dumps({"selected": best, "parent": parent_metrics}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--onpolicy", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261112)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args)


if __name__ == "__main__":
    main()
