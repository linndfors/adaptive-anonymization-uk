"""Train autonomous edit-benefit regression, selecting by validation decisions.

Multiple attacker reference populations label counterfactual edits. Neither
targets, person IDs nor live attacker features enter the fitted predictor.
The reserved test is never parsed, scored or used by this command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from action_data import load_action_data, reference_views
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

from anonrisk_uk.action_resume import (
    DEFAULT_SETTINGS,
    SCHEMA,
    VERSION,
    ActionFeatures,
    ActionResumeRegressor,
    action_search,
)
from anonrisk_uk.attack import BM25Index, TfidfIndex, ranking_result
from anonrisk_uk.e3jsi import regex_detect
from anonrisk_uk.learned_resume_risk import FrozenCodec, TrainingStatistics
from anonrisk_uk.policy import fixed_policies
from anonrisk_uk.resume_adaptive import render
from anonrisk_uk.resume_parser import parse_resume
from anonrisk_uk.resume_resources import load_resume_resources
from anonrisk_uk.workua import ResumeCodec

ROOT = Path(__file__).resolve().parents[1]
MODEL_GRID = [{"max_iter": 120, "max_leaf_nodes": 15, "min_samples_leaf": 30, "l2_regularization": 2.},
              {"max_iter": 160, "max_leaf_nodes": 31, "min_samples_leaf": 40, "l2_regularization": 5.},
              {"max_iter": 120, "max_leaf_nodes": 7, "min_samples_leaf": 40, "l2_regularization": 5.}]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def attack_views(data, split, size, count, seed, texts):
    result = []
    for number, positions in enumerate(reference_views(data, split, count=count, size=size, seed=seed)):
        # Different sizes and distractor samples, separately fitted vocabularies
        # and IDFs. Preserve complete same-person query clusters in each view.
        if count > 1 and number == 0:
            required = set(query_positions(data, split))
            distractors = [k for k in positions if k not in required]
            desired = max(len(required) + 50, size // 2)
            positions = sorted(required) + distractors[:max(0, desired - len(required))]
        values = [texts[k] for k in positions]
        indexes = {"bm25": BM25Index(max_features=30000).fit(values),
                   "char": TfidfIndex(max_features=60000).fit(values)}
        by_person = {}
        for local, global_pos in enumerate(positions):
            by_person.setdefault(str(data["people"][global_pos]), []).append(local)
        result.append({"positions": positions, "local": {k: j for j, k in enumerate(positions)},
                       "people": by_person, "indexes": indexes})
        print(f"{split} teacher reference {number + 1}/{count}: {len(positions)} documents", flush=True)
    return result


def register_policy(features, policy, seen, policies):
    output = features.state(policy)[0]
    if output not in seen:
        seen[output] = len(policies)
        policies.append(policy)
    return seen[output]


def teacher_scores(view, person, pos, outputs):
    """Worst-attack RR plus a smooth same-person margin to teach plateau edits.

    Each constituent is bounded [0,1]. The continuous term is supplementary,
    never an identification probability or an inference-time input.
    """
    excluded = [view["local"][pos]] if pos in view["local"] else []
    targets = [i for i in view["people"][str(person)] if i not in excluded]
    if not targets:
        raise ValueError("A counterfactual query has no independent same-person target")
    risks, metrics = [], {}
    for name, index in view["indexes"].items():
        actual = ranking_result(index, outputs, [targets] * len(outputs), [excluded] * len(outputs),
                                batch_size=64, ks=(1,))[0]
        scores = index.scores(outputs)
        best = scores[:, targets].max(axis=1)
        scores[:, targets + excluded] = -np.inf
        competitor = np.maximum(0., scores.max(axis=1))
        separation = .5 + .5 * (best - competitor) / (np.abs(best) + np.abs(competitor) + 1e-9)
        risks.append(.65 * actual.reciprocal_ranks + .35 * separation)
        metrics[name] = actual
    return np.maximum.reduce(risks), metrics


def query_positions(data, split):
    positions = data["query_positions"]
    if isinstance(positions, dict):
        return list(map(int, positions[split]))
    return [int(k) for k in positions if data["splits"][k] == split]


def label_split(data, split, estimator, views, texts, out, seed, max_contexts, detector=regex_detect,
                max_train_queries_per_person=1):
    records, outcomes, documents = [], [], {}
    rng = np.random.default_rng(seed)
    positions = query_positions(data, split)
    if split == "train" and max_train_queries_per_person:
        counts, retained = {}, []
        for k in rng.permutation(positions):
            person = str(data["people"][k])
            if counts.get(person, 0) < max_train_queries_per_person:
                retained.append(int(k))
                counts[person] = counts.get(person, 0) + 1
        positions = sorted(retained)
    for number, k in enumerate(positions, 1):
        # WorkUA serialization has public headings. The exact same parser is
        # used in validation/test; no hidden extracted fields are supplied.
        doc = parse_resume(texts[k], estimator.codec, detector)
        documents[k] = doc
        features = ActionFeatures(doc, estimator)
        groups = sorted(features.groups)
        contexts = [{}, {group: 1 for group in groups}]
        contexts.extend({group: int(rng.integers(0, 4)) for group in groups}
                        for _ in range(max(0, max_contexts - 2)))
        seen, transitions, policies = {}, [], []

        for context_id, current in enumerate(contexts):
            base = register_policy(features, current, seen, policies)
            candidates = list(features.candidates(current))
            action_records = features.records(current, candidates)
            for candidate, record in zip(candidates, action_records):
                after = register_policy(features, candidate.policy, seen, policies)
                transitions.append((base, after, context_id, record, candidate))
        outputs = [features.state(policy)[0] for policy in policies]
        teacher = np.mean([teacher_scores(view, data["people"][k], k, outputs)[0] for view in views], axis=0)
        for base, after, context_id, record, candidate in transitions:
            records.append(record)
            outcomes.append({"person": str(data["people"][k]), "id": str(data["ids"][k]),
                             "context": context_id, "target_gain": float(teacher[base] - teacher[after]),
                             "base_teacher": float(teacher[base]), "after_teacher": float(teacher[after]),
                             "group": candidate.group, "level": candidate.level, "scope": candidate.scope})
        if number % 10 == 0 or number == len(positions):
            print(f"{split} labelled {number}/{len(positions)} resumes; {len(records)} actions", flush=True)
    with (out / f"{split}_features.pkl").open("xb") as handle:
        pickle.dump(records, handle)
    pd.DataFrame(outcomes).to_csv(out / f"{split}_labels.csv", index=False)
    with (out / f"{split}_documents.pkl").open("xb") as handle:
        pickle.dump(documents, handle)
    return records, pd.DataFrame(outcomes), documents


def decision_metrics(frame, predictions, cost_weight=.3):
    predictions = np.asarray(predictions)
    regrets, selected, oracle, correlations = [], [], [], []
    # Validation decisions, not global calibration, determine model selection.
    for ii in frame.groupby(["id", "context"], sort=False).indices.values():
        target = frame.iloc[ii].target_gain.to_numpy()
        cost = frame.iloc[ii].cost.to_numpy()
        truth = target - cost_weight * cost
        estimate = predictions[ii] - cost_weight * cost
        j = int(np.argmax(estimate))
        chosen = float(truth[j]) if estimate[j] > .005 else 0.
        best = max(0., float(truth.max()))
        regrets.append(best - chosen)
        selected.append(chosen)
        oracle.append(best)
        if np.std(target) > 1e-9 and np.std(predictions[ii]) > 1e-9:
            correlations.append(float(spearmanr(target, predictions[ii]).statistic))
    weights = 1 / frame.groupby("person").person.transform("size").to_numpy()
    return {"mean_decision_regret": float(np.mean(regrets)),
            "mean_selected_net_benefit": float(np.mean(selected)),
            "mean_oracle_net_benefit": float(np.mean(oracle)),
            "within_context_spearman": float(np.mean(correlations)) if correlations else None,
            "person_weighted_gain_mae": float(np.average(np.abs(frame.target_gain - predictions), weights=weights)),
            "contexts": len(regrets)}


def score_policies(data, positions, texts, outputs, views, retention):
    records = []
    for name, values in outputs.items():
        row = {"method": name}
        recalls = {"bm25": [], "char": []}
        rr = {"bm25": [], "char": []}
        for view in views:
            targets, excludes = [], []
            for k in positions:
                excluded = [view["local"][k]] if k in view["local"] else []
                excludes.append(excluded)
                targets.append([j for j in view["people"][str(data["people"][k])] if j not in excluded])
            for name_ix, index in view["indexes"].items():
                actual = ranking_result(index, values, targets, excludes, ks=(1,))[0]
                recalls[name_ix].append(actual.recalls[1])
                rr[name_ix].append(actual.reciprocal_ranks)
        for attacker, recall_arrays in recalls.items():
            row[f"{attacker}_recall_at_1"] = float(np.mean(recall_arrays))
            row[f"{attacker}_mrr"] = float(np.mean(rr[attacker]))
        source = retention.transform([texts[k] for k in positions])
        row["lexical_retention"] = float(retention.transform(values).multiply(source).sum(axis=1).mean())
        records.append(row)
    return records


def run(args):
    started = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=False)
    data = load_action_data(args.data_dir)
    resources = load_resume_resources(args.resources_config)
    train = query_positions(data, "train")
    validation = query_positions(data, "validation")
    test = query_positions(data, "test")
    train_people = {str(data["people"][k]) for k in train}
    val_people = {str(data["people"][k]) for k in validation}
    test_people = {str(data["people"][k]) for k in test}
    if train_people & val_people or train_people & test_people or val_people & test_people:
        raise ValueError("Supervised person splits overlap")
    # The test is reserved but not rendered/parsed/scored. Fit statistics on
    # train-reference rows only, excluding all validation and test people.
    base_codec = ResumeCodec(min_freq=20)
    texts = [base_codec.render(extracted, {}) if str(person) not in test_people else ""
             for extracted, person in zip(data["extracted"], data["people"])]
    train_positions = reference_views(data, "train", count=1, size=args.reference_size, seed=args.seed)[0]
    if any(str(data["people"][k]) in val_people | test_people for k in train_positions):
        raise ValueError("Nontraining people enter frozen statistics")
    codec = FrozenCodec.from_codec(ResumeCodec(min_freq=20).fit([data["extracted"][k] for k in train_positions]))
    statistics = TrainingStatistics.fit([texts[k] for k in train_positions])
    retention = TfidfVectorizer(sublinear_tf=True).fit([texts[k] for k in train_positions])
    estimator = ActionResumeRegressor(None, statistics, codec, retention, dict(DEFAULT_SETTINGS), {})
    protocol = {"method": VERSION, "schema": SCHEMA, "purpose": "development_only", "seed": args.seed,
                "train_people": len(train_people), "validation_people": len(val_people),
                "reserved_test_people": len(test_people), "test_used": False,
                "reference_size": args.reference_size, "reference_views": args.reference_views,
                "parser": "public-heading parser + configured local regex/NER; no hidden fields or API",
                "resources_config": args.resources_config,
                "target": "mean over reference views of decrease in max_attack(0.65*S2_RR+0.35*bounded_S2_score_separation)",
                "counterfactual_contexts": args.contexts,
                "train_label_queries_per_person": args.train_queries_per_person,
                "synthetic_note": "counterfactual edited states are augmentation, not independent people; no fabricated person labels",
                "selection": "model by validation action regret, operating point by validation privacy/content tradeoff",
                "model_grid": MODEL_GRID, "inference_access": "one resume and frozen artifact only",
                "data_manifest_sha256": hashlib.sha256((args.data_dir / "manifest.json").read_bytes()).hexdigest()}
    sources = [Path(__file__), ROOT / "scripts/action_data.py", *sorted((ROOT / "src/anonrisk_uk").glob("*.py"))]
    protocol["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    write_json(args.out / "protocol.json", protocol)
    with (args.out / "feature_resources.pkl").open("xb") as handle:
        pickle.dump(estimator, handle)
    views = {split: attack_views(data, split, args.reference_size, args.reference_views, args.seed, texts)
             for split in ("train", "validation")}
    train_X, train_y, _ = label_split(data, "train", estimator, views["train"], texts, args.out,
                                     args.seed, args.contexts, resources.detector, args.train_queries_per_person)
    val_X, val_y, val_docs = label_split(data, "validation", estimator, views["validation"], texts,
                                       args.out, args.seed + 1, args.contexts, resources.detector)
    for X, y in ((train_X, train_y), (val_X, val_y)):
        y["cost"] = [.5 * max(0., x["lexical_loss"]) + .5 * max(0., x["detail_loss"]) for x in X]
    weights = 1 / train_y.groupby("person").person.transform("size").to_numpy()
    weights *= len(weights) / weights.sum()
    models, scores = [], []
    for i, params in enumerate(MODEL_GRID):
        model = make_pipeline(DictVectorizer(sparse=False, dtype=np.float32),
                              HistGradientBoostingRegressor(**params, learning_rate=.05,
                                  early_stopping=False, random_state=args.seed))
        model.fit(train_X, train_y.target_gain, histgradientboostingregressor__sample_weight=weights)
        scores.append({"candidate": i, **params, **decision_metrics(val_y, model.predict(val_X))})
        models.append(model)
        print(f"fitted model {i}: {scores[-1]}", flush=True)
    best = min(scores, key=lambda r: (r["mean_decision_regret"], r["person_weighted_gain_mae"]))
    estimator.model = models[best["candidate"]]
    constant = np.full(len(val_y), np.average(train_y.target_gain, weights=weights))
    baselines = {"constant_gain": decision_metrics(val_y, constant),
                 "no_action": {"mean_decision_regret": best["mean_oracle_net_benefit"]}}
    outputs = {"historical_L2": [codec.render(data["extracted"][k], fixed_policies()["L2_generalize"])
                                  for k in validation]}
    settings_grid = [{"utility_floor": floor, "detail_floor": .35, "cost_weight": cost}
                     for floor in (.55, .65) for cost in (0., .15, .4, .8)]
    outputs.update({f"action_{i}": [] for i in range(len(settings_grid))})
    for number, k in enumerate(validation, 1):
        features = ActionFeatures(val_docs[k], estimator)
        for i, settings in enumerate(settings_grid):
            policy, _ = action_search(val_docs[k], estimator, _features=features, **settings)
            outputs[f"action_{i}"].append(render(val_docs[k], policy))
        if number % 10 == 0 or number == len(validation):
            print(f"validation policies {number}/{len(validation)} resumes", flush=True)
    policy_rows = score_policies(data, validation, texts, outputs, views["validation"], retention)
    l2 = policy_rows[0]
    # Strong privacy is primary; tradeoffs/negative findings remain in the table.
    # A candidate must not worsen either attack point estimate versus L2. Among
    # feasible candidates maximize lexical retention, then minimize linkage.
    candidates = policy_rows[1:]
    feasible = [r for r in candidates if all(r[f"{a}_recall_at_1"] <= l2[f"{a}_recall_at_1"] for a in ("bm25", "char"))]
    selected = (max(feasible, key=lambda r: r["lexical_retention"]) if feasible else
                min(candidates, key=lambda r: max(r["bm25_recall_at_1"], r["char_recall_at_1"])))
    selected_index = int(selected["method"].split("_")[-1])
    estimator.settings.update(settings_grid[selected_index])
    estimator.metadata = {"protocol": protocol, "validation": {"selected_model": best,
        "baselines": baselines, "selected_policy": selected, "privacy_point_constraint_met": bool(feasible)},
        "risk_calibration": "none; action gains are not absolute risk"}
    if any(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != digest
           for relative, digest in protocol["source_sha256"].items()):
        raise RuntimeError("Training sources changed during the run; do not publish this artifact")
    report = {**protocol, "model_search": scores, "action_baselines": baselines,
              "policy_validation": policy_rows, "selected_settings": estimator.settings,
              "selected_policy": selected, "privacy_point_constraint_met": bool(feasible),
              "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              "elapsed_seconds": time.monotonic() - started,
              "limitations": "validation selection; no absolute safety prediction; semantic/factual quality requires separate evaluation"}
    write_json(args.out / "report.json", report)
    pd.DataFrame(scores).to_csv(args.out / "model_search.csv", index=False)
    pd.DataFrame(policy_rows).to_csv(args.out / "policy_validation.csv", index=False)
    with (args.out / "model.pkl").open("xb") as handle:
        pickle.dump(estimator, handle)
    print(json.dumps({"selected_model": best, "selected_policy": selected, "settings": estimator.settings}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--resources-config", default="configs/baselines/adaptive_resume_v3_3.json")
    parser.add_argument("--reference-size", type=int, default=6000)
    parser.add_argument("--reference-views", type=int, default=2)
    parser.add_argument("--contexts", type=int, default=3)
    parser.add_argument("--train-queries-per-person", type=int, default=1,
                        help="Label a seeded representative query per independent training person; 0 uses all")
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args)


if __name__ == "__main__":
    main()
