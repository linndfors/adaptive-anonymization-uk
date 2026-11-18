"""Retrospective matched-controller controls on an already disclosed v5 test.

The deployed regressor is frozen. Two outcome-independent schedules and a
training-only action-type prior use the same candidate tools, feasibility floors
and step cap. No controller settings are selected on test outcomes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from action_data import load_action_data
from evaluate_action_resume import paired_comparisons, score_outputs
from threadpoolctl import threadpool_limits

from anonrisk_uk.action_resume import DEFAULT_SETTINGS, ActionFeatures, action_search
from anonrisk_uk.attack import BM25Index, TfidfIndex
from anonrisk_uk.experiment import Population
from anonrisk_uk.resume_adaptive import render
from anonrisk_uk.resume_parser import parse_resume
from anonrisk_uk.resume_resources import load_resume_resources

PRIORITY = ("age", "city", "exp_company", "exp_dates", "edu_name", "edu_dates", "salary",
            "add_education", "title", "exp_position", "skills", "exp_resp", "edu_faculty")
CONTROL_NAMES = ("fixed_round_robin_budget", "fixed_type_priority_budget", "action_type_prior")


class ActionTypePrior:
    """Person/source-balanced training mean; no resume context or test labels."""

    def __init__(self, frame):
        frame = frame.copy()
        required = {"person", "source", "group", "level", "scope", "target_gain"}
        if not required <= set(frame):
            raise ValueError("Incomplete training-only action labels")
        counts = frame.groupby(["person", "source"]).person.transform("size")
        sources = frame.groupby("person").source.transform("nunique")
        frame["weight"] = 1. / (counts * sources)
        frame["weighted_gain"] = frame.target_gain * frame.weight
        grouped = frame.groupby(["group", "level", "scope"])[["weighted_gain", "weight"]].sum()
        self.values = {(str(group), int(level), str(scope)): float(row.weighted_gain / row.weight)
                       for (group, level, scope), row in grouped.iterrows()}
        self.fallback = float(frame.weighted_gain.sum() / frame.weight.sum())

    def predict(self, records):
        return np.array([self.values.get((r["group"], int(r["level"]), r["scope"]), self.fallback)
                         for r in records])

    def export(self):
        return {"conditioning": ["group", "level", "scope"], "fallback": self.fallback,
                "weighting": "equal people and equal offline/on-policy source mass",
                "values": [{"group": g, "level": level, "scope": scope, "mean_gain": value}
                           for (g, level, scope), value in sorted(self.values.items())]}


def fixed_schedule(doc, estimator, *, priority=False, _features=None):
    """Fixed edit order; skip infeasible edits, never consult predicted gains."""
    settings = {**DEFAULT_SETTINGS, **estimator.settings}
    features = _features or ActionFeatures(doc, estimator)
    order = {group: i for i, group in enumerate(PRIORITY)}
    policy, trace = {}, []
    for _ in range(settings["max_steps"]):
        candidates, _ = features.context(policy, settings["fragment_limit"])
        feasible = [c for c in candidates if features.state(c.policy)[1] >= settings["utility_floor"]
                    and features.state(c.policy)[2] >= settings["detail_floor"]]
        if not feasible:
            break
        def key(candidate):
            suffix = (candidate.group, candidate.scope != "group", candidate.units)
            if priority:
                return (order.get(candidate.group, len(order)), candidate.level, *suffix)
            return (candidate.level, *suffix)
        chosen = min(feasible, key=key)
        policy = chosen.policy
        trace.append({"group": chosen.group, "level": chosen.level, "scope": chosen.scope,
                      "retention": features.state(policy)[1], "detail_retention": features.state(policy)[2]})
    return policy, trace


def run(args):
    if args.out.exists():
        raise FileExistsError("Follow-up controls already exist")
    original_protocol = json.loads((args.run / "protocol.json").read_text())
    model_hash = hashlib.sha256(args.model.read_bytes()).hexdigest()
    if model_hash != original_protocol["model_sha256"]:
        raise ValueError("Model differs from the disclosed frozen test")
    for relative, expected in original_protocol["source_sha256"].items():
        if hashlib.sha256(Path(relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen runtime changed: {relative}")
    if hashlib.sha256((args.data_dir / "manifest.json").read_bytes()).hexdigest() != original_protocol["data_manifest_sha256"]:
        raise ValueError("Frozen data changed")
    with args.model.open("rb") as handle:
        estimator = pickle.load(handle)
    data = load_action_data(args.data_dir)
    records = [json.loads(line) for line in (args.run / "releases.jsonl").read_text().splitlines()]
    by_id = {str(rid): i for i, rid in enumerate(data["ids"])}
    positions = [by_id[str(r["id"])] for r in records]
    labels, label_sources = [], {}
    for source, directory in (("offline", args.offline), ("onpolicy", args.onpolicy)):
        path = directory / "train_labels.csv"
        if not path.exists() and source == "onpolicy":
            path = directory / "train_onpolicy_labels.csv"
        frame = pd.read_csv(path, dtype={"person": str, "id": str})
        train_people = set(map(str, data["people"][data["splits"] == "train"]))
        if not set(frame.person) <= train_people:
            raise ValueError("Nontraining people in action-type prior")
        frame["source"] = source
        labels.append(frame)
        label_sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    prior = ActionTypePrior(pd.concat(labels, ignore_index=True))
    resources = load_resume_resources(args.resources_config)
    runtime = [Path(__file__), Path("scripts/evaluate_action_resume.py")]
    protocol = {"method": estimator.version, "selected": "action_regression", "purpose": "retrospective_controls",
                "queries": len(records), "persons": len({r["person"] for r in records}),
                "model_sha256": model_hash, "settings": estimator.settings,
                "data_manifest_sha256": original_protocol["data_manifest_sha256"],
                "parent_run": str(args.run), "seed": original_protocol["seed"],
                "pool_size": original_protocol["pool_size"], "reference_views": 1,
                "source_releases_sha256": hashlib.sha256((args.run / "releases.jsonl").read_bytes()).hexdigest(),
                "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in runtime},
                "training_labels_sha256": label_sources,
                "fixed_priority_order": list(PRIORITY),
                "fixed_round_robin": "lowest available level; alphabetical group; group scope before fragment",
                "fixed_type_priority": "listed type order; lowest level; group scope before fragment",
                "fixed_stop": "no feasible edit or identical v5 step cap; no outcome/gain stop",
                "ablation": "action_type_prior replaces only contextual predictor; identical v5 controller/settings",
                "same_candidate_tools_and_floors": True, "equal_actual_information_loss": False,
                "selection": "predeclared schedules; prior fitted only on original training labels; no test tuning",
                "limitation": "Original test outcomes were already disclosed. New controls are retrospective, not a new confirmatory holdout.",
                "bootstrap_samples": 2000, "bootstrap_unit": "person", "absolute_risk_prediction": None}
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    (args.out / "action_type_prior.json").write_text(json.dumps(prior.export(), indent=2))
    output_rows, traces = [], []
    saved_traces = {r["id"]: r for r in json.loads((args.run / "decision_traces.json").read_text())}
    with (args.out / "releases.jsonl").open("x") as handle:
        for number, (record, position) in enumerate(zip(records, positions), 1):
            if record["original"] != estimator.codec.render(data["extracted"][position], {}):
                raise ValueError("Source record no longer matches")
            doc = parse_resume(record["original"], estimator.codec, resources.detector)
            if render(doc, saved_traces[record["id"]]["policy"]) != record["outputs"]["action_regression"]:
                raise ValueError("Parser/renderer changed since frozen test")
            working = copy.copy(estimator)
            features = ActionFeatures(doc, working)
            outputs = {method: record["outputs"][method] for method in (
                "L0", "historical_L2", "matched_all_quasi", "action_regression")}
            for name, priority in ((CONTROL_NAMES[0], False), (CONTROL_NAMES[1], True)):
                policy, trace = fixed_schedule(doc, working, priority=priority, _features=features)
                outputs[name] = render(doc, policy)
                traces.append({"id": record["id"], "method": name, "policy": policy, "trace": trace})
            working.model = prior
            policy, trace = action_search(doc, working, _features=features)
            outputs[CONTROL_NAMES[2]] = render(doc, policy)
            traces.append({"id": record["id"], "method": CONTROL_NAMES[2], "policy": policy, "trace": trace})
            row = {**record, "outputs": outputs}
            output_rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"matched controls {number}/{len(records)}", flush=True)
    primary_ids = list(pd.read_csv(args.run / "pool_ids.csv", dtype=str).id)
    primary = [by_id[rid] for rid in primary_ids]
    texts = [estimator.codec.render(data["extracted"][i], {}) for i in primary]
    indexes = {"bm25": BM25Index(max_features=20000).fit(texts),
               "char": TfidfIndex(max_features=60000).fit(texts)}
    population = Population(primary_ids, [data["extracted"][i] for i in primary], data["people"][primary],
                            estimator.codec, indexes["bm25"], texts)
    local = {position: i for i, position in enumerate(primary)}
    outputs = {method: [r["outputs"][method] for r in output_rows] for method in output_rows[0]["outputs"]}
    frame, summary = score_outputs(population, indexes, [local[i] for i in positions], outputs, estimator.retention_vectorizer)
    paired = paired_comparisons(frame, "action_regression", ["historical_L2", *CONTROL_NAMES], indexes)
    frame.to_csv(args.out / "per_query.csv", index=False)
    pd.DataFrame({"id": primary_ids}).to_csv(args.out / "pool_ids.csv", index=False)
    (args.out / "decision_traces.json").write_text(json.dumps(traces, ensure_ascii=False))
    report = {**protocol, "summary": summary, "paired_differences_95ci": paired}
    for path, digest in {str(args.model): model_hash, **label_sources, **protocol["source_sha256"]}.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Frozen follow-up input changed: {path}")
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame(summary).to_csv(args.out / "summary.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("artifacts/evaluations/action_v5_test_20261110"))
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/action_v5_data_20261110"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/models/resume_action_v5/model.pkl"))
    parser.add_argument("--offline", type=Path, default=Path("artifacts/evaluations/action_v5_development_v2_20261110"))
    parser.add_argument("--onpolicy", type=Path, default=Path("artifacts/evaluations/action_v5_onpolicy_20261110"))
    parser.add_argument("--resources-config", default="configs/adaptive_resume.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args)


if __name__ == "__main__":
    main()
