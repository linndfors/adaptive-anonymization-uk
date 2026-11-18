"""Evaluate a frozen autonomous action regressor on reserved real people.

Reference indexes are available only to declared search controls and attack
scoring. Saved releases are compatible with evaluate_resume_content.py. Raw
texts, IDs and per-person results belong in ignored local artifact directories.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from action_fingerprints import frozen_features_digest
from group_policy_eval import person_bootstrap
from joblib import hash as model_hash
from threadpoolctl import threadpool_limits

from anonrisk_uk.action_learning import lexical_retention
from anonrisk_uk.attack import BM25Index, TfidfIndex, ranking_result
from anonrisk_uk.experiment import Population
from anonrisk_uk.learned_resume import anonymize_learned
from anonrisk_uk.policy import fixed_policies
from anonrisk_uk.resume_adaptive import NeighbourProxy, check_direct_identifiers, options, render, search
from anonrisk_uk.resume_parser import parse_resume
from anonrisk_uk.resume_resources import load_resume_resources

ROOT = Path(__file__).resolve().parents[1]
SELECTED = "action_regression"


def deployment_digest(estimator):
    """Ignore descriptive metadata when identifying identical ablation policies."""
    digest = hashlib.sha256()
    digest.update(frozen_features_digest(estimator).encode())
    # Boosting losses contain Cython objects whose repr includes a process
    # address. Hash fitted state instead of addresses or pickle alias layout.
    normalized = pickle.loads(pickle.dumps(estimator.model, protocol=5))
    digest.update(model_hash(normalized, hash_name="sha1").encode())
    digest.update(json.dumps(estimator.settings, sort_keys=True).encode())
    return digest.hexdigest()


def score_outputs(population, indexes, queries, outputs, retention_vectorizer):
    """Score identical saved releases under each attacker, preserving person rows."""
    if not queries or not outputs or any(len(values) != len(queries) for values in outputs.values()):
        raise ValueError("Every method needs exactly one output per query")
    frame = pd.DataFrame({"id": [str(population.ids[k]) for k in queries],
                          "person": [str(population.clusters[k]) for k in queries]})
    targets, excludes = population.targets_and_excludes(queries, "S2")
    summary = []
    for method, values in outputs.items():
        row = {"method": method, "queries": len(queries)}
        reciprocal_ranks = []
        for attacker, index in indexes.items():
            result = ranking_result(index, values, targets, excludes, batch_size=32)[0]
            frame[f"{method}_{attacker}_r1"] = result.recalls[1]
            frame[f"{method}_{attacker}_rr"] = result.reciprocal_ranks
            row[f"{attacker}_recall_at_1"] = float(result.recalls[1].mean())
            row[f"{attacker}_mrr"] = float(result.reciprocal_ranks.mean())
            reciprocal_ranks.append(result.reciprocal_ranks)
        frame[f"{method}_actual_worst_rr"] = np.maximum.reduce(reciprocal_ranks)
        lexical = [float(lexical_retention(retention_vectorizer, population.texts[k], [value])[0])
                   for k, value in zip(queries, values)]
        frame[f"{method}_lexical"] = lexical
        row["lexical_retention"] = float(np.mean(lexical))
        row["known_direct_pattern_leaks"] = sum(bool(check_direct_identifiers(t)["issues"]) for t in values)
        summary.append(row)
    return frame, summary


def paired_comparisons(frame, selected, controls, attackers):
    """Paired person-cluster bootstrap, never state-level pseudoreplication."""
    results = {}
    for control in controls:
        for metric in [*[f"{name}_r1" for name in attackers], "lexical"]:
            a, b = f"{selected}_{metric}", f"{control}_{metric}"
            if a in frame and b in frame:
                results[f"vs_{control}_{metric}"] = person_bootstrap(frame[a] - frame[b], frame.person)
    return results


def assert_split_disjointness(training_people, validation_people, test_people):
    """Fail closed on person overlap even when query record IDs differ."""
    groups = [set(map(str, x)) for x in (training_people, validation_people, test_people)]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Training, validation and reserved test people overlap")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["test", "validation"], default="test")
    parser.add_argument("--reference-size", type=int, default=6000)
    parser.add_argument("--reference-views", type=int, default=1)
    parser.add_argument("--max-persons", type=int, help="Validation smoke runs only; forbidden on the reserved test")
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--risk-model", type=Path, default=ROOT / "artifacts/models/resume_risk_v4/model.pkl")
    parser.add_argument("--without-previous-risk", action="store_const", const=None, dest="risk_model",
                        help="Explicitly omit the historical v4 control")
    parser.add_argument("--offline-action-model", type=Path,
                        help="Validation-selected offline model for paired ablation; never selected on this test")
    parser.add_argument("--onpolicy-action-model", type=Path,
                        help="Validation-selected on-policy model for paired ablation; never selected on this test")
    parser.add_argument("--resources-config", default="configs/baselines/adaptive_resume_v3_3.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args)


def run(args):
    from action_data import load_action_data, reference_views

    from anonrisk_uk.action_resume import action_search

    if args.split == "test" and args.max_persons is not None:
        raise ValueError("Reserved test must evaluate every reserved person; use validation for smoke runs")
    if args.reference_views < 1 or args.reference_size < 2:
        raise ValueError("At least one reference view and two reference records are required")
    if not args.out.resolve().is_relative_to((ROOT / "artifacts").resolve()):
        raise ValueError("Private evaluation outputs must stay under ignored artifacts/")
    if args.out.exists():
        raise FileExistsError("Evaluation output exists; frozen tests are never silently overwritten")
    with args.model.open("rb") as handle:
        estimator = pickle.load(handle)
    action_controls, action_control_metadata = {}, {}
    for argument, name in [("offline_action_model", "action_offline"), ("onpolicy_action_model", "action_onpolicy")]:
        path = getattr(args, argument, None)
        if path is None:
            continue
        with path.open("rb") as handle:
            control = pickle.load(handle)
        if frozen_features_digest(control) != frozen_features_digest(estimator):
            raise ValueError("Action ablation frozen statistics/codec/vectorizer differ from the selected model")
        action_controls[name] = control
        action_control_metadata[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                         "settings": control.settings,
                                         "controlled_identical_to_selected": deployment_digest(control) == deployment_digest(estimator)}
    previous = None
    if args.risk_model is not None:
        with args.risk_model.open("rb") as handle:
            previous = pickle.load(handle)
    data = load_action_data(args.data_dir)
    people = np.asarray(data["people"], dtype=str)
    splits = np.asarray(data["splits"], dtype=str)
    assert_split_disjointness(people[splits == "train"], people[splits == "validation"], people[splits == "test"])
    queries = list(map(int, data["query_positions"][args.split]))
    if args.max_persons:
        selected_people = set(sorted(set(people[queries]))[:args.max_persons])
        queries = [k for k in queries if people[k] in selected_people]
    if not queries:
        raise ValueError("Reserved query set is empty")
    views = reference_views(data, args.split, count=args.reference_views, size=args.reference_size, seed=args.seed)
    if any(not set(queries) <= set(view) for view in views):
        raise ValueError("Each reference view must contain all query records and their partners")
    texts = [estimator.codec.render(ex, {}) for ex in data["extracted"]]
    resources = load_resume_resources(args.resources_config)
    settings = estimator.settings
    floor = float(settings.get("utility_floor", .55))
    # Only these explicitly constructed objects are passed to autonomous code.
    old_resources = SimpleNamespace(config={"adaptive": {"utility_floor": .55}},
                                    risk_estimator=previous, detector=resources.detector,
                                    background_texts=[], feature_index=None)
    runtime_sources = [Path(__file__), ROOT / "scripts/action_fingerprints.py", *sorted((ROOT / "src/anonrisk_uk").glob("*.py"))]
    if (ROOT / "scripts/action_data.py").exists():
        runtime_sources.append(ROOT / "scripts/action_data.py")
    protocol = {"method": estimator.version, "purpose": "test" if args.split == "test" else "development",
                "selected": SELECTED, "queries": len(queries), "persons": len(set(people[queries])),
                "split": args.split, "seed": args.seed, "pool_size": len(views[0]),
                "reference_views": len(views), "utility_floor": floor,
                "model_settings": settings, "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
                "previous_model_sha256": hashlib.sha256(args.risk_model.read_bytes()).hexdigest() if previous else None,
                "action_model_controls": action_control_metadata,
                "data_manifest_sha256": hashlib.sha256((args.data_dir / "manifest.json").read_bytes()).hexdigest(),
                "query_ids_sha256": hashlib.sha256("\n".join(str(data["ids"][k]) for k in queries).encode()).hexdigest(),
                "prediction_access": "frozen artifact and current resume only; no corpus, person links or attack results",
                "bootstrap_unit": "person", "bootstrap_samples": 2000,
                "scope": "conditional same-person linkage in reserved WorkUA records, not internet identification",
                "semantic_evaluation": "separate frozen-output evaluate_resume_content.py; not factual entailment",
                "selection": "model and policy selected on development only; no test-dependent tuning",
                "live_pool_used_for_prediction": False,
                "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in runtime_sources}}
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    methods = ["L0", "historical_L2", "historical_L3", "matched_direct_only", "matched_all_quasi",
               "matched_all_quasi_broad", "matched_all_quasi_mask", "search_v3_3", SELECTED]
    if previous:
        methods.append("regression_v4")
    methods.extend(action_controls)
    outputs = {name: [] for name in methods}
    traces = []
    fixed = fixed_policies()
    # A single matched search-control corpus is frozen before seeing releases.
    primary = list(map(int, views[0]))
    primary_texts = [texts[k] for k in primary]
    primary_indexes = {"bm25": BM25Index(max_features=20000).fit(primary_texts),
                       "char": TfidfIndex(max_features=60000).fit(primary_texts)}
    with (args.out / "releases.jsonl").open("x") as release_file:
        for number, k in enumerate(queries, 1):
            doc = parse_resume(texts[k], estimator.codec, resources.detector)
            if action_controls:
                from anonrisk_uk.action_resume import ActionFeatures

                working = copy.copy(estimator)
                features = ActionFeatures(doc, working)
                policy, trace = action_search(doc, working, _features=features)
                for name, control in action_controls.items():
                    working.model, working.settings = control.model, dict(control.settings)
                    control_policy, _ = action_search(doc, working, _features=features)
                    outputs[name].append(render(doc, control_policy))
            else:
                policy, trace = action_search(doc, estimator)
            value = render(doc, policy)
            if check_direct_identifiers(value)["issues"]:
                raise ValueError("The learned release failed its known direct-pattern guard")
            outputs[SELECTED].append(value)
            traces.append({"id": str(data["ids"][k]), "person": str(people[k]), "policy": policy, "trace": trace})
            outputs["L0"].append(texts[k])
            for name, key in [("historical_L2", "L2_generalize"), ("historical_L3", "L3_strong")]:
                outputs[name].append(estimator.codec.render(data["extracted"][k], fixed[key]))
            outputs["matched_direct_only"].append(render(doc, {}))
            groups = {u.group for u in doc.units if len(options(doc, u)) > 1}
            for name, level in [("matched_all_quasi", 1), ("matched_all_quasi_broad", 2),
                                ("matched_all_quasi_mask", 3)]:
                outputs[name].append(render(doc, {group: level for group in groups}))
            proxy = NeighbourProxy(primary_indexes, primary_texts, texts[k])
            search_policy, _ = search(doc, proxy, estimator.retention_vectorizer, target=0., utility_floor=floor)
            outputs["search_v3_3"].append(render(doc, search_policy))
            if previous:
                outputs["regression_v4"].append(anonymize_learned(old_resources, texts[k]).releases["adaptive"].text)
            release_file.write(json.dumps({"id": str(data["ids"][k]), "person": str(people[k]),
                                           "original": texts[k],
                                           "outputs": {name: values[-1] for name, values in outputs.items()}},
                                          ensure_ascii=False) + "\n")
            print(f"evaluated {number}/{len(queries)}", flush=True)
    (args.out / "decision_traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2))
    pd.DataFrame({"id": [data["ids"][k] for k in primary]}).to_csv(args.out / "pool_ids.csv", index=False)
    controls = [name for name in methods if name not in {"L0", SELECTED}]
    view_reports = []
    for number, view in enumerate(views):
        view = list(map(int, view))
        local = {global_position: j for j, global_position in enumerate(view)}
        if number == 0:
            indexes = primary_indexes
        else:
            indexes = {"bm25": BM25Index(max_features=20000).fit([texts[k] for k in view]),
                       "char": TfidfIndex(max_features=60000).fit([texts[k] for k in view])}
        pop = Population([data["ids"][k] for k in view], [data["extracted"][k] for k in view],
                         people[view], estimator.codec, indexes["bm25"], [texts[k] for k in view])
        frame, summary = score_outputs(pop, indexes, [local[k] for k in queries], outputs, estimator.retention_vectorizer)
        paired = paired_comparisons(frame, SELECTED, controls, indexes)
        frame.to_csv(args.out / ("per_query.csv" if number == 0 else f"per_query_view_{number}.csv"), index=False)
        view_reports.append({"view": number, "pool_size": len(view), "summary": summary,
                             "paired_differences_95ci": paired})
    report = {**protocol, "summary": view_reports[0]["summary"],
              "paired_differences_95ci": view_reports[0]["paired_differences_95ci"],
              "reference_view_results": view_reports,
              "statuses": {str(k): int(v) for k, v in pd.Series(
                  [r["trace"][-1].get("stop_reason", "unknown") for r in traces]).value_counts().items()},
              "absolute_risk_prediction": None,
              "absolute_risk_note": "Action scores estimate expected improvement; no calibrated acceptance threshold"}
    # Recheck that neither weights nor runtime code changed during scoring.
    if hashlib.sha256(args.model.read_bytes()).hexdigest() != protocol["model_sha256"]:
        raise RuntimeError("Model changed during frozen evaluation")
    if previous and hashlib.sha256(args.risk_model.read_bytes()).hexdigest() != protocol["previous_model_sha256"]:
        raise RuntimeError("Historical control model changed during frozen evaluation")
    for control in protocol["action_model_controls"].values():
        if hashlib.sha256(Path(control["path"]).read_bytes()).hexdigest() != control["sha256"]:
            raise RuntimeError("Action ablation model changed during frozen evaluation")
    if hashlib.sha256((args.data_dir / "manifest.json").read_bytes()).hexdigest() != protocol["data_manifest_sha256"]:
        raise RuntimeError("Reserved data manifest changed during frozen evaluation")
    for path, expected in protocol["source_sha256"].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Runtime source changed during frozen evaluation: {path}")
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame(report["summary"]).to_csv(args.out / "summary.csv", index=False)
    print(pd.DataFrame(report["summary"]).to_string(index=False))
    return report


if __name__ == "__main__":
    main()
