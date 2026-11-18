"""Correct saved-release content measurement using original field/edit provenance.

No anonymization policy is selected or changed. Optional control replay recovers
missing edit plans and must reproduce every saved release exactly before it can
be used for attribution. Prior output-header-based metrics remain archived.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
from pathlib import Path

import pandas as pd
from action_data import load_action_data
from action_fingerprints import frozen_features_digest
from group_policy_eval import person_bootstrap
from threadpoolctl import threadpool_limits

from anonrisk_uk.action_resume import ActionFeatures, action_search
from anonrisk_uk.attack import BM25Index, TfidfIndex
from anonrisk_uk.content_evaluation import MODEL_ID, MODEL_REVISION, SemanticEncoder, professional_view, semantic_pair
from anonrisk_uk.content_provenance import field_view, operation_plan, project_operations, serialized_fields
from anonrisk_uk.learned_resume import learned_search
from anonrisk_uk.policy import fixed_policies
from anonrisk_uk.resume_adaptive import NeighbourProxy, options, render, search
from anonrisk_uk.resume_parser import parse_resume
from anonrisk_uk.resume_resources import load_resume_resources

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def release_records(run, extra_run=None):
    rows = [json.loads(line) for line in (run / "releases.jsonl").read_text().splitlines()]
    if extra_run:
        extra = {r["id"]: r for r in map(json.loads, (extra_run / "releases.jsonl").read_text().splitlines())}
        if len(extra) != len(rows) or set(extra) != {r["id"] for r in rows}:
            raise ValueError("Additional releases have a different query set")
        for row in rows:
            other = extra[row["id"]]
            if row["person"] != other["person"] or row["original"] != other["original"]:
                raise ValueError("Additional releases have different sources/people")
            for method, text in other["outputs"].items():
                if method in row["outputs"] and row["outputs"][method] != text:
                    raise ValueError("Shared frozen output changed")
                row["outputs"][method] = text
    return rows


def run(args):
    if args.out.exists():
        raise FileExistsError("Corrected measurement exists; preserve historical results")
    original = json.loads((args.run / "protocol.json").read_text())
    if digest(args.model) != original["model_sha256"] or digest(args.data_dir / "manifest.json") != original["data_manifest_sha256"]:
        raise ValueError("Frozen model/data differ")
    for name, expected in original["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Frozen inference source differs: {name}")
    records = release_records(args.run, args.extra_run)
    if len(records) != original["queries"]:
        raise ValueError("Incomplete source releases")
    with args.model.open("rb") as handle:
        estimator = pickle.load(handle)
    data = load_action_data(args.data_dir)
    by_id = {str(rid): i for i, rid in enumerate(data["ids"])}
    selected_policies = {str(r["id"]): r["policy"] for r in json.loads((args.run / "decision_traces.json").read_text())}
    extra_policies = {}
    if args.extra_run:
        extra_policies = {(str(r["id"]), r["method"]): r["policy"] for r in
                          json.loads((args.extra_run / "decision_traces.json").read_text())}
    cached_policies = {}
    if args.policies_from:
        parent = json.loads((args.policies_from.parent / "protocol.json").read_text())
        if parent["model_sha256"] != original["model_sha256"] or parent["queries"] != len(records):
            raise ValueError("Cached edit plans belong to another model/cohort")
        cached_policies = {(str(r["id"]), r["method"]): r["policy"]
                           for r in json.loads(args.policies_from.read_text())}
    models = sorted(records[0]["outputs"])
    if any(sorted(row["outputs"]) != models for row in records):
        raise ValueError("Each query must have identical method coverage")
    paths = [args.model, args.data_dir / "manifest.json", args.run / "releases.jsonl",
             args.run / "decision_traces.json", args.run / "protocol.json", Path(__file__),
             ROOT / "src/anonrisk_uk/content_provenance.py", ROOT / "src/anonrisk_uk/content_evaluation.py"]
    if args.extra_run:
        paths.extend([args.extra_run / "releases.jsonl", args.extra_run / "decision_traces.json"])
    if args.policies_from:
        paths.extend([args.policies_from, args.policies_from.parent / "protocol.json"])
    metadata = args.model_dir / ".cache/huggingface/download/model.safetensors.metadata"
    if metadata.read_text().splitlines()[0] != MODEL_REVISION:
        raise ValueError("Semantic revision differs")
    protocol = {"schema": "source_anchored_content_v1", "method": original["method"],
                "purpose": "post_hoc_measurement_correction", "queries": len(records),
                "persons": len({r["person"] for r in records}), "selected": "action_regression",
                "encoder": MODEL_ID, "revision": MODEL_REVISION, "device": "cpu",
                "weights_sha256": digest(args.model_dir / "model.safetensors"),
                "model_sha256": digest(args.model), "model_settings_changed": False,
                "professional_fields": ["title", "exp_position", "exp_resp", "skills", "edu_faculty"],
                "field_boundaries": "exact original WorkUA serialized value spans; never parse transformed headings",
                "attribution": "exact verified edit provenance; historical field renderer uses verified transformed fields",
                "mask_cleanup": "known mask labels only; preserve unknown literals and approximate-duration payloads",
                "ambiguous_replacements": "word-bearing edits touching multiple field bodies are unscorable; never assign guessed credit",
                "long_text": "all 126-token windows plus special tokens; token-weighted normalized pooling",
                "interpretation": "format-specific semantic proxy, not factual entailment or percent retained meaning",
                "selection": "no model/policy tuning; prior metric is retained separately",
                "control_recovery": args.recover_missing_controls, "bootstrap_unit": "person", "bootstrap_samples": 2000,
                "cached_policies_reverified": str(args.policies_from) if args.policies_from else None,
                "input_sha256": {str(path): digest(path) for path in paths}}
    args.out.mkdir(parents=True)
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    resources = load_resume_resources(args.resources_config)
    previous, offline, primary_indexes, primary_texts = None, None, None, None
    if args.recover_missing_controls:
        if "action_offline" in models and any((row["id"], "action_offline") not in cached_policies for row in records):
            info = original["action_model_controls"]["action_offline"]
            path = ROOT / info["path"]
            if digest(path) != info["sha256"]:
                raise ValueError("Offline control artifact changed")
            with path.open("rb") as handle:
                offline = pickle.load(handle)
            if frozen_features_digest(offline) != frozen_features_digest(estimator):
                raise ValueError("Offline control feature provenance differs")
            protocol["input_sha256"][str(path)] = digest(path)
        if "regression_v4" in models:
            if digest(args.previous_model) != original["previous_model_sha256"]:
                raise ValueError("Historical regressor changed")
            with args.previous_model.open("rb") as handle:
                previous = pickle.load(handle)
            protocol["input_sha256"][str(args.previous_model)] = digest(args.previous_model)
        if "search_v3_3" in models and any((row["id"], "search_v3_3") not in cached_policies for row in records):
            pool_ids = pd.read_csv(args.run / "pool_ids.csv", dtype=str).id.tolist()
            primary_texts = [estimator.codec.render(data["extracted"][by_id[rid]], {}) for rid in pool_ids]
            primary_indexes = {"bm25": BM25Index(max_features=20000).fit(primary_texts),
                               "char": TfidfIndex(max_features=60000).fit(primary_texts)}
            protocol["input_sha256"][str(args.run / "pool_ids.csv")] = digest(args.run / "pool_ids.csv")
    # Freeze complete provenance before creating corrected views or scores.
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    views, recovery_rows, unsupported = [], [], set()
    fixed = fixed_policies()
    with (args.out / "projected_views.jsonl").open("x") as handle:
        for number, row in enumerate(records, 1):
            extracted = data["extracted"][by_id[row["id"]]]
            source, fields = serialized_fields(extracted)
            if source != row["original"]:
                raise ValueError("Source serialization provenance differs")
            doc = parse_resume(source, estimator.codec, resources.detector)
            groups = {unit.group for unit in doc.units if len(options(doc, unit)) > 1}
            source_view = field_view(extracted)
            outputs, diagnostics = {}, {}
            for method in models:
                text, policy, method_doc = row["outputs"][method], None, doc
                if method == "L0":
                    if text != source:
                        raise ValueError("Unchanged source release differs")
                    outputs[method] = source_view
                    diagnostics[method] = {"scorable": True, "provenance": "original_fields"}
                    continue
                if method in {"historical_L2", "historical_L3"}:
                    name = "L2_generalize" if method.endswith("L2") else "L3_strong"
                    if estimator.codec.render(extracted, fixed[name]) != text:
                        raise ValueError("Historical field output differs")
                    outputs[method] = field_view(estimator.codec.transform(extracted, fixed[name]))
                    diagnostics[method] = {"scorable": True, "provenance": "verified_field_transform"}
                    continue
                if method == "action_regression" or (method == "action_onpolicy" and text == row["outputs"]["action_regression"]):
                    policy = selected_policies[row["id"]]
                elif (row["id"], method) in extra_policies:
                    policy = extra_policies[(row["id"], method)]
                elif (row["id"], method) in cached_policies:
                    policy = cached_policies[(row["id"], method)]
                    if method == "regression_v4":
                        if previous is None:
                            raise ValueError("Historical codec is required to verify cached v4 plans")
                        method_doc = parse_resume(source, previous.codec, resources.detector)
                elif method == "matched_direct_only":
                    policy = {}
                elif method in {"matched_all_quasi", "matched_all_quasi_broad", "matched_all_quasi_mask"}:
                    level = {"matched_all_quasi": 1, "matched_all_quasi_broad": 2, "matched_all_quasi_mask": 3}[method]
                    policy = {group: level for group in groups}
                elif method == "action_offline" and offline is not None:
                    working = copy.copy(estimator)
                    working.model, working.settings = offline.model, dict(offline.settings)
                    policy, _ = action_search(doc, working, _features=ActionFeatures(doc, working))
                elif method == "search_v3_3" and primary_indexes is not None:
                    proxy = NeighbourProxy(primary_indexes, primary_texts, source)
                    policy, _ = search(doc, proxy, estimator.retention_vectorizer, target=0.,
                                       utility_floor=estimator.settings["utility_floor"])
                elif method == "regression_v4" and previous is not None:
                    method_doc = parse_resume(source, previous.codec, resources.detector)
                    policy, _ = learned_search(method_doc, previous, utility_floor=.55)
                else:
                    unsupported.add(method)
                    continue
                if render(method_doc, policy) != text:
                    raise ValueError(f"Recovered plan fails exact frozen-release parity: {method}")
                projected = project_operations(source, fields, operation_plan(method_doc, policy), expected_output=text)
                outputs[method] = projected["text"]
                diagnostics[method] = {"scorable": projected["scorable"],
                                       "ambiguous_cross_field_edits": len(projected["ambiguous_cross_field_edits"]),
                                       "provenance": "verified_edit_plan"}
                recovery_rows.append({"id": row["id"], "method": method, "policy": policy})
            view = {"id": row["id"], "person": row["person"], "source": source_view, "outputs": outputs,
                    "diagnostics": diagnostics, "legacy_source": professional_view(source),
                    "legacy_outputs": {method: professional_view(row["outputs"][method]) for method in outputs}}
            views.append(view)
            handle.write(json.dumps(view, ensure_ascii=False) + "\n")
            if number % 10 == 0:
                print(f"verified provenance {number}/{len(records)}", flush=True)
    (args.out / "recovered_policies.json").write_text(json.dumps(recovery_rows, ensure_ascii=False))
    encoder = SemanticEncoder(args.model_dir)
    scores = []
    for number, row in enumerate(views, 1):
        names = sorted(row["outputs"])
        vectors = encoder.encode([row["source"], *[row["outputs"][method] for method in names]])
        for method, vector in zip(names, vectors[1:]):
            valid = row["diagnostics"][method]["scorable"]
            metrics = semantic_pair(vectors[0], vector) if valid else {"cosine": None, "chunk_recall": None, "chunk_precision": None}
            scores.append({"id": row["id"], "person": row["person"], "method": method,
                           "professional_cosine": metrics["cosine"], "professional_chunk_recall": metrics["chunk_recall"],
                           "professional_chunk_precision": metrics["chunk_precision"],
                           "provenance_scorable": valid,
                           "source_view_changed": row["source"] != row["legacy_source"],
                           "output_view_changed": row["outputs"][method] != row["legacy_outputs"][method],
                           "source_words": len(row["source"].split()), "output_words": len(row["outputs"][method].split())})
        if number % 10 == 0:
            pd.DataFrame(scores).to_csv(args.out / "per_query.csv", index=False)
            print(f"corrected semantics {number}/{len(views)}", flush=True)
    frame = pd.DataFrame(scores)
    frame.to_csv(args.out / "per_query.csv", index=False)
    metric_names = ["professional_cosine", "professional_chunk_recall", "professional_chunk_precision"]
    summary = []
    for method, part in frame.groupby("method"):
        summary.append({"method": method, "queries": len(part), "scorable": int(part.professional_cosine.notna().sum()),
                        "ambiguous_provenance": int((~part.provenance_scorable).sum()),
                        "source_views_changed": int(part.source_view_changed.sum()),
                        "output_views_changed": int(part.output_view_changed.sum()),
                        **{metric: float(part[metric].mean()) for metric in metric_names}})
    paired = {}
    for metric in metric_names:
        pivot = frame.pivot(index=["id", "person"], columns="method", values=metric)
        for method in pivot.columns:
            if method in {"L0", "action_regression", "action_onpolicy"}:
                continue
            diff = (pivot.action_regression - pivot[method]).dropna()
            paired[f"vs_{method}_{metric}"] = person_bootstrap(diff.to_numpy(), diff.index.get_level_values("person"))
    for path, expected in protocol["input_sha256"].items():
        if digest(path) != expected:
            raise RuntimeError(f"Frozen provenance input changed: {path}")
    report = {**protocol, "summary": summary, "paired_differences_95ci": paired,
              "unsupported_methods_without_provenance": sorted(unsupported),
              "provenance_verified_for_all_scored_releases": True,
              "independent_human_evaluation": False, "private_texts_ids_included": False}
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame(summary).to_csv(args.out / "summary.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--extra-run", type=Path)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "artifacts/action_v5_data_20261110")
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts/models/resume_action_v5/model.pkl")
    parser.add_argument("--previous-model", type=Path, default=ROOT / "artifacts/models/resume_risk_v4/model.pkl")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "artifacts/models/semantic_minilm")
    parser.add_argument("--resources-config", default=str(ROOT / "configs/adaptive_resume.json"))
    parser.add_argument("--recover-missing-controls", action="store_true")
    parser.add_argument("--policies-from", type=Path, help="Reuse recorded plans only after exact output-parity verification")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.out.resolve().is_relative_to(ROOT / "artifacts"):
        raise ValueError("Private views and per-query rows must remain under artifacts/")
    import torch
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        run(args)


if __name__ == "__main__":
    main()
