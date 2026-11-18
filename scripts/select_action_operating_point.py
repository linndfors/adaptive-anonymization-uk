"""Select action-controller settings on validation privacy and semantic proxies.

The fixed rule is declared before any candidate output is generated. Reserved
test documents are never rendered, parsed or scored. The source model remains
unchanged; this command writes a new artifact with provenance and selected
settings. Embedding similarity is not factual entailment or a safety guarantee.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import pickle
import time
from pathlib import Path

import pandas as pd
from action_data import load_action_data, reference_views
from action_fingerprints import frozen_features_digest
from threadpoolctl import threadpool_limits
from train_action_resume import attack_views, query_positions, score_policies

from anonrisk_uk.action_resume import ActionFeatures, ActionResumeRegressor, action_search
from anonrisk_uk.content_evaluation import MODEL_ID, MODEL_REVISION, SemanticEncoder, professional_view, semantic_pair
from anonrisk_uk.policy import fixed_policies
from anonrisk_uk.resume_adaptive import check_direct_identifiers, render

ROOT = Path(__file__).resolve().parents[1]
SETTINGS_GRID = [{"utility_floor": floor, "detail_floor": .35, "cost_weight": cost, "min_gain": gain}
                 for floor in (.55, .65) for cost in (0., .15, .4, .8) for gain in (0., .005)]
RULE = {"privacy": "each validation Recall@1 <= corresponding L2 point estimate",
        "professional_cosine_margin": .02,
        "utility_objective": "mean(lexical_retention, professional_cosine)",
        "tie_break": "lower worst Recall@1, then stable setting index",
        "fallback": "lowest worst Recall@1, then highest utility objective; constraints remain failed",
        "risk_calibration": "none; action gains are not absolute risk or identification probabilities"}


def select_setting(rows, *, professional_margin=.02):
    """One predeclared constrained decision, without significance/safety claims."""
    l2 = next(row for row in rows if row["method"] == "historical_L2")
    if not math.isfinite(l2["professional_cosine"]):
        raise ValueError("L2 professional content is unscorable")
    candidates = []
    for original in rows:
        if not original["method"].startswith("action_"):
            continue
        row = dict(original)
        row["privacy_constraint_met"] = all(row[f"{a}_recall_at_1"] <= l2[f"{a}_recall_at_1"]
                                             for a in ("bm25", "char"))
        row["semantic_constraint_met"] = (math.isfinite(row["professional_cosine"])
                                          and row["professional_cosine"] >= l2["professional_cosine"] - professional_margin)
        row["constraints_met"] = row["privacy_constraint_met"] and row["semantic_constraint_met"]
        row["quality_objective"] = .5 * (row["lexical_retention"] + row["professional_cosine"])
        if not math.isfinite(row["quality_objective"]):
            row["quality_objective"] = -math.inf
        row["worst_recall_at_1"] = max(row[f"{a}_recall_at_1"] for a in ("bm25", "char"))
        candidates.append(row)
    if not candidates:
        raise ValueError("No action settings were evaluated")
    eligible = [row for row in candidates if row["constraints_met"]]
    if eligible:
        selected = min(eligible, key=lambda row: (-row["quality_objective"], row["worst_recall_at_1"],
                                                 int(row["method"].split("_")[-1])))
    else:
        selected = min(candidates, key=lambda row: (row["worst_recall_at_1"], -row["quality_objective"],
                                                   int(row["method"].split("_")[-1])))
    return selected, candidates, bool(eligible)


def semantic_scores(records, encoder):
    """Cache repeated views; empty outputs lose all source professional content."""
    per_query = []
    for number, record in enumerate(records, 1):
        methods = list(record["outputs"])
        texts = [record["original"], *[record["outputs"][method] for method in methods]]
        encoded = encoder.encode([professional_view(text) for text in texts])
        for j, method in enumerate(methods, 1):
            metrics = semantic_pair(encoded[0], encoded[j])
            per_query.append({"id": record["id"], "person": record["person"], "method": method,
                              "professional_cosine": metrics["cosine"],
                              "professional_chunk_recall": metrics["chunk_recall"],
                              "professional_chunk_precision": metrics["chunk_precision"],
                              "empty_output": len(encoded[j]["chunks"]) == 0})
        if number % 10 == 0 or number == len(records):
            print(f"validation semantic {number}/{len(records)}", flush=True)
    frame = pd.DataFrame(per_query)
    return frame, {method: {"professional_cosine": float(part.professional_cosine.mean()),
                           "professional_scorable": int(part.professional_cosine.notna().sum()),
                           "empty_professional_outputs": int(part.empty_output.sum())}
                   for method, part in frame.groupby("method", sort=False)}


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_feature_hash(estimator):
    """Sharing cached features is valid only for identical frozen representations."""
    return frozen_features_digest(estimator)


def run(args):
    started = time.monotonic()
    if not args.out.resolve().is_relative_to((ROOT / "artifacts").resolve()):
        raise ValueError("Validation records and models must stay in ignored artifacts/")
    if args.out.exists():
        raise FileExistsError("Operating-point output exists; source/evidence is never overwritten")
    source_protocol = json.loads((args.model.parent / "protocol.json").read_text())
    if source_protocol.get("test_used") is not False:
        raise ValueError("Source model must explicitly declare no reserved test use")
    if _sha(args.data_dir / "manifest.json") != source_protocol["data_manifest_sha256"]:
        raise ValueError("Source model and reserved data manifest differ")
    with args.model.open("rb") as handle:
        estimator = pickle.load(handle)
    if not isinstance(estimator, ActionResumeRegressor):
        raise TypeError("Expected a fitted action-value model")
    model_paths = [args.model, *(args.candidate_model or [])]
    estimators = [estimator]
    feature_hash = frozen_feature_hash(estimator)
    for path in model_paths[1:]:
        with path.open("rb") as handle:
            candidate = pickle.load(handle)
        if not isinstance(candidate, ActionResumeRegressor):
            raise TypeError("Candidate is not an action-value model")
        if frozen_feature_hash(candidate) != feature_hash:
            raise ValueError("Candidate frozen statistics/codec/vectorizer differ; shared caching is invalid")
        estimators.append(candidate)
    documents_path = args.documents or args.model.parent / "validation_documents.pkl"
    with documents_path.open("rb") as handle:
        documents = pickle.load(handle)
    data = load_action_data(args.data_dir)
    validation = query_positions(data, "validation")
    if set(documents) != set(validation):
        raise ValueError("Saved documents do not match the full reserved validation set")
    test_people = {str(data["people"][k]) for k in query_positions(data, "test")}
    val_people = {str(data["people"][k]) for k in validation}
    if test_people & val_people:
        raise ValueError("Validation and test people overlap")
    size, count, seed = (source_protocol[key] for key in ("reference_size", "reference_views", "seed"))
    positions = reference_views(data, "validation", count=count, size=size, seed=seed)
    reference_positions = {k for view in positions for k in view}
    if any(str(data["people"][k]) in test_people for k in reference_positions):
        raise ValueError("Reserved test people entered validation references")
    texts = [""] * len(data["ids"])
    for k in reference_positions:
        texts[k] = estimator.codec.render(data["extracted"][k], {})
    if any(documents[k].text != texts[k] for k in validation):
        raise ValueError("Validation documents no longer match the frozen serialization")
    metadata = args.model_dir / ".cache/huggingface/download/model.safetensors.metadata"
    if metadata.read_text().splitlines()[0] != MODEL_REVISION:
        raise ValueError("Semantic encoder revision differs from the declared frozen model")
    sources = [Path(__file__), ROOT / "scripts/action_fingerprints.py", ROOT / "scripts/train_action_resume.py", ROOT / "scripts/action_data.py",
               *sorted((ROOT / "src/anonrisk_uk").glob("*.py"))]
    protocol = {"method": estimator.version, "purpose": "validation_selection", "test_used": False,
                "selected": "action_0", "queries": len(validation), "persons": len(val_people),
                "source_model_sha256": _sha(args.model), "source_documents_sha256": _sha(documents_path),
                "candidate_models": [{"candidate": i, "path": str(path), "sha256": _sha(path)}
                                     for i, path in enumerate(model_paths)],
                "frozen_feature_sha256": feature_hash,
                "data_manifest_sha256": _sha(args.data_dir / "manifest.json"),
                "reference_size": size, "reference_views": count, "seed": seed,
                "settings_grid": SETTINGS_GRID, "selection_rule": RULE,
                "policy_catalog": [{"method": f"action_{i * len(SETTINGS_GRID) + j}",
                                    "candidate_model": i, "settings_index": j, "settings": settings}
                                   for i in range(len(estimators)) for j, settings in enumerate(SETTINGS_GRID)],
                "semantic_encoder": MODEL_ID, "semantic_revision": MODEL_REVISION,
                "semantic_weights_sha256": _sha(args.model_dir / "model.safetensors"),
                "semantic_note": "validation utility proxy, not factual entailment or independent final-test evidence",
                "source_sha256": {str(p.relative_to(ROOT)): _sha(p) for p in sources}}
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    views = attack_views(data, "validation", size, count, seed, texts)
    outputs = {"historical_L2": [], **{r["method"]: [] for r in protocol["policy_catalog"]}}
    records = []
    with (args.out / "releases.jsonl").open("x") as handle:
        for number, k in enumerate(validation, 1):
            doc = documents[k]
            working = copy.copy(estimator)
            features = ActionFeatures(doc, working)
            outputs["historical_L2"].append(estimator.codec.render(data["extracted"][k], fixed_policies()["L2_generalize"]))
            for i, candidate in enumerate(estimators):
                working.model = candidate.model
                working.settings = dict(candidate.settings)
                for j, settings in enumerate(SETTINGS_GRID):
                    policy, _ = action_search(doc, working, _features=features, **settings)
                    output = render(doc, policy)
                    if check_direct_identifiers(output)["issues"]:
                        raise ValueError("A candidate release failed the direct-pattern guard")
                    outputs[f"action_{i * len(SETTINGS_GRID) + j}"].append(output)
            record = {"id": str(data["ids"][k]), "person": str(data["people"][k]), "original": texts[k],
                      "outputs": {name: values[-1] for name, values in outputs.items()}}
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if number % 10 == 0 or number == len(validation):
                print(f"validation settings {number}/{len(validation)} resumes", flush=True)
    rows = score_policies(data, validation, texts, outputs, views, estimator.retention_vectorizer)
    semantic_frame, semantic = semantic_scores(records, SemanticEncoder(args.model_dir))
    semantic_frame.to_csv(args.out / "semantic_per_query.csv", index=False)
    for row in rows:
        row.update(semantic[row["method"]])
    selected, candidates, eligible = select_setting(rows)
    index = int(selected["method"].split("_")[-1])
    selected_candidate, selected_settings = divmod(index, len(SETTINGS_GRID))
    estimator = copy.copy(estimators[selected_candidate])
    estimator.settings = {**estimator.settings, **SETTINGS_GRID[selected_settings]}
    selection = {"selected": selected, "constraints_met": eligible, "rule": RULE,
                 "source_model_sha256": protocol["candidate_models"][selected_candidate]["sha256"],
                 "selected_candidate_model": selected_candidate, "test_used": False,
                 "source_validation_metrics": estimator.metadata.get("validation", {}),
                 "semantic_note": protocol["semantic_note"]}
    estimator.metadata = {**estimator.metadata, "validation_operating_point": selection}
    for relative, digest in protocol["source_sha256"].items():
        if _sha(ROOT / relative) != digest:
            raise RuntimeError(f"Source changed during frozen validation selection: {relative}")
    for candidate in protocol["candidate_models"]:
        if _sha(Path(candidate["path"])) != candidate["sha256"]:
            raise RuntimeError("Source model changed during validation selection")
    if _sha(documents_path) != protocol["source_documents_sha256"]:
        raise RuntimeError("Validation documents changed during selection")
    with (args.out / "model.pkl").open("xb") as handle:
        pickle.dump(estimator, handle)
    # Freeze a separately selected operating point for each training design.
    # These are ablations, not alternatives to select after viewing the test.
    candidate_selection = []
    for candidate_index, source_estimator in enumerate(estimators):
        own_rows = [rows[0], *[row for row in rows[1:]
                              if int(row["method"].split("_")[-1]) // len(SETTINGS_GRID) == candidate_index]]
        own_selected, _, own_eligible = select_setting(own_rows)
        own_setting_index = int(own_selected["method"].split("_")[-1]) % len(SETTINGS_GRID)
        own = copy.copy(source_estimator)
        own.settings = {**source_estimator.settings, **SETTINGS_GRID[own_setting_index]}
        own.metadata = {**source_estimator.metadata, "validation_operating_point": {
            "selected": own_selected, "constraints_met": own_eligible, "rule": RULE,
            "source_model_sha256": protocol["candidate_models"][candidate_index]["sha256"],
            "selected_candidate_model": candidate_index, "test_used": False,
            "selection_scope": "settings restricted to this candidate training design",
            "semantic_note": protocol["semantic_note"]}}
        candidate_path = args.out / f"candidate_{candidate_index}_model.pkl"
        with candidate_path.open("xb") as handle:
            pickle.dump(own, handle)
        candidate_selection.append({"candidate": candidate_index, "selected_policy": own_selected,
                                    "settings": own.settings, "constraints_met": own_eligible,
                                    "source_model_sha256": protocol["candidate_models"][candidate_index]["sha256"],
                                    "artifact": candidate_path.name, "artifact_sha256": _sha(candidate_path),
                                    "test_used": False})
    (args.out / "candidate_selection.json").write_text(json.dumps(candidate_selection, indent=2))
    protocol["selected"] = selected["method"]
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    report = {**protocol, "selected_settings": estimator.settings, "selected_policy": selected,
              "selected_candidate_model": selected_candidate,
              "selected_source_model_sha256": protocol["candidate_models"][selected_candidate]["sha256"],
              "candidate_selection": candidate_selection,
              "constraints_met": eligible, "summary": [rows[0], *candidates],
              "output_model_sha256": _sha(args.out / "model.pkl"), "absolute_risk_prediction": None,
              "elapsed_seconds": time.monotonic() - started,
              "limitation": "validation point-estimate constraints, not safety, significance or final-test confirmation"}
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame(report["summary"]).to_csv(args.out / "summary.csv", index=False)
    print(pd.DataFrame(report["summary"]).to_string(index=False))
    print(json.dumps({"selected": selected, "settings": estimator.settings}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--candidate-model", type=Path, action="append", default=[],
                        help="Additional fitted regressor; same frozen statistics/codec required; validation only")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--documents", type=Path)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "artifacts/models/semantic_minilm")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        run(args)


if __name__ == "__main__":
    main()
