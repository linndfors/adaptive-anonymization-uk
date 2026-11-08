"""Reproducible workflow from entity detection to adaptive text release."""

from __future__ import annotations

import gc
import hashlib
import json
import pickle
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm

from .adaptive_text import (
    TextRiskController,
    adaptive_release,
    apply_actions,
    make_document,
    mandatory_actions,
    train_unit_contributions,
)
from .attack import BM25Index, TfidfIndex, ranking_result
from .detectors import ApiBudget, build_detector
from .e3jsi import (
    evaluate_detection,
    gold_spans,
    load_jsonl,
    make_split,
    repair_docs,
    to_gliner_records,
    validate_offsets,
)
from .experiment import Population
from .pipeline import SpanFrequencyReference
from .policy import fixed_policies
from .risk_training import fit_risk_model
from .statistics import metric_draws, metric_intervals, paired_differences
from .utility import DownstreamTask
from .vacancies import VacancyFitModel, load_vacancies, retrieval_sanity_check
from .workua import ResumeCodec, build_person_clusters, extract, load_workua
from .workua_views import (
    VIEWS,
    distinct_targets,
    duplicate_diagnostics,
    query_cohorts,
    render_view,
    source_audit,
    split_persons_and_copies,
)


def _save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint(config, stage, key, compute):
    """Persist atomic checkpoints under a data, configuration, and code signature."""
    if not config.get("resume_checkpoints", False):
        return compute()
    if config.get("_prepared_checkpoint_dir") and stage in {"documents", "contributions"}:
        source = Path(config["_prepared_checkpoint_dir"]) / stage / f"{key}.pkl"
        if not source.is_file():
            raise ValueError(f"Completed checkpoint {stage}/{key} is missing; resume aborted")
        with source.open("rb") as handle:
            return pickle.load(handle)
    namespace = config.get("_checkpoint_signatures", {}).get(stage, config["_checkpoint_signature"])
    path = Path(config["out"]) / "checkpoints" / namespace / stage / f"{key}.pkl"
    if path.is_file():
        with path.open("rb") as handle:
            return pickle.load(handle)
    value = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle)
    temporary.replace(path)
    return value


def _exclude_development_test(split, config):
    """Exclude entire copy blocks containing IDs evaluated during development."""
    source = config.get("development_out")
    if not source:
        return split
    previous = pd.read_csv(Path(source) / "workua" / "split.csv", dtype={"id": str})
    queries = pd.read_csv(Path(source) / "pipeline" / "test_per_query.csv")
    seen_ids = set(previous.iloc[queries.pos.unique().astype(int)].id)
    blocks = set(split.loc[split.id.astype(str).isin(seen_ids), "split_block"])
    excluded = split.split.eq("test") & split.split_block.isin(blocks)
    split = split.copy()
    split.loc[excluded, "split"] = "development"
    print(
        f"Held-out test: excluded {int(excluded.sum())} records from blocks previously used during development.",
        flush=True,
    )
    return split


def _development_fingerprints(config):
    source = config.get("development_out")
    return (
        {name: _fingerprint(Path(source) / name) for name in ["workua/split.csv", "pipeline/test_per_query.csv"]}
        if source
        else None
    )


def _verify_prepared(config):
    """Verify and recover completed training artifacts from an interrupted run."""
    source = Path(config["prepared_source_out"])
    record = json.loads((source / "prepared_recovery.json").read_text())
    keys = [
        "seed",
        "pool_sample",
        "n_train",
        "n_val",
        "n_test",
        "min_freq",
        "extra_raw_columns",
        "release_detectors",
        "gliner_model",
        "gliner_threshold",
        "bm25_max_features",
        "use_char_tfidf",
        "include_residual",
        "max_document_chars",
        "max_units_per_doc",
    ]
    for key in keys:
        if config.get(key) != record["config"].get(key):
            raise ValueError(f"Resume is incompatible because {key} changed")
    if _fingerprint(config["workua_path"]) != record["signature"]["csv"]:
        raise ValueError("The CSV file changed after training")
    from importlib.metadata import PackageNotFoundError, version

    for package, expected in record["signature"]["dependencies"].items():
        try:
            actual = version(package)
        except PackageNotFoundError:
            actual = None
        if actual != expected:
            raise ValueError(f"The {package} version changed after training")
    for name, expected in record["files"].items():
        if _fingerprint(source / name) != expected:
            raise ValueError(f"Resume artifact changed: {name}")
    for name, expected in record["signature"]["code"].items():
        if name not in {"research.py", "adaptive_text.py"} and _fingerprint(Path(__file__).parent / name) != expected:
            raise ValueError(f"Training or data code changed: {name}")
    print(
        "Resume check passed for documents, contributions, the risk model, and retention; "
        "training will not be repeated.",
        flush=True,
    )
    return record, source / "checkpoints" / record["checkpoint_namespace"]


def _tuning_positions(positions, clusters, n_persons, seed):
    if n_persons is None:
        return positions
    if n_persons < 1:
        raise ValueError("tuning_val_persons must be greater than zero")
    persons = np.unique(clusters[positions])
    selected = np.random.default_rng(seed).choice(persons, min(n_persons, len(persons)), replace=False)
    return np.asarray([k for k in positions if clusters[k] in selected], dtype=int)


def _pick(positions, limit, seed):
    positions = np.asarray(positions, dtype=int)
    return (
        np.sort(np.random.default_rng(seed).choice(positions, min(limit, len(positions)), replace=False))
        if limit
        else positions
    )


def _select_gliner_threshold(table, minimum_precision):
    """Select a validation-feasible threshold or fail before touching test data."""
    feasible = table[table.overlap_P >= minimum_precision]
    if feasible.empty:
        best = table.sort_values(["overlap_P", "entity_leak_rate"], ascending=[False, True]).iloc[0]
        raise ValueError(
            "No GLiNER validation threshold satisfies the overlap-precision gate: "
            f"required={minimum_precision:.3f}, best={float(best.overlap_P):.3f}. "
            "Fix or exclude GLiNER instead of selecting a degenerate high-recall model."
        )
    return float(
        feasible.sort_values(
            ["entity_leak_rate", "char_recall", "overlap_P", "threshold"],
            ascending=[True, False, False, False],
        )
        .iloc[0]
        .threshold
    )


def _select_adaptive_point(points, target_recall, reference_recall=None):
    """Choose an operating point on validation only.

    The absolute target remains the primary objective. If it is unattainable,
    a declared fixed-policy reference (normally L2) becomes the fallback target.
    This prevents selecting a weak adaptive release merely because every point
    missed an unrealistically strict absolute target.
    """
    feasible = points[points.worst_recall <= target_recall]
    if not feasible.empty:
        selected = feasible.sort_values(["retention", "worst_recall", "policy"], ascending=[False, True, True]).iloc[0]
        return selected, "target_met_on_validation"
    if reference_recall is not None:
        comparable = points[points.worst_recall <= reference_recall]
        if not comparable.empty:
            selected = comparable.sort_values(
                ["retention", "worst_recall", "policy"], ascending=[False, True, True]
            ).iloc[0]
            return selected, "reference_met_on_validation"
    selected = points.sort_values(["worst_recall", "retention", "policy"], ascending=[True, False, True]).iloc[0]
    return selected, "target_unmet_on_validation"


def resolve_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    # Repository-relative paths are anchored by the config location, not by the
    # installation location of the Python package. This also works from a wheel.
    release = path.parent.parent
    for key in [
        "out",
        "workua_path",
        "e3_path",
        "e3_split_path",
        "vacancies_path",
        "detection_source_out",
        "development_out",
        "prepared_source_out",
        "workua_source_out",
        "api_budget_path",
    ]:
        if config.get(key):
            p = Path(config[key]).expanduser()
            config[key] = str(p if p.is_absolute() else release / p)
    # Hugging Face model identifiers and local model directories share the same
    # configuration field. Resolve it only when it points to a repository path.
    if config.get("gliner_model"):
        candidate = Path(config["gliner_model"]).expanduser()
        local_candidate = candidate if candidate.is_absolute() else release / candidate
        if local_candidate.exists():
            config["gliner_model"] = str(local_candidate.resolve())
    config["release"] = str(release)
    return config


def detector_set(config, names=None):
    names = names or config["detectors"]
    budget_path = config.get("api_budget_path") or Path(config["release"]) / "artifacts" / "api_budget_v3.json"
    budget = ApiBudget(budget_path, config.get("openai_budget_usd", 0), config.get("openai_max_calls", 0))
    objects = []
    for name in names:
        if name == "openai" and not config.get("allow_paid_api", False):
            raise ValueError("OpenAI was selected explicitly, but allow_paid_api is false")
        obj = build_detector(
            name,
            cache_dir=Path(config["out"]) / "detector_cache" / name,
            local_files_only=config.get("local_files_only", False),
            device=config.get("detection_device", "cpu"),
            allow_paid=config.get("allow_paid_api", False),
            budget=budget,
            local_url=config.get("lapa_url", "http://127.0.0.1:8080/v1"),
            local_model=config.get("lapa_model", "lapa"),
            gliner_model=config.get("gliner_model", "urchade/gliner_multi_pii-v1"),
            gliner_threshold=config.get("gliner_threshold", 0.4),
        )
        objects.append(obj)

    def detect(text):
        return [span for obj in objects for span in obj.detect(text)]

    return detect, objects


def prepare_detection_data(config):
    """Freeze the E3-JSI split and export GLiNER train/validation records."""
    out = Path(config["out"]) / "detection"
    out.mkdir(parents=True, exist_ok=True)
    docs, repairs = repair_docs(load_jsonl(config["e3_path"]))
    if validate_offsets(docs):
        raise ValueError("Invalid E3 offsets")
    split_path = Path(config.get("e3_split_path") or out / "split.json")
    if split_path.is_file():
        if split_path.suffix == ".csv":
            saved = pd.read_csv(split_path, dtype=str)
            split = dict(zip(saved.iloc[:, 0], saved.split))
        else:
            split = json.loads(split_path.read_text(encoding="utf-8"))
    else:
        split = {str(k): v for k, v in make_split(docs, seed=config["seed"]).items()}
    if set(split) != {str(d["source_row_id"]) for d in docs}:
        raise ValueError("The E3 split does not match the documents")
    _save_json(out / "split.json", split)
    repairs.to_csv(out / "repairs_for_review.csv", index=False)
    cohorts = {s: [d for d in docs if split[str(d["source_row_id"])] == s] for s in ["train", "val", "test"]}
    for cohort in ["train", "val"]:
        _save_json(out / f"gliner_{cohort}.json", to_gliner_records(cohorts[cohort]))
    _save_json(
        out / "data_manifest.json",
        {
            "source_sha256": _fingerprint(config["e3_path"]),
            "seed": config["seed"],
            "n_train": len(cohorts["train"]),
            "n_validation": len(cohorts["val"]),
            "n_test": len(cohorts["test"]),
            "test_used": False,
        },
    )
    return docs, cohorts


def run_detection(config):
    started = time.perf_counter()
    out = Path(config["out"]) / "detection"
    print("Stage 1: detect PII in E3-JSI; tune on validation and evaluate on test.", flush=True)
    docs, cohorts = prepare_detection_data(config)
    if (
        config.get("require_finetuned_gliner", False)
        and config.get("gliner_model", "urchade/gliner_multi_pii-v1") == "urchade/gliner_multi_pii-v1"
    ):
        raise ValueError(
            "The full run requires a validation-selected fine-tuned GLiNER model. "
            "Complete README steps 2-3 and update gliner_model before evaluating test."
        )
    for s, key in [("val", "detection_val_limit"), ("test", "detection_test_limit")]:
        if config.get(key):
            cohorts[s] = cohorts[s][: config[key]]
    results, statuses, tuning, predictions, validation_predictions = [], [], [], {}, {}
    # Evaluate budget-limited LLMs on one fixed pilot cohort.
    pilot = cohorts["test"][: config.get("llm_test_documents", 5)]
    _save_json(out / "pilot_ids.json", [str(d["source_row_id"]) for d in pilot])
    selected_threshold = config.get("gliner_threshold", 0.4)
    for name in tqdm(config["detectors"], desc="Detectors"):
        try:
            detector, objects = detector_set(config, [name])
            tests = pilot if name in {"openai", "lapa"} else cohorts["test"]
            if name == "gliner":
                for th in tqdm(config.get("gliner_threshold_grid", [0.25, 0.4, 0.55]), desc="GLiNER val", leave=False):
                    tuning_config = {**config, "gliner_threshold": th}
                    tuned, tuned_objects = detector_set(tuning_config, [name])
                    pred = {
                        str(d["source_row_id"]): tuned(d["text_uk"])
                        for d in tqdm(cohorts["val"], desc=f"val th={th}", leave=False)
                    }
                    gold = {str(d["source_row_id"]): gold_spans(d) for d in cohorts["val"]}
                    metric = evaluate_detection(gold, pred)["summary"]
                    tuning.append({"detector": name, "threshold": th, **metric})
                    del tuned, tuned_objects
                    gc.collect()
                table = pd.DataFrame(tuning)
                selected_threshold = _select_gliner_threshold(table, config.get("min_detection_precision", 0.8))
                del detector, objects
                gc.collect()
                detector, objects = detector_set({**config, "gliner_threshold": selected_threshold}, [name])
            if name not in {"openai", "lapa"}:
                validation_predictions[name] = {
                    str(d["source_row_id"]): detector(d["text_uk"])
                    for d in tqdm(cohorts["val"], desc=f"{name}: validation", leave=False)
                }
            pred = {}
            failures = []
            for d in tqdm(tests, desc=f"{name}: test", leave=False):
                key = str(d["source_row_id"])
                try:
                    pred[key] = detector(d["text_uk"])
                except Exception as exc:
                    # Preserve LLM failures explicitly instead of treating them as empty detections.
                    failures.append({"doc_id": key, "error_type": type(exc).__name__})
                    if not config.get("continue_detection_errors", True):
                        raise
                    pred[key] = []
            gold = {str(d["source_row_id"]): gold_spans(d) for d in tests}
            evaluated = evaluate_detection(gold, pred)
            test_cohort = "test_subset" if config.get("detection_test_limit") else "full_test"
            results.append(
                {
                    "detector": name,
                    "cohort": "pilot" if name in {"openai", "lapa"} else test_cohort,
                    "n_docs": len(tests),
                    "n_failed": len(failures),
                    **evaluated["summary"],
                }
            )
            evaluated["per_class"].to_csv(out / f"{name}_per_class.csv", index=False)
            _save_json(out / f"{name}_failures.json", failures)
            _save_json(out / f"{name}_predictions.json", {k: [asdict(s) for s in v] for k, v in pred.items()})
            # Use the same pilot cohort for comparisons with LLM detectors.
            pilot_gold = {str(d["source_row_id"]): gold_spans(d) for d in pilot}
            pilot_pred = {k: pred[k] for k in pilot_gold}
            if name not in {"openai", "lapa"}:
                results.append(
                    {
                        "detector": name,
                        "cohort": "pilot",
                        "n_docs": len(pilot),
                        "n_failed": sum(f["doc_id"] in pilot_gold for f in failures),
                        **evaluate_detection(pilot_gold, pilot_pred)["summary"],
                    }
                )
            predictions[name] = pred
            statuses.append({"detector": name, "status": "completed" if not failures else "completed_with_failures"})
            del detector, objects
            gc.collect()
        except Exception as exc:  # noqa: BLE001 - every detector failure is recorded independently
            statuses.append(
                {
                    "detector": name,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "message": str(exc)[:240]
                    if name not in {"openai", "lapa"}
                    else "Check dependencies and API budget; credentials are not logged",
                }
            )
            print(
                f"{name}: FAILED ({type(exc).__name__}); no substitute detector output is used.",
                flush=True,
            )
    # Merge stored predictions without rerunning models; never label a partial or
    # validation-infeasible ensemble as ready.
    local = [n for n in config["detectors"] if n not in {"openai", "lapa"}]
    release_names = config.get("release_detectors", local)
    if not release_names or not set(release_names) <= set(local):
        raise ValueError("release_detectors must be a non-empty subset of configured local detectors")
    completed = all(any(s["detector"] == n and s["status"] == "completed" for s in statuses) for n in release_names)
    validation_quality = None
    ready = False
    if completed:
        from .pipeline import select_spans

        validation_gold = {str(d["source_row_id"]): gold_spans(d) for d in cohorts["val"]}
        validation_text = {str(d["source_row_id"]): d["text_uk"] for d in cohorts["val"]}
        validation_ensemble = {
            key: select_spans(
                validation_text[key],
                [span for name in release_names for span in validation_predictions[name][key]],
            )
            for key in validation_gold
        }
        validation_quality = evaluate_detection(validation_gold, validation_ensemble)["summary"]
        results.append(
            {
                "detector": "+".join(release_names),
                "cohort": "validation_selection",
                "n_docs": len(validation_gold),
                "n_failed": 0,
                **validation_quality,
            }
        )
        ready = validation_quality["overlap_P"] >= config.get("min_detection_precision", 0.8)
    if ready:
        gold = {str(d["source_row_id"]): gold_spans(d) for d in cohorts["test"]}
        text_by_id = {str(d["source_row_id"]): d["text_uk"] for d in cohorts["test"]}
        ensemble = {
            key: select_spans(text_by_id[key], [span for name in release_names for span in predictions[name][key]])
            for key in gold
        }
        results.append(
            {
                "detector": "+".join(release_names),
                "cohort": "test_subset" if config.get("detection_test_limit") else "full_test",
                "n_docs": len(gold),
                "n_failed": 0,
                **evaluate_detection(gold, ensemble)["summary"],
            }
        )
    pd.DataFrame(results).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(statuses).to_csv(out / "status.csv", index=False)
    pd.DataFrame(tuning).to_csv(out / "validation_thresholds.csv", index=False)
    _save_json(
        out / "selected_detector.json",
        {
            "detectors": release_names,
            "gliner_model": (config.get("gliner_model") if "gliner" in release_names else None),
            "gliner_threshold": (selected_threshold if "gliner" in release_names else None),
            "validation_quality": validation_quality,
            "minimum_overlap_precision": config.get("min_detection_precision", 0.8),
            "precision_constraint_met": bool(ready),
            "status": "ready" if ready else "incomplete",
        },
    )
    _save_json(
        out / "manifest.json",
        {
            "config": config,
            "source_sha256": _fingerprint(config["e3_path"]),
            "elapsed_seconds": time.perf_counter() - started,
            "n_docs": len(docs),
            "statuses": statuses,
        },
    )
    print(pd.DataFrame(results).to_string(index=False))
    return pd.DataFrame(results)


def prepare_workua(config):
    out = Path(config["out"]) / "workua"
    out.mkdir(parents=True, exist_ok=True)
    print(
        "Stage 2: CSV to structured/free-text views, identity clusters, duplicates, and splits.",
        flush=True,
    )
    df = load_workua(config["workua_path"], sample=config.get("pool_sample"), seed=config["seed"])
    clusters = build_person_clusters(df).astype(str).to_numpy()
    # Avoid materialising a second full list of raw row dictionaries. The full
    # WorkUA frame is large enough for that temporary copy to matter in Colab.
    columns = list(df.columns)
    records = (dict(zip(columns, values)) for values in df.itertuples(index=False, name=None))
    extracted = [
        extract(record)
        for record in tqdm(
            records,
            total=len(df),
            desc="Extracting structured and free-text fields",
        )
    ]
    # Additional raw columns require explicit configuration.
    raw_columns = config.get("extra_raw_columns", [])
    for column in raw_columns:
        if column not in df:
            raise ValueError(f"Configured raw column is missing: {column}")
    for k, ex in enumerate(extracted):
        for column in raw_columns:
            value = df.iloc[k][column]
            if isinstance(value, str) and value.strip() and value not in ex["additional_info"]:
                ex["additional_info"].append(value)
    initial_codec = ResumeCodec()
    texts = [render_view(initial_codec, ex) for ex in extracted]
    split = _exclude_development_test(split_persons_and_copies(df.id.tolist(), clusters, texts, config["seed"]), config)
    train = np.flatnonzero(split.split.eq("train"))
    codec = ResumeCodec(min_freq=config.get("min_freq", 20)).fit([extracted[k] for k in train])
    reference = SpanFrequencyReference().fit([extracted[k] for k in train])
    source_audit(df, extracted).to_csv(out / "field_coverage.csv", index=False)
    pd.DataFrame([df.attrs.get("load_report", {})]).to_csv(out / "data_load_report.csv", index=False)
    split.to_csv(out / "split.csv", index=False)
    for view in tqdm(VIEWS, desc="Duplicate diagnostics by text view"):
        view_texts = [render_view(codec, ex, view) for ex in extracted]
        duplicate_diagnostics(
            df.id.tolist(), clusters, view_texts, view, config.get("near_duplicate_threshold", 0.9)
        ).to_csv(out / f"pairs_{view}.csv", index=False)
    bundle = {
        "ids": df.id.tolist(),
        "extracted": extracted,
        "clusters": clusters,
        "codec": codec,
        "reference": reference,
        "split": split,
        "texts": texts,
    }
    with (out / "bundle.pkl").open("wb") as handle:
        pickle.dump(bundle, handle)
    _save_json(
        out / "manifest.json",
        {
            "config": config,
            "source_sha256": _fingerprint(config["workua_path"]),
            "frequency_version": "document_union_v3",
            "n_resumes": len(df),
            "n_persons": len(set(clusters)),
            "extra_raw_columns": raw_columns,
            "development_source_sha256": _development_fingerprints(config),
        },
    )
    return bundle


def _load_bundle(config):
    path = Path(config["out"]) / "workua" / "bundle.pkl"
    if not path.is_file():
        return prepare_workua(config)
    manifest = json.loads(path.with_name("manifest.json").read_text())
    if manifest["source_sha256"] != _fingerprint(config["workua_path"]):
        raise ValueError("The CSV changed; rerun the workua stage in a new output directory")
    if manifest.get("development_source_sha256") != _development_fingerprints(config):
        raise ValueError("Development artifacts changed; rerun the workua stage")
    for key in ["pool_sample", "seed", "min_freq", "extra_raw_columns", "development_out"]:
        if manifest["config"].get(key) != config.get(key):
            raise ValueError(f"Configuration key {key} does not match the bundle; rerun workua")
    with path.open("rb") as handle:
        return pickle.load(handle)


def run_workua(config):
    # Reuse a validated population bundle when a longer baseline evaluation is
    # resumed after a notebook or managed runtime interruption.
    bundle = _load_bundle(config)
    out = Path(config["out"]) / "workua"
    rows, details, intervals, paired, stores = [], [], [], [], {}
    for view in tqdm(VIEWS, desc="S2: structured/free_text/hybrid"):
        texts = [render_view(bundle["codec"], ex, view) for ex in bundle["extracted"]]
        index = BM25Index(max_features=config.get("bm25_max_features", 150000)).fit(texts)
        pop = Population(bundle["ids"], bundle["extracted"], bundle["clusters"], bundle["codec"], index, texts)
        positions = np.array([k for k in pop.linked_positions() if bundle["split"].split.iloc[k] == "test"], dtype=int)
        positions = _pick(positions, config.get("n_test", 40), config["seed"])
        cohorts = query_cohorts(pop, positions, texts, config.get("near_duplicate_threshold", 0.9))
        eligible, dt, de = distinct_targets(pop, positions, texts, config.get("near_duplicate_threshold", 0.9))
        for policy_name, policy in tqdm(fixed_policies().items(), desc=view + ": L0–L3", leave=False):
            queries = [render_view(bundle["codec"], bundle["extracted"][k], view, policy) for k in positions]
            targets, excludes = pop.targets_and_excludes(positions, "S2")
            result = ranking_result(index, queries, targets, excludes, batch_size=16)[0]
            detail = result.to_frame().assign(pos=positions, person=bundle["clusters"][positions])
            detail = detail.merge(cohorts.drop(columns="person"), on="pos", validate="one_to_one")
            stores[(view, "S2", policy_name)] = detail
            intervals.append(
                metric_intervals(detail, n=config.get("n_bootstrap", 300), seed=config["seed"]).assign(
                    view=view, scenario="S2", policy=policy_name
                )
            )
            for cohort, part in [("all", detail), *list(detail.groupby("cohort"))]:
                row = {m: v[0] for m, v in metric_draws(part).items()}
                rows.append(
                    dict(
                        row,
                        view=view,
                        scenario="S2",
                        policy=policy_name,
                        cohort=cohort,
                        n_resumes=len(part),
                        n_persons=part.person.nunique(),
                    )
                )
            details.append(detail.assign(view=view, scenario="S2", policy=policy_name))
            if len(eligible):
                distinct_queries = [
                    render_view(bundle["codec"], bundle["extracted"][k], view, policy) for k in eligible
                ]
                result = ranking_result(index, distinct_queries, dt, de, batch_size=16)[0]
                detail = result.to_frame().assign(pos=eligible, person=bundle["clusters"][eligible])
                stores[(view, "S2_distinct", policy_name)] = detail
                intervals.append(
                    metric_intervals(detail, n=config.get("n_bootstrap", 300), seed=config["seed"]).assign(
                        view=view, scenario="S2_distinct", policy=policy_name
                    )
                )
                row = {m: v[0] for m, v in metric_draws(detail).items()}
                rows.append(
                    dict(
                        row,
                        view=view,
                        scenario="S2_distinct",
                        policy=policy_name,
                        cohort="all",
                        n_resumes=len(detail),
                        n_persons=detail.person.nunique(),
                    )
                )
                details.append(detail.assign(view=view, scenario="S2_distinct", policy=policy_name))
        del index, pop
        gc.collect()
    for (view, scenario, policy), detail in stores.items():
        if policy != "L0_none":
            paired.append(
                paired_differences(
                    detail, stores[(view, scenario, "L0_none")], n=config.get("n_bootstrap", 300), seed=config["seed"]
                ).assign(view=view, scenario=scenario, policy=policy, reference="L0_none")
            )
    pd.DataFrame(rows).to_csv(out / "view_risk_summary.csv", index=False)
    pd.concat(details, ignore_index=True).to_csv(out / "view_per_query.csv", index=False)
    pd.concat(intervals, ignore_index=True).to_csv(out / "view_intervals.csv", index=False)
    pd.concat(paired, ignore_index=True).to_csv(out / "view_paired_differences.csv", index=False)
    print(pd.DataFrame(rows).query("cohort == 'all'").to_string(index=False))
    return pd.DataFrame(rows)


def _fit_and_save_risk_model(config, out, features, labels, val_features, val_labels):
    """Select a risk model on validation data and persist every audit artifact."""
    training = fit_risk_model(
        features,
        labels,
        val_features,
        val_labels,
        seed=config["seed"],
        model_grid=config.get("risk_model_grid"),
        context_weights=config.get("risk_context_weights", [0.0, 0.25, 0.5, 1.0]),
        selection_metric=config.get("risk_selection_metric", "within_doc_spearman"),
        learning_curve_persons=config.get("risk_learning_curve_persons", [50, 100, 200, 400, None]),
    )
    training.hyperparameter_search.to_csv(out / "risk_hyperparameter_search.csv", index=False)
    training.learning_curve.to_csv(out / "risk_learning_curve.csv", index=False)
    training.validation_units.to_csv(out / "risk_validation_per_unit.csv", index=False)
    training.validation_summary.to_csv(out / "risk_model_validation.csv", index=False)
    _save_json(out / "risk_model_selection.json", training.selection)
    with (out / "unit_risk_model.pkl").open("wb") as handle:
        pickle.dump(training.model, handle)
    return training


def run_risk_training(config):
    """Train from a previously generated contribution bank without rebuilding it."""
    out = Path(config["out"]) / "pipeline"
    names = [
        "unit_training_features.csv",
        "unit_training_labels.csv",
        "unit_validation_features.csv",
        "unit_validation_labels.csv",
    ]
    missing = [name for name in names if not (out / name).is_file()]
    if missing:
        raise ValueError(f"Missing risk-label artifacts: {missing}. Run the risk-labels stage first.")
    features, labels, val_features, val_labels = [pd.read_csv(out / name) for name in names]
    training = _fit_and_save_risk_model(config, out, features, labels, val_features, val_labels)
    print(training.validation_summary.to_string(index=False))
    return training.validation_summary


def run_risk_labels(config):
    """Generate and persist the full train/validation S2 contribution bank."""
    return run_pipeline({**config, "_pipeline_action": "risk-labels"})


def run_pipeline(config):
    started = time.perf_counter()
    out = Path(config["out"]) / "pipeline"
    out.mkdir(parents=True, exist_ok=True)
    label_only = config.get("_pipeline_action") == "risk-labels"
    if not label_only and config.get("vacancies_path") and not Path(config["vacancies_path"]).is_file():
        raise ValueError("Vacancy data are missing; run scripts/fetch_vacancies.py or set vacancies_path to null")
    print(
        "Stage 3: detection and known fields to learned contributions, self-linkage control, S2, and utility.",
        flush=True,
    )
    bundle = _load_bundle({**config, "out": config.get("workua_source_out") or config["out"]})
    codec, exs = bundle["codec"], bundle["extracted"]
    split, clusters = bundle["split"], bundle["clusters"]
    train_pos = np.flatnonzero(split.split.eq("train"))
    detection_out = Path(config.get("detection_source_out") or config["out"]) / "detection"
    selected_path = detection_out / "selected_detector.json"
    if not selected_path.is_file():
        raise ValueError("Detector calibration is incomplete; run detection or set detection_source_out")
    selected = json.loads(selected_path.read_text())
    detection_manifest = json.loads((detection_out / "manifest.json").read_text())
    if selected["status"] != "ready" or not set(config["release_detectors"]) <= set(selected["detectors"]):
        raise ValueError("Detection is incomplete or does not match release_detectors")
    if detection_manifest["source_sha256"] != _fingerprint(config["e3_path"]):
        raise ValueError("E3 data changed; rerun detector calibration")
    config = {
        **config,
        "gliner_model": selected.get("gliner_model", config.get("gliner_model", "urchade/gliner_multi_pii-v1")),
        "gliner_threshold": selected["gliner_threshold"],
    }
    if config.get("prepared_source_out"):
        record, directory = _verify_prepared(config)
        config = {**config, "_prepared_checkpoint_dir": str(directory)}
    print(
        f"Detector calibration: {detection_out}; GLiNER threshold={selected['gliner_threshold']}",
        flush=True,
    )
    if config.get("resume_checkpoints", False):
        from importlib.metadata import PackageNotFoundError, version

        dependency_versions = {}
        for package in ["numpy", "scikit-learn", "torch", "transformers", "gliner", "spacy", "spacy-transformers"]:
            try:
                dependency_versions[package] = version(package)
            except PackageNotFoundError:
                dependency_versions[package] = None
        signature_config = {key: value for key, value in config.items() if key != "_pipeline_action"}
        signature = {
            "config": signature_config,
            "csv": _fingerprint(config["workua_path"]),
            "prepared_record": _fingerprint(Path(config["prepared_source_out"]) / "prepared_recovery.json")
            if config.get("prepared_source_out")
            else None,
            "calibration": _fingerprint(selected_path),
            "e3": detection_manifest["source_sha256"],
            "dependencies": dependency_versions,
            "development_split": _fingerprint(Path(config["development_out"]) / "workua" / "split.csv")
            if config.get("development_out")
            else None,
            "development_queries": _fingerprint(Path(config["development_out"]) / "pipeline" / "test_per_query.csv")
            if config.get("development_out")
            else None,
            "code": {p.name: _fingerprint(p) for p in Path(__file__).parent.glob("*.py")},
        }
        config["_checkpoint_signature"] = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[
            :24
        ]
        label_irrelevant = {
            "strategies",
            "taus",
            "utility_floors",
            "target_validation_recall",
            "candidate_limit",
            "max_candidates_per_step",
            "max_steps",
            "tuning_val_persons",
            "run_downstream",
            "n_bootstrap",
            "expand_on_plateau",
            "risk_model_grid",
            "risk_context_weights",
            "risk_selection_metric",
            "risk_learning_curve_persons",
            "use_char_tfidf",
            "controller_attackers",
            "require_margin",
            "vacancies_path",
            "vacancy_model_path",
            "vacancy_limit",
            "vacancy_k",
            "vacancy_components",
            "vacancy_max_features",
            "vacancy_validation_queries",
        }
        label_signature = {
            **signature,
            "config": {key: value for key, value in signature_config.items() if key not in label_irrelevant},
        }
        label_namespace = hashlib.sha256(json.dumps(label_signature, sort_keys=True).encode()).hexdigest()[:24]
        config["_checkpoint_signatures"] = {
            "documents": label_namespace,
            "contributions": label_namespace,
        }
        print(
            "Checkpoints are enabled; reruns reuse completed computations for this profile.",
            flush=True,
        )
    # Paid or local LLMs never run on WorkUA without explicit selection.
    names = config.get("release_detectors", ["regex", "ner_uk", "gliner"])
    detector, _detector_objects = detector_set(config, names)
    texts = bundle["texts"]
    print("BM25: building the complete original candidate pool for this profile...", flush=True)
    bm25 = BM25Index(max_features=config.get("bm25_max_features", 150000)).fit(texts)
    pop = Population(bundle["ids"], exs, clusters, codec, bm25, texts)
    indexes = {"BM25": bm25}
    if config.get("use_char_tfidf", False) and not label_only:
        print("Adding character TF-IDF; this stage is more expensive...", flush=True)
        indexes["char-TF-IDF"] = TfidfIndex(max_features=100000).fit(texts)
    control_names = ["BM25"] if label_only else (config.get("controller_attackers") or list(indexes))
    if not set(control_names) <= set(indexes):
        raise ValueError("Unknown attacker in controller_attackers")
    controller = TextRiskController(
        {name: indexes[name] for name in control_names}, require_margin=config.get("require_margin", False)
    )
    print("Transformation search:", control_names, "| independent evaluation:", list(indexes), flush=True)
    linked = pop.linked_positions()
    max_chars = config.get("max_document_chars")
    eligible_queries = [k for k in linked if max_chars is None or len(texts[k]) <= max_chars]
    positions = {
        s: _pick([k for k in eligible_queries if split.split.iloc[k] == s], config.get("n_" + s), config["seed"])
        for s in ["train", "val", "test"]
    }
    print(
        f"Linked queries: {len(linked)}; eligible under the configured length limit: {len(eligible_queries)}.",
        flush=True,
    )
    if any(not len(v) for v in positions.values()):
        raise ValueError("No linked train/validation/test queries; increase pool_sample")
    documents = {}
    document_splits = ["train", "val"] if label_only else ["train", "val", "test"]
    document_positions = np.concatenate([positions[split_name] for split_name in document_splits])
    for k in tqdm(document_positions, desc="Fields, entities, and residual text"):
        documents[int(k)] = _checkpoint(
            config,
            "documents",
            str(k),
            lambda k=k: make_document(exs[k], codec, detector, include_residual=config.get("include_residual", True)),
        )
        if config.get("max_units_per_doc") is not None and len(documents[int(k)].units) > config["max_units_per_doc"]:
            raise ValueError("Too many units; increase max_units_per_doc explicitly")
    pd.DataFrame(
        [
            {
                "pos": k,
                "person": clusters[k],
                "n_units": len(doc.units),
                "n_quasi": sum(u.role == "QUASI" for u in doc.units),
                "n_residual": sum(u.source == "residual_text" for u in doc.units),
            }
            for k, doc in documents.items()
        ]
    ).to_csv(out / "document_units.csv", index=False)
    if config.get("prepared_source_out"):
        source = Path(config["prepared_source_out"]) / "pipeline"
        for name in record["files"]:
            if name.startswith("pipeline/"):
                shutil.copy2(Path(config["prepared_source_out"]) / name, out / Path(name).name)
        with (source / "unit_risk_model.pkl").open("rb") as handle:
            model = pickle.load(handle)
        with (source / "retention_vectorizer.pkl").open("rb") as handle:
            vec = pickle.load(handle)
    else:
        print("Retention vocabulary: training data only...", flush=True)
        vec = TfidfVectorizer(token_pattern=r"(?u)\b\w[\w'’+#.\-]*\b", sublinear_tf=True, max_features=150000).fit(
            texts[k] for k in train_pos
        )

        def contributions(pos, desc):
            parts = [
                _checkpoint(
                    config,
                    "contributions",
                    str(k),
                    lambda k=k: train_unit_contributions(pop, [k], documents, bundle["reference"]),
                )
                for k in tqdm(pos, desc=desc)
            ]
            return tuple(pd.concat([part[i] for part in parts], ignore_index=True) for i in [0, 1])

        features, labels = contributions(positions["train"], "Unit risk: training S2, two contexts")
        labels.to_csv(out / "unit_training_labels.csv", index=False)
        features.to_csv(out / "unit_training_features.csv", index=False)
        with (out / "retention_vectorizer.pkl").open("wb") as handle:
            pickle.dump(vec, handle)
        val_features, val_labels = contributions(
            positions["val"], "Contribution prediction evaluation: validation only"
        )
        val_labels.to_csv(out / "unit_validation_labels.csv", index=False)
        val_features.to_csv(out / "unit_validation_features.csv", index=False)
        if config.get("_pipeline_action") == "risk-labels":
            _save_json(
                out / "risk_label_manifest.json",
                {
                    "target": "mean_forward_addback_delta_S2_train_units",
                    "n_train_people": int(labels.person.nunique()),
                    "n_train_queries": int(labels.pos.nunique()),
                    "n_train_units": len(labels),
                    "n_validation_people": int(val_labels.person.nunique()),
                    "n_validation_queries": int(val_labels.pos.nunique()),
                    "n_validation_units": len(val_labels),
                },
            )
            return pd.DataFrame(
                [{"stage": "risk-labels", "n_train_units": len(labels), "n_validation_units": len(val_labels)}]
            )
        training = _fit_and_save_risk_model(config, out, features, labels, val_features, val_labels)
        model = training.model

    def evaluate_queries(pos, queries, source, task_predictions=None):
        targets, excludes = pop.targets_and_excludes(pos, "S2")
        rows, details = [], []
        A = vec.transform([texts[k] for k in pos])
        B = vec.transform(queries)
        retention = np.asarray(A.multiply(B).sum(axis=1)).ravel()
        for aname, ix in indexes.items():
            result = ranking_result(ix, queries, targets, excludes, batch_size=16)[0]
            detail = result.to_frame().assign(pos=pos, person=clusters[pos], retention=retention)
            if task_predictions is not None:
                for col, values in task_predictions.items():
                    detail[col] = values
            row = {m: v[0] for m, v in metric_draws(detail, task_classes).items()}
            row.update(
                policy=source,
                attacker=aname,
                cohort="all",
                scenario="S2",
                n_resumes=len(pos),
                n_persons=detail.person.nunique(),
            )
            rows.append(row)
            details.append(detail.assign(policy=source, attacker=aname))
        return rows, details

    tune_pos = _tuning_positions(positions["val"], clusters, config.get("tuning_val_persons"), config["seed"])
    pd.DataFrame({"pos": tune_pos, "person": clusters[tune_pos]}).to_csv(out / "tuning_positions.csv", index=False)
    print(
        f"Parameter tuning: {len(tune_pos)} resumes / {len(set(clusters[tune_pos]))} identities; "
        "the test set remains complete.",
        flush=True,
    )
    task_classes = {}
    validation_rows, validation_details = [], []
    grid = [
        (strategy, tau, floor)
        for strategy in config.get("strategies", ["category", "model"])
        for tau in config.get("taus", [0.2, 0.5])
        for floor in config.get("utility_floors", [0.25, 0.5])
    ]
    for strategy, tau, floor in tqdm(grid, desc="Policies and parameters: validation only"):
        output = []
        for k in tqdm(tune_pos, desc=f"{strategy} τ={tau} utility≥{floor}", leave=False):
            output.append(
                _checkpoint(
                    config,
                    "validation",
                    f"{strategy}_{tau}_{floor}_{k}",
                    lambda k=k, strategy=strategy, tau=tau, floor=floor: adaptive_release(
                        documents[int(k)],
                        int(k),
                        controller,
                        model,
                        bundle["reference"],
                        bm25,
                        vec,
                        tau=tau,
                        strategy=strategy,
                        utility_floor=floor,
                        max_steps=config.get("max_steps"),
                        candidate_limit=config.get("candidate_limit", 8),
                        max_candidates_per_step=config.get("max_candidates_per_step"),
                        progress_steps=config.get("progress_steps", False),
                        expand_on_plateau=config.get("expand_on_plateau", True),
                    ),
                )
            )
        name = f"{strategy}_tau{tau}_floor{floor}"
        rows, detail_frames = evaluate_queries(tune_pos, [r[0] for r in output], name)
        for frame in detail_frames:
            frame["proxy_threshold_met"] = [r[2][-1]["threshold_met"] for r in output]
            frame["proxy_rr"] = [r[2][-1]["proxy_rr"] for r in output]
            frame["stop_reason"] = [r[2][-1]["stop_reason"] for r in output]
            validation_details.append(frame)
        for row in rows:
            validation_rows.append(
                {
                    **row,
                    "split": "val",
                    "strategy": strategy,
                    "tau": tau,
                    "utility_floor": floor,
                    "proxy_met_share": np.mean([r[2][-1]["threshold_met"] for r in output]),
                    **{
                        reason + "_count": sum(r[2][-1]["stop_reason"] == reason for r in output)
                        for reason in [
                            "max_steps",
                            "search_budget",
                            "utility_floor",
                            "actions_exhausted",
                            "threshold_met",
                        ]
                    },
                }
            )
        pd.DataFrame(validation_rows).to_csv(out / "validation_grid.csv", index=False)
        pd.concat(validation_details, ignore_index=True).to_csv(out / "validation_per_query.csv", index=False)
    validation = pd.DataFrame(validation_rows)
    validation.to_csv(out / "validation_grid.csv", index=False)
    # Evaluate fixed references before selecting an adaptive point. Selection is
    # validation-only and can explicitly require matching L2 before maximizing
    # retention; test remains untouched.
    aggregate = validation.groupby(["policy", "strategy", "tau", "utility_floor"], as_index=False).agg(
        worst_recall=("recall@1", "max"), retention=("retention", "first"), proxy_met_share=("proxy_met_share", "first")
    )
    baseline_rows = []
    for name, policy in tqdm(fixed_policies().items(), desc="Baseline policies: validation"):
        queries = [codec.render(exs[k], policy) for k in tune_pos]
        baseline_rows.extend(evaluate_queries(tune_pos, queries, name)[0])
    masked = [
        apply_actions(
            documents[int(k)],
            {
                **{
                    i: "mask"
                    for i, u in enumerate(documents[int(k)].units)
                    if u.source != "residual_text" and u.role != "CONTEXT"
                },
                **mandatory_actions(documents[int(k)]),
            },
        )
        for k in tune_pos
    ]
    baseline_rows.extend(evaluate_queries(tune_pos, masked, "mask_detected")[0])
    baseline_frame = pd.DataFrame(baseline_rows)
    baseline_frame.to_csv(out / "validation_baselines.csv", index=False)
    reference_policy = config.get("selection_reference_policy")
    reference_recall = None
    if reference_policy:
        reference_rows = baseline_frame[baseline_frame.policy.eq(reference_policy)]
        if reference_rows.empty:
            raise ValueError(f"Unknown or unavailable selection_reference_policy: {reference_policy}")
        reference_recall = float(reference_rows["recall@1"].max())
    selected, status = _select_adaptive_point(
        aggregate,
        config.get("target_validation_recall", 0.2),
        reference_recall=reference_recall,
    )
    _save_json(
        out / "operating_point.json",
        {
            **selected.to_dict(),
            "status": status,
            "target": config.get("target_validation_recall", 0.2),
            "selection_reference_policy": reference_policy,
            "selection_reference_worst_recall": reference_recall,
        },
    )
    print("Validation:", status, selected.to_dict(), flush=True)
    frontier = (
        pd.concat([validation, baseline_frame], ignore_index=True)
        .groupby("policy", as_index=False)
        .agg(worst_recall=("recall@1", "max"), retention=("retention", "first"))
    )
    frontier["dominated_on_validation"] = [
        bool(
            (
                (frontier.worst_recall <= row.worst_recall)
                & (frontier.retention >= row.retention)
                & ((frontier.worst_recall < row.worst_recall) | (frontier.retention > row.retention))
            ).any()
        )
        for row in frontier.itertuples()
    ]
    frontier.to_csv(out / "validation_tradeoffs.csv", index=False)

    tasks = []
    if config.get("run_downstream", True):
        for task in tqdm(
            [DownstreamTask("industry", {"exp_industry"}), DownstreamTask("salary", {"salary"})],
            desc="Fixed utility classifiers: training",
        ):
            try:
                tasks.append(task.fit(codec, [exs[k] for k in train_pos]))
            except ValueError:
                if config.get("allow_missing_utility", False):
                    print(f"{task.name}: training data have fewer than two classes; metric unavailable", flush=True)
                else:
                    raise
        task_classes = {task.name: sorted(task.classes_) for task in tasks}
    test_pos = positions["test"]
    cohorts = query_cohorts(pop, test_pos, texts, config.get("near_duplicate_threshold", 0.9))
    cohorts.to_csv(out / "test_query_cohorts.csv", index=False)
    eligible, distinct_tg, distinct_ex = distinct_targets(
        pop, test_pos, texts, config.get("near_duplicate_threshold", 0.9)
    )
    summary_rows, all_details, all_ci, paired, stores, trace_rows, decision_rows = [], [], [], [], {}, [], []
    release_cache = {}
    # Compare each method at its own best validation point and at the common selected point.
    specs = [(name, policy, None) for name, policy in fixed_policies().items()]
    strategy_settings = []
    for strategy in config.get("strategies", ["category", "model"]):
        subset = aggregate[aggregate.strategy.eq(strategy)]
        best, strategy_status = _select_adaptive_point(
            subset,
            config.get("target_validation_recall", 0.2),
            reference_recall=reference_recall,
        )
        specs.append(("adaptive_" + strategy, None, best))
        strategy_settings.append(
            {
                **best.to_dict(),
                "release_policy": "adaptive_" + strategy,
                "status": strategy_status,
                "selection_reference_policy": reference_policy,
                "selection_reference_worst_recall": reference_recall,
            }
        )
    pd.DataFrame(strategy_settings).to_csv(out / "strategy_operating_points.csv", index=False)
    specs.append(("mask_detected", None, "mask_detected"))
    for name, policy, setting in tqdm(specs, desc="Final comparison: test"):
        outputs = []
        if policy is not None:
            queries = [codec.render(exs[k], policy) for k in test_pos]
        else:
            for k in tqdm(test_pos, desc=name + ": resumes", leave=False):
                doc = documents[int(k)]
                if isinstance(setting, str):
                    actions = {
                        i: "mask"
                        for i, u in enumerate(doc.units)
                        if u.source != "residual_text" and u.role != "CONTEXT"
                    }
                    actions.update(mandatory_actions(doc))
                    output = (apply_actions(doc, actions), actions, [])
                else:
                    output = _checkpoint(
                        config,
                        "test_release",
                        f"{name}_{k}",
                        lambda doc=doc, k=k, setting=setting: adaptive_release(
                            doc,
                            int(k),
                            controller,
                            model,
                            bundle["reference"],
                            bm25,
                            vec,
                            tau=float(setting.tau),
                            strategy=str(setting.strategy),
                            utility_floor=float(setting.utility_floor),
                            max_steps=config.get("max_steps"),
                            candidate_limit=config.get("candidate_limit", 8),
                            max_candidates_per_step=config.get("max_candidates_per_step"),
                            progress_steps=config.get("progress_steps", False),
                            expand_on_plateau=config.get("expand_on_plateau", True),
                        ),
                    )
                outputs.append(output)
                trace_rows.extend(
                    {**row, "pos": int(k), "person": str(clusters[k]), "policy": name} for row in output[2]
                )
                for i, operation in output[1].items():
                    u = doc.units[i]
                    decision_rows.append(
                        {
                            "pos": int(k),
                            "policy": name,
                            "unit": i,
                            "start": u.start,
                            "end": u.end,
                            "label": u.label,
                            "role": u.role,
                            "source": u.source,
                            "group": u.group,
                            "operation": operation,
                        }
                    )
            queries = [r[0] for r in outputs]
        release_cache[name] = queries
        utility_predictions = {}
        for task in tasks:
            original_ex = [exs[k] for k in test_pos]
            truth = task.labels(original_ex)
            valid = np.array([v in task.classes_ for v in truth])
            utility_texts = []
            for j, k in enumerate(tqdm(test_pos, desc=f"{name}: utility {task.name}", leave=False)):
                if not valid[j]:
                    utility_texts.append("")
                    continue
                if policy is not None:
                    utility_texts.append(codec.render(exs[k], policy, exclude=task.exclude))
                else:
                    # Remove the target group before detection and policy decisions, then
                    # replay the same actions on the utility view.
                    doc = _checkpoint(
                        config,
                        "utility_documents",
                        f"{task.name}_{k}",
                        lambda k=k, task=task: make_document(
                            exs[k],
                            codec,
                            detector,
                            exclude=task.exclude,
                            include_residual=config.get("include_residual", True),
                        ),
                    )
                    if isinstance(setting, str):
                        actions = {
                            i: "mask"
                            for i, u in enumerate(doc.units)
                            if u.source != "residual_text" and u.role != "CONTEXT"
                        }
                        actions.update(mandatory_actions(doc))
                        utility_texts.append(apply_actions(doc, actions))
                    else:
                        utility_texts.append(
                            _checkpoint(
                                config,
                                "utility_release",
                                f"{name}_{task.name}_{k}",
                                lambda doc=doc, k=k, setting=setting: adaptive_release(
                                    doc,
                                    int(k),
                                    controller,
                                    model,
                                    bundle["reference"],
                                    bm25,
                                    vec,
                                    tau=float(setting.tau),
                                    strategy=str(setting.strategy),
                                    utility_floor=float(setting.utility_floor),
                                    max_steps=config.get("max_steps"),
                                    candidate_limit=config.get("candidate_limit", 8),
                                    max_candidates_per_step=config.get("max_candidates_per_step"),
                                    progress_steps=config.get("progress_steps", False),
                                    expand_on_plateau=config.get("expand_on_plateau", True),
                                ),
                            )[0]
                        )
            prediction = np.full(len(test_pos), None, dtype=object)
            if valid.any():
                prediction[valid] = task.model.predict([utility_texts[j] for j in np.flatnonzero(valid)])
            utility_predictions[task.name + "_true"] = [v if keep else None for v, keep in zip(truth, valid)]
            utility_predictions[task.name + "_pred"] = prediction
        rows, detail_frames = evaluate_queries(test_pos, queries, name, utility_predictions)
        for row, detail in zip(rows, detail_frames):
            if outputs and not isinstance(setting, str):
                row["proxy_unmet_count"] = sum(not r[2][-1]["threshold_met"] for r in outputs)
                row["utility_floor_stops"] = sum(r[2][-1]["stop_reason"] == "utility_floor" for r in outputs)
                row["max_steps_stops"] = sum(r[2][-1]["stop_reason"] == "max_steps" for r in outputs)
                row["search_budget_stops"] = sum(r[2][-1]["stop_reason"] == "search_budget" for r in outputs)
                row["actions_exhausted_stops"] = sum(r[2][-1]["stop_reason"] == "actions_exhausted" for r in outputs)
                row["tau"], row["utility_floor"] = float(setting.tau), float(setting.utility_floor)
                row["calibration_status"] = next(s["status"] for s in strategy_settings if s["release_policy"] == name)
                detail["proxy_threshold_met"] = [r[2][-1]["threshold_met"] for r in outputs]
                detail["proxy_rr"] = [r[2][-1]["proxy_rr"] for r in outputs]
                detail["stop_reason"] = [r[2][-1]["stop_reason"] for r in outputs]
                detail["utility_floor_met"] = [r[2][-1]["utility_floor_met"] for r in outputs]
            for task in tasks:
                row[task.name + "_n"] = int(detail[task.name + "_true"].notna().sum())
                row[task.name + "_n_persons"] = detail.loc[detail[task.name + "_true"].notna(), "person"].nunique()
            summary_rows.append(row)
            stores[(row["attacker"], name)] = detail
            all_details.append(detail)
            all_ci.append(
                metric_intervals(detail, task_classes, n=config.get("n_bootstrap", 300), seed=config["seed"]).assign(
                    policy=name, attacker=row["attacker"]
                )
            )
            # Report standard S2 separately for copy, format-only, near, and distinct cohorts.
            joined = detail.merge(cohorts.drop(columns="person"), on="pos", validate="one_to_one")
            for cohort, part in joined.groupby("cohort"):
                points = {m: v[0] for m, v in metric_draws(part, task_classes).items()}
                summary_rows.append(
                    {
                        **points,
                        "policy": name,
                        "attacker": row["attacker"],
                        "cohort": cohort,
                        "n_resumes": len(part),
                        "n_persons": part.person.nunique(),
                        "scenario": "S2",
                    }
                )
        if len(eligible):
            lookup = dict(zip(test_pos, queries))
            for aname, ix in indexes.items():
                result = ranking_result(ix, [lookup[k] for k in eligible], distinct_tg, distinct_ex, batch_size=16)[0]
                detail = result.to_frame().assign(pos=eligible, person=clusters[eligible])
                summary_rows.append(
                    {
                        **{m: v[0] for m, v in metric_draws(detail).items()},
                        "policy": name,
                        "attacker": aname,
                        "cohort": "all",
                        "scenario": "S2_distinct",
                        "n_resumes": len(detail),
                        "n_persons": detail.person.nunique(),
                    }
                )
    for aname in indexes:
        for method in ["adaptive_" + s for s in config.get("strategies", ["category", "model"])]:
            for baseline in [
                "L0_none",
                "L2_generalize",
                "L3_strong",
                "mask_detected",
                *(["adaptive_category"] if method == "adaptive_model" and "category" in config.get("strategies", ["category", "model"]) else []),
            ]:
                paired.append(
                    paired_differences(
                        stores[(aname, method)],
                        stores[(aname, baseline)],
                        task_classes,
                        n=config.get("n_bootstrap", 300),
                        seed=config["seed"],
                    ).assign(attacker=aname, policy=method, reference=baseline)
                )
    pd.DataFrame(summary_rows).to_csv(out / "test_summary.csv", index=False)
    pd.concat(all_details, ignore_index=True).to_csv(out / "test_per_query.csv", index=False)
    pd.concat(all_ci, ignore_index=True).to_csv(out / "test_intervals.csv", index=False)
    pd.concat(paired, ignore_index=True).to_csv(out / "paired_differences.csv", index=False)
    pd.DataFrame(trace_rows).to_csv(out / "release_traces.csv", index=False)
    final_traces = pd.DataFrame(trace_rows)
    if len(final_traces):
        final_traces[final_traces.stop_reason.notna()].groupby(["policy", "stop_reason"]).size().rename(
            "n_resumes"
        ).reset_index().to_csv(out / "stop_reasons.csv", index=False)
    pd.DataFrame(decision_rows).to_csv(out / "release_operations.csv", index=False)
    with (out / "release_queries.pkl").open("wb") as handle:
        pickle.dump({"positions": test_pos, "queries": release_cache}, handle)
    _save_json(
        out / "manifest.json",
        {
            "config": config,
            "metrics_version": "exact_random_ties_v3",
            "release_version": "budgeted_search_v3_2",
            "risk_target": "mean_forward_addback_delta_S2_train_units",
            "risk_training_weights": "equal_total_weight_per_person",
            "context_weight": model.context_weight,
            "context_status": "category_fallback_on_validation"
            if model.context_weight == 0
            else "context_correction_enabled",
            "utility_protocol": "target_excluded_before_release",
            "n_train_queries": len(positions["train"]),
            "n_val_queries": len(positions["val"]),
            "n_test_queries": len(test_pos),
            "n_distinct_test_queries": len(eligible),
            "n_linked_queries": len(linked),
            "n_length_eligible_queries": len(eligible_queries),
            "calibration_status": status,
            "strategy_operating_points": strategy_settings,
            "n_train_persons": len(set(clusters[positions["train"]])),
            "n_val_persons": len(set(clusters[positions["val"]])),
            "n_test_persons": len(set(clusters[test_pos])),
            "n_tuning_persons": len(set(clusters[tune_pos])),
            "n_tuning_queries": len(tune_pos),
            "task_classes": task_classes,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    if config.get("vacancies_path"):
        run_vacancy_evaluation(config, bundle=bundle)
    print(pd.DataFrame(summary_rows).to_string(index=False))
    return pd.DataFrame(summary_rows)


def run_vacancy_training(config):
    """Train and persist the vacancy-fit model independently of resume evaluation."""
    vacancy_path = config.get("vacancies_path")
    model_path = config.get("vacancy_model_path")
    if not vacancy_path or not model_path:
        raise ValueError("Set vacancies_path and vacancy_model_path in the configuration")
    frame = load_vacancies(vacancy_path, config.get("vacancy_limit", 10_000))
    model = VacancyFitModel(
        k=int(config.get("vacancy_k", 10)),
        n_components=int(config.get("vacancy_components", 128)),
        max_features=int(config.get("vacancy_max_features", 100_000)),
        random_state=int(config["seed"]),
    ).fit(frame)
    model.save(model_path)
    diagnostics = retrieval_sanity_check(
        model,
        n=int(config.get("vacancy_validation_queries", 500)),
        seed=int(config["seed"]),
    )
    sampling_manifest = Path(vacancy_path).with_suffix(".manifest.json")
    manifest = {
        "model": "tfidf_word_1_2gram_truncated_svd_cosine",
        "model_path": str(model_path),
        "vacancies_file_sha256": _fingerprint(vacancy_path),
        "sampling": json.loads(sampling_manifest.read_text()) if sampling_manifest.is_file() else None,
        "training": model.training_summary,
        "diagnostics": diagnostics,
        "excluded_fields": ["company_id", "salary", "location_details", "description_text"],
        "interpretation": "job_fit_utility_proxy_not_reidentification_risk",
    }
    _save_json(Path(model_path).with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return model


def run_vacancy_evaluation(config, bundle=None):
    """Re-evaluate utility from saved texts without NER, LLM, or risk retraining."""
    from .statistics import cluster_mean_ci

    if not config.get("vacancies_path"):
        raise ValueError("Set vacancies_path in a copy of the configuration")
    bundle = (
        _load_bundle({**config, "out": config.get("workua_source_out") or config["out"]}) if bundle is None else bundle
    )
    out = Path(config["out"]) / "pipeline"
    with (out / "release_queries.pkl").open("rb") as handle:
        release = pickle.load(handle)
    positions = release["positions"]
    originals = [bundle["texts"][k] for k in positions]
    model_path = config.get("vacancy_model_path")
    if not model_path or not Path(model_path).is_file():
        raise FileNotFoundError("Train the vacancy model first: anonrisk vacancy-train --config <config>")
    model_manifest = Path(model_path).with_suffix(".manifest.json")
    if model_manifest.is_file():
        training = json.loads(model_manifest.read_text())
        expected = training.get("vacancies_file_sha256")
        current = _fingerprint(config["vacancies_path"])
        if expected and expected != current:
            raise ValueError("The vacancy model was trained on a different vacancies_path version")
    metric = VacancyFitModel.load(model_path)
    frames, rows = [], []
    for name, queries in tqdm(release["queries"].items(), desc="Prepared texts to top-k vacancies"):
        frame = metric.score(originals, queries).assign(
            pos=positions, person=bundle["clusters"][positions], policy=name
        )
        metric_columns = [
            column
            for column in frame
            if column.startswith(("vacancy_overlap@", "vacancy_ndcg@"))
            or column in {"vacancy_signal_retention", "vacancy_job_fit_utility", "vacancy_utility_loss_risk"}
        ]
        for column in metric_columns:
            low, high = cluster_mean_ci(
                frame[column].to_numpy(), frame.person.to_numpy(), n=config.get("n_bootstrap", 300), seed=config["seed"]
            )
            rows.append(
                {
                    "policy": name,
                    "metric": column,
                    "estimate": frame[column].mean(),
                    "ci_low": low,
                    "ci_high": high,
                    "n_scorable": int(frame.vacancy_scorable.sum()),
                    "n_persons": frame.loc[frame.vacancy_scorable, "person"].nunique(),
                }
            )
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_csv(out / "vacancy_per_query.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "vacancy_summary.csv", index=False)
    sampling_manifest = Path(config["vacancies_path"]).with_suffix(".manifest.json")
    _save_json(
        out / "vacancy_manifest.json",
        {
            "source_sha256": _fingerprint(config["vacancies_path"]),
            "sampling": json.loads(sampling_manifest.read_text()) if sampling_manifest.is_file() else None,
            "config": config,
            "model_manifest": json.loads(model_manifest.read_text()) if model_manifest.is_file() else None,
            "metric": "trained_lsa_fixed_reference_job_fit_v5",
            "formula": "0.45*NDCG + 0.35*weighted_recall + 0.20*signal_retention",
            "excluded_fields": ["company_id", "salary", "location_details", "description_text"],
        },
    )
    print(pd.DataFrame(rows).to_string(index=False))
    return pd.DataFrame(rows)
