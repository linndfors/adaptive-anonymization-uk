"""Benchmark frozen v5 and matched live-search inference in isolated workers.

All inputs and timings stay private under artifacts/. Only aggregate reports
may be published. Workers are sequential, use one CPU thread, and each warm
resume's repeated timings collapse to one observation before aggregation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("action_regression", "search_v3_3")
STRATA = ("short", "medium", "long")
THREAD_ENV = {name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                     "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "LOKY_MAX_CPU_COUNT")}
THREAD_ENV["TOKENIZERS_PARALLELISM"] = "false"


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2))


def peak_rss_bytes(raw=None, system=None):
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if raw is None else raw
    system = platform.system() if system is None else system
    if system == "Darwin":
        return int(raw)
    if system == "Linux":
        return int(raw * 1024)
    return None


def uncached_detection(detectors):
    """Run the same models and span validation without stored document hits."""
    from anonrisk_uk.detectors import CachedDetector, validate_spans

    inner = []
    for detector in detectors:
        while isinstance(detector, CachedDetector):
            detector = detector.detector
        inner.append(detector)

    def detect(text):
        return [span for detector in inner for span in validate_spans(text, detector.detect(text))]
    return detect


def select_cases(records, per_stratum=6, seed=20261111):
    """One seeded resume/person, followed by equal length-tertile sampling."""
    import numpy as np

    if per_stratum < 1:
        raise ValueError("Each stratum needs at least one person")
    grouped = {}
    for row in records:
        grouped.setdefault(str(row["person"]), []).append(row)
    rng = np.random.default_rng(seed)
    representatives = []
    for person in sorted(grouped):
        choices = sorted(grouped[person], key=lambda row: str(row["id"]))
        representatives.append(choices[int(rng.integers(len(choices)))])
    representatives.sort(key=lambda row: (len(row["original"].split()), str(row["person"])))
    chosen, descriptions = [], {}
    for label, indices in zip(STRATA, np.array_split(np.arange(len(representatives)), 3)):
        if len(indices) < per_stratum:
            raise ValueError("Insufficient distinct people for length-stratified timing")
        selected = rng.choice(indices, size=per_stratum, replace=False)
        lengths = [len(representatives[int(i)]["original"].split()) for i in indices]
        for index in selected:
            row = representatives[int(index)]
            chosen.append({"case": f"case_{len(chosen) + 1:03d}", "length_stratum": label,
                           "words": len(row["original"].split()), "record": row})
        descriptions[label] = {"available_people": len(indices), "sampled_people": per_stratum,
                               "population_source_words_min": min(lengths),
                               "population_source_words_max": max(lengths)}
    return chosen, descriptions


def runtime_source_paths():
    names = ("action_resume.py", "action_learning.py", "resume_adaptive.py", "resume_parser.py",
             "resume_context.py", "resume_fields.py", "category_taxonomy.py", "learned_resume_risk.py",
             "workua.py", "research.py", "detectors.py", "e3jsi.py", "attack.py", "resume_resources.py")
    return [Path(__file__), *[ROOT / "src/anonrisk_uk" / name for name in names]]


def prepare(args):
    import numpy as np
    import pandas as pd
    from action_data import load_action_data

    if args.out.exists():
        raise FileExistsError("Benchmark directory exists; use --execute-existing to run a prepared protocol")
    parent = json.loads((args.run / "protocol.json").read_text())
    if digest(args.model) != parent["model_sha256"]:
        raise ValueError("Benchmark must use the original frozen model")
    if digest(args.data_dir / "manifest.json") != parent["data_manifest_sha256"]:
        raise ValueError("Cohort differs from the original test")
    records = [json.loads(line) for line in (args.run / "releases.jsonl").read_text().splitlines()]
    if len(records) != parent["queries"] or any(not all(m in row["outputs"] for m in METHODS) for row in records):
        raise ValueError("Complete frozen releases for both methods are required")
    chosen, descriptions = select_cases(records, args.per_stratum, args.seed)
    if args.repeats < 1:
        raise ValueError("At least one warm timing repetition is required")
    data = load_action_data(args.data_dir)
    with args.model.open("rb") as handle:
        estimator = pickle.load(handle)
    settings = json.loads(args.config.read_text())
    from anonrisk_uk.resume_resources import resource_paths

    _, profile, paths = resource_paths(args.config)
    by_id = {str(rid): i for i, rid in enumerate(data["ids"])}
    pool_ids = list(pd.read_csv(args.run / "pool_ids.csv", dtype=str).id)
    pool = [estimator.codec.render(data["extracted"][by_id[rid]], {}) for rid in pool_ids]
    if len(pool) != parent["pool_size"]:
        raise ValueError("Frozen reference pool is incomplete")
    args.out.mkdir(parents=True)
    queries = []
    manifest = []
    for case in chosen:
        row = case["record"]
        if estimator.codec.render(data["extracted"][by_id[str(row["id"])]], {}) != row["original"]:
            raise ValueError("Selected source no longer matches the frozen codec")
        queries.append({key: case[key] for key in ("case", "length_stratum", "words")}
                       | {"text": row["original"],
                          "expected_output_sha256": {method: hashlib.sha256(row["outputs"][method].encode()).hexdigest()
                                                      for method in METHODS}})
        manifest.append({key: case[key] for key in ("case", "length_stratum", "words")}
                        | {"id": str(row["id"]), "person": str(row["person"])})
    save_json(args.out / "queries.json", queries)
    pd.DataFrame(manifest).to_csv(args.out / "sample_manifest.csv", index=False)
    with (args.out / "reference_texts.pkl").open("xb") as handle:
        pickle.dump(pool, handle, protocol=5)
    with (args.out / "search_resources.pkl").open("xb") as handle:
        pickle.dump({"codec": estimator.codec, "retention_vectorizer": estimator.retention_vectorizer}, handle,
                    protocol=5)
    rng = np.random.default_rng(args.seed + 1)
    cases = [row["case"] for row in queries]
    schedule = [list(map(str, rng.permutation(cases))) for _ in range(args.repeats)]
    method_order = list(map(str, rng.permutation(METHODS)))
    sources = runtime_source_paths()
    detector_profile = ROOT / settings["resources_config"]
    sources.extend([args.config, detector_profile, paths["detector"]])
    protocol = {
        "schema": "frozen_runtime_benchmark_v1", "purpose": "descriptive deployment measurement, no model selection",
        "model_path": str(args.model.resolve()), "model_sha256": digest(args.model),
        "config_path": str(args.config.resolve()), "settings": estimator.settings,
        "source_run": str(args.run), "source_releases_sha256": digest(args.run / "releases.jsonl"),
        "data_manifest_sha256": parent["data_manifest_sha256"], "seed": args.seed,
        "people": len(chosen), "queries": len(chosen), "repeats": args.repeats,
        "per_stratum": args.per_stratum, "strata": descriptions,
        "selection": "one seeded resume/person; length tertiles; equal seeded person sample; no outcome selection",
        "method_order": method_order, "warm_schedule": schedule, "cold_first_case": cases[0],
        "warm_observation": "median wall/CPU latency across repeated requests for one distinct person",
        "timed_request": "parse current text + construct controller state/proxy + choose edits + render release",
        "cold_start": "new process to resources ready; imports, detector/model loading, and search index build included",
        "cold_cache_note": "cold process only; OS filesystem caches are not flushed; one startup per method",
        "warmup": "one first request per method, timed separately and excluded from warm aggregates",
        "extraction_modes": ["uncached_extraction", "document_cache_hot"],
        "extraction_cache_note": "Primary uncached extraction bypasses only CachedDetector document storage; underlying NER/regex models and span validation are unchanged. Secondary cache-hot mode measures the real persistent detector cache after all sampled texts have been primed outside request timing.",
        "search_pool_size": len(pool), "search_target": 0., "search_utility_floor": parent["utility_floor"],
        "search_index": {"bm25_max_features": 20000, "char_max_features": 60000},
        "live_pool_used_for_prediction": {"action_regression": False, "search_v3_3": True},
        "resource_isolation": "v5 worker loads queries+model+detector only; search worker additionally loads corpus/indexes",
        "memory": "per-worker lifetime peak RSS through inference completion, before optional index serialization; not per-request RAM or isolated incremental model memory",
        "artifact_sizes": "deployment-specific serialized artifacts; common NER weights/environment excluded from both",
        "thread_env": THREAD_ENV, "detector_device": profile.get("detection_device", "cpu"),
        "source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in sources},
        "input_sha256": {path.name: digest(path) for path in (args.out / "queries.json", args.out / "reference_texts.pkl",
                                                                args.out / "search_resources.pkl", args.out / "sample_manifest.csv")},
        "scope": "local seeded 18-person stratified benchmark; p95 per 6-person stratum is unstable; not a service SLA",
    }
    save_json(args.out / "protocol.json", protocol)
    print(json.dumps({"prepared_people": len(chosen), "strata": descriptions, "method_order": method_order}, indent=2))


def verify_protocol(out, protocol, worker_method=None):
    if worker_method != "search_v3_3" and digest(protocol["model_path"]) != protocol["model_sha256"]:
        raise ValueError("Frozen model changed")
    for path, expected in protocol["source_sha256"].items():
        if digest(ROOT / path) != expected:
            raise ValueError(f"Frozen benchmark source/config changed: {path}")
    needed = ({"queries.json"} if worker_method == "action_regression" else
              {"queries.json", "reference_texts.pkl", "search_resources.pkl"} if worker_method == "search_v3_3" else
              set(protocol["input_sha256"]))
    for name, expected in protocol["input_sha256"].items():
        if name not in needed:
            continue
        if digest(out / name) != expected:
            raise ValueError(f"Frozen benchmark input changed: {name}")


def worker(out, method):
    started = time.perf_counter()
    import torch
    from threadpoolctl import threadpool_limits

    from anonrisk_uk.action_resume import action_search
    from anonrisk_uk.attack import BM25Index, TfidfIndex
    from anonrisk_uk.research import detector_set
    from anonrisk_uk.resume_adaptive import NeighbourProxy, render, search
    from anonrisk_uk.resume_parser import parse_resume
    from anonrisk_uk.resume_resources import resource_paths

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    protocol = json.loads((out / "protocol.json").read_text())
    verify_protocol(out, protocol, worker_method=method)
    cases = {row["case"]: row for row in json.loads((out / "queries.json").read_text())}
    with threadpool_limits(limits=1):
        resource_started = time.perf_counter()
        if method == "action_regression":
            with Path(protocol["model_path"]).open("rb") as handle:
                estimator = pickle.load(handle)
            codec, retention = estimator.codec, estimator.retention_vectorizer
        else:
            with (out / "search_resources.pkl").open("rb") as handle:
                resources = pickle.load(handle)
            codec, retention = resources["codec"], resources["retention_vectorizer"]
        settings, profile, paths = resource_paths(protocol["config_path"])
        selection = json.loads(paths["detector"].read_text())
        names = profile.get("release_detectors", ["regex", "ner_uk"])
        if "openai" in names or profile.get("detection_device", "cpu") != "cpu":
            raise ValueError("This benchmark requires local CPU extraction")
        config = {**profile, "adaptive": settings["adaptive"], "local_files_only": True,
                  "gliner_threshold": selection.get("gliner_threshold")}
        cached_detector, detector_objects = detector_set(config, names)
        detector = uncached_detection(detector_objects)
        resource_seconds = time.perf_counter() - resource_started
        index_seconds, pool, indexes = 0., None, None
        if method == "search_v3_3":
            index_started = time.perf_counter()
            with (out / "reference_texts.pkl").open("rb") as handle:
                pool = pickle.load(handle)
            indexes = {"bm25": BM25Index(max_features=20000).fit(pool),
                       "char": TfidfIndex(max_features=60000).fit(pool)}
            index_seconds = time.perf_counter() - index_started
        ready = {"event": "resources_ready", "method": method,
                 "worker_imports_and_resource_seconds": time.perf_counter() - started,
                 "model_and_detector_load_seconds": resource_seconds,
                 "reference_load_and_index_build_seconds": index_seconds,
                 "peak_rss_bytes_at_ready": peak_rss_bytes()}
        print(json.dumps(ready), flush=True)

        def infer(case, extraction_mode="uncached_extraction"):
            wall_start, cpu_start = time.perf_counter(), time.process_time()
            current_detector = detector if extraction_mode == "uncached_extraction" else cached_detector
            doc = parse_resume(case["text"], codec, current_detector)
            if method == "action_regression":
                policy, _ = action_search(doc, estimator)
            else:
                proxy = NeighbourProxy(indexes, pool, case["text"])
                policy, _ = search(doc, proxy, retention, target=0., utility_floor=protocol["search_utility_floor"])
            output = render(doc, policy)
            elapsed = {"wall_seconds": time.perf_counter() - wall_start,
                       "cpu_seconds": time.process_time() - cpu_start}
            if hashlib.sha256(output.encode()).hexdigest() != case["expected_output_sha256"][method]:
                raise ValueError(f"Frozen output parity failed for {method} {case['case']}")
            return elapsed

        first = cases[protocol["cold_first_case"]]
        cold = {"case": first["case"], "source_words": first["words"], **infer(first)}
        print(json.dumps({"event": "cold_first_request", "method": method, **cold}), flush=True)
        timings = []
        for mode in protocol["extraction_modes"]:
            if mode == "document_cache_hot":
                for case in cases.values():
                    cached_detector(case["text"])
            for repeat, order in enumerate(protocol["warm_schedule"]):
                for code in order:
                    case = cases[code]
                    timings.append({"case": code, "length_stratum": case["length_stratum"],
                                    "words": case["words"], "repeat": repeat, "extraction_mode": mode,
                                    **infer(case, mode)})
                    if len(timings) % 6 == 0:
                        print(json.dumps({"event": "warm_progress", "method": method, "extraction_mode": mode,
                                          "requests": len(timings), "total": protocol["queries"] * protocol["repeats"] *
                                          len(protocol["extraction_modes"])}), flush=True)
        peak = peak_rss_bytes()
        index_bytes = 0
        if indexes is not None:
            path = out / "search_indexes.pkl"
            with path.open("xb") as handle:
                pickle.dump(indexes, handle, protocol=5)
            index_bytes = path.stat().st_size
        verify_protocol(out, protocol, worker_method=method)
        result = {"method": method, "protocol_sha256": digest(out / "protocol.json"), **ready,
                  "cold_first_request": cold, "warm_timings": timings,
                  "peak_rss_bytes_through_inference": peak, "serialized_index_bytes": index_bytes,
                  "output_parity_checks_passed": len(timings) + 1,
                  "live_reference_pool_loaded": pool is not None}
        save_json(out / f"worker_{method}.json", result)
        print(json.dumps({"event": "complete", "method": method}), flush=True)


def aggregate_timings(rows):
    """Repeated requests never masquerade as independent resume observations."""
    import numpy as np
    import pandas as pd

    frame = pd.DataFrame(rows)
    frame["extraction_mode"] = frame.get("extraction_mode", "uncached_extraction")
    if frame.duplicated(["case", "repeat", "extraction_mode"]).any():
        raise ValueError("Duplicate case/repeat timings")
    if (frame[["wall_seconds", "cpu_seconds"]] < 0).any().any():
        raise ValueError("Negative timing")
    medians = frame.groupby(["case", "length_stratum", "extraction_mode"], as_index=False)[["wall_seconds", "cpu_seconds"]].median()
    summary = []
    for mode, subset in medians.groupby("extraction_mode"):
        for label in ("all", *STRATA):
            part = subset if label == "all" else subset[subset.length_stratum.eq(label)]
            if not len(part):
                continue
            summary.append({"extraction_mode": mode, "length_stratum": label, "people": len(part),
                            "wall_p50_seconds": float(np.quantile(part.wall_seconds, .5)),
                            "wall_p95_seconds": float(np.quantile(part.wall_seconds, .95)),
                            "cpu_p50_seconds": float(np.quantile(part.cpu_seconds, .5)),
                            "cpu_p95_seconds": float(np.quantile(part.cpu_seconds, .95))})
    return summary, medians


def execute(args):
    import pandas as pd
    from group_policy_eval import person_bootstrap

    protocol = json.loads((args.out / "protocol.json").read_text())
    verify_protocol(args.out, protocol)
    if (args.out / "report.json").exists():
        raise FileExistsError("Completed benchmark must not be rerun or overwritten")
    environment = {**os.environ, **THREAD_ENV}
    startup = {}
    for method in protocol["method_order"]:
        output = args.out / f"worker_{method}.json"
        if output.exists():
            result = json.loads(output.read_text())
            if result["protocol_sha256"] != digest(args.out / "protocol.json"):
                raise ValueError("Completed worker belongs to a different protocol")
            startup[method] = result["parent_start_to_resources_ready_seconds"]
            continue
        with (args.out / f"worker_{method}.stderr.log").open("x") as error_log, \
                (args.out / f"worker_{method}.events.jsonl").open("x") as event_log:
            launched = time.perf_counter()
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--out", str(args.out.resolve()),
                                        "--worker", method], cwd=ROOT, env=environment,
                                       stdout=subprocess.PIPE, stderr=error_log, text=True)
            for line in process.stdout:
                event_log.write(line)
                event_log.flush()
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("event") == "resources_ready":
                    startup[method] = time.perf_counter() - launched
                print(json.dumps(event), flush=True)
            if process.wait() != 0:
                raise RuntimeError(f"Benchmark worker failed: {method}; inspect its private stderr log")
        result = json.loads(output.read_text())
        result["parent_start_to_resources_ready_seconds"] = startup[method]
        save_json(output, result)
    summaries, memory, medians = [], [], {}
    for method in METHODS:
        result = json.loads((args.out / f"worker_{method}.json").read_text())
        summary, medians[method] = aggregate_timings(result["warm_timings"])
        summaries.extend({"method": method, **row} for row in summary)
        specialized = (Path(protocol["model_path"]).stat().st_size if method == "action_regression" else
                       (args.out / "reference_texts.pkl").stat().st_size +
                       (args.out / "search_resources.pkl").stat().st_size + result["serialized_index_bytes"])
        memory.append({"method": method, "cold_process_to_resources_ready_seconds": startup[method],
                       "model_and_detector_load_seconds": result["model_and_detector_load_seconds"],
                       "reference_load_and_index_build_seconds": result["reference_load_and_index_build_seconds"],
                       "cold_first_request_seconds": result["cold_first_request"]["wall_seconds"],
                       "cold_first_source_words": result["cold_first_request"]["source_words"],
                       "peak_worker_rss_bytes": result["peak_rss_bytes_through_inference"],
                       "method_specific_serialized_artifact_bytes": specialized,
                       "live_reference_pool_loaded": result["live_reference_pool_loaded"],
                       "output_parity_checks_passed": result["output_parity_checks_passed"]})
    paired = medians[METHODS[0]].merge(medians[METHODS[1]], on=["case", "length_stratum", "extraction_mode"], validate="one_to_one",
                                        suffixes=("_v5", "_search"))
    delta = {}
    for mode, part in paired.groupby("extraction_mode"):
        difference = part.wall_seconds_v5 - part.wall_seconds_search
        delta[mode] = person_bootstrap(difference.to_numpy(), part.case.to_numpy())
    sample = pd.read_csv(args.out / "sample_manifest.csv")
    lengths = {label: {"words_min": int(part.words.min()), "words_max": int(part.words.max()), "people": len(part)}
               for label, part in sample.groupby("length_stratum")}
    report = {"purpose": protocol["purpose"], "model_sha256": protocol["model_sha256"],
              "protocol_sha256": digest(args.out / "protocol.json"), "people": protocol["people"],
              "queries": protocol["queries"], "repeats_per_person": protocol["repeats"], "sample_lengths": lengths,
              "timed_request": protocol["timed_request"], "summary": summaries, "resources": memory,
              "paired_mean_latency_difference_v5_minus_search_seconds_95ci": delta,
              "hardware": {"platform": platform.platform(), "machine": platform.machine(),
                           "processor_reported": platform.processor(), "logical_cpus_reported": os.cpu_count(),
                           "python": platform.python_version()},
              "thread_env": THREAD_ENV, "method_order": protocol["method_order"],
              "extraction_modes": protocol["extraction_modes"], "extraction_cache_note": protocol["extraction_cache_note"],
              "rss_note": protocol["memory"], "artifact_size_note": protocol["artifact_sizes"],
              "cold_note": protocol["cold_cache_note"],
              "notes": ["Per-resume medians across repeats form the p50/p95 distribution, not all request repeats.",
                        "Sequential isolated worker order is seeded once; thermal, OS cache and unrelated machine load remain possible confounders.",
                        "This is a local deployment measurement, not a performance guarantee, energy benchmark or model quality score.",
                        "Peak RSS is full worker lifetime through inference, including Python, libraries, NER and loaded resources; it is not model-only RAM.",
                        "Both inference paths include local parsing. Search corpus preparation/index loading is separated from warm latency.",
                        "The live-search control has different stopping/tools and needs a reference pool; timing is a deployment strategy comparison."],
              "private_texts_ids_included": False, "model_or_settings_changed": False}
    verify_protocol(args.out, protocol)
    save_json(args.out / "report.json", report)
    pd.DataFrame(summaries).to_csv(args.out / "summary.csv", index=False)
    pd.DataFrame(memory).to_csv(args.out / "resources.csv", index=False)
    print(json.dumps({"complete_people": report["people"], "summary": summaries, "resources": memory}, indent=2))
    if args.publish_dir:
        args.publish_dir.mkdir(parents=True, exist_ok=True)
        for source, target in (("report.json", "performance_report_20261111.json"),
                               ("summary.csv", "performance_summary_20261111.csv"),
                               ("resources.csv", "performance_resources_20261111.csv")):
            destination = args.publish_dir / target
            if destination.exists():
                raise FileExistsError("Public benchmark aggregate already exists")
            destination.write_bytes((args.out / source).read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts/evaluations/action_v5_test_20261110")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "artifacts/action_v5_data_20261110")
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts/models/resume_action_v5/model.pkl")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/adaptive_resume.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-stratum", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--execute-existing", action="store_true")
    parser.add_argument("--publish-dir", type=Path)
    parser.add_argument("--worker", choices=METHODS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.out = args.out.resolve()
    for name in ("run", "data_dir", "model", "config"):
        setattr(args, name, getattr(args, name).resolve())
    if not args.out.is_relative_to((ROOT / "artifacts").resolve()):
        raise ValueError("Private benchmark inputs/timings must stay under ignored artifacts/")
    if args.worker:
        worker(args.out, args.worker)
        return
    if args.prepare_only and args.execute_existing:
        raise ValueError("Select preparation or execution, not both")
    if not args.execute_existing:
        prepare(args)
    if not args.prepare_only:
        execute(args)


if __name__ == "__main__":
    main()
