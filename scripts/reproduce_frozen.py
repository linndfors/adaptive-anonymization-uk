"""Replay frozen evaluation from owner-supplied private inputs, without training.

The output directory must be new. No corpus, individual releases or model file
is downloaded, published or sent to an API by this command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from action_data import load_action_data
from evaluate_action_resume import paired_comparisons, score_outputs
from frozen_runtime import MODEL_SHA256
from threadpoolctl import threadpool_limits

from anonrisk_uk.attack import BM25Index, TfidfIndex
from anonrisk_uk.experiment import Population

ROOT = Path(__file__).resolve().parents[1]
STEPS = ("privacy", "content", "semantic", "benchmark")


def validate_output_path(path):
    if not Path(path).resolve().is_relative_to((ROOT / "artifacts").resolve()):
        raise ValueError("Reproduction writes real texts and IDs; --out must remain under ignored artifacts/")


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_records(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len({str(row["id"]) for row in rows}) != len(rows):
        raise ValueError("Duplicate query IDs in frozen releases")
    return rows


def merged_releases(primary, extra):
    rows = read_records(primary / "releases.jsonl")
    more = {str(row["id"]): row for row in read_records(extra / "releases.jsonl")}
    if set(more) != {str(row["id"]) for row in rows}:
        raise ValueError("Frozen fixed controls use different query IDs")
    for row in rows:
        other = more[str(row["id"])]
        if row["original"] != other["original"] or row["person"] != other["person"]:
            raise ValueError("Frozen fixed controls use different source people/texts")
        for method, value in other["outputs"].items():
            if method in row["outputs"] and row["outputs"][method] != value:
                raise ValueError("Shared controller outputs changed between frozen runs")
            row["outputs"][method] = value
    return rows


def replay_privacy(args, out):
    protocol = json.loads((args.run / "protocol.json").read_text())
    if digest(args.model) != MODEL_SHA256 or protocol["model_sha256"] != MODEL_SHA256:
        raise ValueError("Selected model differs from the recorded frozen v5 model")
    if digest(args.data_dir / "manifest.json") != protocol["data_manifest_sha256"]:
        raise ValueError("Frozen cohort manifest differs")
    rows = merged_releases(args.run, args.fixed_run)
    if len(rows) != protocol["queries"] or len({r["person"] for r in rows}) != protocol["persons"]:
        raise ValueError("Frozen query/person count differs")
    with args.model.open("rb") as handle:
        estimator = pickle.load(handle)
    data = load_action_data(args.data_dir)
    by_id = {str(value): i for i, value in enumerate(data["ids"])}
    pool_ids = list(pd.read_csv(args.run / "pool_ids.csv", dtype=str).id)
    if len(set(pool_ids)) != len(pool_ids) or len(pool_ids) != protocol["pool_size"]:
        raise ValueError("Frozen reference population differs")
    positions = [by_id[value] for value in pool_ids]
    texts = [estimator.codec.render(data["extracted"][k], {}) for k in positions]
    for row in rows:
        k = by_id[str(row["id"])]
        if row["original"] != estimator.codec.render(data["extracted"][k], {}):
            raise ValueError("Saved query source differs from frozen data")
        if str(row["person"]) != str(data["people"][k]):
            raise ValueError("Saved query linkage differs from frozen data")
    indexes = {"bm25": BM25Index(max_features=20000).fit(texts),
               "char": TfidfIndex(max_features=60000).fit(texts)}
    population = Population(pool_ids, [data["extracted"][k] for k in positions], data["people"][positions],
                            estimator.codec, indexes["bm25"], texts)
    local = {value: i for i, value in enumerate(pool_ids)}
    methods = list(rows[0]["outputs"])
    outputs = {name: [row["outputs"][name] for row in rows] for name in methods}
    frame, summary = score_outputs(population, indexes, [local[str(row["id"])] for row in rows],
                                   outputs, estimator.retention_vectorizer)
    # Every shared aggregate must reproduce; fail rather than relabel a changed metric.
    for directory in (args.run, args.fixed_run):
        expected = json.loads((directory / "report.json").read_text())
        table = {row["method"]: row for row in expected["summary"]}
        for row in summary:
            if row["method"] in table:
                for metric in ("bm25_recall_at_1", "char_recall_at_1", "bm25_mrr", "char_mrr", "lexical_retention"):
                    if not np.isclose(row[metric], table[row["method"]][metric], atol=1e-8, rtol=0):
                        raise ValueError(f"Frozen metric did not reproduce: {row['method']} {metric}")
    report = {"method": protocol["method"], "purpose": "frozen_output_replay", "selected": "action_regression",
              "queries": len(rows), "persons": protocol["persons"], "pool_size": len(pool_ids),
              "model_sha256": MODEL_SHA256, "summary": summary,
              "paired_differences_95ci": paired_comparisons(
                  frame, "action_regression", [m for m in methods if m not in {"L0", "action_regression"}], indexes),
              "bootstrap_unit": "person", "bootstrap_samples": 2000,
              "selection": "None: saved releases only, no controller tuning or model fitting",
              "original_metrics_reproduced": True, "source_sha256": digest(Path(__file__)),
              "input_sha256": {str(path): digest(path) for path in [args.run / "releases.jsonl",
                  args.fixed_run / "releases.jsonl", args.data_dir / "manifest.json", args.model]}}
    out.mkdir(parents=True, exist_ok=False)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    frame.to_csv(out / "per_query.csv", index=False)
    (out / "releases.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    shutil.copyfile(args.run / "pool_ids.csv", out / "pool_ids.csv")
    print("Reproduced all frozen privacy/linkage aggregates.", flush=True)


def plan(args):
    steps = args.steps.split(",")
    if len(set(steps)) != len(steps) or any(step not in STEPS for step in steps):
        raise ValueError("Steps must be unique comma-separated choices: " + ",".join(STEPS))
    commands = {}
    if "content" in steps:
        commands["content"] = [sys.executable, "scripts/evaluate_content_provenance.py", "--run", str(args.run),
            "--extra-run", str(args.fixed_run), "--data-dir", str(args.data_dir), "--model", str(args.model),
            "--model-dir", str(args.content_model), "--recover-missing-controls", "--out", str(args.out / "content")]
    if "semantic" in steps:
        commands["semantic"] = [sys.executable, "scripts/evaluate_semantic_attacker.py", "--run", str(args.run),
            "--fixed-run", str(args.fixed_run), "--data-dir", str(args.data_dir), "--model", str(args.model),
            "--encoder-dir", str(args.attack_model), "--device", "cpu", "--out", str(args.out / "semantic")]
    if "benchmark" in steps:
        commands["benchmark"] = [sys.executable, "scripts/benchmark_action_runtime.py", "--run", str(args.run),
            "--data-dir", str(args.data_dir), "--model", str(args.model), "--out", str(args.out / "benchmark")]
    prerequisites = [args.model, args.data_dir / "cohort.pkl", args.data_dir / "manifest.json"]
    prerequisites += [directory / name for directory in (args.run, args.fixed_run)
                      for name in ("protocol.json", "report.json", "releases.jsonl", "pool_ids.csv")]
    if "content" in steps:
        prerequisites.extend([args.content_model / "model.safetensors", args.content_model / "config.json",
                              args.content_model / "tokenizer.json",
                              args.content_model / ".cache/huggingface/download/model.safetensors.metadata",
                              Path("artifacts/models/resume_risk_v4/model.pkl"),
                              Path("artifacts/evaluations/action_v5_validation_selection_v4_20261110/candidate_0_model.pkl"),
                              Path("artifacts/evaluations/action_v5_validation_selection_v4_20261110/candidate_1_model.pkl"),
                              args.run / "decision_traces.json", args.fixed_run / "decision_traces.json",
                              args.fixed_run / "action_type_prior.json"])
    if "semantic" in steps:
        from download_semantic_attacker import FILES
        prerequisites.extend([args.attack_model / name for name in (*FILES, "snapshot_manifest.json")])
    for command in commands.values():
        prerequisites.append(Path(command[1]))
    missing = sorted({str(path) for path in prerequisites if not path.is_file()})
    return steps, commands, prerequisites, missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("artifacts/evaluations/action_v5_test_20261110"))
    parser.add_argument("--fixed-run", type=Path, default=Path("artifacts/evaluations/action_v5_fixed_controls_20261111"))
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/action_v5_data_20261110"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/models/resume_action_v5/model.pkl"))
    parser.add_argument("--content-model", type=Path, default=Path("artifacts/models/semantic_minilm"))
    parser.add_argument("--attack-model", type=Path, default=Path("artifacts/models/semantic_attacker_e5"))
    parser.add_argument("--steps", default="privacy,content,semantic",
                        help="privacy,content,semantic,benchmark; omitted phases are explicit")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="Only show prerequisite status and exact commands")
    args = parser.parse_args()
    validate_output_path(args.out)
    steps, commands, prerequisites, missing = plan(args)
    if args.check:
        print(json.dumps({"steps": steps, "missing": missing, "commands": commands,
                          "private_inputs_downloaded": False, "model_training": False}, indent=2))
        raise SystemExit(bool(missing))
    if missing:
        raise FileNotFoundError("Missing private prerequisites/public model caches:\n" + "\n".join(missing)
                                + "\nSee docs/reproduction.md; no inputs will be replaced or downloaded.")
    if args.out.exists():
        raise FileExistsError("Verification output exists; use a new directory")
    frozen = {str(path): digest(path) for path in prerequisites}
    args.out.mkdir(parents=True)
    protocol = {"steps": steps, "input_sha256": frozen, "commands": commands,
                "model_training": False, "paid_api_used": False,
                "purpose": "Reproduction/diagnostics on already disclosed frozen people, not a new independent test"}
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "LOKY_MAX_CPU_COUNT": "1", "TOKENIZERS_PARALLELISM": "false"}
    for step in steps:
        if step == "privacy":
            with threadpool_limits(limits=1):
                replay_privacy(args, args.out / step)
        else:
            subprocess.run(commands[step], cwd=ROOT, env=environment, check=True)
    for name, expected in frozen.items():
        if digest(name) != expected:
            raise RuntimeError(f"Frozen input changed during verification: {name}")
    (args.out / "completion.json").write_text(json.dumps({"completed_steps": steps, "input_hashes_unchanged": True}, indent=2))
    print(f"Completed frozen verification: {args.out}")


if __name__ == "__main__":
    main()
