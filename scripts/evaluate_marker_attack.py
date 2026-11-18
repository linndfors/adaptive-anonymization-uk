"""Exploratory attack preprocessing: discard known anonymization labels.

Uses already frozen releases only, never retrains/selects a model. Real content
such as [SQL] and approximate-duration payloads is retained. Release utility
is unchanged: only the attacker's query preprocessing changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from action_data import load_action_data, reference_views
from group_policy_eval import person_bootstrap
from threadpoolctl import threadpool_limits

from anonrisk_uk.attack import BM25Index, TfidfIndex, ranking_result
from anonrisk_uk.experiment import Population
from anonrisk_uk.resume_adaptive import GENERAL, MASKS
from anonrisk_uk.resume_parser import FIELD_TYPES, FIELDS
from anonrisk_uk.workua import GROUPS

ROOT = Path(__file__).resolve().parents[1]
LABELS = ({value[1:-1] for value in [*GENERAL.values(), *MASKS.values()]} | set(FIELDS)
          | set(FIELD_TYPES.values()) | {group.upper() for group in GROUPS}
          | {"OTHER_QUASI", "QUASI_IDENT", "OCCUPATION_EDU", "DATE_TIME", "ORGANIZATION"})


def normalize_query(text):
    def replace(match):
        label = match[1]
        if label.startswith("ПЕРІОД РОБОТИ:"):
            return label.split(":", 1)[1].strip()
        return " " if label in LABELS else match[0]
    return re.sub(r"\[([^\]\n]+)\]", replace, text)


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    primary = json.loads((args.run / "report.json").read_text())
    data = load_action_data(args.data_dir)
    if hashlib.sha256((args.data_dir / "manifest.json").read_bytes()).hexdigest() != primary["data_manifest_sha256"]:
        raise ValueError("Frozen cohort differs from the primary evaluation")
    if hashlib.sha256(args.model.read_bytes()).hexdigest() != primary["model_sha256"]:
        raise ValueError("The selected frozen model differs")
    with args.model.open("rb") as handle:
        estimator = pickle.load(handle)
    positions = reference_views(data, primary["split"], count=1, size=primary["pool_size"], seed=primary["seed"])[0]
    expected_ids = pd.read_csv(args.run / "pool_ids.csv", dtype=str).id.tolist()
    if [str(data["ids"][k]) for k in positions] != expected_ids:
        raise ValueError("Attacker background differs from the primary frozen view")
    texts = [estimator.codec.render(data["extracted"][k], {}) for k in positions]
    indexes = {"bm25": BM25Index(max_features=20000).fit(texts), "char": TfidfIndex(max_features=60000).fit(texts)}
    population = Population(expected_ids, [data["extracted"][k] for k in positions],
                            np.asarray(data["people"])[positions], estimator.codec, indexes["bm25"], texts)
    local = {rid: k for k, rid in enumerate(expected_ids)}
    records = [json.loads(line) for line in (args.run / "releases.jsonl").read_text().splitlines()]
    if len(records) != primary["queries"]:
        raise ValueError("Partial release records cannot be scored")
    queries = [local[record["id"]] for record in records]
    if any(record["original"] != texts[k] for record, k in zip(records, queries)):
        raise ValueError("Original query serialization differs")
    targets, excludes = population.targets_and_excludes(queries, "S2")
    people = np.array([record["person"] for record in records])
    frame = pd.DataFrame({"person": people, "id": [record["id"] for record in records]})
    summary = []
    utility = {r["method"]: r["lexical_retention"] for r in primary["summary"]}
    for method in records[0]["outputs"]:
        original = [record["outputs"][method] for record in records]
        values = original if method == "L0" else [normalize_query(text) for text in original]
        row = {"method": method, "lexical_retention": utility[method],
               "queries_with_markers_removed": sum(a != b for a, b in zip(original, values))}
        for attacker, index in indexes.items():
            score = ranking_result(index, values, targets, excludes, batch_size=32)[0]
            row[f"{attacker}_recall_at_1"] = float(score.recalls[1].mean())
            row[f"{attacker}_mrr"] = float(score.reciprocal_ranks.mean())
            frame[f"{method}_{attacker}_r1"] = score.recalls[1]
        summary.append(row)
    paired = {f"vs_{control}_{attacker}_r1": person_bootstrap(
        frame[f"{primary['selected']}_{attacker}_r1"] - frame[f"{control}_{attacker}_r1"], people)
        for control in ("historical_L2", "search_v3_3", "regression_v4")
        if control in records[0]["outputs"] for attacker in indexes}
    report = {"method": primary["method"], "purpose": "exploratory_frozen_release_robustness",
              "selected": primary["selected"], "queries": len(records), "persons": len(set(people)),
              "pool_size": len(positions), "model_or_settings_changed": False,
              "preprocessing": "remove exact known typed labels; retain approximate-duration payloads and unknown bracketed literals",
              "interpretation": "sensitivity to anonymization-label tokens, not an optimal or universal attacker",
              "primary_report_sha256": hashlib.sha256((args.run / "report.json").read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "summary": summary, "paired_differences_95ci": paired}
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame(summary).to_csv(args.out / "summary.csv", index=False)
    frame.to_csv(args.out / "per_query.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.out.resolve().is_relative_to((ROOT / "artifacts").resolve()):
        raise ValueError("Per-query results must remain under ignored artifacts/")
    with threadpool_limits(limits=1):
        run(args)


if __name__ == "__main__":
    main()
