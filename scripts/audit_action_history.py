"""Recheck frozen person splits against persisted pre-reservation supervision.

Includes legacy position-indexed unit labels, which the original sampler's
document-label counter did not cover. Writes aggregates only and never changes
the immutable cohort, weights or experiment outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from action_data import load_action_data

ROOT = Path(__file__).resolve().parents[1]


def audit(data_dir, root=ROOT):
    data_dir, root = Path(data_dir), Path(root)
    manifest_path = data_dir / "manifest.json"
    cutoff = manifest_path.stat().st_mtime_ns
    data = load_action_data(data_dir)
    unit_ids, document_ids, pools = set(), set(), set()
    unit_sources, document_sources, unavailable = [], [], []
    # Only files already present at reservation belong to historical evidence.
    for path in sorted((root / "artifacts/runs").glob("*/pipeline/unit_*labels.csv")):
        if path.stat().st_mtime_ns > cutoff:
            continue
        columns = pd.read_csv(path, nrows=0).columns
        if "pos" not in columns:
            continue
        bundle = path.parent.parent / "workua/bundle.pkl"
        if not bundle.exists():
            unavailable.append(str(path.relative_to(root)))
            continue
        with bundle.open("rb") as handle:
            ids = pickle.load(handle)["ids"]
        positions = pd.read_csv(path, usecols=["pos"]).pos.dropna().astype(int).unique()
        if any(k < 0 or k >= len(ids) for k in positions):
            raise ValueError(f"Historical unit positions outside source bundle: {path}")
        unit_ids.update(str(ids[k]) for k in positions)
        unit_sources.append(str(path.relative_to(root)))
    for path in sorted((root / "artifacts/evaluations").glob("**/labels.csv")):
        if path.stat().st_mtime_ns > cutoff:
            continue
        columns = set(pd.read_csv(path, nrows=0).columns)
        if not {"id", "person", "target", "split"} <= columns:
            continue
        document_ids.update(pd.read_csv(path, usecols=["id"], dtype=str).id)
        document_sources.append(str(path.relative_to(root)))
    # Replay the original recorded pool history, excluding this experiment and
    # any pools subsequently written by validation or final evaluation.
    for name in data["manifest"]["historical_sources"]:
        path = root / name
        if path.name == "bundle.pkl":
            with path.open("rb") as handle:
                pools.update(map(str, pickle.load(handle)["ids"]))
        elif path.name in {"pool_ids.csv", "holdout_pool_ids.csv"}:
            pools.update(pd.read_csv(path, usecols=["id"], dtype=str).id)
    identifiers = np.asarray(data["ids"], dtype=str)
    people = np.asarray(data["people"], dtype=str)
    splits = np.asarray(data["splits"], dtype=str)
    prior_unit_people = set(people[np.isin(identifiers, list(unit_ids))])
    prior_document_people = set(people[np.isin(identifiers, list(document_ids))])
    summary = {}
    for split in ("train", "validation", "test"):
        members = set(people[splits == split])
        a, b = members & prior_unit_people, members & prior_document_people
        summary[split] = {"prior_unit_risk_supervised_people": len(a),
                          "prior_v4_document_risk_supervised_people": len(b),
                          "prior_both_supervisions_people": len(a & b),
                          "prior_either_supervision_people": len(a | b)}
    return {"purpose": "Post-reservation aggregate history audit; frozen splits are unchanged",
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(), "splits": summary,
            "reserved_test_record_overlap_with_historical_pools": len(set(identifiers[splits == "test"]) & pools),
            "shared_background_record_overlap_with_historical_pools": len(set(identifiers[splits == "background"]) & pools),
            "current_train_validation_person_overlap": len(set(people[splits == "train"]) & set(people[splits == "validation"])),
            "unit_sources": unit_sources, "document_sources": document_sources,
            "legacy_labels_without_available_source_bundle": unavailable,
            "scope": "available persisted pre-reservation history; missing original bundles cannot be reconstructed",
            "private_identifiers_included": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Audit output exists; choose a new path")
    report = audit(args.data_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"splits": report["splits"],
                      "reserved_test_record_overlap": report["reserved_test_record_overlap_with_historical_pools"]}, indent=2))


if __name__ == "__main__":
    main()
