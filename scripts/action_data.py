"""Frozen linked-person cohorts for autonomous action learning.

Only linkage columns are scanned globally. Full nested records are parsed for a
bounded candidate sample. All record-level outputs belong in ignored artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from anonrisk_uk.workua import ResumeCodec, _as_list, _identifier, extract, load_workua, norm


def historical_used_ids(root):
    """Exclude complete historical backgrounds, including prior held-out tests."""
    root = Path(root)
    used, sources = set(), []
    for path in sorted((root / "artifacts/runs").glob("*/workua/bundle.pkl")):
        with path.open("rb") as handle:
            used.update(map(str, pickle.load(handle)["ids"]))
        sources.append(str(path.relative_to(root)))
    paths = set()
    for directory in (root / "artifacts/evaluations", root / "artifacts/models"):
        paths.update(directory.glob("**/pool_ids.csv"))
        paths.update(directory.glob("**/holdout_pool_ids.csv"))
    paths.update((root / "artifacts").glob("action*_data*/pool_ids.csv"))
    for path in sorted(paths):
        used.update(pd.read_csv(path, dtype=str)["id"])
        sources.append(str(path.relative_to(root)))
    return used, sources


def historical_query_ids(root):
    """Actual past queries and reserved test IDs, distinct from background use."""
    root = Path(root)
    queried, supervised, sources = set(), set(), []
    for base in (root / "artifacts", root / "results"):
        for path in sorted(base.glob("**/per_query.csv")):
            columns = pd.read_csv(path, nrows=0).columns
            if "id" in columns:
                queried.update(pd.read_csv(path, usecols=["id"], dtype=str).id)
                sources.append(str(path.relative_to(root)))
        for path in sorted(base.glob("**/releases.jsonl")):
            with path.open() as handle:
                for line in handle:
                    if line.strip():
                        rid = json.loads(line).get("id")
                        if rid is not None:
                            queried.add(str(rid))
            sources.append(str(path.relative_to(root)))
        for path in sorted(base.glob("**/labels.csv")):
            columns = pd.read_csv(path, nrows=0).columns
            if "id" in columns:
                supervised.update(pd.read_csv(path, usecols=["id"], dtype=str).id)
                sources.append(str(path.relative_to(root)))
    for path in sorted((root / "artifacts/runs").glob("*/workua/split.csv")):
        frame = pd.read_csv(path, dtype=str)
        queried.update(frame.loc[frame.split.eq("test"), "id"])
        sources.append(str(path.relative_to(root)))
    return queried, supervised, sources


def linkage_members(csv_path, excluded_ids=()):
    """Resolve transitive person links without parsing unrelated nested fields."""
    identifiers, edges, seen = [], [], set()
    for chunk in pd.read_csv(csv_path, usecols=["id", "other_resumes"], dtype=str,
                             keep_default_na=False, encoding="utf-8-sig", chunksize=10000):
        for raw_id, raw_links in chunk[["id", "other_resumes"]].itertuples(index=False, name=None):
            rid = _identifier(raw_id)
            if rid is None or rid in seen:
                continue
            seen.add(rid)
            identifiers.append(rid)
            for item in _as_list(raw_links, strict=True):
                oid = _identifier(item.get("resume_id")) if isinstance(item, dict) else None
                if oid and oid != rid:
                    edges.append((rid, oid))
    parent = dict.fromkeys(identifiers)
    for rid in identifiers:
        parent[rid] = rid

    def find(rid):
        while parent[rid] != rid:
            parent[rid] = parent[parent[rid]]
            rid = parent[rid]
        return rid

    for rid, oid in edges:
        if oid in parent:
            a, b = find(rid), find(oid)
            if a != b:
                parent[max(a, b)] = min(a, b)
    members = defaultdict(list)
    for rid in identifiers:
        members[find(rid)].append(rid)
    excluded = set(map(str, excluded_ids))
    blocked = {person for person, ids in members.items() if excluded.intersection(ids)}
    return {person: ids for person, ids in members.items() if person not in blocked}, {
        "source_records": len(identifiers), "source_persons": len(members),
        "excluded_historical_persons": len(blocked),
        "eligible_linked_persons": sum(2 <= len(ids) <= 8 for p, ids in members.items() if p not in blocked),
    }


def _selected_records(csv_path, identifiers, temporary_dir):
    """Reuse the canonical field validator on a streamed subset CSV."""
    with tempfile.TemporaryDirectory(dir=temporary_dir) as temporary:
        subset = Path(temporary) / "selected.csv"
        first = True
        for chunk in pd.read_csv(csv_path, dtype=str, keep_default_na=False,
                                 encoding="utf-8-sig", chunksize=5000):
            selected = chunk.loc[chunk.id.map(_identifier).isin(identifiers)]
            if len(selected):
                selected.to_csv(subset, mode="w" if first else "a", header=first, index=False)
                first = False
        if first:
            raise ValueError("No selected records were found")
        frame = load_workua(subset)
    return {str(record["id"]): extract(record) for record in frame.to_dict("records")}


def prepare_action_data(csv_path, out_dir, *, excluded_ids=None, root=None,
                        train_persons=800, validation_persons=150, test_persons=150,
                        reference_size=6000, seed=20261110, reuse_historical_background=False):
    """Reserve test people before fitting; freeze all splits and backgrounds.

    No attack outcomes are inspected. Exact normalized rendered duplicates across
    supervised people are rejected, and matching background people are removed.
    """
    csv_path, out_dir = Path(csv_path), Path(out_dir)
    requests = {"train": train_persons, "validation": validation_persons, "test": test_persons}
    if any(n < 1 for n in requests.values()) or reference_size < 2:
        raise ValueError("Each split needs at least one person and a positive reference size")
    if out_dir.exists():
        raise FileExistsError(f"Frozen data already exists: {out_dir}")
    sources, old_supervised, fresh_members = [], set(), None
    full_history = set()
    if root is None:
        root = Path(__file__).resolve().parents[1]
    if excluded_ids is None:
        full_history, sources = historical_used_ids(root)
        if reuse_historical_background:
            excluded_ids, old_supervised, query_sources = historical_query_ids(root)
            sources += query_sources
            fresh_members, _ = linkage_members(csv_path, full_history | excluded_ids)
        else:
            excluded_ids = full_history
    members, audit = linkage_members(csv_path, excluded_ids)
    rng = np.random.default_rng(seed)
    eligible = sorted(person for person, ids in members.items() if 2 <= len(ids) <= 8)
    needed = sum(requests.values())
    if len(eligible) < needed and not reuse_historical_background:
        raise ValueError(f"Need {needed} fresh linked people, found {len(eligible)}")
    candidate_people = list(rng.permutation(eligible)[:min(len(eligible), needed * 2)])
    candidates = set(candidate_people)
    background_source = members if fresh_members is None else fresh_members
    background_people = [p for p in sorted(background_source) if p not in candidates]
    background_people = list(rng.permutation(background_people))
    chosen_background, count = [], 0
    for person in background_people:
        chosen_background.append(person)
        count += len(background_source[person])
        if count >= int(reference_size * 1.5):
            break
    # Fresh backgrounds are a subset of members; all complete clusters are retained.
    wanted = {rid for person in candidate_people + chosen_background for rid in members[person]}
    out_dir.mkdir(parents=True)
    records = _selected_records(csv_path, wanted, out_dir)
    codec = ResumeCodec()
    signatures = {}
    for person in candidate_people + chosen_background:
        signatures[person] = {hashlib.sha256(norm(codec.render(records[rid], {})).encode()).hexdigest()
                              for rid in members[person] if rid in records}
    selected, seen_hashes, rejected, assigned = {}, set(), 0, set()
    # Reserve test first. Neither text nor attack scores are exposed to fitting.
    for split in ("test", "validation", "train"):
        selected[split] = []
        split_candidates = [p for p in candidate_people if p not in assigned]
        if split == "test" and fresh_members is not None:
            split_candidates = [p for p in split_candidates if p in fresh_members]
        if split == "validation":
            split_candidates = [p for p in split_candidates if not old_supervised.intersection(members[p])]
        order = iter(split_candidates)
        while len(selected[split]) < requests[split]:
            try:
                person = next(order)
            except StopIteration:
                if reuse_historical_background and split == "train" and len(selected[split]) >= 20:
                    break
                raise ValueError("Insufficient fresh nonduplicate valid people; reduce requested counts") from None
            texts = [codec.render(records[rid], {}) for rid in members[person] if rid in records]
            if len(texts) != len(members[person]) or any(not t.strip() or len(t) > 20000 for t in texts):
                rejected += 1
                continue
            if signatures[person] & seen_hashes:
                rejected += 1
                continue
            selected[split].append(person)
            assigned.add(person)
            seen_hashes.update(signatures[person])
    clean_background = [p for p in chosen_background if not signatures[p] & seen_hashes]
    if sum(len(members[p]) for p in clean_background) < reference_size:
        raise ValueError("Insufficient clean background records")
    final_people = {p: split for split, group in selected.items() for p in group}
    final_people.update({p: "background" for p in clean_background})
    identifiers, extracted, people, splits = [], [], [], []
    for person, split in final_people.items():
        for rid in members[person]:
            identifiers.append(rid)
            extracted.append(records[rid])
            people.append(person)
            splits.append(split)
    queries = {split: [i for i, value in enumerate(splits) if value == split] for split in requests}
    with csv_path.open("rb") as handle:
        source_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    manifest = {"schema": "action_data_v1", "seed": seed, "reference_size": reference_size,
                "source_sha256": source_hash,
                **audit, "historical_sources": sources, "requested_people": requests,
                "persons": {s: len(v) for s, v in selected.items()},
                "queries": {s: len(v) for s, v in queries.items()},
                "background_records": splits.count("background"), "cohort_records": len(identifiers),
                "rejected_candidate_persons": rejected,
                "person_overlap_between_splits": 0, "normalized_text_overlap_between_splits": 0,
                "historical_record_overlap": len(set(identifiers) & set(map(str, excluded_ids))),
                "prior_background_reuse_allowed_for_development": reuse_historical_background,
                "records_previously_seen_as_background": len(set(identifiers) & full_history),
                "prior_supervised_people_reused_for_train": sum(
                    bool(old_supervised.intersection(members[p])) for p in selected["train"]),
                "test_people_never_in_historical_backgrounds": fresh_members is not None
                    or not reuse_historical_background,
                "background_never_in_historical_backgrounds": fresh_members is not None
                    or not reuse_historical_background,
                "sampling": "linked-person first; complete clusters; bounded full-record extraction",
                "test_reserved_before_training": True,
                "reference_rule": "each split's complete query clusters plus shared background-only people",
                "synthetic_in_test": False,
                "privacy": "cohort.pkl and CSV manifests contain private records/IDs; never publish"}
    data = {"ids": identifiers, "extracted": extracted, "people": np.asarray(people),
            "splits": np.asarray(splits), "query_positions": queries, "manifest": manifest}
    with (out_dir / "cohort.pkl").open("xb") as handle:
        pickle.dump(data, handle)
    pd.DataFrame({"id": identifiers}).to_csv(out_dir / "pool_ids.csv", index=False)
    pd.DataFrame({"id": identifiers, "person": people, "split": splits}).to_csv(
        out_dir / "query_manifest.csv", index=False)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_action_data(data_dir, split=None):
    """Replay frozen data; never redraw people when evaluating a model."""
    with (Path(data_dir) / "cohort.pkl").open("rb") as handle:
        data = pickle.load(handle)
    if data["manifest"]["schema"] != "action_data_v1":
        raise ValueError("Unsupported action-data schema")
    if split is not None:
        if split not in data["query_positions"]:
            raise ValueError(f"Unknown supervised split: {split}")
        data["queries"] = list(data["query_positions"][split])
    return data


def reference_views(data, split, *, count=3, size=None, seed=None):
    """Return global positions; self-exclusion remains the attack scorer's job."""
    if split not in data["query_positions"] or count < 1:
        raise ValueError("Invalid split or view count")
    size = data["manifest"]["reference_size"] if size is None else size
    seed = data["manifest"]["seed"] if seed is None else seed
    required = list(data["query_positions"][split])
    if size < len(required):
        raise ValueError("Reference size cannot omit complete query clusters")
    background = np.flatnonzero(data["splits"] == "background")
    if size > len(required) + len(background):
        raise ValueError("Not enough background records for requested reference size")
    offset = {"train": 0, "validation": 10000, "test": 20000}[split]
    return [sorted(required + np.random.default_rng(seed + offset + i).choice(
        background, size=size - len(required), replace=False).tolist()) for i in range(count)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("data/raw/all_resumes.csv"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-persons", type=int, default=800)
    parser.add_argument("--validation-persons", type=int, default=150)
    parser.add_argument("--test-persons", type=int, default=150)
    parser.add_argument("--reference-size", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--reuse-historical-background", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare_action_data(args.csv, args.out, train_persons=args.train_persons,
                     validation_persons=args.validation_persons, test_persons=args.test_persons,
                     reference_size=args.reference_size, seed=args.seed,
                     reuse_historical_background=args.reuse_historical_background), indent=2))


if __name__ == "__main__":
    main()
