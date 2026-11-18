"""Evaluate frozen releases with an E5 attacker absent from action training.

All text is encoded locally with complete token windows. No policy, model,
threshold or encoder is selected using these retrospective outcomes.
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
from download_semantic_attacker import EXPECTED_SHA256, MODEL_ID, REVISION, verify_existing_cache
from evaluate_marker_attack import normalize_query as legacy_marker_cleanup
from group_policy_eval import person_bootstrap
from threadpoolctl import threadpool_limits

from anonrisk_uk.attack import BM25Index, TfidfIndex, ranking_result
from anonrisk_uk.content_provenance import KNOWN_LABELS, remove_mask_labels

ROOT = Path(__file__).resolve().parents[1]
SELECTED = "action_regression"


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def window_inputs(tokenizer, text, prefix, max_tokens=512):
    """Repeat the retrieval prefix without dropping any payload token."""
    prefix_ids = tokenizer(prefix, add_special_tokens=False, truncation=False)["input_ids"]
    tokens = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    capacity = max_tokens - tokenizer.num_special_tokens_to_add(pair=False) - len(prefix_ids)
    if capacity < 1:
        raise ValueError("Prefix and special tokens exceed the encoder capacity")
    windows, weights = [], []
    for start in range(0, len(tokens), capacity):
        payload = tokens[start:start + capacity]
        windows.append(tokenizer.prepare_for_model(prefix_ids + payload, return_attention_mask=True,
                                                   return_token_type_ids=False))
        weights.append(len(payload))
    return windows, weights


def unit_vector(vector):
    length = float(np.linalg.norm(vector))
    return vector / length if length else np.zeros_like(vector)


class E5Encoder:
    def __init__(self, model_dir, *, device="cpu", batch_size=16):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch, self.device, self.batch_size = torch, device, batch_size
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS requested but unavailable; do not silently switch devices")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_dir, local_files_only=True).float().eval().to(device)
        self.dim = self.model.config.hidden_size
        self.cache, self.token_counts, self.window_counts = {}, {}, {}

    def encode(self, texts, prefix):
        pending = list(dict.fromkeys(t for t in texts if (prefix, t) not in self.cache))
        encoded, owners, weights, keys = [], [], [], []
        for text in pending:
            key = (prefix, text)
            windows, lengths = window_inputs(self.tokenizer, text, prefix)
            index = len(keys)
            keys.append(key)
            encoded.extend(windows)
            weights.extend(lengths)
            owners.extend([index] * len(windows))
            self.token_counts[key], self.window_counts[key] = sum(lengths), len(windows)
        totals = np.zeros((len(keys), self.dim), dtype=np.float64)
        # Stable length ordering reduces padding without changing the window set.
        order = sorted(range(len(encoded)), key=lambda i: (-len(encoded[i]["input_ids"]), i))
        for start in range(0, len(order), self.batch_size):
            indices = order[start:start + self.batch_size]
            batch = self.tokenizer.pad([encoded[i] for i in indices], padding=True, return_tensors="pt")
            batch = {name: tensor.to(self.device) for name, tensor in batch.items()}
            with self.torch.inference_mode():
                hidden = self.model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).bool()
                pooled = hidden.masked_fill(~mask, 0.).sum(dim=1) / mask.sum(dim=1)
                vectors = self.torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy()
            for i, vector in zip(indices, vectors):
                totals[owners[i]] += vector * weights[i]
            if start % (self.batch_size * 100) == 0:
                print(f"E5 windows {min(start + len(indices), len(order))}/{len(order)}", flush=True)
        for key, total in zip(keys, totals):
            self.cache[key] = unit_vector(total).astype(np.float32)
        return np.stack([self.cache[(prefix, text)] for text in texts])


class EmbeddedIndex:
    def __init__(self, passages, query_vectors):
        self.passages, self.query_vectors = passages, query_vectors
        self.n_docs = len(passages)

    def scores(self, texts):
        return np.stack([self.query_vectors[text] for text in texts]) @ self.passages.T


def same_person_targets(pool_ids, pool_people, records):
    local = {str(rid): i for i, rid in enumerate(pool_ids)}
    targets, excludes = [], []
    people = np.asarray(pool_people, dtype=str)
    for row in records:
        own = local[str(row["id"])]
        if people[own] != str(row["person"]):
            raise ValueError("Query person differs from the frozen linkage manifest")
        target = np.flatnonzero(people == str(row["person"]))
        target = target[target != own]
        if not len(target):
            raise ValueError("Every query needs another same-person target")
        targets.append(target.tolist())
        excludes.append([own])
    return targets, excludes


def merge_releases(primary, additional):
    rows = [{**row, "outputs": dict(row["outputs"])} for row in primary]
    if additional is None:
        return rows
    if len(rows) != len(additional):
        raise ValueError("Fixed controls have different query counts")
    for row, extra in zip(rows, additional):
        if any(str(row[key]) != str(extra[key]) for key in ("id", "person", "original")):
            raise ValueError("Fixed controls must use the identical ordered queries and sources")
        for method, value in extra["outputs"].items():
            if method in row["outputs"] and row["outputs"][method] != value:
                raise ValueError("Shared frozen release differs")
            row["outputs"][method] = value
    return rows


def load_embedding_cache(directory, *, parent, snapshot, pool_ids, texts):
    """Bind an earlier cache to frozen model, encoder, corpus, pool and lookup."""
    directory = Path(directory)
    previous = json.loads((directory / "report.json").read_text())
    if (previous["model_sha256"] != parent["model_sha256"]
            or previous["data_manifest_sha256"] != parent["data_manifest_sha256"]
            or previous["attacker"] != MODEL_ID or previous["revision"] != REVISION
            or previous["encoder_snapshot"]["files_sha256"] != snapshot["files_sha256"]
            or previous["max_sequence_tokens"] != 512):
        raise ValueError("Embedding cache does not match frozen model, data or encoder")
    if pd.read_csv(directory / "pool_ids.csv", dtype=str).id.tolist() != pool_ids:
        raise ValueError("Embedding cache uses a different ordered background")
    passages = np.load(directory / "passage_vectors.npy", allow_pickle=False)
    query_texts = json.loads((directory / "query_vector_texts.json").read_text())
    vectors = np.load(directory / "query_vectors.npy", allow_pickle=False)
    if (passages.shape != (len(pool_ids), 384) or vectors.shape != (len(query_texts), 384)
            or len(set(query_texts)) != len(query_texts)
            or not np.isfinite(passages).all() or not np.isfinite(vectors).all()):
        raise ValueError("Embedding cache shapes, unique lookup or numeric values are invalid")
    binding = {"origin_report_sha256": digest(directory / "report.json"),
               "origin_device": previous["device"], "origin_batch_size": previous["batch_size"],
               "origin_encoding": previous["encoding"],
               "source_texts_sha256": hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest(),
               "pool_ids_sha256": hashlib.sha256("\n".join(pool_ids).encode()).hexdigest(),
               "cache_files_sha256": {str(directory / name): digest(directory / name) for name in (
                   "report.json", "pool_ids.csv", "passage_vectors.npy", "query_vectors.npy",
                   "query_vector_texts.json", "per_query.csv")},
               "scope": "Same frozen encoder/model/cohort and reconstructed ordered passages; original raw per-query ranks must reproduce exactly"}
    return passages, dict(zip(query_texts, vectors)), previous, binding


def run(args):
    if args.out.exists():
        raise FileExistsError("Semantic attack output already exists")
    if not args.out.resolve().is_relative_to((ROOT / "artifacts").resolve()):
        raise ValueError("Private per-query/embedding outputs must remain under artifacts/")
    parent = json.loads((args.run / "report.json").read_text())
    data = load_action_data(args.data_dir)
    if digest(args.model) != parent["model_sha256"] or digest(args.data_dir / "manifest.json") != parent["data_manifest_sha256"]:
        raise ValueError("Frozen model or cohort changed")
    snapshot = json.loads((args.encoder_dir / "snapshot_manifest.json").read_text())
    if snapshot["model_id"] != MODEL_ID or snapshot["revision"] != REVISION:
        raise ValueError("Wrong semantic attacker snapshot; do not substitute a model")
    verify_existing_cache(args.encoder_dir)
    if snapshot["files_sha256"] != EXPECTED_SHA256:
        raise ValueError("Encoder manifest differs from checked-in pinned file hashes")
    for name, expected in EXPECTED_SHA256.items():
        if digest(args.encoder_dir / name) != expected:
            raise ValueError(f"Semantic attacker file hash differs: {name}")
    paths = [args.run / "report.json", args.run / "pool_ids.csv", args.run / "releases.jsonl",
             args.data_dir / "manifest.json", args.data_dir / "cohort.pkl", args.model, Path(__file__),
             ROOT / "scripts/download_semantic_attacker.py", ROOT / "scripts/evaluate_marker_attack.py",
             ROOT / "scripts/group_policy_eval.py", ROOT / "src/anonrisk_uk/attack.py",
             ROOT / "src/anonrisk_uk/content_provenance.py"]
    original = [json.loads(line) for line in (args.run / "releases.jsonl").read_text().splitlines()]
    extra = None
    if args.fixed_run:
        fixed_protocol = json.loads((args.fixed_run / "report.json").read_text())
        if fixed_protocol["model_sha256"] != parent["model_sha256"]:
            raise ValueError("Fixed-control model differs")
        if pd.read_csv(args.fixed_run / "pool_ids.csv", dtype=str).id.tolist() != pd.read_csv(args.run / "pool_ids.csv", dtype=str).id.tolist():
            raise ValueError("Fixed controls used a different background")
        extra = [json.loads(line) for line in (args.fixed_run / "releases.jsonl").read_text().splitlines()]
        paths += [args.fixed_run / "report.json", args.fixed_run / "releases.jsonl", args.fixed_run / "pool_ids.csv"]
    records = merge_releases(original, extra)
    if len(records) != parent["queries"] or len({r["id"] for r in records}) != len(records):
        raise ValueError("Frozen releases are incomplete or duplicated")
    pool_ids = pd.read_csv(args.run / "pool_ids.csv", dtype=str).id.tolist()
    expected_positions = reference_views(data, parent["split"], count=1, size=parent["pool_size"], seed=parent["seed"])[0]
    if pool_ids != [str(data["ids"][i]) for i in expected_positions] or len(set(pool_ids)) != len(pool_ids):
        raise ValueError("Background differs from the reconstructed original reference view")
    global_positions = {str(rid): i for i, rid in enumerate(data["ids"])}
    positions = [global_positions[rid] for rid in pool_ids]
    with args.model.open("rb") as handle:
        estimator = pickle.load(handle)
    texts = [estimator.codec.render(data["extracted"][i], {}) for i in positions]
    local_text = dict(zip(pool_ids, texts))
    if any(row["original"] != local_text[str(row["id"])] for row in records):
        raise ValueError("Query originals differ from background serialization")
    targets, excludes = same_person_targets(pool_ids, np.asarray(data["people"])[positions], records)
    methods = list(records[0]["outputs"])
    if any(set(row["outputs"]) != set(methods) for row in records):
        raise ValueError("Every query must contain all methods")
    people = np.array([str(row["person"]) for row in records])
    cache, cache_binding, previous, passages = {}, None, None, None
    if args.reuse_embeddings:
        passages, cache, previous, cache_binding = load_embedding_cache(
            args.reuse_embeddings, parent=parent, snapshot=snapshot, pool_ids=pool_ids, texts=texts)
        paths.extend(Path(path) for path in cache_binding["cache_files_sha256"])
    protocol = {"method": parent["method"], "selected": SELECTED, "purpose": "retrospective_unseen_attacker_canonical_masks",
                "queries": len(records), "persons": len(set(people)), "pool_size": len(pool_ids),
                "attacker": MODEL_ID, "revision": REVISION, "device": args.device,
                "precision": "float32 model and stored vectors; float64 document accumulation",
                "batch_size": args.batch_size, "max_sequence_tokens": 512,
                "encoding": "all non-overlapping payload-token windows; repeat query:/passage: prefix in every window; attention-mask mean pooling then L2 normalization per window; payload-token-weighted window mean then L2 normalization per document",
                "empty_payload": "zero vector, exact random-tie ranking; no invented semantic content",
                "attacker_conditions": ["raw", "known_markers_removed"],
                "marker_rule": "canonical content_provenance.remove_mask_labels; includes YEAR/NUMBER/period and all role-of-class labels; preserve all other characters, bracketed literals and approximate-duration payloads; original passages unchanged",
                "known_mask_labels": sorted(KNOWN_LABELS),
                "lexical_marker_diagnostic": args.lexical_diagnostic,
                "embedding_cache_binding": cache_binding,
                "ranking": "cosine; other same-person CV is target; self excluded; exact expected random-tie ranks/recalls",
                "bootstrap_unit": "person", "bootstrap_samples": 2000,
                "model_sha256": parent["model_sha256"], "data_manifest_sha256": parent["data_manifest_sha256"],
                "encoder_snapshot": snapshot, "input_source_sha256": {str(path): digest(path) for path in paths},
                "training_exposure": "E5 was absent from action-label training and utility-based model selection; frozen public pretrained snapshot",
                "model_or_settings_changed": False, "paid_api_calls": 0,
                "interpretation": "conditional corpus linkage under a new fixed attacker; not identification probability; retrospective without test-dependent tuning",
                "multiple_comparisons": "exploratory intervals; no multiplicity correction"}
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    encoder = E5Encoder(args.encoder_dir, device=args.device, batch_size=args.batch_size)
    encoder.cache.update({("query: ", text): vector for text, vector in cache.items()})
    passage_vectors = passages if passages is not None else encoder.encode(texts, "passage: ")
    np.save(args.out / "passage_vectors.npy", passage_vectors)
    pd.DataFrame({"id": pool_ids}).to_csv(args.out / "pool_ids.csv", index=False)
    condition_values = {}
    for condition in protocol["attacker_conditions"]:
        condition_values[condition] = {method: [row["outputs"][method] if condition == "raw" else
                                             remove_mask_labels(row["outputs"][method]) for row in records]
                                       for method in methods}
    all_queries = list(dict.fromkeys(text for outputs in condition_values.values() for values in outputs.values() for text in values))
    cache_hits = sum(("query: ", text) in encoder.cache for text in all_queries)
    vectors = encoder.encode(all_queries, "query: ")
    np.save(args.out / "query_vectors.npy", vectors)
    # Stored lookup texts contain private data, never publish this file.
    (args.out / "query_vector_texts.json").write_text(json.dumps(all_queries, ensure_ascii=False))
    index = EmbeddedIndex(passage_vectors, dict(zip(all_queries, vectors)))
    lexical = ({"bm25": BM25Index(max_features=20000).fit(texts),
                "char": TfidfIndex(max_features=60000).fit(texts)} if args.lexical_diagnostic else {})
    frames, summary, paired = [], [], {}
    raw_cache_reproduced = None
    original_raw = (pd.read_csv(args.reuse_embeddings / "per_query.csv", dtype={"id": str, "person": str},
                                float_precision="round_trip").query("condition == 'raw'")
                    if args.reuse_embeddings else None)
    for condition, outputs in condition_values.items():
        frame = pd.DataFrame({"id": [str(row["id"]) for row in records], "person": people})
        frame["condition"] = condition
        for method, values in outputs.items():
            attack = ranking_result(index, values, targets, excludes, batch_size=32)[0]
            for metric, scores in (("r1", attack.recalls[1]), ("r5", attack.recalls[5]),
                                   ("r10", attack.recalls[10]), ("rr", attack.reciprocal_ranks)):
                frame[f"{method}_{metric}"] = scores
            row = {"condition": condition, "method": method, "queries": len(records),
                   "semantic_recall_at_1": float(attack.recalls[1].mean()),
                   "semantic_recall_at_5": float(attack.recalls[5].mean()),
                   "semantic_recall_at_10": float(attack.recalls[10].mean()),
                   "semantic_mrr": float(attack.reciprocal_ranks.mean()),
                   "r1_person_bootstrap_95ci": person_bootstrap(attack.recalls[1], people),
                   "queries_changed_by_marker_removal": sum(a != row["outputs"][method] for a, row in zip(values, records))}
            for attacker_name, lexical_index in lexical.items():
                result = ranking_result(lexical_index, values, targets, excludes, batch_size=32)[0]
                frame[f"{method}_{attacker_name}_r1"] = result.recalls[1]
                frame[f"{method}_{attacker_name}_rr"] = result.reciprocal_ranks
                row[f"{attacker_name}_recall_at_1"] = float(result.recalls[1].mean())
                row[f"{attacker_name}_mrr"] = float(result.reciprocal_ranks.mean())
            summary.append(row)
            print(f"scored {condition}: {method}", flush=True)
        for control in methods:
            if control in {"L0", SELECTED}:
                continue
            for metric in ("r1", "rr"):
                paired[f"{condition}_vs_{control}_{metric}"] = person_bootstrap(
                    frame[f"{SELECTED}_{metric}"] - frame[f"{control}_{metric}"], people)
            for attacker_name in lexical:
                paired[f"{condition}_vs_{control}_{attacker_name}_r1"] = person_bootstrap(
                    frame[f"{SELECTED}_{attacker_name}_r1"] - frame[f"{control}_{attacker_name}_r1"], people)
        if condition == "raw" and original_raw is not None:
            old = original_raw.set_index(["id", "person"])
            now = frame.set_index(["id", "person"])
            if set(old.index) != set(now.index):
                raise ValueError("Cache raw query/person rows differ")
            raw_cache_reproduced = all(np.array_equal(now[column].to_numpy(), old.loc[now.index, column].to_numpy())
                                      for column in now if column in old and column != "condition")
            if not raw_cache_reproduced:
                raise RuntimeError("Cached embeddings failed exact raw-result reproduction")
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_csv(args.out / "per_query.csv", index=False)
    for path, expected in protocol["input_source_sha256"].items():
        if digest(path) != expected:
            raise RuntimeError(f"Frozen input/source changed during evaluation: {path}")
    for name, expected in snapshot["files_sha256"].items():
        if digest(args.encoder_dir / name) != expected:
            raise RuntimeError(f"Encoder file changed during evaluation: {name}")
    if previous:
        window_counts = previous["passage_window_counts"]
    else:
        counts = np.array([encoder.window_counts[("passage: ", text)] for text in texts])
        window_counts = {"min": int(counts.min()), "max": int(counts.max()), "mean": float(counts.mean()),
                         "multi_window_documents": int((counts > 1).sum())}
    labels = []
    for label in sorted(KNOWN_LABELS):
        token = f"[{label}]"
        occurrences = [(method, i) for method in methods for i, record in enumerate(records)
                       if token in record["outputs"][method]]
        if occurrences:
            labels.append({"label": label, "method_query_rows": len(occurrences),
                           "unique_queries": len({i for _, i in occurrences})})
    residual_masks = sum(bool(re.search(r"\[([^\]\n]+)\]", remove_mask_labels(text)))
                         for values in condition_values["raw"].values() for text in values)
    report = {**protocol, "summary": summary, "paired_differences_95ci": paired,
              "passage_window_counts": window_counts,
              "raw_results_exactly_reproduced_from_cache": raw_cache_reproduced,
              "reused_query_encodings": cache_hits, "new_query_encodings": len(all_queries) - cache_hits,
              "mask_label_affected_rows": labels,
              "canonical_vs_legacy_marker_rows_changed": {
                  method: sum(remove_mask_labels(record["outputs"][method]) != legacy_marker_cleanup(record["outputs"][method])
                              for record in records) for method in methods},
              "rows_retaining_bracketed_literals_or_payloads": residual_masks,
              "unique_query_encodings": len(all_queries), "private_texts_ids_embeddings_included": False}
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame(summary).to_csv(args.out / "summary.csv", index=False)
    print(pd.DataFrame(summary)[["condition", "method", "semantic_recall_at_1", "semantic_mrr"]].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--fixed-run", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--encoder-dir", type=Path, default=Path("artifacts/models/semantic_attacker_e5"))
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--reuse-embeddings", type=Path, help="Verified prior E5 run; exact raw-result parity is mandatory")
    parser.add_argument("--lexical-diagnostic", action="store_true", help="Also rescore raw/canonical marker-cleaned BM25 and char attacks")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive")
    import torch
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        run(args)


if __name__ == "__main__":
    main()
