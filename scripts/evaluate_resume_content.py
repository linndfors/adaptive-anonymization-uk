"""Post-hoc semantic evaluation of saved local releases. No API calls or tuning.

Scores are embedding similarities, not percentages of retained meaning. Export
a deterministic, method-blinded review packet for separate factual assessment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from group_policy_eval import person_bootstrap
from threadpoolctl import threadpool_limits

from anonrisk_uk.content_evaluation import (
    MODEL_ID,
    MODEL_REVISION,
    SemanticEncoder,
    professional_view,
    semantic_pair,
)

ROOT = Path(__file__).resolve().parents[1]


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    records = [json.loads(line) for line in (args.run / "releases.jsonl").read_text().splitlines()]
    protocol = json.loads((args.run / "protocol.json").read_text())
    if len(records) != protocol["queries"]:
        raise ValueError("Incomplete release records")
    models = sorted({name for row in records for name in row["outputs"]})
    semantic_protocol = {"method": protocol["method"], "purpose": protocol["purpose"],
                         "queries": len(records), "persons": len({r["person"] for r in records}),
                         "encoder": MODEL_ID, "revision": MODEL_REVISION, "device": "cpu",
                         "max_sequence_tokens": 128, "long_text": "all non-overlapping token windows; weighted pooling",
                         "professional_fields": ["title", "exp_position", "skills", "exp_resp", "edu_faculty"],
                         "metric_note": "embedding cosine, not factual entailment or a percent of meaning",
                         "review": {"seed": 20261101, "people": 12, "methods": 3,
                                    "rubric_sha256": hashlib.sha256(
                                        (ROOT / "docs/CONTENT_REVIEW_PROTOCOL_UK.md").read_bytes()).hexdigest()},
                         "contrast_fixture_sha256": hashlib.sha256(
                             (ROOT / "tests/fixtures/content_contrasts_uk.json").read_bytes()).hexdigest(),
                         "policy_selection": "none; frozen outputs, no feedback to anonymization",
                         "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                           for p in [Path(__file__), ROOT / "src/anonrisk_uk/content_evaluation.py"]},
                         "records_sha256": hashlib.sha256((args.run / "releases.jsonl").read_bytes()).hexdigest()}
    # Freeze protocol before scoring. Do not silently substitute another model.
    config = json.loads((args.model_dir / "config.json").read_text())
    metadata = args.model_dir / ".cache/huggingface/download/model.safetensors.metadata"
    if metadata.read_text().splitlines()[0] != MODEL_REVISION:
        raise ValueError("Local semantic weights do not match the frozen revision")
    semantic_protocol["model_files_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                               for p in sorted(args.model_dir.glob("*.json"))}
    semantic_protocol["embedding_dimension"] = config["hidden_size"]
    semantic_protocol["weights_sha256"] = hashlib.sha256((args.model_dir / "model.safetensors").read_bytes()).hexdigest()
    (args.out / "protocol.json").write_text(json.dumps(semantic_protocol, ensure_ascii=False, indent=2))
    encoder = SemanticEncoder(args.model_dir)
    rows = []
    for i, record in enumerate(records):
        full_texts = [record["original"], *[record["outputs"][m] for m in models]]
        views = [professional_view(t) for t in full_texts]
        full = encoder.encode(full_texts)
        professional = encoder.encode(views)
        for j, method in enumerate(models, 1):
            metrics = semantic_pair(professional[0], professional[j])
            rows.append({"id": record["id"], "person": record["person"], "method": method,
                         "professional_cosine": metrics["cosine"],
                         "professional_chunk_recall": metrics["chunk_recall"],
                         "professional_chunk_precision": metrics["chunk_precision"],
                         "full_text_cosine": semantic_pair(full[0], full[j])["cosine"],
                         "source_professional_chunks": len(professional[0]["chunks"]),
                         "output_professional_chunks": len(professional[j]["chunks"])})
        if i % 25 == 0:
            pd.DataFrame(rows).to_csv(args.out / "per_query.csv", index=False)
            print(f"semantic {i + 1}/{len(records)}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.out / "per_query.csv", index=False)
    summary = []
    metrics = ["professional_cosine", "professional_chunk_recall", "professional_chunk_precision", "full_text_cosine"]
    for method, part in frame.groupby("method", sort=True):
        row = {"method": method, "n": len(part), "n_professional_scorable": int(part.professional_cosine.notna().sum()),
               "empty_professional_outputs": int((part.output_professional_chunks == 0).sum())}
        row.update({metric: float(part[metric].mean()) for metric in metrics})
        summary.append(row)
    adaptive = f"adaptive_{protocol['utility_floors'][0]:g}"
    paired = {}
    for control in ["historical_L2", "historical_L3", "historical_v2", "matched_all_quasi"]:
        for metric in metrics:
            table = frame.pivot(index=["id", "person"], columns="method", values=metric)
            diff = (table[adaptive] - table[control]).dropna()
            paired[f"vs_{control}_{metric}"] = person_bootstrap(diff.to_numpy(), diff.index.get_level_values("person"))
    report = {**semantic_protocol, "summary": summary, "paired_differences_95ci": paired,
              "review_status": "packet only; no human ratings yet"}
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    pd.DataFrame(summary).to_csv(args.out / "summary.csv", index=False)
    # One resume per sampled person, not multiple near-duplicates per reviewer.
    rng = np.random.default_rng(20261101)
    people = sorted({r["person"] for r in records})
    chosen = set(rng.permutation(people)[:min(12, len(people))])
    packet, key, seen = [], [], set()
    for record in records:
        if record["person"] not in chosen or record["person"] in seen:
            continue
        seen.add(record["person"])
        for method in rng.permutation([adaptive, "historical_L2", "matched_all_quasi"]):
            code = f"review_{len(packet) + 1:03d}"
            packet.append({"code": code, "source": professional_view(record["original"]),
                           "output": professional_view(record["outputs"][method]),
                           "professional_meaning": None, "factual_distortion": None,
                           "readability": None, "notes": ""})
            key.append({"code": code, "id": record["id"], "person": record["person"], "method": method})
    (args.out / "review_packet.json").write_text(json.dumps(packet, ensure_ascii=False, indent=2))
    pd.DataFrame(key).to_csv(args.out / "review_key.csv", index=False)
    fixture = ROOT / "tests/fixtures/content_contrasts_uk.json"
    contrasts = []
    for case in json.loads(fixture.read_text()):
        vectors = encoder.encode([case[k] for k in ("source", "paraphrase", "contradiction", "unrelated")])
        scores = {key: semantic_pair(vectors[0], vector)["cosine"] for key, vector in
                  zip(("paraphrase", "contradiction", "unrelated"), vectors[1:])}
        contrasts.append({"id": case["id"], **scores,
                          "paraphrase_above_contradiction": scores["paraphrase"] > scores["contradiction"],
                          "paraphrase_above_unrelated": scores["paraphrase"] > scores["unrelated"]})
    (args.out / "contrast_report.json").write_text(json.dumps({
        "kind": "eight synthetic contrast groups, not natural-resume validation",
        "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(), "cases": contrasts,
        "paraphrase_above_contradiction": sum(r["paraphrase_above_contradiction"] for r in contrasts),
        "paraphrase_above_unrelated": sum(r["paraphrase_above_unrelated"] for r in contrasts),
        "n": len(contrasts)}, ensure_ascii=False, indent=2))
    print(pd.DataFrame(summary).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "artifacts/models/semantic_minilm")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        run(args)


if __name__ == "__main__":
    main()
