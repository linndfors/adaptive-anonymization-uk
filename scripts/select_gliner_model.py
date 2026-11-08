#!/usr/bin/env python3
"""Select a fine-tuned GLiNER run and threshold on E3-JSI validation data only."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from anonrisk_uk.e3jsi import (
    GlinerDetector,
    evaluate_detection,
    gold_spans,
    load_jsonl,
    repair_docs,
)


def _model(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use NAME=MODEL_PATH for every --model argument")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("Use NAME=MODEL_PATH for every --model argument")
    return name, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--model", type=_model, action="append", required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.2, 0.3, 0.4, 0.5, 0.6])
    parser.add_argument("--min-overlap-precision", type=float, default=0.8)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device for validation selection: auto, cpu, cuda, or a concrete torch device",
    )
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if any(not 0 < threshold < 1 for threshold in args.thresholds):
        raise ValueError("Every threshold must be between zero and one")
    import torch

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {device!r} was requested but CUDA is unavailable")
    split = json.loads(args.split.read_text(encoding="utf-8"))
    docs, _ = repair_docs(load_jsonl(args.data))
    validation = [doc for doc in docs if split.get(str(doc["source_row_id"])) == "val"]
    if not validation:
        raise ValueError("The split contains no validation documents")
    gold = {str(doc["source_row_id"]): gold_spans(doc) for doc in validation}

    rows = []
    for name, model_path in args.model:
        detector = GlinerDetector(
            model_path,
            threshold=args.thresholds[0],
            device=device,
            local_files_only=args.local_files_only,
        )
        for threshold in args.thresholds:
            detector.threshold = threshold
            predictions = {str(doc["source_row_id"]): detector.detect(doc["text_uk"]) for doc in validation}
            rows.append(
                {
                    "candidate": name,
                    "model": model_path,
                    "threshold": threshold,
                    **evaluate_detection(gold, predictions)["summary"],
                }
            )
        del detector
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    table = pd.DataFrame(rows)
    feasible = table[table.overlap_P >= args.min_overlap_precision]
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "gliner_validation_search.csv", index=False)
    if feasible.empty:
        best = table.sort_values(["overlap_P", "entity_leak_rate"], ascending=[False, True]).iloc[0]
        failure = {
            "status": "rejected",
            "reason": "minimum_overlap_precision_not_met",
            "minimum_overlap_precision": args.min_overlap_precision,
            "best_overlap_precision": float(best.overlap_P),
            "selection_split": "validation",
            "test_data_used": False,
        }
        (args.out / "selection_failure.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise RuntimeError(
            "No GLiNER candidate satisfies the validation precision constraint: "
            f"required={args.min_overlap_precision:.3f}, best={float(best.overlap_P):.3f}. "
            "The detector was rejected before test evaluation."
        )
    selected = feasible.sort_values(
        ["entity_leak_rate", "char_recall", "overlap_P", "threshold", "candidate"],
        ascending=[True, False, False, False, True],
    ).iloc[0]
    result = {
        "candidate": selected.candidate,
        "gliner_model": selected.model,
        "gliner_threshold": float(selected.threshold),
        "selection_split": "validation",
        "selection_rule": "minimum_entity_leak_then_maximum_char_recall",
        "minimum_overlap_precision": args.min_overlap_precision,
        "precision_constraint_met": bool(len(feasible)),
        "validation_documents": len(validation),
        "inference_device": device,
        "test_data_used": False,
    }
    (args.out / "selected_gliner.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
