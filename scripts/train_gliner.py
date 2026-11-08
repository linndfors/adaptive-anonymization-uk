#!/usr/bin/env python3
"""Fine-tune GLiNER on E3-JSI train data and select a checkpoint by validation loss.

This script never reads the E3-JSI test split. Use a base model that was not
trained on the English E3-JSI source documents; the default model satisfies that
requirement for this project.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_records(path: Path) -> list[dict]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError(f"Expected a non-empty JSON list in {path}")
    for index, record in enumerate(records):
        if not isinstance(record.get("tokenized_text"), list) or not isinstance(record.get("ner"), list):
            raise TypeError(f"Invalid GLiNER record {index} in {path}")
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--base", default="urchade/gliner_multi_pii-v1")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=15.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--fp16", action="store_true", help="Use CUDA FP16 mixed-precision training")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the most recent checkpoint under OUT/checkpoints when one exists",
    )
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch < 1 or args.gradient_accumulation < 1 or args.patience < 1:
        raise ValueError("Epochs, batch size, gradient accumulation, and patience must be positive")
    if args.cpu and args.fp16:
        raise ValueError("--fp16 cannot be combined with --cpu")
    train = _load_records(args.train)
    validation = _load_records(args.val)
    args.out.mkdir(parents=True, exist_ok=True)

    import torch
    from gliner import GLiNER
    from gliner.data_processing.collator import DataCollator
    from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed
    from transformers.trainer_utils import get_last_checkpoint

    if not args.cpu and not torch.cuda.is_available():
        print("WARNING: CUDA is unavailable; training will use the default non-CUDA device.", flush=True)
    if args.fp16 and not torch.cuda.is_available():
        raise RuntimeError("--fp16 requires a CUDA GPU")

    set_seed(args.seed)
    if args.local_files_only:
        # GLiNER 0.2.22 does not propagate this flag to the encoder tokenizer.
        from unittest.mock import patch

        with patch("transformers.utils.hub._is_offline_mode", True):
            model = GLiNER.from_pretrained(args.base, local_files_only=True)
    else:
        model = GLiNER.from_pretrained(args.base)

    class StrictTrainer(Trainer):
        """Evaluate the native GLiNER loss without materialising unused logits."""

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            inputs = self._prepare_inputs(inputs)
            with torch.no_grad():
                loss = self.compute_loss(model, inputs)
            return loss.detach(), None, None

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            output = model(alpha=0.75, gamma=2.0, reduction="sum", **inputs)
            if output.loss is None or not torch.isfinite(output.loss):
                raise RuntimeError("GLiNER produced a non-finite loss")
            return (output.loss, output) if return_outputs else output.loss

    training_args = TrainingArguments(
        output_dir=str(args.out / "checkpoints"),
        learning_rate=args.lr,
        weight_decay=0.01,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.gradient_accumulation,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        prediction_loss_only=True,
        warmup_ratio=0.1,
        use_cpu=args.cpu,
        fp16=args.fp16,
        bf16=False,
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = StrictTrainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=validation,
        data_collator=DataCollator(
            model.config,
            data_processor=model.data_processor,
            prepare_labels=True,
        ),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )
    checkpoint_dir = args.out / "checkpoints"
    resume_checkpoint = get_last_checkpoint(str(checkpoint_dir)) if args.resume and checkpoint_dir.is_dir() else None
    if resume_checkpoint:
        print(f"Resuming from {resume_checkpoint}", flush=True)
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    final_dir = args.out / "final"
    trainer.model.save_pretrained(final_dir)
    final_files = [
        {
            "path": str(path.relative_to(final_dir)),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(final_dir.rglob("*"))
        if path.is_file()
    ]
    manifest = {
        "base_model": args.base,
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "learning_rate": args.lr,
        "batch_size": args.batch,
        "gradient_accumulation": args.gradient_accumulation,
        "early_stopping_patience": args.patience,
        "fp16": args.fp16,
        "device": str(trainer.args.device),
        "resumed_from_checkpoint": resume_checkpoint,
        "train_records": len(train),
        "validation_records": len(validation),
        "train_sha256": _sha256(args.train),
        "validation_sha256": _sha256(args.val),
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_validation_loss": trainer.state.best_metric,
        "global_steps": trainer.state.global_step,
        "train_metrics": train_result.metrics,
        "test_data_used": False,
        "local_files_only": args.local_files_only,
        "final_model": str(final_dir.resolve()),
        "final_files": final_files,
    }
    (args.out / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
