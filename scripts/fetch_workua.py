#!/usr/bin/env python3
"""Download the public WorkUA resumes dataset into a local, Git-ignored file."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/raw/all_resumes.csv"))
    args = parser.parse_args()

    from datasets import load_dataset

    from anonrisk_uk.workua import LIST_COLUMNS, _as_list

    frame = load_dataset("KSE-RESEARCH-Group/Work_UA_resumes", split="train").to_pandas()
    # Arrow list columns become ndarrays in pandas. Their string representation
    # omits commas (and can truncate long lists), so serialize explicitly as JSON.
    for column in LIST_COLUMNS:
        if column in frame:
            frame[column] = frame[column].map(
                lambda value: json.dumps(_as_list(value, strict=True), ensure_ascii=False)
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    manifest = {
        "source": "KSE-RESEARCH-Group/Work_UA_resumes",
        "rows": len(frame),
        "sha256": sha256(args.out),
    }
    args.out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(frame)} resumes to {args.out}")


if __name__ == "__main__":
    main()
