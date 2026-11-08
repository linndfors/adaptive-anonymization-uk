"""Create a deterministic vacancy sample from the complete Work.ua stream."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import urllib.request
from pathlib import Path

from tqdm import tqdm

SOURCE_URL = "https://huggingface.co/datasets/KSE-RESEARCH-Group/Work_UA_vacancies/resolve/main/vacancies.ndjson"
REQUIRED_FIELDS = {"id", "title", "description_text"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit < 1:
        raise ValueError("limit must be greater than zero")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(".download")
    digest = hashlib.sha256()
    rows_seen = 0

    # Selecting the lowest deterministic hashes gives every source record the
    # same inclusion probability and avoids bias from source ordering.
    sample: list[tuple[int, str, bytes]] = []
    try:
        with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
            for line in tqdm(response, desc="Vacancies: deterministic hash sample"):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not REQUIRED_FIELDS <= set(row):
                    raise ValueError("The upstream vacancy schema has changed")
                digest.update(line)
                rows_seen += 1
                key = int.from_bytes(
                    hashlib.sha256(f"{args.seed}:{row['id']}".encode()).digest()[:8],
                    "big",
                )
                item = (-key, str(row["id"]), line)
                if len(sample) < args.limit:
                    heapq.heappush(sample, item)
                elif key < -sample[0][0]:
                    heapq.heapreplace(sample, item)

        selected = sorted(sample, key=lambda item: (-item[0], item[1]))
        with temporary.open("wb") as destination:
            for _negative_key, _identifier, line in selected:
                destination.write(line)
        temporary.replace(args.out)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    manifest = {
        "source": SOURCE_URL,
        "source_rows_seen": rows_seen,
        "n_vacancies": len(sample),
        "selection": "deterministic_uniform_hash_sample",
        "seed": args.seed,
        "source_stream_sha256": digest.hexdigest(),
    }
    args.out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(sample)} of {rows_seen} vacancies to {args.out}")


if __name__ == "__main__":
    main()
