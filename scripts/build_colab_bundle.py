#!/usr/bin/env python3
"""Build deterministic public or private Colab input bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = "adaptive-anonymization-uk"
ROOT_FILES = [
    ".python-version",
    "MANIFEST.in",
    "README.md",
    "pyproject.toml",
    "uv.lock",
]
TREE_PATHS = [
    "configs",
    "data/processed",
    "src",
]
EXPLICIT_FILES = [
    "notebooks/04_colab_gliner_training.ipynb",
    "notebooks/COLAB.md",
    "scripts/train_gliner.py",
    "scripts/select_gliner_model.py",
]
PRIVATE_PIPELINE_FILES = ["notebooks/05_colab_full_pipeline.ipynb"]
PRIVATE_FILES = [
    "data/raw/all_resumes.csv",
    "data/raw/vacancies_sample.ndjson",
    "data/raw/vacancies_sample.manifest.json",
]
SKIP_PARTS = {"__pycache__", ".ipynb_checkpoints"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _candidate_files(include_private_data: bool) -> list[Path]:
    requested = ROOT_FILES + EXPLICIT_FILES
    if include_private_data:
        requested += PRIVATE_PIPELINE_FILES + PRIVATE_FILES
    paths = [PROJECT_ROOT / name for name in requested]
    for name in TREE_PATHS:
        paths.extend(path for path in (PROJECT_ROOT / name).rglob("*") if path.is_file())
    files = sorted(set(paths), key=lambda path: path.relative_to(PROJECT_ROOT).as_posix())
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing bundle input: {missing[0]}")
    return [
        path
        for path in files
        if not (set(path.relative_to(PROJECT_ROOT).parts) & SKIP_PARTS)
        and not any(part.endswith(".egg-info") for part in path.relative_to(PROJECT_ROOT).parts)
    ]


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    return info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--include-private-data",
        action="store_true",
        help="Include the local Git-ignored WorkUA resumes and vacancy sample",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out is None:
        filename = (
            "adaptive-anonymization-uk-colab-private.zip"
            if args.include_private_data
            else "adaptive-anonymization-uk-colab.zip"
        )
        args.out = PROJECT_ROOT / "artifacts" / "exports" / filename
    files = _candidate_files(args.include_private_data)
    manifest_files = []
    for path in files:
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        manifest_files.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
                "private": relative in PRIVATE_FILES,
            }
        )

    manifest = {
        "schema_version": 1,
        "archive_root": ARCHIVE_ROOT,
        "purpose": (
            "Complete adaptive-anonymization research pipeline in Google Colab"
            if args.include_private_data
            else "GLiNER fine-tuning and validation-only model selection in Google Colab"
        ),
        "contains_private_workua_data": args.include_private_data,
        "contains_e3_jsi_test_annotations": True,
        "test_split_used_by_training_notebook": False,
        "files": manifest_files,
    }
    manifest_data = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w") as archive:
        for path in files:
            relative = path.relative_to(PROJECT_ROOT).as_posix()
            with path.open("rb") as source, archive.open(_zip_info(f"{ARCHIVE_ROOT}/{relative}"), "w") as target:
                while block := source.read(1024 * 1024):
                    target.write(block)
        archive.writestr(_zip_info(f"{ARCHIVE_ROOT}/colab_bundle_manifest.json"), manifest_data)

    archive_hash = _sha256_file(args.out)
    print(f"Created: {args.out}")
    print(f"Size: {args.out.stat().st_size} bytes")
    print(f"SHA-256: {archive_hash}")


if __name__ == "__main__":
    main()
