"""Export/restore a checksum-bound local v5 runtime; smoke-test without a pool.

Model pickle archives are executable Python objects. Restore only an archive
from the project owner whose SHA-256 you verified through a trusted channel.
Never publish this private runtime archive automatically.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
MODEL = "artifacts/models/resume_action_v5/model.pkl"
SELECTION = "artifacts/models/resume_action_v5/selection.json"
DETECTOR = "artifacts/runs/entity_policy_v4c_local/detection/selected_detector.json"
TAXONOMY = "data/taxonomy/category_map.json"
PAYLOAD = (MODEL, SELECTION, DETECTOR, TAXONOMY)
MODEL_SHA256 = "5c43ea2774448840c7a395d4ecbf31a3d9f34a5ee6f8f331588b1957ea9084e8"
NER_ID = "dchaplinsky/uk_ner_web_trf_13class"
NER_REVISION = "5982492393dd308a04e6989cac655499fa840c69"
SAMPLE = ("Мене звати Андрій Мельник. Я Python-розробник у Києві. "
          "Володію Python, SQL і Docker. Контакти: andrii@example.com, +380 50 123 45 67.")


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def code_files(root):
    # Evaluation-only projections do not participate in deployed inference.
    evaluation_modules = {"content_evaluation.py", "content_provenance.py"}
    return [Path("pyproject.toml"), Path("uv.lock"), Path(".python-version"),
            Path("configs/adaptive_resume.json"), Path("configs/research_entity_policy_v4c_local.json"),
            *[path.relative_to(root) for path in sorted((root / "src/anonrisk_uk").glob("*.py"))
              if path.name not in evaluation_modules]]


def cached_ner():
    from huggingface_hub import snapshot_download
    path = Path(snapshot_download(NER_ID, local_files_only=True, ignore_patterns=["*.whl"], token=False))
    if path.name != NER_REVISION:
        raise ValueError("Local NER default differs from the recorded v5 snapshot")
    return {"model": NER_ID, "revision": path.name,
            "files_sha256": {str(p.relative_to(path)): sha256(p) for p in sorted(path.rglob("*")) if p.is_file()},
            "included_in_archive": False}


def make_manifest(root=ROOT, *, include_ner=True):
    missing = [name for name in PAYLOAD if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing local runtime assets: " + ", ".join(missing))
    if sha256(root / MODEL) != MODEL_SHA256:
        raise ValueError("The selected v5 weights differ from the recorded frozen model")
    packages = {}
    for name in ("numpy", "scikit-learn", "pandas", "torch", "transformers", "spacy", "huggingface-hub"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"schema": "private_frozen_runtime_v1", "method": "adaptive_resume_v5_action_regression",
            "model_sha256": MODEL_SHA256, "python": platform.python_version(), "packages": packages,
            "payload_sha256": {name: sha256(root / name) for name in PAYLOAD},
            "source_sha256": {str(name): sha256(root / name) for name in code_files(root)},
            "external_detector": cached_ner() if include_ner else None,
            "contains_real_resume_records": False, "contains_api_credentials": False,
            "learned_statistics_note": "Fitted vocabularies/statistics can retain source-derived information; keep this archive private.",
            "trust": "Checksums detect differences, not malicious pickle code. Use a trusted owner-supplied archive."}


def export_archive(destination, root=ROOT, *, include_ner=True):
    if destination.exists():
        raise FileExistsError("Archive output exists")
    manifest = make_manifest(root, include_ner=include_ner)
    payload = json.dumps(manifest, indent=2).encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(destination, "w:gz") as archive:
            metadata = tarfile.TarInfo("runtime_manifest.json")
            metadata.size = len(payload)
            metadata.mode = 0o600
            archive.addfile(metadata, io.BytesIO(payload))
            for name in PAYLOAD:
                archive.add(root / name, arcname=name, recursive=False)
        # Refuse a snapshot assembled while inputs were changing.
        for name, digest in {**manifest["payload_sha256"], **manifest["source_sha256"]}.items():
            if sha256(root / name) != digest:
                raise RuntimeError(f"Runtime source changed during export: {name}")
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    destination.chmod(0o600)
    return manifest


def restore_archive(source, root=ROOT):
    with tarfile.open(source, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(set(names)) != len(names) or set(names) != {*PAYLOAD, "runtime_manifest.json"}:
            raise ValueError("Archive members differ from the explicit runtime allowlist")
        for member in members:
            name = PurePosixPath(member.name)
            if not member.isfile() or name.is_absolute() or ".." in name.parts:
                raise ValueError("Unsafe archive member")
            if member.size > 100_000_000:
                raise ValueError("Oversized runtime member")
        manifest = json.load(archive.extractfile("runtime_manifest.json"))
        if manifest.get("schema") != "private_frozen_runtime_v1" or set(manifest["payload_sha256"]) != set(PAYLOAD):
            raise ValueError("Unsupported runtime manifest")
        # Check everything, including existing files, before performing mutations.
        extracted = {}
        for name in PAYLOAD:
            data = archive.extractfile(name).read()
            digest = hashlib.sha256(data).hexdigest()
            if digest != manifest["payload_sha256"][name]:
                raise ValueError(f"Archive checksum mismatch: {name}")
            target = root / name
            if target.exists() and (not target.is_file() or target.is_symlink() or sha256(target) != digest):
                raise FileExistsError(f"Conflicting local runtime asset: {name}")
            # Parent symlinks must not redirect extraction outside the checkout.
            if not target.resolve().is_relative_to(root.resolve()):
                raise ValueError("Archive target escapes checkout")
            extracted[name] = data
        for name, digest in manifest["source_sha256"].items():
            candidate = PurePosixPath(name)
            allowed_source = {str(path) for path in code_files(root)}
            if (candidate.is_absolute() or ".." in candidate.parts or name not in allowed_source
                    or not (root / name).is_file()):
                raise ValueError(f"Missing or unsafe matching source: {name}")
            if sha256(root / name) != digest:
                raise ValueError(f"Checkout differs from archived runtime source: {name}")
        if manifest["payload_sha256"][MODEL] != MODEL_SHA256:
            raise ValueError("Archive is not the recorded selected v5 model")
        record = root / "artifacts/models/resume_action_v5/runtime_manifest.json"
        if record.is_symlink() or not record.resolve().is_relative_to(root.resolve()):
            raise ValueError("Runtime manifest target escapes checkout")
        for name, data in extracted.items():
            target = root / name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as handle:
                    handle.write(data)
                target.chmod(0o600)
        record.write_text(json.dumps(manifest, indent=2))
    return manifest


def verify_manifest(manifest, root=ROOT, *, detector=True):
    for name, digest in {**manifest["source_sha256"], **manifest["payload_sha256"]}.items():
        if not (root / name).is_file() or sha256(root / name) != digest:
            raise ValueError(f"Missing or changed frozen file: {name}")
    for name, expected in manifest.get("packages", {}).items():
        if expected is None:
            continue
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        if actual != expected:
            raise ValueError(f"Runtime dependency differs: {name} expected {expected}, found {actual}; use uv sync --locked")
    expected = manifest.get("external_detector")
    if detector and expected:
        actual = cached_ner()
        if actual["revision"] != expected["revision"] or actual["files_sha256"] != expected["files_sha256"]:
            raise ValueError("Local NER snapshot differs; prepare the pinned revision before inference")


def prepare_detector(manifest):
    """Download public weights only, then bind the local default ref exactly."""
    from huggingface_hub import snapshot_download
    expected = manifest.get("external_detector")
    if not expected:
        raise ValueError("Manifest does not pin a detector snapshot")
    if expected["model"] != NER_ID or expected["revision"] != NER_REVISION:
        raise ValueError("Manifest requests a different detector than the recorded v5 runtime")
    path = Path(snapshot_download(NER_ID, revision=NER_REVISION, ignore_patterns=["*.whl"], token=False))
    actual = {str(p.relative_to(path)): sha256(p) for p in sorted(path.rglob("*")) if p.is_file()}
    if actual != expected["files_sha256"]:
        raise ValueError("Downloaded detector files differ from the archived snapshot")
    reference = path.parent.parent / "refs/main"
    reference.parent.mkdir(parents=True, exist_ok=True)
    reference.write_text(expected["revision"])


def smoke():
    # Set offline flags before importing the runtime's Hugging Face integrations.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from unittest.mock import patch

    from threadpoolctl import threadpool_limits

    from anonrisk_uk.resume_adaptive import anonymize_resume
    from anonrisk_uk.resume_resources import load_resume_resources
    # Any attempted construction of a live corpus index fails this check.
    with (threadpool_limits(limits=1),
          patch("anonrisk_uk.attack.BM25Index.fit", side_effect=AssertionError("Live index forbidden"))):
        resources = load_resume_resources(ROOT / "configs/adaptive_resume.json")
        if resources.background_texts or resources.feature_index is not None:
            raise RuntimeError("Autonomous runtime unexpectedly contains a search corpus")
        result = anonymize_resume(resources, SAMPLE)
    value = result.releases["adaptive"].text
    if "andrii@example.com" in value or "+380 50 123 45 67" in value:
        raise RuntimeError("Synthetic contact was not removed")
    if result.diagnostics.get("live_reference_pool_used") is not False:
        raise RuntimeError("No-pool inference was not confirmed")
    return {"synthetic_smoke": "passed", "model_sha256": sha256(ROOT / MODEL),
            "live_reference_pool_used": False, "paid_api_used": False,
            "output": value, "note": "Implementation smoke test, not a quality benchmark"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["export", "restore", "check", "prepare-detector", "smoke"])
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--manifest", type=Path, default=ROOT / "artifacts/models/resume_action_v5/runtime_manifest.json")
    args = parser.parse_args()
    if args.action == "export":
        if args.archive is None:
            parser.error("export requires --archive")
        manifest = export_archive(args.archive)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2))
        print(json.dumps({"archive": str(args.archive), "bytes": args.archive.stat().st_size,
                          "sha256": sha256(args.archive), "manifest": str(args.manifest)}, indent=2))
    elif args.action == "restore":
        if args.archive is None:
            parser.error("restore requires --archive")
        restore_archive(args.archive)
        print("Restored missing runtime assets; identical existing files preserved.")
    elif args.action == "prepare-detector":
        prepare_detector(json.loads(args.manifest.read_text()))
        print("Pinned public detector snapshot prepared; no resumes or credentials were transmitted.")
    elif args.action == "check":
        manifest = json.loads(args.manifest.read_text()) if args.manifest.exists() else make_manifest()
        verify_manifest(manifest)
        print(json.dumps({"runtime": "ready", "model_sha256": manifest["model_sha256"],
                          "ner_revision": manifest["external_detector"]["revision"]}, indent=2))
    else:
        if args.manifest.exists():
            verify_manifest(json.loads(args.manifest.read_text()))
        elif sha256(ROOT / MODEL) != MODEL_SHA256:
            raise ValueError("Selected model changed")
        print(json.dumps(smoke(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
