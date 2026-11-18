"""Download only public E5 files at a pinned revision; never send resume text."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

MODEL_ID = "intfloat/multilingual-e5-small"
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
FILES = ("README.md", "config.json", "tokenizer.json", "tokenizer_config.json",
         "special_tokens_map.json", "sentencepiece.bpe.model", "model.safetensors")
EXPECTED_SHA256 = {
    "README.md": "0038de97aee16258cecbad7ffda4b4febd6953e747a00e0ddbc8e6ed241e9c1c",
    "config.json": "69137736cab8b8903a07fe8afaafdda25aac55415a12a55d1bffa9f581abf959",
    "tokenizer.json": "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39",
    "tokenizer_config.json": "a1d6bc8734a6f635dc158508bef000f8e2e5a759c7d92f984b2c86e5ff53425b",
    "special_tokens_map.json": "d05497f1da52c5e09554c0cd874037a083e1dc1b9cfd48034d1c717f1afc07a7",
    "sentencepiece.bpe.model": "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865",
    "model.safetensors": "1a55775f53449dac10a2bcbc312469fac40b96d53198c407081a831f81c98477",
}


def verified_file_sha(path, name):
    """Trust the checked-in pinned digest, never a freshly rewritten manifest."""
    with Path(path).open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual != EXPECTED_SHA256[name]:
        raise ValueError(f"Pinned public model file is corrupt or from another revision: {name}")
    return actual


def verify_existing_cache(directory):
    directory = Path(directory)
    manifest_path = directory / "snapshot_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get("model_id") != MODEL_ID or manifest.get("revision") != REVISION
                or manifest.get("files_sha256") != EXPECTED_SHA256):
            raise ValueError("Existing snapshot manifest is not the pinned public model")
    for name in FILES:
        if (directory / name).exists():
            verified_file_sha(directory / name, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("artifacts/models/semantic_attacker_e5"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    verify_existing_cache(args.out)
    hashes = {}
    for name in FILES:
        path = args.out / name
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".download")
            request = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}"
            with urlopen(request, timeout=90) as response, temporary.open("wb") as target:
                while block := response.read(1024 * 1024):
                    target.write(block)
            verified_file_sha(temporary, name)
            temporary.replace(path)
        hashes[name] = verified_file_sha(path, name)
        print(f"downloaded {name}: {path.stat().st_size} bytes", flush=True)
    manifest = {"model_id": MODEL_ID, "revision": REVISION, "files_sha256": hashes,
                "network_payload": "public model requests only; no resume data", "paid_api_calls": 0}
    (args.out / "snapshot_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
