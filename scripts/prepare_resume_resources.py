"""Prepare only the validated detector, WorkUA bundle and lexical vectorizer.

Existing complete artifacts are reused. No historical policy model is trained
and no paid provider is called. Run from the repository root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from threadpoolctl import threadpool_limits

from anonrisk_uk.research import prepare_workua, run_detection
from anonrisk_uk.resume_resources import resource_paths


def detector_ready(path, names):
    if not path.is_file():
        return False
    selection = json.loads(path.read_text())
    return selection.get("status") == "ready" and set(names) <= set(selection.get("detectors", []))


def prepare(config_path, *, check_only=False):
    settings, profile, paths = resource_paths(config_path)
    names = profile.get("release_detectors", ["regex", "ner_uk"])
    if set(names) - {"regex", "ner_uk"}:
        raise ValueError("This setup command supports the canonical local regex + ner_uk detector set")
    complete = all(p.is_file() for p in paths.values()) and detector_ready(paths["detector"], names)
    if complete:
        print("Inference artifacts are present; no files changed.")
        return True
    if check_only:
        for name, path in paths.items():
            valid = detector_ready(path, names) if name == "detector" else path.is_file()
            print(f"{name}: {'ready' if valid else 'missing or incomplete'} ({path})")
        return False
    if not paths["bundle"].is_file() and not Path(profile["workua_path"]).is_file():
        raise FileNotFoundError("Download WorkUA first: python scripts/fetch_workua.py --out data/raw/all_resumes.csv")
    # Bootstrap must also work before any historical experiment has been run.
    config = {**profile, "development_out": None, "allow_paid_api": False}
    if not detector_ready(paths["detector"], names):
        run_detection(config)
        if not detector_ready(paths["detector"], names):
            raise RuntimeError("Detector validation failed; inspect the detection metrics before using the demo")
    if not paths["bundle"].is_file():
        prepare_workua(config)
    if not paths["retention"].is_file():
        with paths["bundle"].open("rb") as handle:
            bundle = pickle.load(handle)
        vectorizer = TfidfVectorizer(sublinear_tf=True).fit(bundle["texts"])
        paths["retention"].parent.mkdir(parents=True, exist_ok=True)
        with paths["retention"].open("xb") as handle:
            pickle.dump(vectorizer, handle)
    manifest = {"method": settings["method"], "inference_resources_only": True,
                "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
                "historical_policy_training": False, "published_evaluation_reproduced": False}
    (Path(profile["out"]) / "resume_resource_preparation.json").write_text(json.dumps(manifest, indent=2))
    print("Inference artifacts prepared. Run: streamlit run app/text_demo.py")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/adaptive_resume.json"))
    parser.add_argument("--check", action="store_true", help="Check local artifact files without downloads or changes")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        return 0 if prepare(args.config, check_only=args.check) else 1


if __name__ == "__main__":
    raise SystemExit(main())
