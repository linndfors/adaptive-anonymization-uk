"""Local resources for the canonical adaptive engine, without legacy policy weights."""

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

from .attack import BM25Index
from .research import detector_set, resolve_config

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/adaptive_resume.json"


@dataclass
class ResumeResources:
    config: dict
    background_texts: list
    codec: object
    retention_vectorizer: object
    detector: object
    feature_index: object
    artifact_dir: Path
    risk_estimator: object | None = None


def resource_paths(config_path=DEFAULT_CONFIG):
    settings = json.loads(Path(config_path).read_text())
    profile = resolve_config(ROOT / settings["resources_config"])
    out = Path(profile["out"])
    if settings.get("controller") in {"regression", "action_regression"}:
        return settings, profile, {
            "risk_model": ROOT / settings["risk_model"],
            "detector": out / "detection/selected_detector.json",
        }
    return settings, profile, {
        "bundle": out / "workua/bundle.pkl",
        "retention": out / "pipeline/retention_vectorizer.pkl",
        "detector": out / "detection/selected_detector.json",
    }


def load_resume_resources(config_path=DEFAULT_CONFIG):
    settings, profile, paths = resource_paths(config_path)
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing local adaptive resource: {path}")
    selection = json.loads(paths["detector"].read_text())
    names = profile.get("release_detectors", ["regex", "ner_uk"])
    if selection.get("status") != "ready" or not set(names) <= set(selection.get("detectors", [])):
        raise ValueError("Local detector selection is not ready")
    config = {**profile, "adaptive": settings["adaptive"], "local_files_only": True,
              "gliner_threshold": selection.get("gliner_threshold")}
    config["controller"] = settings.get("controller", "search")
    if config["controller"] in {"regression", "action_regression"}:
        from .action_resume import SCHEMA as ACTION_SCHEMA
        from .action_resume import ActionResumeRegressor
        from .learned_resume_risk import SCHEMA, ResumeRiskRegressor

        with paths["risk_model"].open("rb") as handle:
            estimator = pickle.load(handle)
        supported = ((isinstance(estimator, ResumeRiskRegressor) and estimator.schema == SCHEMA)
                     if config["controller"] == "regression" else
                     (isinstance(estimator, ActionResumeRegressor) and estimator.schema == ACTION_SCHEMA))
        if not supported:
            raise ValueError("Model artifact does not match the selected learned controller")
        detector, _ = detector_set(config, names)
        return ResumeResources(config, [], estimator.codec, estimator.retention_vectorizer,
                               detector, None, paths["risk_model"].parent, estimator)
    with paths["bundle"].open("rb") as handle:
        bundle = pickle.load(handle)
    with paths["retention"].open("rb") as handle:
        retention = pickle.load(handle)
    detector, _ = detector_set(config, names)
    index = BM25Index(max_features=config.get("bm25_max_features", 20000)).fit(bundle["texts"])
    return ResumeResources(config, bundle["texts"], bundle["codec"], retention, detector, index, Path(profile["out"]))
