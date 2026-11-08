"""Command-line interface for the reproducible research workflow."""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .research import (
    prepare_detection_data,
    resolve_config,
    run_detection,
    run_pipeline,
    run_risk_labels,
    run_risk_training,
    run_vacancy_evaluation,
    run_vacancy_training,
    run_workua,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def default_config() -> Path:
    """Prefer a checkout in the current directory, then an editable source checkout."""
    candidates = [Path.cwd() / "configs" / "research_pilot.json", PROJECT_ROOT / "configs" / "research_pilot.json"]
    return next((candidate for candidate in candidates if candidate.is_file()), candidates[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "detection-data",
            "detection",
            "workua",
            "risk-labels",
            "risk-train",
            "pipeline",
            "vacancy-train",
            "vacancies",
            "all",
            "doctor",
        ],
    )
    parser.add_argument("--config", type=Path, default=default_config())
    parser.add_argument("--allow-paid", action="store_true", help="Allow OpenAI calls within the configured budget")
    args = parser.parse_args()

    config = resolve_config(args.config)
    if not args.allow_paid:
        config["allow_paid_api"] = False

    if args.stage == "doctor":
        print("Project:", PROJECT_ROOT)
        for package in ["numpy", "torch", "transformers", "gliner", "spacy", "spacy-transformers", "openai"]:
            try:
                print(package, version(package))
            except PackageNotFoundError:
                print(package, "MISSING")
        print("WorkUA:", Path(config["workua_path"]).is_file())
        print("E3-JSI:", Path(config["e3_path"]).is_file())
        print("Paid APIs:", config.get("allow_paid_api", False))
        return

    Path(config["out"]).mkdir(parents=True, exist_ok=True)
    if config.get("approach") in {"typed_actions_v1", "entity_decision_v2"}:
        from threadpoolctl import threadpool_limits

        from . import action_research
        from .research import prepare_workua

        if config.get("approach") == "entity_decision_v2":
            from . import entity_research as workflow
        else:
            workflow = action_research

        functions = {
            "detection-data": prepare_detection_data,
            "detection": run_detection,
            "workua": prepare_workua,
            "risk-labels": workflow.run_labels,
            "risk-train": workflow.run_training,
            "vacancy-train": run_vacancy_training,
            "vacancies": run_vacancy_evaluation,
            "pipeline": workflow.run_pipeline,
        }
        stages = (
            ["detection", "workua", "risk-labels", "risk-train", "vacancy-train", "pipeline"]
            if args.stage == "all"
            else [args.stage]
        )
        with threadpool_limits(limits=config.get("cpu_threads", 1)):
            for stage in stages:
                if stage == "vacancy-train" and not config.get("vacancies_path"):
                    print("vacancy-train: skipped (no vacancy dataset in this diagnostic profile)", flush=True)
                    continue
                print(f"\n=== {stage}: typed action policy ===", flush=True)
                functions[stage](config)
        return
    if args.stage == "vacancy-train":
        run_vacancy_training(config)
        return
    if args.stage == "detection-data":
        prepare_detection_data(config)
        return
    if args.stage == "vacancies":
        run_vacancy_evaluation(config)
        return
    if args.stage == "risk-labels":
        run_risk_labels(config)
        return
    if args.stage == "risk-train":
        run_risk_training(config)
        return

    stages = [("detection", run_detection), ("workua", run_workua), ("pipeline", run_pipeline)]
    for stage, function in stages:
        if args.stage in {stage, "all"}:
            print(f"\n=== {stage} ===", flush=True)
            function(config)


if __name__ == "__main__":
    main()
