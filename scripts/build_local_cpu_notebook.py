#!/usr/bin/env python3
"""Generate the resumable local-CPU research notebook."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "notebooks" / "05_local_cpu_full_pipeline.ipynb"


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": f"md-{len(cells):02d}",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": f"code-{len(cells):02d}",
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


cells: list[dict] = []

cells.append(
    markdown("""# Full adaptive-anonymization pipeline — local CPU

This notebook runs the complete pipeline locally without CUDA, CuPy, Google
Drive, or Colab-specific APIs. It uses the privacy-oriented `regex + ner_uk`
detector configuration, rebuilds risk labels for that detector set, and performs
a validation-only adaptive-policy search relative to the fixed L2 baseline.

The run is resumable. Every expensive document, contribution, validation
release, and test release is checkpointed under
`artifacts/runs/cpu_v5_regex_ner/`. Interrupting the kernel does not remove
completed checkpoints; start the notebook again and run all cells.

Before opening the notebook, run from the repository root:

```bash
uv sync --extra data --extra detection --extra notebooks --extra dev
uv run jupyter lab
```

This is an exploratory rerun because results from the previous held-out test
have already been inspected. Use validation for development decisions; a new
untouched test split is required for a new confirmatory claim.
""")
)

cells.append(
    code("""# Local CPU controls. These values do not change the research split.
import os

CPU_THREADS = max(1, (os.cpu_count() or 4) - 1)
FORCE_STAGE = None  # Set to a stage name only when you intentionally want to rerun it.

print("Logical CPUs:", os.cpu_count())
print("Threads reserved for the pipeline:", CPU_THREADS)
print("GPU is not required and will not be used.")
""")
)

cells.append(
    markdown("""## 1. Locate the repository and verify private inputs

The notebook can be opened either from the repository root or from its parent
workspace. Raw WorkUA resumes remain local and must not be committed.
""")
)

cells.append(
    code("""from pathlib import Path
import json
import shutil

cwd = Path.cwd().resolve()
candidates = [cwd, cwd / "adaptive-anonymization-uk", Path("..") / "adaptive-anonymization-uk"]
PROJECT_ROOT = next(
    (path.resolve() for path in candidates if (path / "pyproject.toml").is_file()
     and (path / "src/anonrisk_uk").is_dir()),
    None,
)
if PROJECT_ROOT is None:
    raise FileNotFoundError("Open this notebook from the project or its parent workspace")

CONFIG_PATH = PROJECT_ROOT / "configs/research_cpu.json"
WORKUA_DATA = PROJECT_ROOT / "data/raw/all_resumes.csv"
E3_DATA = PROJECT_ROOT / "data/processed/e3_jsi_uk.jsonl"
VACANCY_DATA = PROJECT_ROOT / "data/raw/vacancies_sample.ndjson"

for required in [CONFIG_PATH, WORKUA_DATA, E3_DATA, VACANCY_DATA]:
    if not required.is_file():
        raise FileNotFoundError(required)

config_preview = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
RUN_OUTPUT = (PROJECT_ROOT / config_preview["out"]).resolve()
VACANCY_MODEL = (PROJECT_ROOT / config_preview["vacancy_model_path"]).resolve()

print("Project:", PROJECT_ROOT)
print("Run output:", RUN_OUTPUT)
print("WorkUA MiB:", round(WORKUA_DATA.stat().st_size / 2**20, 1))
print("Free disk GiB:", round(shutil.disk_usage(PROJECT_ROOT).free / 2**30, 1))
""")
)

cells.append(
    markdown("""## 2. Reproduce the environment and force CPU execution

The environment deliberately does not install or verify CuPy. Transformer NER
uses spaCy on CPU. BLAS and tokenizer thread counts are bounded so the laptop
remains responsive.
""")
)

cells.append(
    code("""import shlex
import subprocess
import sys

UV = shutil.which("uv")
if UV is None:
    raise RuntimeError("Install uv first: https://docs.astral.sh/uv/")

VENV = PROJECT_ROOT / ".venv"
PYTHON = VENV / "bin/python"
INSTALL_ENV = os.environ.copy()
INSTALL_ENV["UV_PROJECT_ENVIRONMENT"] = str(VENV)

def print_command(command):
    print("$", shlex.join(str(part) for part in command), flush=True)

sync_command = [
    UV, "sync", "--project", PROJECT_ROOT, "--frozen",
    "--extra", "data", "--extra", "detection",
    "--extra", "notebooks", "--extra", "dev",
]
print_command(sync_command)
subprocess.run([str(part) for part in sync_command], cwd=PROJECT_ROOT, env=INSTALL_ENV, check=True)

RUN_ENV = os.environ.copy()
RUN_ENV.update({
    "TOKENIZERS_PARALLELISM": "false",
    "OMP_NUM_THREADS": str(CPU_THREADS),
    "MKL_NUM_THREADS": str(CPU_THREADS),
    "OPENBLAS_NUM_THREADS": str(CPU_THREADS),
    "VECLIB_MAXIMUM_THREADS": str(CPU_THREADS),
    "NUMEXPR_NUM_THREADS": str(CPU_THREADS),
    "LOKY_MAX_CPU_COUNT": str(CPU_THREADS),
    "PYTHONUNBUFFERED": "1",
})

probe = subprocess.check_output(
    [PYTHON, "-c", "import json,platform,torch; print(json.dumps({'python': platform.python_version(), 'machine': platform.machine(), 'torch': torch.__version__, 'cuda_used': False}))"],
    cwd=PROJECT_ROOT, env=RUN_ENV, text=True,
)
print(json.dumps(json.loads(probe), indent=2))
""")
)

cells.append(
    code("""runtime_config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
assert runtime_config["detection_device"] == "cpu"
assert runtime_config["detectors"] == ["regex", "ner_uk"]
assert runtime_config["release_detectors"] == ["regex", "ner_uk"]
assert runtime_config["require_finetuned_gliner"] is False
assert runtime_config["selection_reference_policy"] == "L2_generalize"
assert runtime_config["max_steps"] >= 20
assert runtime_config["expand_on_plateau"] is True

print(json.dumps({
    key: runtime_config[key]
    for key in [
        "detectors", "detection_device", "strategies", "taus",
        "utility_floors", "selection_reference_policy", "max_steps",
        "max_candidates_per_step", "expand_on_plateau",
    ]
}, ensure_ascii=False, indent=2))

subprocess.run(
    [PYTHON, "-m", "anonrisk_uk.cli", "doctor", "--config", CONFIG_PATH],
    cwd=PROJECT_ROOT, env=RUN_ENV, check=True,
)
""")
)

cells.append(
    markdown("""## 3. Resumable stage runner

A stage is skipped only after all of its declared final artifacts exist. The
fine-grained checkpoints inside `risk-labels` and `pipeline` are reused even
when a top-level stage was interrupted before producing its manifest.
""")
)

cells.append(
    code("""import time

LOG_DIR = RUN_OUTPUT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

def files_ready(paths):
    return all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in paths)

def run_stage(stage, outputs):
    outputs = [Path(path) for path in outputs]
    if FORCE_STAGE != stage and files_ready(outputs):
        print(f"{stage}: verified final artifacts already exist; skipped.")
        return
    log_path = LOG_DIR / f"{stage}.log"
    command = [PYTHON, "-m", "anonrisk_uk.cli", stage, "--config", CONFIG_PATH]
    print_command(command)
    print("Log:", log_path, flush=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [str(part) for part in command], cwd=PROJECT_ROOT, env=RUN_ENV,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
        print("\\n".join(tail))
        raise RuntimeError(f"{stage} failed with exit code {return_code}; see {log_path}")
    if not files_ready(outputs):
        raise RuntimeError(f"{stage} finished without all expected outputs: {outputs}")
    print(f"{stage}: complete in {(time.perf_counter() - started) / 60:.1f} min")
""")
)

cells.append(
    markdown("""## 4. Detection quality gate

This run excludes the rejected GLiNER model. The `regex + ner_uk` release
ensemble must satisfy the declared overlap-precision constraint on validation;
otherwise the pipeline stops before risk-label generation.
""")
)

cells.append(
    code("""run_stage("detection", [
    RUN_OUTPUT / "detection/manifest.json",
    RUN_OUTPUT / "detection/selected_detector.json",
    RUN_OUTPUT / "detection/metrics.csv",
])

selected_detector = json.loads(
    (RUN_OUTPUT / "detection/selected_detector.json").read_text(encoding="utf-8")
)
print(json.dumps(selected_detector, ensure_ascii=False, indent=2))
assert selected_detector["status"] == "ready"
assert selected_detector["precision_constraint_met"] is True
""")
)

cells.append(
    markdown("""## 5. WorkUA population and contribution labels

Changing the release detector changes document units and therefore invalidates
old contribution labels. These stages intentionally rebuild them. On CPU this
is the longest preparation stage, but completed documents and contributions are
saved atomically.
""")
)

cells.append(
    code("""run_stage("workua", [
    RUN_OUTPUT / "workua/bundle.pkl",
    RUN_OUTPUT / "workua/manifest.json",
    RUN_OUTPUT / "workua/split.csv",
])
""")
)

cells.append(
    code("""run_stage("risk-labels", [
    RUN_OUTPUT / "pipeline/risk_label_manifest.json",
    RUN_OUTPUT / "pipeline/unit_training_features.csv",
    RUN_OUTPUT / "pipeline/unit_training_labels.csv",
    RUN_OUTPUT / "pipeline/unit_validation_features.csv",
    RUN_OUTPUT / "pipeline/unit_validation_labels.csv",
])
""")
)

cells.append(
    code("""run_stage("risk-train", [
    RUN_OUTPUT / "pipeline/unit_risk_model.pkl",
    RUN_OUTPUT / "pipeline/risk_model_selection.json",
    RUN_OUTPUT / "pipeline/risk_model_validation.csv",
    RUN_OUTPUT / "pipeline/risk_learning_curve.csv",
])
""")
)

cells.append(
    markdown("""## 6. Utility reference and adaptive-policy evaluation

The adaptive search uses up to 20 transformations, does not truncate the
candidate list, expands across score plateaus, and selects its operating point
on validation. If the absolute target is unattainable, it must at least match
the validation risk of L2 before retention is maximized.
""")
)

cells.append(
    code("""run_stage("vacancy-train", [
    VACANCY_MODEL,
    VACANCY_MODEL.with_suffix(".manifest.json"),
])
""")
)

cells.append(
    code("""run_stage("pipeline", [
    RUN_OUTPUT / "pipeline/manifest.json",
    RUN_OUTPUT / "pipeline/operating_point.json",
    RUN_OUTPUT / "pipeline/validation_tradeoffs.csv",
    RUN_OUTPUT / "pipeline/test_summary.csv",
    RUN_OUTPUT / "pipeline/test_intervals.csv",
    RUN_OUTPUT / "pipeline/vacancy_summary.csv",
])
""")
)

cells.append(
    markdown("""## 7. Inspect the completed run

The final cell prints the detector gate, risk-model validation, validation
trade-offs, test metrics, and stop reasons. Do not use the displayed test table
to retune the configuration.
""")
)

cells.append(
    code("""import pandas as pd

def show_csv(path, **kwargs):
    frame = pd.read_csv(path, **kwargs)
    display(frame)
    return frame

print("Detection metrics")
detection_metrics = show_csv(RUN_OUTPUT / "detection/metrics.csv")

print("Risk-model validation")
risk_validation = show_csv(RUN_OUTPUT / "pipeline/risk_model_validation.csv")

print("Validation privacy/utility frontier")
validation_tradeoffs = show_csv(RUN_OUTPUT / "pipeline/validation_tradeoffs.csv")

print("Selected operating point")
operating_point = json.loads(
    (RUN_OUTPUT / "pipeline/operating_point.json").read_text(encoding="utf-8")
)
print(json.dumps(operating_point, ensure_ascii=False, indent=2))

print("Held-out evaluation (exploratory rerun; do not tune on this table)")
test_summary = pd.read_csv(RUN_OUTPUT / "pipeline/test_summary.csv")
display(test_summary[
    test_summary["cohort"].eq("all") & test_summary["scenario"].eq("S2")
].sort_values(["attacker", "recall@1"]))

stop_path = RUN_OUTPUT / "pipeline/stop_reasons.csv"
if stop_path.is_file():
    print("Adaptive stop reasons")
    display(pd.read_csv(stop_path))

print("Complete output:", RUN_OUTPUT)
""")
)

cells.append(
    markdown("""## Recovery and interpretation

- If the notebook is interrupted, restart it and choose **Run All**. Completed
  stages and item-level checkpoints are reused.
- Do not delete the run directory while a stage is incomplete.
- If a detector fails the validation gate, fix the detector; do not lower the
  gate after looking at test results.
- Success over L2 requires lower or equal validation/test linkage risk with
  higher utility. Merely lowering risk relative to the original text is not
  enough to claim that the adaptive policy is better than L2.
""")
)

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (adaptive-anonymization-uk)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

DESTINATION.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(DESTINATION)
