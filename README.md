# Adaptive Anonymization of Ukrainian Resumes

A trained action regressor selects which resume fragments to mask or generalize.
The active implementation is `adaptive_resume_v5_action_regression`, using
`HistGradientBoostingRegressor` and a controller constrained by information loss.
The CLI and Streamlit demo use the same inference engine.

```text
Training: linked resumes → candidate edits → measured attack effects → action regressor
Inference: one resume + trained weights → typed transformations → anonymized text
```

Inference runs locally without a resume pool, search index or OpenAI key.
The regressor predicts the relative benefit of an edit. Its score is not a
probability of identification; the output requires review before publication.

## Requirements

- Python **3.12**.
- [uv](https://docs.astral.sh/uv/) for the locked environment.
- The separately supplied runtime archive for the trained thesis model.
- Internet access for initial dependency and public NER-weight downloads.

Run all commands from the repository root. The repository contains source code,
synthetic software tests and technical instructions. Datasets,
trained weights, generated reports and thesis drafts are supplied separately.

## 1. Install

```bash
uv sync --locked --extra detection --extra ui --extra dev
```

The lock file fixes package versions. No `.env` file is needed for the local
pipeline. The supported environment uses Python 3.12; do not substitute a newer
Python version for exact reproduction.

## 2. Restore the trained model

Obtain `resume_action_v5_runtime_20261111.tar.gz` from the project owner. Its
SHA-256 is:

```text
477007c9c299d12218b90a97b847765aefb3d6e0ea1db9497465601b2a4c772b
```

Check the archive checksum, restore the resources and run the local smoke test:

```bash
shasum -a 256 /path/to/resume_action_v5_runtime_20261111.tar.gz
uv run python scripts/frozen_runtime.py restore \
  --archive /path/to/resume_action_v5_runtime_20261111.tar.gz
uv run python scripts/frozen_runtime.py prepare-detector
uv run python scripts/frozen_runtime.py check
uv run python scripts/frozen_runtime.py smoke
```

The archive contains the model, selection record, detector selection and
category lookup. Restore verifies checksums and matching runtime sources.
`prepare-detector` downloads the pinned public NER snapshot. The smoke test
processes a synthetic resume offline and verifies that inference uses no pool.
Use the archive supplied with this source revision; missing weights cause an
explicit error.

## 3. Run

Start the demo and open the local address printed by Streamlit:

```bash
uv run streamlit run app/text_demo.py
```

Choose a synthetic example, paste a resume or upload a UTF-8 `.txt` file. Select
**Анонімізувати** to view the result, highlighted changes and selected actions.
The result can be downloaded as text. Examples are demonstrations, not test data
for measuring generalization.

Process a file from the command line:

```bash
mkdir -p artifacts/outputs
uv run python scripts/anonymize_resume.py \
  --input tests/fixtures/resume_prose_regression.txt \
  --output artifacts/outputs/resume_anonymized.txt
```

Use a new output filename for each run; the CLI refuses to overwrite files.
Inputs and outputs must be plain text. PDF parsing and OCR are outside this
implementation.

## 4. Check the code

These checks need no trained weights or datasets:

```bash
uv sync --locked --extra dev
uv run ruff check src app scripts tests
uv run pytest -q
uv run python scripts/check_repository.py
```

Tests use synthetic inputs and mocks. Streamlit-specific tests require the
`ui` extra. CI runs the checks and builds the source and wheel distributions.

## 5. Reproduce experiments or train

[Reproduction instructions](docs/reproduction.md) provide the required inputs
and commands for three separate workflows:

1. Run the existing trained model.
2. Recompute frozen corpus metrics from the original evaluation inputs.
3. Reserve new splits, train a model and evaluate it on held-out people.

Exact corpus reproduction requires the separately supplied original data and
evaluation bundle. The runtime archive alone is sufficient for inference.
Downloading data and training again creates a new experiment; it does not
guarantee the original split, weights or numerical results.

## Source map

| Path | Purpose |
|---|---|
| `src/anonrisk_uk/action_resume.py` | Trained action model, features and controller |
| `src/anonrisk_uk/resume_parser.py` | Typed fragment extraction |
| `src/anonrisk_uk/resume_adaptive.py` | Shared inference entry point and rendering |
| `src/anonrisk_uk/attack.py` | Retrieval attackers used offline |
| `app/` | Streamlit interface, styles and synthetic examples |
| `configs/adaptive_resume.json` | Active model and resource paths |
| `configs/baselines/` | Explicit comparison configurations |
| `scripts/` | Preparation, training, evaluation and reproduction commands |
| `tests/` | Software tests and synthetic fixtures |
| `data/README.md` | External input formats and resource locations |
| `docs/` | Method and reproduction instructions |

The older search and absolute-risk implementations remain as evaluation
controls and shared dependencies. They are not alternative demo defaults.
The `research_cpu.json` and `research_pilot.json` profiles support the retained
stage-based research CLI; the active demo uses `adaptive_resume.json`.

Read [the method description](docs/action-model-v5.md) for the learning target,
features, transformations and metric definitions. Raw data, models, reports,
caches and credentials are excluded by `.gitignore` and the distribution
manifest. `scripts/check_repository.py` checks files eligible for publication.

No source-code license has been selected. Dataset and pretrained-model licenses
apply separately.
