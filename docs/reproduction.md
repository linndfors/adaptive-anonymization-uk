# Reproduction

Use Python 3.12 and run commands from the repository root. The workflows below
distinguish inference, replay of the frozen thesis experiment and training a
new model. Generated files belong under ignored `artifacts/` paths.

## 1. Run the trained model

Follow [README steps 1–3](../README.md). Required external file:
`resume_action_v5_runtime_20261111.tar.gz`, supplied by the project owner.

| Archive property | Expected value |
|---|---|
| Size | 12,062,994 bytes |
| SHA-256 | `477007c9c299d12218b90a97b847765aefb3d6e0ea1db9497465601b2a4c772b` |
| Selected model SHA-256 | `5c43ea2774448840c7a395d4ecbf31a3d9f34a5ee6f8f331588b1957ea9084e8` |

Restore only the trusted owner-supplied archive: pickle models are executable
Python objects. Restore verifies archive members, source checksums and conflicts
before writing files. The archive contains exactly these resources plus a manifest:

```text
artifacts/models/resume_action_v5/model.pkl
artifacts/models/resume_action_v5/selection.json
artifacts/runs/entity_policy_v4c_local/detection/selected_detector.json
data/taxonomy/category_map.json
```

NER weights are prepared separately by `frozen_runtime.py prepare-detector`.
The pinned public model is `dchaplinsky/uk_ner_web_trf_13class`, revision
`5982492393dd308a04e6989cac655499fa840c69`. The recorded spaCy/transformer stack
can emit its state-loading compatibility warning; replacing the stack changes
the frozen environment. `frozen_runtime.py check` verifies installed dependency
versions and cached detector files.

The runtime needs no WorkUA dataset. `smoke` disables network access and rejects
live BM25 index construction. It checks a synthetic example, not privacy quality.

## 2. Replay the frozen corpus experiment

Obtain the original research-input bundle separately. Preserve the paths below
when copying its contents into the checkout. The small runtime archive does not
contain these record-level inputs.

| Input path | Contents |
|---|---|
| `artifacts/action_v5_data_20261110/` | Original `cohort.pkl`, manifest and reserved people |
| `artifacts/evaluations/action_v5_test_20261110/` | Primary releases, report, protocol, pool IDs and decision traces |
| `artifacts/evaluations/action_v5_fixed_controls_20261111/` | Matched releases, traces, protocol and type-prior parameters |
| `artifacts/models/resume_action_v5/` | Selected model and selection record |
| `artifacts/models/resume_risk_v4/model.pkl` | Previous regressor for exact legacy edit reconstruction |
| `artifacts/evaluations/action_v5_validation_selection_v4_20261110/candidate_0_model.pkl` | Frozen offline control |
| `artifacts/evaluations/action_v5_validation_selection_v4_20261110/candidate_1_model.pkl` | Frozen on-policy control |
| `artifacts/models/semantic_minilm/` | Pinned content encoder |
| `artifacts/models/semantic_attacker_e5/` | Pinned semantic attacker |

For a privacy-only replay, the selected model, cohort and both sets of saved
releases/protocols are sufficient. Inspect prerequisites first:

```bash
uv run python scripts/reproduce_frozen.py \
  --steps privacy --out artifacts/verification/privacy_replay --check
uv run python scripts/reproduce_frozen.py \
  --steps privacy --out artifacts/verification/privacy_replay
```

The replay rebuilds the recorded 6000-record attacker pool, excludes self
records, scores all saved methods and checks original privacy/lexical
aggregates. It does not run training or select parameters. A new output
directory is required. The command fails if original data or hashes differ.

### Content and semantic attack

Download the pinned content encoder once if it is absent from the bundle:

```bash
uv run python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    revision='e8f8c211226b894fcb81acc59f3b34ba3efd5f42',
    token=False,
    local_dir='artifacts/models/semantic_minilm',
    allow_patterns=['*.json', 'model.safetensors', 'tokenizer.model',
                    'sentencepiece.bpe.model', 'unigram.json'],
)
PY
uv run python scripts/download_semantic_attacker.py \
  --out artifacts/models/semantic_attacker_e5
```

The semantic attacker is `intfloat/multilingual-e5-small`, revision
`614241f622f53c4eeff9890bdc4f31cfecc418b3`. Models run locally; downloads send
no resume text. Preserve model metadata files because evaluators verify revisions.

Run all three evaluation phases:

```bash
uv run python scripts/reproduce_frozen.py \
  --out artifacts/verification/full_replay --check
uv run python scripts/reproduce_frozen.py \
  --out artifacts/verification/full_replay
```

The phases are `privacy,content,semantic`. Source-anchored content evaluation
reconstructs exact edit plans and checks equality to saved releases. E5 is
evaluated on the original and marker-cleaned output. All token windows are
processed. CPU replay can have small floating-point or tie differences from
the original MPS semantic run; report them explicitly.

Add `--steps privacy,content,semantic,benchmark` to include runtime measurements.
Timings and peak process memory depend on hardware and cache state; numerical
equality of latency is not expected across machines.

Each replay records input checksums before execution and checks them again
before writing completion. Partial output is not a completed experiment.
Re-evaluating disclosed people is not an additional independent test.

## 3. Train and evaluate a new model

This workflow creates a new experiment. It cannot guarantee the published
weights, cohorts or metrics. Do not tune on the reserved test people.

### Prepare inputs and detector resources

```bash
uv sync --locked --extra data --extra detection --extra ui --extra dev
uv run python scripts/fetch_workua.py --out data/raw/all_resumes.csv
```

Obtain the prepared E3-JSI Ukrainian JSONL and the category lookup separately.
Put them at `data/processed/e3_jsi_uk.jsonl` and
`data/taxonomy/category_map.json`. The JSONL schema is described in
[data resources](../data/README.md). This translation/annotation input is not
regenerated by the training script.

Prepare the local detector selection and training bootstrap:

```bash
uv run python scripts/prepare_resume_resources.py \
  --config configs/baselines/adaptive_resume_v3_3.json
```

The baseline configuration is used to prepare shared detector, corpus and
lexical resources. This command does not train an old adaptive policy and does
not call a paid API. It stops if detector validation fails.

### Reserve splits and fit action models

```bash
uv run python scripts/action_data.py \
  --train-persons 300 --validation-persons 80 --test-persons 60 \
  --reference-size 6000 --reuse-historical-background \
  --out artifacts/evaluations/new_data
uv run python scripts/train_action_resume.py \
  --data-dir artifacts/evaluations/new_data \
  --reference-size 6000 --reference-views 2 --contexts 3 \
  --out artifacts/evaluations/new_offline
uv run python scripts/train_action_onpolicy.py \
  --model artifacts/evaluations/new_offline/model.pkl \
  --data-dir artifacts/evaluations/new_data --contexts 2 \
  --out artifacts/evaluations/new_onpolicy
```

Requested people counts depend on source eligibility and saved experiment
history. Use achieved counts from the manifest. Historical-background reuse
allows development exposure while excluding recorded queries and reserved
test people. It does not make an already evaluated person a fresh participant.
Training uses one seeded query per training person by default. Edited states
do not increase the independent person count.

### Select on validation and test once

Prepare the pinned content encoder as above. Select the working point using
validation privacy and professional similarity:

```bash
uv run python scripts/select_action_operating_point.py \
  --model artifacts/evaluations/new_offline/model.pkl \
  --candidate-model artifacts/evaluations/new_onpolicy/model.pkl \
  --documents artifacts/evaluations/new_offline/validation_documents.pkl \
  --data-dir artifacts/evaluations/new_data \
  --out artifacts/models/new_action_model
uv run python scripts/evaluate_action_resume.py \
  --model artifacts/models/new_action_model/model.pkl \
  --data-dir artifacts/evaluations/new_data --split test \
  --without-previous-risk \
  --offline-action-model artifacts/evaluations/new_offline/model.pkl \
  --onpolicy-action-model artifacts/evaluations/new_onpolicy/model.pkl \
  --out artifacts/evaluations/new_test
```

The historical v4 control is explicitly omitted unless its separate weights
are available. Results remain under the chosen new output paths. To try a
newly trained model in the demo, use a copy of `configs/adaptive_resume.json`
with `risk_model` pointing to that artifact and load it explicitly in the CLI.
The demo reads the canonical config; change it deliberately to deploy a new
model. A newly trained artifact will not pass the frozen thesis SHA check.

## Files for review

Submit source code and instructions from Git. Supply the runtime archive
separately for a runnable demo, and the original research-input bundle only
when the reviewer needs exact corpus replay. Keep `.env`, original resumes,
review quotes, model binaries and generated reports outside the source submission.
