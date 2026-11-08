# Adaptive Anonymization of Ukrainian Resumes

A research prototype that chooses which parts of a resume to generalize or mask,
using a record-linkage risk proxy and a lexical-retention constraint. The current
controller is **`adaptive_resume_v3.3`**. The Streamlit demo and file-based CLI use
the same implementation and configuration.

```text
Resume → local entity detection → mandatory identifier masking
       → adaptive search over transformations → release checks → annotated result
```

The controller is an optimization algorithm built around pretrained NER, rules,
a category taxonomy and search indexes. It is not a neural model trained from
scratch. Local anonymization does not require OpenAI or an `.env` file.

## Review the project

- [Architecture and source map](docs/architecture.md): components, formulas and decision rules.
- [Evaluation](docs/evaluation.md): cohorts, policies, metrics and reproduction commands.
- [Latest results](results/adaptive_resume_v3.3/README.md): full v3.3 findings and limitations.
- [Setup and troubleshooting](docs/setup.md): data, local artifacts and optional API extraction.
- [Research history](docs/research-history.md): earlier methods retained for comparison.

## Install and run checks

Use **Python 3.12** and `uv`, from the repository root:

```bash
uv sync --locked --extra dev
uv run ruff check src app scripts tests
uv run pytest -q
```

These checks use synthetic fixtures and mocks. They do not require private
WorkUA records, downloaded neural weights or API credentials. GitHub Actions
runs the same checks and builds the package.

## Run the demo

Install the data, detection and interface dependencies:

```bash
uv sync --locked --extra data --extra detection --extra ui --extra dev
```

For the first run, download WorkUA and prepare the local inference resources:

```bash
uv run python scripts/fetch_workua.py --out data/raw/all_resumes.csv
uv run python scripts/prepare_resume_resources.py
```

Preparation validates the detector, constructs the reference corpus and fits the
lexical vectorizer. It reuses complete existing artifacts. Initial preparation
requires internet access for data and pretrained NER weights and can take time;
the source CSV is parsed before sampling. See [setup](docs/setup.md) for details.

Check the artifact files and start Streamlit:

```bash
uv run python scripts/prepare_resume_resources.py --check
uv run streamlit run app/text_demo.py
```

Paste a resume or load a UTF-8 `.txt` file, then select **Анонімізувати**.
The interface intentionally remains Ukrainian. It shows the transformed text,
exact edit highlighting, a download button and diagnostics. Corpus results are
available in the collapsed quality panel.

For file-based processing:

```bash
uv run python scripts/anonymize_resume.py \
  --input tests/fixtures/resume_prose_regression.txt \
  --output artifacts/anonymized-example.txt
```

The CLI refuses to overwrite an existing output. Its optional `--audit` file
contains original fragments and should stay under the ignored `artifacts/` path.

## Current evidence

The fresh v3.3 test contains **385 resumes from 192 people**, with a background
pool of **12,000 resumes**. People from previous saved pools were excluded.
The policy parameters were frozen before this test.

| Metric | Fixed L2 | Adaptive v3.3 |
|---|---:|---:|
| BM25 linkage Recall@1 ↓ | 66.0% | 48.1% |
| Character TF-IDF linkage Recall@1 ↓ | 63.1% | 47.0% |
| Lexical similarity ↑ | 63.8% | 68.7% |
| Professional-field embedding cosine ↑ | 0.869 | 0.871 |

The reduction in linkage success and increase in lexical similarity are supported
by person-level paired bootstrap intervals. The professional semantic difference
is inconclusive: +0.0015, 95% interval −0.0263 to +0.0301. Earlier v3.2 numbers
refer to another version and must not be attributed to the current implementation.

Important limitations:

- The search score is a relative margin, not a re-identification probability.
  The internal target was met for only **115/385** queries; linkage attacks
  sometimes still succeeded among those releases.
- Embedding similarity is not factual correctness. An assisted review of 12
  people found unsupported category assignments in two adaptive outputs and
  ambiguous generalizations in three more. This was not independent human annotation.
- The evaluation uses serialized WorkUA fields with mixed-language content,
  not arbitrary PDF layouts or an exclusively Ukrainian natural-prose corpus.
- Historical L2/L3 use a different renderer. Same-parser controls are included
  in the full report. Stronger fixed policies trade more information for lower linkage.

The software does not guarantee anonymity. The [full report](results/adaptive_resume_v3.3/README.md)
includes all seven policies, regression results, semantic diagnostics and known errors.

## Repository layout

```text
app/text_demo.py          Current Streamlit interface
configs/adaptive_resume.json
                         Current controller settings and resource-profile reference
src/anonrisk_uk/          Runtime package and research implementations
scripts/                 Resource setup, anonymization and evaluation commands
tests/                   Unit tests and synthetic regression fixtures
docs/                    Current English documentation and a frozen evaluation rubric
results/                 Aggregate evaluation evidence; no real resume texts or IDs
data/                    Synthetic corpus, category map and provenance notes
notebooks/               Optional historical research notebooks
artifacts/               Local data products, weights and individual outputs; Git-ignored
```

Ukrainian strings in entity rules, the category taxonomy, UI and fixtures are
functional language data. Explanatory documentation is in English; the original
review rubric is retained unchanged to preserve its recorded evaluation hash.

## Data and reproducibility

WorkUA records, model binaries, detector caches, budgets and `.env` files are not
committed. See [data provenance](data/README.md). Newly prepared resources make
the demo runnable but do not reproduce the published experiment by themselves;
exact replay also requires the recorded code, input resources and private pool IDs.
[Evaluation instructions](docs/evaluation.md) distinguish a fresh test from replay.

## License

No source-code license has been selected. Dataset and pretrained-model licenses
apply independently; their inclusion or use does not license the project code.
