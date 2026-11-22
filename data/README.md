# Data resources

Training requires the WorkUA dataset, prepared Ukrainian entity annotations
and a category lookup. These inputs are kept outside Git.

All inputs and generated resources are external and Git-ignored:

| Input | Local path | Used by |
|---|---|---|
| WorkUA CSV | `data/raw/all_resumes.csv` | New training and corpus evaluation |
| Prepared E3-JSI Ukrainian JSONL | `data/processed/e3_jsi_uk.jsonl` | Detector preparation and evaluation |
| Category lookup | `data/taxonomy/category_map.json` | Local transformation rules |

Use a prepared E3-JSI Ukrainian JSONL file. Each record has
`source_row_id`, `text_uk` and `entities_uk`; annotations contain character
offsets, entity text and type labels. Changing translations, annotation repairs
or source data changes the detector evaluation.

For a new experiment, download WorkUA with:

```bash
uv run python scripts/fetch_workua.py --out data/raw/all_resumes.csv
```

The command requires the `data` extra.

The category lookup has an `entries` dictionary. Keys are formed as
`LABEL::normalized value`; entries contain `label`, `value` and the level-one
category ID `l1`. Category IDs are defined in
`src/anonrisk_uk/category_taxonomy.py`. The offline builder
`scripts/build_category_taxonomy.py` supports `collect`, `classify` and `map`
stages. Its `classify` stage uses a paid OpenAI API and requires explicit
`--allow-paid` authorization and credentials. An existing prepared lookup can
be used without running that stage.

Once training is complete, inference needs the trained model, local detector
and category lookup. It does not read the WorkUA CSV or E3-JSI documents.
See the [main instructions](../README.md) for the command sequence.

## Optional UI corpus

Enabling the corpus toggle automatically loads `data/corpus/default.jsonl`,
if available. This local JSONL uses the same `id`, `text` and optional `person_id`
fields as uploads and remains Git-ignored. It can hold a prepared WorkUA
background; it is demonstration data, not a new independent evaluation.
Without that file, the app uses its six authored examples and labels the corpus
as **Демо**. Selecting **Власний** replaces the default with uploaded files.

Upload CSV, JSON, JSONL or individual UTF-8 TXT resumes. A minimal CSV has
`id,text`; JSON records have the same fields. Optional `person_id` identifies
multiple resumes belonging to one person. Structured WorkUA rows can also be
uploaded. The UI supplies a synthetic CSV example.

Limits: 50 MB total, at most 6000 unique records and 20,000 characters per
resume. Limits and skipped duplicates are reported explicitly. Uploaded corpus
texts and indexes are held in the current session, not written to the project.
The same deterministic content view is applied to both query variants and
every corpus record: known service masks, recognized field and section headings,
and punctuation-only formatting are removed. Field values and narrative remain.
This prevents empty fields such as `Email: [EMAIL]` from identifying an otherwise
masked input merely because the background uses different headings. Exact-copy
consolidation still uses the original text, so distinct records with identical
normalized content remain real retrieval ties.

For ordinary input without a declared link (S1), it is added once as the target. Exact normalized copies
already in the background are consolidated into that one target. Both attacks
are fitted on this pool once and use the same indexes for the original and all
anonymized variants. The UI reports the largest expected reciprocal rank (RR)
over BM25 and character TF-IDF; ranks and Recall@1 remain available in details.
RR is 1 for a unique first-place target; without ties it is 1/rank. With ties,
the exact expected reciprocal rank is used, not 1/expected_rank. A fully empty
query receives the random-ranking baseline rather than zero risk.
RR stays 1 as long as at least one attacker ranks the target first. The main UI
metric reports before/after RR and its relative change; this percentage is a
change in ranking score, not an identification probability. Values come
directly from the same fixed index and may increase or decrease for a new input.
Details show target/competitor scores
and signed relative separation `(target - competitor)/(abs(target) + abs(competitor))`.
These explain rank plateaus; decreasing similarity alone does not establish
successful prevention of linkage. UI content-view scores are separate from
previous whole-text results and from the frozen thesis S2 evaluation.

Selecting a declared related resume switches to S2: the input is not inserted,
exact copies of it are excluded, and `person_id` expands the target to the
known person's records. S1 and S2 answer different attack questions. Neither
is a calibrated probability of identification outside the supplied corpus.

The six synthetic S2 showcase pairs are declared in `app/demo_examples.py`.
Selecting one enables the standard background and supplies one explicitly
fictional related target in session memory. This does not change the background
file, model decisions or the attack calculation. Selecting S1 or editing the
input removes the paired target from scoring. Their outcomes were selected
after inspecting the current model and prepared background; they are curated
demonstrations and must not be counted as an independent evaluation sample.
