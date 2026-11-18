# Data resources

The category lookup is an external runtime resource. The trusted model archive
restores it to `data/taxonomy/category_map.json` with a verified checksum.
It contains term-to-category assignments used by the transformation rules.

All inputs and generated resources are external and Git-ignored:

| Input | Local path | Used by |
|---|---|---|
| WorkUA CSV | `data/raw/all_resumes.csv` | New training and corpus evaluation |
| Prepared E3-JSI Ukrainian JSONL | `data/processed/e3_jsi_uk.jsonl` | Detector preparation and evaluation |
| Category lookup | `data/taxonomy/category_map.json` | Local transformation rules |
| Original split and evaluation bundle | `artifacts/` | Exact experiment reproduction |

The original prepared E3-JSI file is supplied separately. Each record has
`source_row_id`, `text_uk` and `entities_uk`; annotations contain character
offsets, entity text and type labels. Changing translations, annotation repairs
or source data changes the detector evaluation.

For a new experiment, download WorkUA with:

```bash
uv run python scripts/fetch_workua.py --out data/raw/all_resumes.csv
```

The command requires the `data` extra. Original input hashes and saved person
splits are needed to reproduce an existing experiment. A fresh download cannot
replace them while claiming the same results.

Inference with the trained runtime needs no WorkUA CSV or E3-JSI documents.
See [reproduction](../docs/reproduction.md) for installation and input delivery.
