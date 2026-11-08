# Data and derived resources

## Included in the repository

- `processed/e3_jsi_uk.jsonl`: translated synthetic E3-JSI span-detection data.
- `taxonomy/category_map.json`: cached assignments of frequent professional
  values to the fixed category hierarchy. This derived lookup is required by
  local semantic generalization; it is not a record-level resume dataset.
- `taxonomy/manifest.json` and `taxonomy/validation*`: construction provenance,
  coverage and taxonomy-review diagnostics. An earlier offline LLM-assisted
  assignment step produced the lookup; inference uses the saved map locally.
- `metadata/vacancies_sample.manifest.json`: provenance of the historical
  vacancy sample, without its record-level contents.

Taxonomy assignments include known contextual mistakes. Coverage and category
validation are not proof of factual preservation; see the
[current evaluation](../results/adaptive_resume_v3.3/README.md).

## External datasets

WorkUA resume and vacancy records are not committed. Download them locally:

```bash
uv run python scripts/fetch_workua.py --out data/raw/all_resumes.csv
uv run python scripts/fetch_vacancies.py \
  --out data/raw/vacancies_sample.ndjson --limit 10000 --seed 42
```

Vacancies are used only by historical downstream-utility experiments, not by the
current resume demo. Source dataset names and extraction parameters are recorded
by the fetch scripts. The `data/raw/` directory and generated manifests remain
Git-ignored. Upstream dataset terms apply independently of the project code.

The current vacancy fetcher scans the full source and uses a seeded hash of the
vacancy ID. The historical 10,000-vacancy run used `first_n_not_random`; refetching
with the current sampler does not reproduce that old table.

WorkUA sampling preserves whole person clusters. The full CSV is parsed before
sampling, so a small pool reduces model/index costs but not the initial load.
Raw records, per-person keys, detector caches and API-response caches must not
be added to either Git or source distributions.
