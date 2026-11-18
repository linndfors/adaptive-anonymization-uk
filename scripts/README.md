# Command map

Run commands from the repository root with `uv run python scripts/<name>.py`.
The full sequence and required inputs are in [reproduction](../docs/reproduction.md).

| Workflow | Commands |
|---|---|
| Restore and verify the thesis runtime | `frozen_runtime.py restore`, `prepare-detector`, `check`, `smoke` |
| Anonymize a text file | `anonymize_resume.py` |
| Download new WorkUA input | `fetch_workua.py` |
| Prepare detector/training resources | `prepare_resume_resources.py` |
| Reserve person-disjoint splits | `action_data.py` |
| Audit train/test participant history | `audit_action_history.py` |
| Train action regression | `train_action_resume.py` |
| Add controller-visited training states | `train_action_onpolicy.py` |
| Compare additional learners | `train_action_candidates.py` |
| Select validation operating point | `select_action_operating_point.py` |
| Evaluate held-out data | `evaluate_action_resume.py` |
| Add fixed-order and type-prior controls | `evaluate_action_fixed_controls.py` |
| Source-anchored content metrics | `evaluate_content_provenance.py` |
| Semantic attacker | `download_semantic_attacker.py`, `evaluate_semantic_attacker.py` |
| Replay frozen experiments | `reproduce_frozen.py` |
| Runtime measurements | `benchmark_action_runtime.py` |
| Synthetic quality checks | `check_action_quality.py`, `check_demo_examples.py` |
| Validate source submission | `check_repository.py` |

`action_fingerprints.py` and `group_policy_eval.py` are shared helpers.
The remaining evaluation commands support earlier search/risk baselines,
taxonomy preparation and optional vacancy diagnostics.
They remain for controlled comparisons; the active inference entry point is
`anonymize_resume.py`.
