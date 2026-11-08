"""Compare completed local/LLM action runs and export a human review worksheet."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RATINGS = ["meaning_preserved_0_2", "no_invented_facts_0_2", "specificity_reduced_0_2",
           "professional_value_0_2", "grammar_0_2", "acceptable_yes_no", "reviewer", "notes"]


def load_run(config_path):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    out = Path(config["out"])
    if not out.is_absolute():
        out = config_path.parent.parent / out
    state = json.loads((out / "cpu_pipeline_state.json").read_text(encoding="utf-8"))
    if state["stages"]["pipeline"]["status"] != "completed":
        raise ValueError(f"Run has not completed: {out.name}")
    pipeline = out / "pipeline"
    with (pipeline / "release_queries.pkl").open("rb") as handle:
        releases = pickle.load(handle)
    frames = pd.read_csv(pipeline / "test_per_query.csv")
    return config, out, releases, frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", type=Path, default=ROOT / "configs/research_actions_smoke.json")
    parser.add_argument("--llm", type=Path, default=ROOT / "configs/research_actions_openai.json")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/evaluations/actions_v6_local_vs_openai")
    args = parser.parse_args()
    local, llm = load_run(args.local.resolve()), load_run(args.llm.resolve())
    ignored = {"out", "allow_paid_api", "llm_provider", "max_stage_seconds"}
    if {k: v for k, v in local[0].items() if k not in ignored} != {k: v for k, v in llm[0].items() if k not in ignored}:
        raise ValueError("Conditions differ beyond the LLM tool/output/time budget; not a controlled comparison")
    if list(local[2]["positions"]) != list(llm[2]["positions"]):
        raise ValueError("Conditions use different test positions")
    if local[2]["queries"]["L0_none"] != llm[2]["queries"]["L0_none"]:
        raise ValueError("Original test texts differ")
    local_split = pd.read_csv(local[1] / "workua/split.csv", dtype=str)
    llm_split = pd.read_csv(llm[1] / "workua/split.csv", dtype=str)
    pd.testing.assert_frame_equal(local_split, llm_split)
    signatures = [json.loads((run[1] / "pipeline/action_signature.json").read_text()) for run in (local, llm)]
    if signatures[0]["code"] != signatures[1]["code"]:
        raise ValueError("Source code differs between conditions")
    args.out.mkdir(parents=True, exist_ok=True)
    keys = ["policy", "attacker", "scenario", "cohort"]
    summaries = [pd.read_csv(run[1] / "pipeline/test_summary.csv") for run in (local, llm)]
    comparison = summaries[0].merge(summaries[1], on=keys, suffixes=("_local", "_llm"), validate="one_to_one")
    for metric in ("recall@1", "mrr", "retention", "content_retention"):
        comparison[f"{metric}_delta_llm_minus_local"] = comparison[f"{metric}_llm"] - comparison[f"{metric}_local"]
    comparison.to_csv(args.out / "metrics_comparison.csv", index=False)
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from anonrisk_uk.statistics import paired_differences

    pairs = []
    for (policy, attacker), baseline in local[3].groupby(["policy", "attacker"]):
        treated = llm[3][llm[3].policy.eq(policy) & llm[3].attacker.eq(attacker)]
        pairs.append(paired_differences(treated, baseline, n=100, seed=local[0]["seed"])
                     .assign(policy=policy, attacker=attacker, comparison="llm_minus_local"))
    pd.concat(pairs, ignore_index=True).to_csv(args.out / "tool_paired_differences.csv", index=False)
    worksheet = []
    for condition, run in (("local", local), ("openai", llm)):
        releases, frames = run[2], run[3]
        persons = frames.groupby("pos").person.first()
        for i, pos in enumerate(releases["positions"]):
            original = releases["queries"]["L0_none"][i]
            for policy, texts in releases["queries"].items():
                if policy == "L0_none":
                    continue
                digest = hashlib.sha256((original + "\n" + texts[i]).encode()).hexdigest()
                worksheet.append({"condition": condition, "pos": int(pos), "person": str(persons.loc[pos]),
                                  "policy": policy, "text_sha256": digest, "original": original, "result": texts[i],
                                  **{name: "" for name in RATINGS}})
    sheet = pd.DataFrame(worksheet)
    review_path = args.out / "human_review.csv"
    if review_path.exists():
        old = pd.read_csv(review_path, keep_default_na=False)
        index = ["condition", "pos", "policy", "text_sha256"]
        # Retain ratings only for byte-identical old/new examples.
        old = old.set_index(index)
        for i, row in sheet.iterrows():
            key = tuple(row[name] for name in index)
            if key in old.index:
                sheet.loc[i, RATINGS] = old.loc[key, RATINGS].to_numpy()
    sheet.to_csv(review_path, index=False)
    budget_path = llm[1] / "openai_budget.json"
    budget = json.loads(budget_path.read_text()) if budget_path.exists() else {}
    usage = [r["usage"] for r in budget.get("requests", []) if "usage" in r]
    manifest = {"role": "small_development_comparison_not_final_evidence",
                "local": str(local[1]), "llm": str(llm[1]), "test_queries": len(local[2]["positions"]),
                "test_people": local[3].person.nunique(), "matching_inputs_splits_code": True,
                "delta_direction": "LLM minus local; lower recall/mrr and higher retention are favorable",
                "api_attempts_reserved": budget.get("calls", 0), "api_reservations_usd": budget.get("reserved_usd", 0),
                "completed_calls_with_usage": len(usage), "review_rows": len(sheet),
                "review_scale": "0=failed, 1=partial/uncertain, 2=passed; blank=unreviewed"}
    (args.out / "comparison_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Human review: {review_path}")


if __name__ == "__main__":
    main()
