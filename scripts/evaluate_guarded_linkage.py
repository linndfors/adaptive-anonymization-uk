"""Fresh-person S2 diagnostic. Save protocol before inference; never overwrite old experiments.

Controls: historical L2/L3 and frozen adaptive v2, plus identical-parser/tool
direct-only and all-quasi controls. This bounded diagnostic is not a powered
confirmatory experiment. Ground-truth person links are used ONLY for scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from adaptive_vs_levels_holdout import Adaptive
from group_policy_eval import holdout_pool, person_bootstrap
from sklearn.feature_extraction.text import TfidfVectorizer
from threadpoolctl import threadpool_limits

from anonrisk_uk.attack import BM25Index, TfidfIndex, ranking_result
from anonrisk_uk.demo import load_demo_resources
from anonrisk_uk.experiment import Population
from anonrisk_uk.policy import fixed_policies
from anonrisk_uk.resume_adaptive import compare_resume
from anonrisk_uk.resume_parser import extractor_from_env
from anonrisk_uk.workua import norm

ROOT = Path(__file__).resolve().parents[1]


def used_ids(root):
    """Exclude all historical background pools, not merely training query IDs."""
    used, sources = set(), []
    for path in sorted((root / "artifacts/runs").glob("*/workua/bundle.pkl")):
        with path.open("rb") as handle:
            used.update(str(i) for i in pickle.load(handle)["ids"])
        sources.append(str(path.relative_to(root)))
    paths = set((root / "artifacts/evaluations").glob("**/pool_ids.csv"))
    paths.update((root / "artifacts/evaluations").glob("**/holdout_pool_ids.csv"))
    for path in sorted(paths):
        used.update(pd.read_csv(path, dtype=str)["id"])
        sources.append(str(path.relative_to(root)))
    return used, sources


def core_survival(ex, output):
    values = [v for g in ("skills", "title", "exp_position") for v in ex.get(g, []) if norm(v)]
    normalized = norm(output)
    return np.mean([norm(v) in normalized for v in values]) if values else np.nan


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    resources = load_demo_resources(args.config)
    used, sources = used_ids(ROOT)
    ids, extracted, clusters, excluded = holdout_pool(ROOT / "data/raw/all_resumes.csv", used,
                                                    args.pool_size, args.seed)
    assert not set(map(str, ids)) & used
    texts = [resources.codec.render(ex, {}) for ex in extracted]
    attacks = {"bm25": BM25Index(max_features=20000).fit(texts),
               "char": TfidfIndex(max_features=60000).fit(texts)}
    clusters = np.array(clusters)
    pop = Population(ids, extracted, clusters, resources.codec, attacks["bm25"], texts)
    linked = pop.linked_positions()
    eligible = [p for p in sorted(set(clusters[linked]))
                if all(0 < len(texts[k]) <= 20000 for k in np.flatnonzero(clusters == p))]
    selected = set(np.random.default_rng(args.seed).permutation(eligible)[:args.max_persons])
    queries = np.array([int(k) for k in linked if clusters[k] in selected])
    if len(selected) < 2:
        raise ValueError("Too few linked persons; increase pool-size")
    resources.background_texts = texts
    resources.retention_vectorizer = TfidfVectorizer(sublinear_tf=True).fit(texts)
    resources.feature_index = attacks["bm25"]
    frozen_path = ROOT / "artifacts/evaluations/group_policy_dev_v2/frozen.json"
    frozen = json.loads(frozen_path.read_text())
    extractor = extractor_from_env(ROOT) if args.openai else None
    protocol = {"purpose": "fresh-person S2 diagnostic; no model selection on this result",
                "seed": args.seed, "pool_size": len(ids), "queries": len(queries), "persons": len(selected),
                "excluded_historical_persons": excluded, "excluded_sources": sources,
                "model": extractor.model if extractor else None, "target": 0.0, "utility_floor": .45,
                "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [
                    Path(__file__), ROOT / "src/anonrisk_uk/resume_adaptive.py",
                    ROOT / "src/anonrisk_uk/resume_parser.py", frozen_path]},
                "privacy_margin": .05, "bootstrap_unit": "person", "bootstrap_samples": 2000}
    (args.out / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2))
    pd.DataFrame({"id": ids}).to_csv(args.out / "pool_ids.csv", index=False)
    old = Adaptive(resources.codec, extracted, texts, attacks, resources.retention_vectorizer)
    policies = fixed_policies()
    released = {k: [] for k in ("L0", "historical_L2", "historical_L3", "historical_v2",
                                "guarded", "matched_direct_only", "matched_all_quasi")}
    statuses = []
    for i, k in enumerate(queries):
        result = compare_resume(resources, texts[k], extractor=extractor, indexes=attacks)
        released["L0"].append(texts[k])
        released["historical_L2"].append(resources.codec.render(extracted[k], policies["L2_generalize"]))
        released["historical_L3"].append(resources.codec.render(extracted[k], policies["L3_strong"]))
        released["historical_v2"].append(old.render(k, old.release(k, frozen["taus"]["q25"])))
        for name, key in (("guarded", "adaptive_guarded"), ("matched_direct_only", "direct_only"),
                          ("matched_all_quasi", "quasi_generalized")):
            released[name].append(result.releases[key].text)
        statuses.append(result.diagnostics["stop_reason"])
        print(f"{i + 1}/{len(queries)}", flush=True)
    targets, excludes = pop.targets_and_excludes(queries, "S2")
    frame = pd.DataFrame({"id": [ids[k] for k in queries], "person": clusters[queries], "status": statuses})
    summary = []
    for name, outputs in released.items():
        row = {"method": name}
        for attacker, index in attacks.items():
            result = ranking_result(index, outputs, targets, excludes)[0]
            values = result.recalls[1]
            frame[f"{name}_{attacker}_r1"] = values
            row[f"{attacker}_recall_at_1"] = float(np.mean(values))
        core = [core_survival(extracted[k], output) for k, output in zip(queries, outputs)]
        frame[f"{name}_core"] = core
        row["core_verbatim_survival"] = float(np.nanmean(core))
        summary.append(row)
    differences = {}
    for control in ("historical_L2", "historical_v2", "matched_all_quasi"):
        for metric in ("bm25_r1", "char_r1", "core"):
            differences[f"vs_{control}_{metric}"] = person_bootstrap(
                frame[f"guarded_{metric}"] - frame[f"{control}_{metric}"], clusters[queries])
    report = {**protocol, "summary": summary, "paired_differences_95ci": differences,
              "statuses": pd.Series(statuses).value_counts().to_dict(),
              "noninferiority_vs_L2_supported": all(differences[f"vs_historical_L2_{m}"][2] <= .05
                                                   for m in ("bm25_r1", "char_r1"))}
    frame.to_csv(args.out / "per_query.csv", index=False)
    pd.DataFrame(summary).to_csv(args.out / "summary.csv", index=False)
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(pd.DataFrame(summary).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/research_entity_policy_v4c_local.json")
    parser.add_argument("--pool-size", type=int, default=4000)
    parser.add_argument("--max-persons", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--openai", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args)


if __name__ == "__main__":
    main()
