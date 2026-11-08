"""Canonical adaptive model: development selection or frozen fresh-person S2 test. Save protocol before inference; never overwrite old experiments.

Controls: historical L2/L3 and frozen adaptive v2, plus identical-parser/tool
direct-only and all-quasi controls. This bounded diagnostic is not a powered
confirmatory experiment. Ground-truth person links are used ONLY for scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from adaptive_vs_levels_holdout import Adaptive
from group_policy_eval import holdout_pool, person_bootstrap
from sklearn.feature_extraction.text import TfidfVectorizer
from threadpoolctl import threadpool_limits

from anonrisk_uk.attack import BM25Index, TfidfIndex, ranking_result
from anonrisk_uk.experiment import Population
from anonrisk_uk.policy import fixed_policies
from anonrisk_uk.resume_adaptive import VERSION, NeighbourProxy, compare_resume
from anonrisk_uk.resume_parser import parse_resume
from anonrisk_uk.resume_resources import load_resume_resources
from anonrisk_uk.workua import build_person_clusters, extract, load_workua, norm

ROOT = Path(__file__).resolve().parents[1]


def used_ids(root, ignore_pool=None):
    """Exclude all historical background pools, not merely training query IDs."""
    used, sources = set(), []
    for path in sorted((root / "artifacts/runs").glob("*/workua/bundle.pkl")):
        with path.open("rb") as handle:
            used.update(str(i) for i in pickle.load(handle)["ids"])
        sources.append(str(path.relative_to(root)))
    paths = set((root / "artifacts/evaluations").glob("**/pool_ids.csv"))
    paths.update((root / "artifacts/evaluations").glob("**/holdout_pool_ids.csv"))
    for path in sorted(paths):
        if ignore_pool is not None and path.resolve() == ignore_pool.resolve():
            continue
        used.update(pd.read_csv(path, dtype=str)["id"])
        sources.append(str(path.relative_to(root)))
    return used, sources


def core_survival(ex, output):
    values = [v for g in ("skills", "title", "exp_position") for v in ex.get(g, []) if norm(v)]
    normalized = norm(output)
    return np.mean([norm(v) in normalized for v in values]) if values else np.nan


def replay_pool(path):
    """Reconstruct exactly a previously used pool for a labelled regression run."""
    ids = pd.read_csv(path, dtype=str)["id"].tolist()
    if len(ids) != len(set(ids)):
        raise ValueError("Replay pool contains duplicate IDs")
    frame = load_workua(ROOT / "data/raw/all_resumes.csv").set_index("id", drop=False).loc[ids].reset_index(drop=True)
    extracted = [extract(row) for row in frame.to_dict("records")]
    return ids, extracted, build_person_clusters(frame).astype(str).to_numpy()


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    resources = load_resume_resources(args.config)
    if args.purpose == "regression":
        if not args.reuse_pool:
            raise ValueError("Regression requires an explicit previously used pool")
        ids, extracted, clusters = replay_pool(args.reuse_pool)
        sources, excluded = [], None
    else:
        if args.reuse_pool and args.purpose != "development":
            raise ValueError("Fresh final test cannot reuse an already-seen pool")
        used, sources = used_ids(ROOT, args.reuse_pool)
        ids, extracted, clusters, excluded = holdout_pool(ROOT / "data/raw/all_resumes.csv", used,
                                                        args.pool_size, args.seed)
        assert not set(map(str, ids)) & used
        if args.reuse_pool:
            assert list(map(str, ids)) == pd.read_csv(args.reuse_pool, dtype=str)["id"].tolist()
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
    if not frozen_path.is_file():
        frozen_path = ROOT / "configs/baselines/adaptive_v2.json"
    frozen = json.loads(frozen_path.read_text())
    floors = args.utility_floors or [resources.config["adaptive"]["utility_floor"]]
    if args.purpose in {"test", "regression"} and args.utility_floors is not None:
        raise ValueError("Final test uses the frozen config; parameter overrides are development-only")
    protocol = {"purpose": args.purpose, "method": VERSION, "utility_floors": floors,
                "reused_development_pool": str(args.reuse_pool) if args.reuse_pool else None,
                "evaluation_note": "previously used holdout; regression only, no parameter selection"
                if args.purpose == "regression" else "development selection or fresh frozen final test",
                "selection_rule": "highest lexical utility with both dev recalls <= L2+0.05; otherwise minimum worst recall",
                "seed": args.seed, "pool_size": len(ids), "queries": len(queries), "persons": len(selected),
                "excluded_historical_persons": excluded, "excluded_sources": sources,
                "model": VERSION, "target": resources.config["adaptive"]["target"],
                "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
                "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [
                    Path(__file__), ROOT / "src/anonrisk_uk/resume_adaptive.py",
                    ROOT / "src/anonrisk_uk/resume_parser.py",
                    ROOT / "src/anonrisk_uk/resume_context.py", frozen_path]},
                "privacy_margin": .05, "bootstrap_unit": "person", "bootstrap_samples": 2000,
                "runtime": {"python": sys.version, "platform": platform.platform(),
                            "packages": {name: version(name) for name in
                                         ("numpy", "scikit-learn", "torch", "transformers", "spacy")}},
                "release_records": "releases.jsonl; local identifying data, never publish",
                "semantic_evaluation": "separate post-hoc evaluator; no semantic scores used for policy selection"}
    protocol["source_sha256"].update({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in sorted((ROOT / "src/anonrisk_uk").glob("*.py"))})
    (args.out / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2))
    (args.out / "model_config.json").write_bytes(Path(args.config).read_bytes())
    pd.DataFrame({"id": ids}).to_csv(args.out / "pool_ids.csv", index=False)
    old = Adaptive(resources.codec, extracted, texts, attacks, resources.retention_vectorizer)
    policies = fixed_policies()
    released = {k: [] for k in ("L0", "historical_L2", "historical_L3", "historical_v2",
                                "matched_direct_only", "matched_all_quasi")}
    for floor in floors:
        released[f"adaptive_{floor:g}"] = []
    statuses, risks, direct_checks = [], [], []
    records = args.out / "releases.jsonl"
    records.touch(exist_ok=False)
    for i, k in enumerate(queries):
        document = parse_resume(texts[k], resources.codec, resources.detector)
        proxy = NeighbourProxy(attacks, texts, texts[k])
        for floor in floors:
            resources.config["adaptive"]["utility_floor"] = floor
            result = compare_resume(resources, texts[k], indexes=attacks, document=document, proxy=proxy)
            released[f"adaptive_{floor:g}"].append(result.releases["adaptive_guarded"].text)
        released["L0"].append(texts[k])
        released["historical_L2"].append(resources.codec.render(extracted[k], policies["L2_generalize"]))
        released["historical_L3"].append(resources.codec.render(extracted[k], policies["L3_strong"]))
        released["historical_v2"].append(old.render(k, old.release(k, frozen["taus"]["q25"])))
        for name, key in (("matched_direct_only", "direct_only"),
                          ("matched_all_quasi", "quasi_generalized")):
            released[name].append(result.releases[key].text)
        statuses.append(result.diagnostics["stop_reason"])
        risks.append(result.diagnostics["metrics"]["risk_after"])
        direct_checks.append(result.diagnostics["direct_check"]["status"])
        with records.open("a") as handle:
            handle.write(json.dumps({"id": str(ids[k]), "person": str(clusters[k]),
                                     "original": texts[k], "status": statuses[-1], "risk": risks[-1],
                                     "direct_check": direct_checks[-1],
                                     "outputs": {name: values[-1] for name, values in released.items()}},
                                    ensure_ascii=False) + "\n")
        print(f"{i + 1}/{len(queries)}", flush=True)
    targets, excludes = pop.targets_and_excludes(queries, "S2")
    frame = pd.DataFrame({"id": [ids[k] for k in queries], "person": clusters[queries], "status": statuses,
                          "proxy_risk": risks, "direct_check": direct_checks})
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
        ret = np.array([float((resources.retention_vectorizer.transform([text]) @
                       resources.retention_vectorizer.transform([texts[k]]).T).toarray()[0, 0])
                        for k, text in zip(queries, outputs)])
        frame[f"{name}_ret"] = ret
        row["lexical_retention"] = float(ret.mean())
        summary.append(row)
    candidates = [r for r in summary if r["method"].startswith("adaptive_")]
    l2 = next(r for r in summary if r["method"] == "historical_L2")
    eligible = [r for r in candidates if r["bm25_recall_at_1"] <= l2["bm25_recall_at_1"] + .05
                and r["char_recall_at_1"] <= l2["char_recall_at_1"] + .05]
    chosen = (max(eligible, key=lambda r: (r["lexical_retention"], r["core_verbatim_survival"])) if eligible
              else min(candidates, key=lambda r: (max(r["bm25_recall_at_1"], r["char_recall_at_1"]),
                                                 -r["lexical_retention"])))
    name = chosen["method"]
    differences = {}
    for control in ("historical_L2", "historical_v2", "matched_all_quasi"):
        for metric in ("bm25_r1", "char_r1", "core", "ret"):
            differences[f"vs_{control}_{metric}"] = person_bootstrap(
                frame[f"{name}_{metric}"] - frame[f"{control}_{metric}"], clusters[queries])
    report = {**protocol, "summary": summary, "selected": name,
              "paired_differences_95ci": differences,
              "statuses_last_setting": pd.Series(statuses).value_counts().to_dict(),
              "direct_pattern_checks": pd.Series(direct_checks).value_counts().to_dict(),
              "noninferiority_vs_L2_supported": all(differences[f"vs_historical_L2_{m}"][2] <= .05
                                                   for m in ("bm25_r1", "char_r1"))}
    frame.to_csv(args.out / "per_query.csv", index=False)
    pd.DataFrame(summary).to_csv(args.out / "summary.csv", index=False)
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(pd.DataFrame(summary).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/adaptive_resume.json")
    parser.add_argument("--pool-size", type=int, default=12000)
    parser.add_argument("--max-persons", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--purpose", choices=["development", "test", "regression"], required=True)
    parser.add_argument("--utility-floors", type=float, nargs="+")
    parser.add_argument("--reuse-pool", type=Path, help="Explicit development replay or labelled regression pool")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args)


if __name__ == "__main__":
    main()
