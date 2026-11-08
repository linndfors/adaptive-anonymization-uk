"""Adaptive v2 vs the hand-designed levels L2/L3 on a fresh pool of people (holdout v2).

Method (frozen on dev, `artifacts/evaluations/group_policy_dev_v2/`): per document, greedy
keep -> generalize -> remove over groups (skills are never removed). Each step maximises the drop
of the summed proxy margin of two attackers (BM25 and char-TF-IDF, each with its own top-3
non-self neighbours of the original) per unit of lexical retention lost. A risk gate releases the
document unchanged when its initial worst proxy margin is below TAU; otherwise the first state
whose worst margin is <= 0 is released (the last state if none).

Pool: complete persons from the full CSV with no resume in the dev pool, holdout v1 or dev2.
Queries: every linked resume; targets: the person's other resumes (S2). Bootstrap resamples persons.

Primary hypotheses (registered before the run; the protocol passes only if all three hold):
  H1 privacy non-inferiority: upper 95% CI of recall@1(adaptive) - recall@1(L2) <= +0.05,
     for BM25 and char-TF-IDF.
  H2 utility superiority: lower 95% CI of core survival(adaptive) - core survival(L2) > 0
     (per-resume mean share of skills, desired title and past positions kept verbatim).
  H3 less unnecessary damage: the same as H2 on queries that neither attacker finds at rank 1
     in the original (L0).
Secondary, descriptive: retention difference vs L2, the comparison with L3, and the gap to the
envelope of fixed policies (L0-L3, the 20 global policies fitted on dev, pairwise mixtures).
The envelope claim already failed on dev2 and is not tested here.

  --smoke  runs on 30 dev queries (already seen data) to check the pipeline; never on the holdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from group_policy_eval import holdout_pool, log, person_bootstrap
from sklearn.feature_extraction.text import TfidfVectorizer

from anonrisk_uk.attack import BM25Index, TfidfIndex, ranking_result
from anonrisk_uk.experiment import Population
from anonrisk_uk.policy import fixed_policies, next_op
from anonrisk_uk.workua import GROUPS, norm

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "artifacts/evaluations/group_policy_dev_v2/frozen.json"
USED_POOLS = [
    ROOT / "artifacts/evaluations/group_policy_holdout_v1/holdout_pool_ids.csv",
    ROOT / "artifacts/evaluations/group_policy_dev2_v2/pool_ids.csv",
]
PROTECT = {"skills"}
NEIGHBOR_K = 3
RETENTION_EPS = 0.01
TARGET = 0.0
NONINFERIORITY = 0.05
CORE_GROUPS = ["skills", "title", "exp_position"]
MIX_GRID = np.linspace(0, 1, 51)


class Adaptive:
    def __init__(self, codec, extracted, texts, attacks, retvec):
        self.codec, self.extracted, self.texts, self.attacks, self.retvec = codec, extracted, texts, attacks, retvec

    def render(self, k, policy):
        return self.codec.render(self.extracted[k], policy)

    def retention(self, k, queries):
        return np.asarray((self.retvec.transform(queries) @ self.retvec.transform([self.texts[k]]).T).todense()).ravel()

    def margins(self, k, queries, targets):
        """Normalised margin (best target - best other) per attacker and query."""
        rows = []
        for name, index in self.attacks.items():
            s = index.scores(queries)
            s[:, k] = -np.inf
            best = s[:, targets[name]].max(axis=1)
            s[:, targets[name]] = -np.inf
            rows.append((best - s.max(axis=1)) / np.maximum(np.abs(best), 1e-9))
        return np.stack(rows)

    def neighbours(self, k):
        out = {}
        for name, index in self.attacks.items():
            s = index.scores([self.texts[k]])[0]
            s[k] = -np.inf
            out[name] = list(np.argsort(-s)[:NEIGHBOR_K])
        return out

    def release(self, k, tau):
        targets = self.neighbours(k)
        policy, r0 = {}, 1.0
        m0 = self.margins(k, [self.texts[k]], targets)[:, 0]
        if m0.max() < tau:
            return policy
        while m0.max() > TARGET:
            cands = []
            for g in GROUPS:
                op = next_op(g, policy.get(g, "keep")) if self.extracted[k].get(g) else None
                if op and not (g in PROTECT and op == "remove"):
                    cands.append({**policy, g: op})
            if not cands:
                break
            queries = [self.render(k, p) for p in cands]
            m, r = self.margins(k, queries, targets), self.retention(k, queries)
            j = int(np.argmax((m0.sum() - m.sum(axis=0)) / (r0 - r + RETENTION_EPS)))
            if m[:, j].sum() >= m0.sum() - 1e-9:
                break
            policy, m0, r0 = cands[j], m[:, j], r[j]
        return policy


def core_survival(codec, ex, policy):
    vals = []
    for g in CORE_GROUPS:
        orig = {norm(v) for v in ex.get(g, []) if norm(v)}
        if orig:
            kept = {norm(v) for v in codec.transform(ex, policy).get(g, [])}
            vals.append(len(orig & kept) / len(orig))
    return np.mean(vals) if vals else np.nan


def envelope_gap(results, persons, n=500, seed=0):
    """Adaptive core survival minus the best fixed policy or pairwise mixture at no worse privacy."""

    def point(res, w):
        m = {k: np.nansum(v * w) / np.sum(w * ~np.isnan(v)) for k, v in res.items()}
        return np.array([max(m["bm25_r1"], m["char_r1"]), m["core"]])

    def gap(w):
        pts = np.array([point(r, w) for name, r in results.items() if name != "adaptive"])
        i, j = np.triu_indices(len(pts), 1)
        mix = pts[i][:, None] * MIX_GRID[None, :, None] + pts[j][:, None] * (1 - MIX_GRID[None, :, None])
        mix = np.concatenate([mix.reshape(-1, 2), pts])
        P, U = point(results["adaptive"], w)
        ok = mix[:, 0] <= P + 1e-12
        return U - mix[ok, 1].max() if ok.any() else np.inf

    _, inv = np.unique(persons, return_inverse=True)
    rng = np.random.default_rng(seed)
    draws = [
        gap(np.bincount(rng.integers(0, inv.max() + 1, inv.max() + 1), minlength=inv.max() + 1)[inv]) for _ in range(n)
    ]
    return float(gap(np.ones(len(persons)))), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dev-run", type=Path, default=ROOT / "artifacts/runs/entity_policy_v4c_pilot")
    parser.add_argument("--csv", type=Path, default=ROOT / "data/raw/all_resumes.csv")
    parser.add_argument("--pool-size", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; results are never overwritten")
    t0 = time.time()
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    frozen = json.loads(FROZEN.read_text())
    tau, global_policies = frozen["taus"]["q25"], frozen["global_policies"]
    with (args.dev_run / "workua" / "bundle.pkl").open("rb") as handle:
        bundle = pickle.load(handle)
    codec = bundle["codec"]
    if args.smoke:
        ids, extracted, clusters = list(bundle["ids"]), bundle["extracted"], np.array(bundle["clusters"])
    else:
        used = list(bundle["ids"]) + [i for p in USED_POOLS for i in pd.read_csv(p, dtype=str).id]
        ids, extracted, clusters, excluded = holdout_pool(args.csv, used, args.pool_size, args.seed)
        clusters = np.array(clusters)
        assert not set(ids) & set(used)
        log(f"pool {len(ids)} resumes, excluded {excluded} persons")
    texts = [codec.render(ex, {}) for ex in extracted]
    attacks = {"bm25": BM25Index(max_features=20000).fit(texts), "char": TfidfIndex(max_features=20000).fit(texts)}
    pop = Population(ids, extracted, clusters, codec, attacks["bm25"], texts)
    retvec = TfidfVectorizer(token_pattern=r"(?u)\b\w[\w'’+#.\-]*\b", sublinear_tf=True).fit(texts)
    queries = [int(k) for k in pop.linked_positions()]
    if args.smoke:
        queries = queries[:30]
    persons = clusters[queries]
    log(f"{len(queries)} linked queries, {len(set(persons))} persons")

    adaptive = Adaptive(codec, extracted, texts, attacks, retvec)
    released = []
    for i, k in enumerate(queries):
        released.append(adaptive.release(k, tau))
        if i % 50 == 0:
            log(f"adaptive {i}/{len(queries)} {time.time() - t0:.0f}s")
    targets, excludes = pop.targets_and_excludes(np.array(queries), "S2")

    def evaluate(policies):
        texts_out = [adaptive.render(k, p) for k, p in zip(queries, policies)]
        res = {}
        for name, index in attacks.items():
            frame = ranking_result(index, texts_out, targets, excludes, batch_size=16)[0].to_frame()
            res[f"{name}_r1"] = frame["recall@1"].to_numpy()
        res["core"] = np.array([core_survival(codec, extracted[k], p) for k, p in zip(queries, policies)])
        res["ret"] = np.array([adaptive.retention(k, [q])[0] for k, q in zip(queries, texts_out)])
        return res

    fixed = {**fixed_policies(), **{f"global_{i + 1}": p for i, p in enumerate(global_policies)}}
    results = {name: evaluate([p] * len(queries)) for name, p in fixed.items()}
    results["adaptive"] = evaluate(released)
    log(f"evaluated {len(results)} policies {time.time() - t0:.0f}s")

    safe = (results["L0_none"]["bm25_r1"] == 0) & (results["L0_none"]["char_r1"] == 0)
    a, l2, l3 = results["adaptive"], results["L2_generalize"], results["L3_strong"]
    diffs = {}
    for metric in ["bm25_r1", "char_r1", "core", "ret"]:
        diffs[f"vs_L2_{metric}"] = person_bootstrap(a[metric] - l2[metric], persons)
        diffs[f"vs_L3_{metric}"] = person_bootstrap(a[metric] - l3[metric], persons)
    diffs["vs_L2_core_on_L0_safe"] = person_bootstrap((a["core"] - l2["core"])[safe], persons[safe])
    h1 = diffs["vs_L2_bm25_r1"][2] <= NONINFERIORITY and diffs["vs_L2_char_r1"][2] <= NONINFERIORITY
    h2 = diffs["vs_L2_core"][1] > 0
    h3 = diffs["vs_L2_core_on_L0_safe"][1] > 0
    gap = envelope_gap(results, persons)

    args.out.mkdir(parents=True)
    pd.DataFrame({"id": ids}).to_csv(args.out / "pool_ids.csv", index=False)
    per_query = pd.DataFrame({"pos": queries, "id": [ids[k] for k in queries], "person": persons, "l0_safe": safe})
    for name in ["L0_none", "L2_generalize", "L3_strong", "adaptive"]:
        for metric, values in results[name].items():
            per_query[f"{name}_{metric}"] = values
    per_query["adaptive_policy"] = [json.dumps(p, ensure_ascii=False, sort_keys=True) for p in released]
    per_query.to_csv(args.out / "per_query.csv", index=False)
    summary = pd.DataFrame(
        [
            {
                "policy": name,
                **{m: float(np.nanmean(v)) for m, v in res.items()},
                "core_on_L0_safe": float(np.nanmean(res["core"][safe])),
            }
            for name, res in results.items()
        ]
    )
    summary["privacy"] = summary[["bm25_r1", "char_r1"]].max(axis=1)
    summary.round(4).to_csv(args.out / "summary.csv", index=False)
    report = {
        "script_sha256": script_hash,
        "smoke": args.smoke,
        "seed": args.seed,
        "queries": len(queries),
        "persons": len(set(persons)),
        "l0_safe_queries": int(safe.sum()),
        "diffs_mean_lo_hi_n": diffs,
        "envelope_gap_mean_lo_hi": gap,
        "H1": bool(h1),
        "H2": bool(h2),
        "H3": bool(h3),
        "verdict": "PASS" if h1 and h2 and h3 else "FAIL",
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=1))
    pd.set_option("display.width", 200)
    print(summary.round(3).to_string(index=False))
    for key, (mean, lo, hi, n) in diffs.items():
        print(f"{key:28s} {mean:+.3f} [{lo:+.3f}, {hi:+.3f}] n={n}")
    print(f"envelope gap {gap[0]:+.3f} [{gap[1]:+.3f}, {gap[2]:+.3f}]")
    log(f"H1 {h1} H2 {h2} H3 {h3} -> {report['verdict']} {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
