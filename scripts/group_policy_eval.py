"""Group-level adaptive policy vs L2: S2 privacy and independent utility.

Method (frozen before the holdout): per document, greedy keep -> generalize -> remove over
groups (skills are never removed). Each step maximises the drop of a release-time proxy
margin (BM25, top-3 non-self neighbours of the original) per unit of lexical retention lost;
the search stops when no step lowers the proxy. The released state is the lowest-proxy state
whose retention is at least that document's L2 retention.

  --mode dev      dev pool of an existing run; queries = linked train+val resumes
  --mode holdout  fresh pool of people disjoint from the dev pool; queries = all linked resumes

Primary hypotheses (decided before the holdout):
  H1 privacy non-inferiority: upper 95% CI of recall@1(adaptive) - recall@1(L2) <= +0.05,
     for BM25 and char-TF-IDF.
  H2 utility superiority: lower 95% CI of core survival(adaptive) - core survival(L2) > 0, where core
     survival is the per-resume mean share of skills, desired title and past positions kept verbatim.
Industry and salary classifier accuracy are secondary. Bootstrap resamples persons, not resumes.
Revised after the dev run and before the holdout: H2 was industry accuracy, which had 115 labelled
dev queries and 0.46 accuracy on originals, too weak to decide anything.
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
from sklearn.feature_extraction.text import TfidfVectorizer

from anonrisk_uk.attack import BM25Index, TfidfIndex, ranking_result
from anonrisk_uk.experiment import Population
from anonrisk_uk.policy import fixed_policies, next_op
from anonrisk_uk.utility import DownstreamTask
from anonrisk_uk.workua import GROUPS, build_person_clusters, extract, load_workua, norm

ROOT = Path(__file__).resolve().parents[1]
PROTECT = {"skills"}
NEIGHBOR_K = 3
RETENTION_EPS = 0.01
NONINFERIORITY = 0.05
SURVIVAL_GROUPS = ["skills", "title", "exp_position", "exp_company", "exp_dates", "edu_faculty"]
CORE_GROUPS = ["skills", "title", "exp_position"]


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def dev_pool(run):
    with (run / "workua" / "bundle.pkl").open("rb") as handle:
        b = pickle.load(handle)
    split = b["split"].split.to_numpy()
    return b, np.array(b["clusters"]), split


def holdout_pool(csv, dev_ids, size, seed):
    """Fresh complete-person sample from the full CSV, excluding every person touching the dev pool."""
    df = load_workua(csv)
    clusters = build_person_clusters(df).to_numpy()
    dev = set(clusters[df["id"].isin(set(dev_ids)).to_numpy()])
    members = pd.Series(np.arange(len(df))).groupby(clusters).apply(list)
    members = members[[c not in dev for c in members.index]]
    order = np.random.default_rng(seed).permutation(len(members))
    positions = []
    for i in order:
        group = members.iloc[i]
        if len(positions) + len(group) <= size:
            positions.extend(group)
        if len(positions) == size:
            break
    df = df.iloc[sorted(positions)].reset_index(drop=True)
    columns = list(df.columns)
    extracted = [extract(dict(zip(columns, row))) for row in df.itertuples(index=False, name=None)]
    return df["id"].tolist(), extracted, build_person_clusters(df).astype(str).to_numpy(), len(dev)


class Search:
    def __init__(self, codec, extracted, texts, index, retvec):
        self.codec, self.extracted, self.texts, self.index, self.retvec = codec, extracted, texts, index, retvec

    def render(self, k, policy):
        return self.codec.render(self.extracted[k], policy)

    def retention(self, k, queries):
        orig = self.retvec.transform([self.texts[k]])
        return np.asarray((self.retvec.transform(queries) @ orig.T).todense()).ravel()

    def margin(self, scores, k, targets):
        s = scores.copy()
        s[:, k] = -np.inf
        t = s[:, targets].max(axis=1)
        s[:, targets] = -np.inf
        return (t - s.max(axis=1)) / np.maximum(np.abs(t), 1e-9)

    def neighbours(self, k):
        s = self.index.scores([self.texts[k]])[0]
        s[k] = -np.inf
        return list(np.argsort(-s)[:NEIGHBOR_K])

    def path(self, k, targets):
        policy, m0, r0 = {}, self.margin(self.index.scores([self.texts[k]]), k, targets)[0], 1.0
        states = [(policy, r0, m0)]
        while True:
            cands = []
            for g in GROUPS:
                op = next_op(g, policy.get(g, "keep")) if self.extracted[k].get(g) else None
                if op and not (g in PROTECT and op == "remove"):
                    cands.append({**policy, g: op})
            if not cands:
                break
            queries = [self.render(k, p) for p in cands]
            m, r = self.margin(self.index.scores(queries), k, targets), self.retention(k, queries)
            j = int(np.argmax((m0 - m) / (r0 - r + RETENTION_EPS)))
            if m[j] >= m0 - 1e-9:
                break
            policy, m0, r0 = cands[j], m[j], r[j]
            states.append((policy, r0, m0))
        return states

    def release(self, k, floor):
        states = self.path(k, self.neighbours(k))
        ok = [(m, -r, i) for i, (_, r, m) in enumerate(states) if r >= floor - 1e-9]
        return states[min(ok)[2]][0]


def survival(codec, ex, policy, group):
    """Share of original values of a group that survive verbatim."""
    orig = {norm(v) for v in ex.get(group, []) if norm(v)}
    if not orig:
        return np.nan
    kept = {norm(v) for v in codec.transform(ex, policy).get(group, [])}
    return len(orig & kept) / len(orig)


def person_bootstrap(diff, persons, n=2000, seed=0):
    frame = pd.DataFrame({"d": diff, "p": persons}).dropna()
    sums, counts = frame.groupby("p").d.sum(), frame.groupby("p").d.size()
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        pick = rng.integers(0, len(sums), len(sums))
        draws.append(sums.iloc[pick].sum() / counts.iloc[pick].sum())
    return float(frame.d.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)), len(frame)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["dev", "holdout"], required=True)
    parser.add_argument("--dev-run", type=Path, default=ROOT / "artifacts/runs/entity_policy_v4c_pilot")
    parser.add_argument("--csv", type=Path, default=ROOT / "data/raw/all_resumes.csv")
    parser.add_argument("--pool-size", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--limit", type=int, default=None, help="Smoke test on the first N queries")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; results are never overwritten")
    t0 = time.time()

    b, dev_clusters, dev_split = dev_pool(args.dev_run)
    codec = b["codec"]
    if args.mode == "dev":
        ids, extracted, clusters = b["ids"], b["extracted"], dev_clusters
        query_mask = np.isin(dev_split, ["train", "val"])
        n_excluded = 0
    else:
        log("Building the holdout pool from the full CSV")
        ids, extracted, clusters, n_excluded = holdout_pool(args.csv, b["ids"], args.pool_size, args.seed)
        query_mask = np.ones(len(ids), dtype=bool)
    texts = [codec.render(ex, {}) for ex in extracted]
    bm = BM25Index(max_features=20000).fit(texts)
    char = TfidfIndex(max_features=20000).fit(texts)
    pop = Population(ids, extracted, clusters, codec, bm, texts)
    pos = np.array([k for k in pop.linked_positions() if query_mask[k]], dtype=int)
    if args.limit:
        pos = pos[: args.limit]
    log(f"pool {len(ids)} resumes, {len(set(clusters))} persons; {len(pos)} linked queries")

    # Utility classifiers are trained on dev originals; in dev mode the query persons are left out.
    train_k = [k for k in range(len(b["ids"])) if dev_split[k] == "train"]
    if args.mode == "dev":
        held = set(dev_clusters[pos])
        train_k = [k for k in train_k if dev_clusters[k] not in held]
    tasks = [DownstreamTask("industry", {"exp_industry"}), DownstreamTask("salary", {"salary"})]
    for task in tasks:
        task.fit(codec, [b["extracted"][k] for k in train_k])

    retvec = TfidfVectorizer(token_pattern=r"(?u)\b\w[\w'’+#.\-]*\b", sublinear_tf=True).fit(texts)
    search = Search(codec, extracted, texts, bm, retvec)
    fixed = fixed_policies()
    policies = {"L0": [{}] * len(pos), "L2": [fixed["L2_generalize"]] * len(pos)}
    l2_ret = np.array([search.retention(k, [search.render(k, fixed["L2_generalize"])])[0] for k in pos])
    adaptive = []
    for i, k in enumerate(pos):
        adaptive.append(search.release(k, l2_ret[i]))
        if i % 50 == 0:
            log(f"adaptive release {i}/{len(pos)}")
    policies["adaptive"] = adaptive

    tg, ex = pop.targets_and_excludes(pos, "S2")
    sub = [extracted[k] for k in pos]
    frames = []
    for name, pols in policies.items():
        queries = [search.render(k, p) for k, p in zip(pos, pols)]
        row = pd.DataFrame({"pos": pos, "id": [ids[k] for k in pos], "person": clusters[pos], "method": name})
        for attack, index in [("bm25", bm), ("char", char)]:
            row[f"{attack}_r1"] = ranking_result(index, queries, tg, ex, batch_size=16)[0].to_frame()["recall@1"].values
        row["retention"] = [search.retention(k, [q])[0] for k, q in zip(pos, queries)]
        for task in tasks:
            y, pred, valid = task.predictions(codec, sub, pols)
            row[f"{task.name}_correct"] = np.where(valid, (pred == y).astype(float), np.nan)
        for g in SURVIVAL_GROUPS:
            row[f"kept_{g}"] = [survival(codec, extracted[k], p, g) for k, p in zip(pos, pols)]
        row["core_survival"] = row[[f"kept_{g}" for g in CORE_GROUPS]].mean(axis=1)
        row["policy"] = [json.dumps(p, sort_keys=True, ensure_ascii=False) for p in pols]
        frames.append(row)
    per_query = pd.concat(frames, ignore_index=True)

    metrics = [c for c in per_query if c.endswith(("_r1", "_correct")) or c == "retention"]
    metrics += ["core_survival"] + [c for c in per_query if c.startswith("kept_")]
    summary = per_query.groupby("method")[metrics].mean().round(4)
    a = per_query[per_query.method == "adaptive"].reset_index(drop=True)
    l2 = per_query[per_query.method == "L2"].reset_index(drop=True)
    diffs = {}
    for m in metrics:
        mean, lo, hi, n = person_bootstrap((a[m] - l2[m]).to_numpy(), a.person.to_numpy())
        diffs[m] = {"diff": mean, "lo": lo, "hi": hi, "n": n}
    h1 = all(diffs[f"{x}_r1"]["hi"] <= NONINFERIORITY for x in ["bm25", "char"])
    h2 = diffs["core_survival"]["lo"] > 0

    args.out.mkdir(parents=True)
    per_query.to_csv(args.out / "per_query.csv", index=False)
    summary.to_csv(args.out / "summary.csv")
    report = {
        "mode": args.mode,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dev_run": str(args.dev_run),
        "seed": args.seed,
        "pool_resumes": len(ids),
        "pool_persons": len(set(clusters)),
        "excluded_dev_persons": n_excluded,
        "queries": len(pos),
        "query_persons": len(set(clusters[pos])),
        "classifier_train_resumes": len(train_k),
        "differences_adaptive_minus_L2": diffs,
        "H1_privacy_noninferior": h1,
        "H2_core_survival_better": h2,
        "pass": h1 and h2,
        "seconds": round(time.time() - t0),
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    if args.mode == "holdout":
        pd.Series(ids, name="id").to_csv(args.out / "holdout_pool_ids.csv", index=False)
    print(summary.T.to_string())
    for m, d in diffs.items():
        print(f"  {m:22s} {d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] n={d['n']}")
    log(f"H1 privacy non-inferior: {h1}; H2 core survival better: {h2}; pass: {h1 and h2}")


if __name__ == "__main__":
    main()
