"""Attack scenarios, document-level ablation, and k-anonymity.

S2 queries an anonymized resume against all other originals and targets another
resume belonging to the same person. It is evaluation-only because production
does not know same-person links. S1 self-linkage includes the original query
record in the pool and targets it directly, making S1 an operational proxy whose
agreement with S2 can be measured experimentally.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .attack import expected_ranks, ranking_result
from .workua import GROUPS, ResumeCodec, nonempty_groups, norm


@dataclass
class Population:
    ids: list[str]
    extracted: list[dict]
    clusters: np.ndarray  # Person-cluster ID for every record
    codec: ResumeCodec
    index: object  # Attacker fitted on original pool texts
    texts: list | None = None  # Pool texts used for exact-copy diagnostics

    def __post_init__(self):
        self.pos = {i: k for k, i in enumerate(self.ids)}
        members = {}
        for k, c in enumerate(self.clusters):
            members.setdefault(c, []).append(k)
        self.members = members
        # Exact copies are diagnostic and do not expand S1 targets.
        self.dups = {}
        if self.texts is not None:
            by_hash = {}
            for k, t in enumerate(self.texts):
                by_hash.setdefault(hash(t), []).append(k)
            self.dups = {k: grp for grp in by_hash.values() if len(grp) > 1 for k in grp}

    def linked_positions(self) -> np.ndarray:
        """Return records with at least one other same-person resume in the pool."""
        return np.array([k for k, c in enumerate(self.clusters) if len(self.members[c]) >= 2])

    def targets_and_excludes(self, positions, scenario: str):
        if scenario == "S2":
            tg = [[m for m in self.members[self.clusters[k]] if m != k] for k in positions]
            ex = [[k] for k in positions]
        elif scenario == "S1":
            # S1 targets one exact record. An equal text from another record is a
            # tie competitor, not evidence that the records belong to one person.
            tg = [[k] for k in positions]
            ex = [[] for _ in positions]
        else:
            raise ValueError(scenario)
        return tg, ex


def render_queries(pop: Population, positions, policies) -> list[str]:
    """Render one shared policy or one policy per query."""
    if isinstance(policies, dict):
        policies = [policies] * len(positions)
    if len(policies) != len(positions):
        raise ValueError("Policies and positions must have the same length")
    return [pop.codec.render(pop.extracted[k], p) for k, p in zip(positions, policies)]


def attack(pop: Population, positions, policies, scenario: str = "S2", batch_size: int = 256) -> np.ndarray:
    tg, ex = pop.targets_and_excludes(positions, scenario)
    return expected_ranks(pop.index, render_queries(pop, positions, policies), tg, ex, batch_size)


def attack_result(pop: Population, positions, policies, scenario: str = "S2", batch_size: int = 256):
    tg, ex = pop.targets_and_excludes(positions, scenario)
    return ranking_result(pop.index, render_queries(pop, positions, policies), tg, ex, batch_size)[0]


# ---------------------------------------------------------------------------
# Document-level group ablation
# ---------------------------------------------------------------------------
def group_contributions(
    pop: Population,
    positions,
    base_policy: dict,
    groups=None,
    mode: str = "forward",
    op: str = "remove",
    scenarios=("S2", "S1"),
    batch_size: int = 512,
) -> pd.DataFrame:
    """Measure forward or add-back reciprocal-rank contribution for each group.

    Forward contribution is the reduction after applying an operation. Add-back
    contribution is the increase after restoring a group while the others remain
    hidden, reducing redundancy effects. RR is exact expected reciprocal rank for
    the individual document.
    """
    groups = groups or GROUPS
    rows, q_pos, q_pol, meta = [], [], [], []
    for k in positions:
        present = [g for g in nonempty_groups(pop.extracted[k]) if g in groups]
        q_pos.append(k)
        q_pol.append(base_policy)
        meta.append((k, None, "base"))
        for g in present:
            pol = dict(base_policy)
            if mode == "forward":
                pol[g] = op
            else:
                if g not in pol:
                    q_pos.append(k)
                    q_pol.append({**base_policy, g: "remove"})
                    meta.append((k, g, "base"))
                pol.pop(g, None)
            q_pos.append(k)
            q_pol.append(pol)
            meta.append((k, g, "variant"))
    out = {}
    for sc in scenarios:
        out[sc] = attack_result(pop, q_pos, q_pol, sc, batch_size).reciprocal_ranks
    base_rr = {sc: {} for sc in scenarios}
    for i, (k, g, kind) in enumerate(meta):
        if kind == "base":
            for sc in scenarios:
                base_rr[sc][(k, g)] = out[sc][i]
    for i, (k, g, kind) in enumerate(meta):
        if kind == "base":
            continue
        baseline_group = g if mode == "addback" and g not in base_policy else None
        row = {
            "pos": k,
            "group": g,
            "mode": mode,
            "baseline_operation": base_policy.get(g, "remove" if mode == "addback" else "keep"),
        }
        for sc in scenarios:
            rr = out[sc][i]
            row[f"rr_base_{sc}"] = base_rr[sc][(k, baseline_group)]
            row[f"rr_var_{sc}"] = rr
            base = row[f"rr_base_{sc}"]
            row[f"delta_{sc}"] = (base - rr) if mode == "forward" else (rr - base)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# k-anonymity on structured quasi-identifiers
# ---------------------------------------------------------------------------
DEFAULT_QI = ["age", "city", "title", "exp_company", "edu_institution", "edu_years"]


def k_anonymity(pop: Population, policy: dict, qi=DEFAULT_QI, positions=None) -> np.ndarray:
    """Compute equivalence-class size and prosecutor risk after a fixed policy.

    Classical k-anonymity is not defined for per-person adaptive policies here.
    """
    keys = []
    for ex in pop.extracted:
        t = pop.codec.transform(ex, policy)
        keys.append(tuple("|".join(sorted(norm(v) for v in t[g])) for g in qi))
    cnt = Counter(keys)
    k = np.array([cnt[key] for key in keys])
    return k if positions is None else k[np.asarray(positions)]


def k_summary(k: np.ndarray) -> dict:
    return {
        "unique_share(k=1)": float((k == 1).mean()),
        "share_k<5": float((k < 5).mean()),
        "share_k<10": float((k < 10).mean()),
        "mean_prosecutor_risk": float((1 / k).mean()),
        "median_k": float(np.median(k)),
    }


def qi_marginal_uniqueness(pop: Population, qi=DEFAULT_QI) -> pd.DataFrame:
    """Measure how removing one QI changes the share of unique records."""
    base = k_summary(k_anonymity(pop, {}, qi))["unique_share(k=1)"]
    rows = [{"removed_qi": "—", "unique_share": base}]
    for g in qi:
        rest = [x for x in qi if x != g]
        rows.append({"removed_qi": g, "unique_share": k_summary(k_anonymity(pop, {}, rest))["unique_share(k=1)"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Attacker background-knowledge model (S1b)
# ---------------------------------------------------------------------------
def field_stability(pop: Population, max_pairs: int = 20000, seed: int = 0, positions=None) -> pd.DataFrame:
    """Estimate same-person field stability with Jaccard similarity.

    Stable fields transfer between resumes and can support an attacker. ``positions``
    restricts estimation to train data so test links remain unused.
    """
    rng = np.random.default_rng(seed)
    allowed = set(positions) if positions is not None else None
    members = [[k for k in m if allowed is None or k in allowed] for m in pop.members.values()]
    pairs = [(a, b) for m in members if len(m) >= 2 for i, a in enumerate(m) for b in m[i + 1 :]]
    if len(pairs) > max_pairs:
        pairs = [pairs[i] for i in rng.choice(len(pairs), max_pairs, replace=False)]
    rows = []
    for g in GROUPS:
        jac, both = [], 0
        for a, b in pairs:
            A = {norm(v) for v in pop.extracted[a].get(g, [])}
            B = {norm(v) for v in pop.extracted[b].get(g, [])}
            if A and B:
                both += 1
                jac.append(len(A & B) / len(A | B))
        rows.append({"group": g, "pairs_with_both": both, "mean_jaccard": float(np.mean(jac)) if jac else np.nan})
    return pd.DataFrame(rows).sort_values("mean_jaccard", ascending=False).reset_index(drop=True)


def background_population(pop: Population, background_policy: dict, index_factory) -> Population:
    """Build a pool containing a stable background view of every document."""
    texts = [pop.codec.render(e, background_policy) for e in pop.extracted]
    return Population(pop.ids, pop.extracted, pop.clusters, pop.codec, index_factory().fit(texts), texts)


def subpopulation(pop: Population, positions, index_factory, policy: dict | None = None) -> tuple[Population, dict]:
    """Build a subpopulation and return its old-to-new position mapping."""
    positions = list(positions)
    texts = [pop.codec.render(pop.extracted[k], policy or {}) for k in positions]
    sub = Population(
        [pop.ids[k] for k in positions],
        [pop.extracted[k] for k in positions],
        pop.clusters[positions],
        pop.codec,
        index_factory().fit(texts),
        texts,
    )
    return sub, {k: i for i, k in enumerate(positions)}


def free_text_residual_pii(pop: Population, groups=("additional_info", "exp_resp")) -> pd.DataFrame:
    """Count residual PII-like patterns in free text after dataset placeholders."""
    import re

    pats = {
        "name_after_placeholder": r"\[(?:FIRST_NAME|LAST_NAME|FULL_NAME|NAME)\]\s*[А-ЯІЇЄҐA-Z]",
        "placeholder_any": r"\[[A-Z_]{3,}\]",
        "email": r"[\w.+\-]+@[\w\-]+\.[\w.\-]+",
        "url": r"https?://\S+|www\.\S+|t\.me/\S+",
        "phone_like": r"(?<!\w)\+?\d[\d\s()\-]{8,}\d",
        "year": r"\b(?:19|20)\d{2}\b",
    }
    rows = []
    for name, p in pats.items():
        rx = re.compile(p)
        n_docs = sum(any(rx.search(v) for g in groups for v in ex.get(g, [])) for ex in pop.extracted)
        rows.append({"pattern": name, "docs_with_pattern": n_docs, "share": n_docs / len(pop.extracted)})
    return pd.DataFrame(rows)
