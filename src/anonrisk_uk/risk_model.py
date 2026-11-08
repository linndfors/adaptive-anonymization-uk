"""Contextual fragment-risk models and within-document ranking metrics.

The target is not corpus uniqueness. It is the measured contribution of a unit
in a particular document to attack success, estimated by counterfactual
ablation. Labels are therefore tied to an explicit threat model.

The central comparison is between a train-only category prior and a contextual
model that also uses rarity, counts, and document context. Improvement in
within-document ordering supports the contextual-risk hypothesis.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

from .workua import GROUPS


def group_features(pop, positions_groups: pd.DataFrame) -> pd.DataFrame:
    """Build legacy group-level features for rows containing ``pos`` and ``group``."""
    vocab, idf, df = pop.index.vec.vocabulary_, pop.index.idf, pop.index.df
    an = pop.index.vec.build_analyzer()
    feats = []
    doc_cache = {}
    for k, g in zip(positions_groups["pos"], positions_groups["group"]):
        ex = pop.extracted[k]
        vals = ex.get(g, [])
        freqs = [pop.codec.population_frequency(g, v) for v in vals] or [0]
        toks = [t for v in vals for t in an(v)]
        tid = [vocab[t] for t in toks if t in vocab]
        idfs = idf[tid] if tid else np.array([0.0])
        dfs = df[tid] if tid else np.array([0])
        if k not in doc_cache:
            doc_cache[k] = (sum(1 for x in GROUPS if ex.get(x)), sum(len(an(v)) for x in GROUPS for v in ex.get(x, [])))
        n_groups, doc_tokens = doc_cache[k]
        feats.append(
            {
                "n_values": len(vals),
                "n_tokens": len(toks),
                "min_log_freq": math.log1p(min(freqs)),
                "mean_log_freq": float(np.mean(np.log1p(freqs))),
                "share_rare_values": float(np.mean([f < 5 for f in freqs])),
                "max_idf": float(idfs.max()),
                "mean_idf": float(idfs.mean()),
                "n_rare_tokens": int((dfs <= 3).sum()),
                "share_group_tokens": len(toks) / max(1, doc_tokens),
                "n_nonempty_groups": n_groups,
                "doc_tokens": doc_tokens,
            }
        )
    X = pd.DataFrame(feats, index=positions_groups.index)
    for g in GROUPS:
        X[f"g_{g}"] = (positions_groups["group"] == g).astype(int).values
    return X


class ContextualRiskModel:
    def __init__(self, **kw):
        self.model = HistGradientBoostingRegressor(
            max_iter=kw.get("max_iter", 300),
            learning_rate=kw.get("learning_rate", 0.05),
            max_leaf_nodes=kw.get("max_leaf_nodes", 31),
            random_state=kw.get("seed", 0),
        )

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.columns = list(X.columns)
        self.model.fit(X[self.columns], y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[self.columns])


def category_prior(train: pd.DataFrame, target: str) -> dict:
    return train.groupby("group")[target].mean().to_dict()


def ranking_records(frame: pd.DataFrame, target: str, score_col: str) -> pd.DataFrame:
    """Measure within-document ordering quality for transformation priority."""
    rows = []
    for pos, d in frame.groupby("pos"):
        if len(d) < 3 or d[target].nunique() < 2:
            continue
        rho = spearmanr(d[score_col], d[target]).correlation if d[score_col].nunique() > 1 else np.nan
        selected = d.loc[d[score_col] == d[score_col].max(), target]
        rows.append(
            {"pos": pos, "within_doc_spearman": rho, "precision@1": float((selected == d[target].max()).mean())}
        )
    return pd.DataFrame(rows, columns=["pos", "within_doc_spearman", "precision@1"])


def ranking_quality(frame: pd.DataFrame, target: str, score_col: str) -> dict:
    records = ranking_records(frame, target, score_col)
    glob = spearmanr(frame[score_col], frame[target]).correlation
    return {
        "global_spearman": float(glob),
        "within_doc_spearman": records.within_doc_spearman.mean(),
        "precision@1": records["precision@1"].mean(),
        "n_docs": len(records),
        "n_total_docs": frame.pos.nunique(),
        "n_spearman_docs": records.within_doc_spearman.notna().sum(),
    }
