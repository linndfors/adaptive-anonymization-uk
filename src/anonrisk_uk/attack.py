"""Record-linkage attackers and exact tie-aware risk metrics.

An attacker ranks a background pool of original documents for an anonymized
query. Ties are resolved by a uniformly random permutation. Expected rank,
expected reciprocal rank, and top-k probability are computed separately because
Recall or MRR cannot be recovered from expected rank alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.special import digamma
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

TOKEN_PATTERN = r"(?u)\b[\w][\w'’+#.\-]*\b"


class BM25Index:
    def __init__(self, k1: float = 1.2, b: float = 0.75, min_df: int = 1, max_features: int | None = None):
        self.k1, self.b = k1, b
        self.vec = CountVectorizer(
            token_pattern=TOKEN_PATTERN, lowercase=True, min_df=min_df, max_features=max_features, dtype=np.float32
        )

    def fit(self, texts: list[str]) -> BM25Index:
        tf = self.vec.fit_transform(texts).tocsr()
        n = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        self.df = df
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        dl = np.asarray(tf.sum(axis=1)).ravel()
        avgdl = dl.mean() if n else 1.0
        tf = tf.tocoo()
        denom = tf.data + self.k1 * (1 - self.b + self.b * dl[tf.row] / avgdl)
        w = self.idf[tf.col] * tf.data * (self.k1 + 1) / denom
        self.W = sp.csr_matrix((w, (tf.row, tf.col)), shape=tf.shape).T.tocsr()  # terms × docs
        self.n_docs = n
        return self

    def query_matrix(self, texts: list[str]) -> sp.csr_matrix:
        q = self.vec.transform(texts)
        q.data[:] = 1.0  # BM25 counts each query term once.
        return q.tocsr()

    def scores(self, texts: list[str]) -> np.ndarray:
        return np.asarray((self.query_matrix(texts) @ self.W).todense(), dtype=np.float32)

    def token_idf(self, token: str) -> float:
        j = self.vec.vocabulary_.get(token.lower())
        return float(self.idf[j]) if j is not None else float(np.log(1 + self.n_docs))


class TfidfIndex:
    """Character n-grams robust to inflection and transliteration."""

    def __init__(self, analyzer: str = "char_wb", ngram_range=(3, 5), max_features: int = 300_000, min_df: int = 2):
        self.vec = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            min_df=min_df,
            max_features=max_features,
            sublinear_tf=True,
            dtype=np.float32,
        )

    def fit(self, texts):
        self.D = self.vec.fit_transform(texts).T.tocsr()
        self.n_docs = self.D.shape[1]
        return self

    def scores(self, texts):
        return np.asarray((self.vec.transform(texts) @ self.D).todense(), dtype=np.float32)


class DenseIndex:
    """Semantic attacker with E5 query and passage prefix support."""

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        batch_size: int = 64,
        device: str | None = None,
        max_chars: int = 2000,
    ):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        self.bs, self.max_chars = batch_size, max_chars
        self.e5 = "e5" in model_name

    def _enc(self, texts, prefix):
        texts = [(prefix if self.e5 else "") + t[: self.max_chars] for t in texts]
        return self.model.encode(texts, batch_size=self.bs, normalize_embeddings=True, show_progress_bar=False).astype(
            np.float32
        )

    def fit(self, texts):
        self.E = self._enc(texts, "passage: ")
        self.n_docs = len(texts)
        return self

    def scores(self, texts):
        return self._enc(texts, "query: ") @ self.E.T


class HybridIndex:
    """Fuse row-wise z-normalized scores with a weighted sum."""

    def __init__(self, indexes: list, weights: list[float] | None = None):
        self.indexes = indexes
        self.weights = weights or [1.0] * len(indexes)

    def fit(self, texts):
        for ix in self.indexes:
            ix.fit(texts)
        self.n_docs = len(texts)
        return self

    def scores(self, texts):
        total = 0
        for ix, w in zip(self.indexes, self.weights):
            s = ix.scores(texts)
            mu, sd = s.mean(axis=1, keepdims=True), s.std(axis=1, keepdims=True) + 1e-9
            total = total + w * (s - mu) / sd
        return total


# ---------------------------------------------------------------------------
# Ranking and metrics
# ---------------------------------------------------------------------------
@dataclass
class AttackResult:
    ranks: np.ndarray
    reciprocal_ranks: np.ndarray
    recalls: dict[int, np.ndarray]
    n_better: np.ndarray
    n_tied_targets: np.ndarray
    n_tied_others: np.ndarray
    n_candidates: np.ndarray

    def to_frame(self):
        import pandas as pd

        return pd.DataFrame(
            {
                "expected_rank": self.ranks,
                "rr": self.reciprocal_ranks,
                **{f"recall@{k}": v for k, v in self.recalls.items()},
                "n_better": self.n_better,
                "n_tied_targets": self.n_tied_targets,
                "n_tied_others": self.n_tied_others,
                "n_candidates": self.n_candidates,
            }
        )


def _tie_expectations(better: int, others: int, targets: int, ks):
    """Compute the exact first-target distribution under a random tie permutation."""
    total = others + targets
    rank = better + 1 + others / (targets + 1)
    if others == 0:
        rr = 1.0 / (better + 1)
    elif targets == 1:
        rr = float((digamma(better + total + 1) - digamma(better + 1)) / total)
    else:
        # P(J=1)=m/T; P(J=j+1)/P(J=j)=(T-j-m+1)/(T-j).
        steps = np.arange(1, others + 1, dtype=float)
        log_p = np.log(targets / total)
        probs = np.exp(np.r_[log_p, log_p + np.cumsum(np.log1p(-(targets - 1) / (total - steps)))])
        rr = float(np.sum(probs / (better + np.arange(1, others + 2))))
    recalls = {}
    for k in ks:
        take = int(k - better)
        if take <= 0:
            recalls[k] = 0.0
        elif take > others:
            recalls[k] = 1.0
        else:
            # Probability of selecting no target in the first ``take`` positions.
            log_none = np.log1p(-targets / (total - np.arange(take, dtype=float))).sum()
            recalls[k] = float(-np.expm1(log_none))
    return rank, rr, recalls


def ranking_result(index, query_texts, targets, exclude=None, batch_size=256, ks=(1, 5, 10)):
    """Return exact tie metrics, target margin, and best target score."""
    n = len(query_texts)
    if len(targets) != n or (exclude is not None and len(exclude) != n):
        raise ValueError("Queries, targets, and exclusion lists must have the same length")
    if batch_size <= 0 or any(k <= 0 for k in ks):
        raise ValueError("batch_size and k must be positive")
    ranks, rr, margins, best_scores = (np.empty(n, dtype=float) for _ in range(4))
    recalls = {k: np.empty(n, dtype=float) for k in ks}
    better, tied_targets, tied_others, candidates = (np.empty(n, dtype=int) for _ in range(4))
    cache = {}
    for b0 in range(0, n, batch_size):
        scores = index.scores(query_texts[b0 : b0 + batch_size])
        for j, s in enumerate(scores):
            i = b0 + j
            if not np.isfinite(s).all():
                raise ValueError("The attacker returned non-numeric or non-finite scores")
            target = np.unique(np.asarray(targets[i], dtype=int))
            excluded = np.asarray(exclude[i] if exclude is not None else [], dtype=int)
            if not len(target) or np.any(target < 0) or np.any(target >= len(s)):
                raise ValueError("Every query must have at least one valid target in the candidate pool")
            if np.any(excluded < 0) or np.any(excluded >= len(s)) or np.intersect1d(target, excluded).size:
                raise ValueError("Exclusions cannot contain targets or positions outside the candidate pool")
            best = float(s[target].max())
            mask = np.ones(len(s), dtype=bool)
            mask[target] = False
            mask[excluded] = False
            other = s[mask]
            g, nt, no = int((other > best).sum()), int((s[target] == best).sum()), int((other == best).sum())
            key = (g, no, nt)
            if key not in cache:
                cache[key] = _tie_expectations(g, no, nt, ks)
            ranks[i], rr[i], probabilities = cache[key]
            for k in ks:
                recalls[k][i] = probabilities[k]
            better[i], tied_targets[i], tied_others[i] = g, nt, no
            candidates[i] = len(s) - len(np.unique(excluded))
            best_scores[i] = best
            margins[i] = best - (float(other.max()) if len(other) else 0.0)
    return AttackResult(ranks, rr, recalls, better, tied_targets, tied_others, candidates), margins, best_scores


def expected_ranks(index, query_texts, targets, exclude=None, batch_size=256) -> np.ndarray:
    """Compatibility interface for expected rank; use ``ranking_result`` for metrics."""
    return ranking_result(index, query_texts, targets, exclude, batch_size)[0].ranks


def risk_metrics(result: AttackResult, ks=(1, 5, 10)) -> dict:
    if not isinstance(result, AttackResult):
        raise TypeError("Recall and MRR require AttackResult, not expected ranks")
    return {
        **{f"recall@{k}": float(result.recalls[k].mean()) for k in ks},
        "mrr": float(result.reciprocal_ranks.mean()),
        "median_rank": float(np.median(result.ranks)),
    }


def bootstrap_ci(values: np.ndarray, stat=np.mean, n: int = 1000, seed: int = 0, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    boots = [stat(values[rng.integers(0, len(values), len(values))]) for _ in range(n)]
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def target_margin(
    index, query_texts: list[str], targets: list[list[int]], exclude: list[list[int]] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return expected rank, raw target margin, and best target score.

    The margin changes continuously while the target remains first and is useful
    for stepwise transformation selection on rank plateaus.
    """
    result, margins, best_scores = ranking_result(index, query_texts, targets, exclude)
    return result.ranks, margins, best_scores
