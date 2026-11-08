"""Utility measures for anonymized resumes.

Fixed classifiers trained on original train resumes evaluate industry and salary
on anonymized test resumes while excluding the label-bearing field. TF-IDF cosine
similarity additionally measures lexical content retention.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline

from .workua import norm


def industry_label(ex: dict) -> str | None:
    v = ex.get("exp_industry") or []
    return norm(v[0]) if v and norm(v[0]) else None


def salary_value(ex: dict) -> float | None:
    v = ex.get("salary") or []
    try:
        return float(v[0].split()[0]) if v else None
    except ValueError:
        return None


class DownstreamTask:
    def __init__(self, name: str, exclude_groups: set[str], top_k_classes: int = 20, max_features: int = 200_000):
        self.name, self.exclude, self.top_k = name, set(exclude_groups), top_k_classes
        self.model = make_pipeline(
            TfidfVectorizer(
                token_pattern=r"(?u)\b\w[\w'’+#.\-]*\b",
                ngram_range=(1, 2),
                min_df=2,
                max_features=max_features,
                sublinear_tf=True,
            ),
            LogisticRegression(max_iter=300, C=4.0),
        )

    def labels(self, extracted: list[dict]) -> np.ndarray:
        if self.name == "industry":
            y = np.array([industry_label(e) for e in extracted], dtype=object)
        elif self.name == "salary":
            vals = np.array([salary_value(e) or np.nan for e in extracted], dtype=float)
            q = getattr(self, "_q", None)
            if q is None:
                self._q = q = np.nanquantile(vals, [1 / 3, 2 / 3])
            y = np.where(np.isnan(vals), None, np.digitize(vals, q).astype(str)).astype(object)
        else:
            raise ValueError(self.name)
        return y

    def fit(self, codec, extracted: list[dict]):
        y = self.labels(extracted)
        if self.name == "industry":
            keep = pd.Series(y).value_counts().head(self.top_k).index
            self.classes_ = set(keep)
        else:
            self.classes_ = {v for v in y if v is not None}
        idx = [i for i, v in enumerate(y) if v in self.classes_]
        if len(self.classes_) < 2:
            raise ValueError(f"Task {self.name}: the training set must contain at least two valid classes")
        X = [codec.render(extracted[i], {}, exclude=self.exclude) for i in idx]
        self.model.fit(X, y[idx])
        return self

    def eval_mask(self, extracted: list[dict]) -> np.ndarray:
        y = self.labels(extracted)
        return np.array([v in self.classes_ for v in y])

    def predictions(self, codec, extracted: list[dict], policies):
        """Return labels and predictions in original order without dropping missing values."""
        if isinstance(policies, dict):
            policies = [policies] * len(extracted)
        if len(policies) != len(extracted):
            raise ValueError("The number of policies must match the number of resumes")
        y = self.labels(extracted)
        idx = [i for i, v in enumerate(y) if v in self.classes_]
        pred = np.full(len(extracted), None, dtype=object)
        if idx:
            X = [codec.render(extracted[i], policies[i], exclude=self.exclude) for i in idx]
            pred[idx] = self.model.predict(X)
        return y, pred, np.array([v in self.classes_ for v in y], dtype=bool)

    def score(self, codec, extracted: list[dict], policies) -> dict:
        y, pred, valid = self.predictions(codec, extracted, policies)
        idx = np.flatnonzero(valid)
        if not len(idx):
            # Industry or salary may be absent from a small real-data sample.
            return {f"{self.name}_acc": np.nan, f"{self.name}_macroF1": np.nan, "n": 0}
        return {
            f"{self.name}_acc": accuracy_score(y[idx], pred[idx]),
            f"{self.name}_macroF1": f1_score(
                y[idx], pred[idx], average="macro", labels=sorted(self.classes_), zero_division=0
            ),
            "n": len(idx),
        }

    def proba(self, texts: list[str]) -> np.ndarray:
        return self.model.predict_proba(texts)


def retention_scores(codec, extracted: list[dict], policies, vectorizer: TfidfVectorizer | None = None):
    """Return per-resume cosine similarity for paired bootstrap analysis."""
    if isinstance(policies, dict):
        policies = [policies] * len(extracted)
    orig = [codec.render(e, {}) for e in extracted]
    anon = [codec.render(e, p) for e, p in zip(extracted, policies)]
    vec = vectorizer or TfidfVectorizer(token_pattern=r"(?u)\b\w[\w'’+#.\-]*\b", sublinear_tf=True).fit(orig)
    A, B = vec.transform(orig), vec.transform(anon)
    return np.asarray(A.multiply(B).sum(axis=1)).ravel()


def retention(codec, extracted: list[dict], policies, vectorizer: TfidfVectorizer | None = None) -> float:
    return float(retention_scores(codec, extracted, policies, vectorizer).mean())


class InstanceUtilityProxy:
    """Estimate label-free per-document utility loss at release time."""

    def __init__(self, codec, tasks: list[DownstreamTask], retention_weight: float = 0.5):
        self.codec, self.tasks, self.w = codec, tasks, retention_weight
        self.vec = None

    def fit_retention(self, extracted):
        self.vec = TfidfVectorizer(token_pattern=r"(?u)\b\w[\w'’+#.\-]*\b", sublinear_tf=True).fit(
            [self.codec.render(e, {}) for e in extracted]
        )
        return self

    def loss(self, ex: dict, policies: list[dict]) -> np.ndarray:
        """Return one loss value for every candidate policy."""
        loss = np.zeros(len(policies))
        for t in self.tasks:
            p0 = t.proba([self.codec.render(ex, {}, exclude=t.exclude)])
            P = t.proba([self.codec.render(ex, p, exclude=t.exclude) for p in policies])
            loss += 0.5 * np.abs(P - p0).sum(axis=1) / len(self.tasks)
        if self.vec is not None and self.w:
            o = self.vec.transform([self.codec.render(ex, {})])
            A = self.vec.transform([self.codec.render(ex, p) for p in policies])
            loss += self.w * (1 - np.asarray(A @ o.T.toarray()).ravel())
        return loss
