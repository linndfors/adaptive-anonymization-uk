"""Document-state risk regression with frozen training statistics, no live pool.

Target: worst reciprocal rank of another same-person resume under the declared
BM25/character attacks. Predictions are vulnerability scores, not probabilities.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .attack import TOKEN_PATTERN
from .resume_adaptive import options, planned_edits, render
from .resume_parser import FIELDS
from .workua import GROUPS, ResumeCodec, norm

VERSION = "adaptive_resume_v4_regression"
SCHEMA = "document_state_risk_v2"
FEATURE_GROUPS = sorted(set(GROUPS) | {v[1] for v in FIELDS.values()})
TOKENIZER = CountVectorizer(token_pattern=TOKEN_PATTERN).build_analyzer()


def fingerprint(value):
    return hashlib.sha256(value.encode()).hexdigest()


class FrozenCodec(ResumeCodec):
    """Keep frequency counts as hashed keys, not original resume fragments."""

    @classmethod
    def from_codec(cls, codec):
        result = cls(min_freq=codec.min_freq)
        result.settlements = set(codec.settlements)
        result.freq = {group: Counter({fingerprint(value): count for value, count in counts.items()})
                       for group, counts in codec.freq.items()}
        return result

    def population_frequency(self, group, value):
        return self.freq.get(group, {}).get(fingerprint(norm(value)), 0)

    def _frequent(self, group, value):
        return self.population_frequency(group, value) >= self.min_freq


@dataclass
class TrainingStatistics:
    token_idf: dict
    documents: int
    vectorizer: object = None
    projection: object = None

    @classmethod
    def fit(cls, texts):
        counts = Counter()
        for text in texts:
            counts.update({fingerprint(token) for token in TOKENIZER(text)})
        n = len(texts)
        vectorizer = TfidfVectorizer(sublinear_tf=True, max_features=20000)
        matrix = vectorizer.fit_transform(texts)
        dimensions = min(32, min(matrix.shape) - 1)
        projection = (TruncatedSVD(n_components=dimensions, random_state=42).fit(matrix)
                      if dimensions > 0 and matrix.shape[1] > 1 else None)
        return cls({key: math.log(1 + (n - df + .5) / (df + .5)) for key, df in counts.items()},
                   n, vectorizer, projection)

    def features(self, doc, policy, *, _output=None, _latent=None):
        output = render(doc, policy) if _output is None else _output
        visible = re.sub(r"\[[^\]]*\]", " ", output)
        tokens = TOKENIZER(visible)
        original_tokens = TOKENIZER(doc.text)
        idfs = np.array([self.token_idf.get(fingerprint(t), 0.) for t in tokens] or [0.])
        row = {"log_chars": math.log1p(len(visible)), "log_tokens": math.log1p(len(tokens)),
               "token_ratio": len(tokens) / max(len(original_tokens), 1),
               "unique_ratio": len(set(tokens)) / max(len(tokens), 1),
               "idf_mean": float(idfs.mean()), "idf_max": float(idfs.max()),
               "idf_p90": float(np.quantile(idfs, .9)),
               "oov_fraction": float((idfs == 0).mean()),
               "numbers": float(sum(t.isdigit() for t in tokens)),
               "years": float(len(re.findall(r"\b(?:19|20)\d{2}\b", visible))),
               "original_units": len(doc.units),
               "changed_fraction": 0., "masked_fraction": 0.}
        latent = np.zeros(32)
        if _latent is not None:
            latent[:len(_latent)] = _latent
        elif self.projection is not None:
            values = self.projection.transform(self.vectorizer.transform([visible]))[0]
            latent[:len(values)] = values
        row.update({f"text_lsa_{i}": float(value) for i, value in enumerate(latent)})
        edits = planned_edits(doc, policy)
        for group in FEATURE_GROUPS:
            indices = [i for i, unit in enumerate(doc.units) if unit.group == group]
            changed = sum(edits[i]["applied"] for i in indices)
            masked = sum(edits[i]["decision"].startswith(("mask", "remove")) for i in indices)
            row[f"{group}_original"] = len(indices)
            row[f"{group}_visible"] = len(indices) - masked
            row[f"{group}_changed"] = changed
            row[f"{group}_level"] = policy.get(group, 0)
        if edits:
            row["changed_fraction"] = sum(e["applied"] for e in edits) / len(edits)
            row["masked_fraction"] = sum(e["decision"].startswith(("mask", "remove")) for e in edits) / len(edits)
        return row


@dataclass
class FeatureSubsetRegressor:
    model: object
    columns: list

    def predict(self, frame):
        return self.model.predict(frame[self.columns])


@dataclass
class ResumeRiskRegressor:
    model: object
    statistics: TrainingStatistics
    columns: list
    codec: FrozenCodec
    retention_vectorizer: object
    target: float
    metadata: dict
    schema: str = SCHEMA

    def predict(self, doc, policies):
        if self.schema != SCHEMA:
            raise ValueError("Unsupported risk feature schema")
        frame = pd.DataFrame([self.statistics.features(doc, policy) for policy in policies])
        if set(frame) != set(self.columns):
            raise ValueError("Risk feature schema differs from the trained model")
        predictions = np.asarray(self.model.predict(frame[self.columns]), dtype=float)
        if not np.isfinite(predictions).all():
            raise ValueError("Risk model returned non-finite predictions")
        return np.clip(predictions, 0., 1.)


def state_policies(doc, rng, random_states=8):
    """Fixed label-generation design, independent of outcomes and test people."""
    groups = sorted({u.group for u in doc.units if len(options(doc, u)) > 1})
    policies = [{}] + [{g: level for g in groups} for level in (1, 2, 3)]
    policies.extend({group: level} for group in groups for level in (1, 2, 3))
    policies.extend({group: int(rng.integers(0, 4)) for group in groups} for _ in range(random_states))
    unique = {}
    for policy in policies:
        unique.setdefault(render(doc, policy), policy)
    return list(unique.values())
