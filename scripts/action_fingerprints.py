"""Stable fingerprints of fitted numerical state, independent of process IDs."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

import numpy as np


def stable_digest(value):
    """Canonicalize sets, mappings and arrays; ignore sklearn's address cache.

    TfidfVectorizer._stop_words_id stores id(stop_words), which changes when a
    fitted vectorizer is loaded and used in another process. It is a cache
    invalidation marker, not a learned parameter. Pickle byte equality therefore
    cannot establish equality of two fitted representations.
    """
    digest = hashlib.sha256()

    def visit(item):
        kind = f"{type(item).__module__}.{type(item).__qualname__}"
        digest.update(kind.encode() + b"\0")
        if isinstance(item, np.ndarray):
            digest.update(str(item.dtype).encode() + repr(item.shape).encode())
            digest.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, np.generic):
            visit(item.item())
        elif isinstance(item, Mapping):
            for key in sorted(item, key=repr):
                visit(key)
                visit(item[key])
        elif isinstance(item, (set, frozenset)):
            for element in sorted(stable_digest(element) for element in item):
                digest.update(element.encode())
        elif isinstance(item, (list, tuple)):
            for element in item:
                visit(element)
        elif isinstance(item, type):
            digest.update(f"{item.__module__}.{item.__qualname__}".encode())
        elif hasattr(item, "__dict__"):
            visit({k: v for k, v in vars(item).items() if k != "_stop_words_id"})
        else:
            digest.update(repr(item).encode())
        digest.update(b"\xff")

    visit(value)
    return digest.hexdigest()


def frozen_features_digest(estimator):
    return stable_digest((estimator.statistics, estimator.codec, estimator.retention_vectorizer))
