"""Resume views, person clusters, and copy diagnostics without identity inference."""

from __future__ import annotations

import hashlib
import itertools
import re
import unicodedata
from collections import defaultdict

import numpy as np
import pandas as pd

from .workua import FREE_TEXT_GROUPS, GROUPS

VIEWS = ("structured", "free_text", "hybrid")


def render_view(codec, ex, view="hybrid", policy=None, exclude=()):
    if view not in VIEWS:
        raise ValueError(view)
    exclude = set(exclude)
    if view == "structured":
        exclude |= FREE_TEXT_GROUPS
    elif view == "free_text":
        exclude |= set(GROUPS) - FREE_TEXT_GROUPS
    return codec.render(ex, policy or {}, exclude=exclude)


def canonical_text(text):
    value = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    # C++ and C# are skill values whose punctuation is semantically meaningful.
    return " ".join(re.findall(r"\w+(?:\+\+|#)?(?:'\w+)*", value))


def shingles(text, n=3):
    words = canonical_text(text).split()
    return {tuple(words[i : i + n]) for i in range(max(1, len(words) - n + 1))} if words else set()


def pair_kind(a, b, threshold=0.9):
    if not canonical_text(a) or not canonical_text(b):
        return "empty_text", np.nan
    if a == b:
        return "exact", 1.0
    if canonical_text(a) == canonical_text(b):
        return "format_only", 1.0
    A, B = shingles(a), shingles(b)
    sim = len(A & B) / len(A | B)
    return ("near_duplicate" if sim >= threshold else "distinct"), sim


def duplicate_diagnostics(ids, clusters, texts, view, threshold=0.9):
    members = defaultdict(list)
    for i, c in enumerate(clusters):
        members[c].append(i)
    rows = []
    for person, positions in members.items():
        for a, b in itertools.combinations(positions, 2):
            kind, score = pair_kind(texts[a], texts[b], threshold)
            rows.append(
                {
                    "person": str(person),
                    "pos_a": a,
                    "pos_b": b,
                    "id_a": ids[a],
                    "id_b": ids[b],
                    "view": view,
                    "pair_kind": kind,
                    "shingle_jaccard": score,
                }
            )
    return pd.DataFrame(
        rows, columns=["person", "pos_a", "pos_b", "id_a", "id_b", "view", "pair_kind", "shingle_jaccard"]
    )


def query_cohorts(pop, positions, texts, threshold=0.9):
    rows = []
    priority = {"exact": 0, "format_only": 1, "near_duplicate": 2, "distinct": 3, "empty_text": 4}
    for k in positions:
        pairs = [pair_kind(texts[k], texts[t], threshold) for t in pop.members[pop.clusters[k]] if t != k]
        cohort = min((kind for kind, _ in pairs), key=priority.get, default="no_other_resume")
        distinct = sum(kind == "distinct" for kind, _ in pairs)
        rows.append(
            {
                "pos": k,
                "person": str(pop.clusters[k]),
                "cohort": cohort,
                "n_same_person_targets": len(pairs),
                "n_distinct_targets": distinct,
                "empty_query": not bool(canonical_text(texts[k])),
            }
        )
    return pd.DataFrame(rows)


def distinct_targets(pop, positions, texts, threshold=0.9):
    """Build a separate scenario targeting only substantively different resumes.

    This does not replace ordinary S2, where every other resume belonging to the
    same person is a valid target.
    """
    eligible, targets, excludes = [], [], []
    for k in positions:
        other = [t for t in pop.members[pop.clusters[k]] if t != k]
        target = [t for t in other if pair_kind(texts[k], texts[t], threshold)[0] == "distinct"]
        if target and canonical_text(texts[k]):
            eligible.append(k)
            targets.append(target)
            excludes.append([k, *[t for t in other if t not in target]])
    return np.asarray(eligible, dtype=int), targets, excludes


def split_persons_and_copies(ids, clusters, texts, seed=42):
    """Place person clusters and shared exact normalized texts in one split block.

    Text equality merges only split blocks, never people or S2 targets. Empty and
    short templates do not merge blocks, nor do cross-person near-duplicates.
    """
    persons = list(dict.fromkeys(str(c) for c in clusters))
    parent = {p: p for p in persons}

    def find(p):
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    seen = {}
    for c, text in zip(clusters, texts):
        canonical = canonical_text(text)
        if len(canonical.split()) < 8:
            continue
        key = hashlib.sha256(canonical.encode()).hexdigest()
        c = str(c)
        if key in seen:
            union(c, seen[key])
        else:
            seen[key] = c
    blocks = defaultdict(list)
    for i, c in enumerate(clusters):
        blocks[find(str(c))].append(i)
    counts = pd.Series(clusters).value_counts()
    linked, singleton = [], []
    for b, positions in blocks.items():
        (linked if any(counts[clusters[k]] >= 2 for k in positions) else singleton).append(b)
    if len(linked) < 3:
        raise ValueError("At least three independent multi-resume blocks are required for train/validation/test")
    rng, assignments = np.random.default_rng(seed), {}
    for group in [linked, singleton]:
        group = list(rng.permutation(group))
        n = len(group)
        n_test = max(1, round(0.2 * n)) if n >= 3 else 0
        n_val = max(1, round(0.1 * n)) if n >= 3 else 0
        for i, block in enumerate(group):
            assignments[block] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return pd.DataFrame(
        {
            "id": ids,
            "person_cluster": [str(c) for c in clusters],
            "split_block": [find(str(c)) for c in clusters],
            "split": [assignments[find(str(c))] for c in clusters],
        }
    )


def source_audit(df, extracted):
    rows = [
        {
            "field": g,
            "kind": "free_text" if g in FREE_TEXT_GROUPS else "structured",
            "n_nonempty": sum(bool(ex.get(g)) for ex in extracted),
        }
        for g in GROUPS
    ]
    for field in ["resume_text", "raw_text", "description", "text"]:
        if field in df:
            rows.append(
                {
                    "field": field,
                    "kind": "additional_raw_column",
                    "n_nonempty": int(df[field].fillna("").astype(str).str.strip().ne("").sum()),
                }
            )
    return pd.DataFrame(rows)
