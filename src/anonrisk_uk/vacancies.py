"""Vacancy retrieval and job-fit preservation for Ukrainian resumes.

There are no resume--vacancy relevance labels in Work.ua. We therefore treat
the professional, privacy-neutral projection of the original resume as a fixed
reference query. An anonymized version is useful when it preserves both the
reference vacancies and their ranking. This is a reproducible proxy, not a
claim that a recruiter would select the candidate.
"""

from __future__ import annotations

import ast
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer

from .attack import TOKEN_PATTERN


def load_vacancies(path, limit=None):
    path = Path(path)
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
        frame = pd.DataFrame(rows)
    else:
        frame = pd.read_csv(path, dtype={"id": str}, nrows=limit, keep_default_na=False)
    if not {"id", "title", "description_text"} <= set(frame):
        raise ValueError("Vacancy data must contain id, title, and description_text columns")
    frame["id"] = frame.id.astype(str)
    return frame.drop_duplicates("id").reset_index(drop=True)


def _skills(value):
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        try:
            values = json.loads(value)
        except json.JSONDecodeError:
            values = ast.literal_eval(value)
        if not isinstance(values, list):
            raise ValueError("The vacancy skills field must be a list")
        return [str(item).strip() for item in values if str(item).strip()]
    return []


def _clean(value) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _normal_title(value) -> str:
    return re.sub(r"\W+", " ", _clean(value).casefold()).strip()


def _clean_query(value: str) -> str:
    """Drop UI placeholders and contact-field labels from retrieval queries."""
    value = re.sub(r"\[[^\]]+\]", " ", value)
    value = re.sub(
        r"(?iu)\b(?:e-?mail|email|електронна\s+пошта|телефон|phone|адреса)\s*:",
        " ",
        value,
    )
    value = re.sub(r"(?iu)https?://\S+|www\.\S+|\b\S+@\S+\b", " ", value)
    return re.sub(r"\s+", " ", value).strip()


@dataclass
class VacancyReference:
    """Frozen relevance judgement induced only from the original resume."""

    query: str
    scores: np.ndarray
    top_indices: tuple[int, ...]
    jobs: list[dict]
    scorable: bool


@dataclass(frozen=True)
class VacancyPreservation:
    utility: float
    ndcg: float
    weighted_recall: float
    signal_retention: float
    overlap: float
    scorable: bool


class VacancyFitModel:
    """Trainable latent vacancy matcher and job-fit preservation metric.

    Company identity, salary and location are deliberately excluded: identity
    signals must not make a resume look professionally more useful. Explicit
    title, skills and employment requirements form the training representation.
    TF-IDF learns corpus weights and truncated SVD learns a latent job space.
    """

    def __init__(
        self, k=10, *, n_components=128, max_features=100_000, random_state=42, max_per_company=2, max_per_title=1
    ):
        if k < 1:
            raise ValueError("k > 0")
        self.k = k
        self.n_components = n_components
        self.max_features = max_features
        self.random_state = random_state
        self.max_per_company = max_per_company
        self.max_per_title = max_per_title

    def fit(self, frame):
        frame = frame.copy().reset_index(drop=True)
        self.ids = frame.id.astype(str).tolist()
        self.titles = [_clean(value) or "Untitled" for value in frame.title]
        skill_values = frame["skills"] if "skills" in frame else pd.Series([[]] * len(frame))
        self.skills = [_skills(value) for value in skill_values]
        company_values = frame["company_id"] if "company_id" in frame else [None] * len(frame)
        self.companies = [
            str(value) if value not in (None, "") and not pd.isna(value) else f"missing:{vacancy_id}"
            for value, vacancy_id in zip(company_values, self.ids, strict=True)
        ]
        remote_values = frame["is_remote"] if "is_remote" in frame else [False] * len(frame)
        self.remote = [bool(value) for value in remote_values]
        employment_values = frame["employment_type"] if "employment_type" in frame else [""] * len(frame)
        language_values = frame["language"] if "language" in frame else [""] * len(frame)
        employment = [_clean(value) for value in employment_values]
        language = [_clean(value) for value in language_values]

        # Repetition is an explicit field prior: occupation and skills dominate
        # generic conditions, while no employer/free-description text enters training.
        documents = [
            " ".join([title, title, title, " ".join(items), " ".join(items), kind, lang])
            for title, items, kind, lang in zip(self.titles, self.skills, employment, language, strict=True)
        ]
        self.vectorizer = TfidfVectorizer(
            token_pattern=TOKEN_PATTERN,
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=self.max_features,
            sublinear_tf=True,
            dtype=np.float32,
        )
        matrix = self.vectorizer.fit_transform(documents)
        components = min(self.n_components, matrix.shape[0] - 1, matrix.shape[1] - 1)
        if components < 2:
            raise ValueError("Insufficient vacancies or features to train the latent vacancy model")
        self.svd = TruncatedSVD(n_components=components, n_iter=7, random_state=self.random_state)
        self.normalizer = Normalizer(copy=False)
        self.embeddings = self.normalizer.fit_transform(self.svd.fit_transform(matrix)).astype(np.float32)
        self.training_summary = {
            "n_vacancies": len(frame),
            "vocabulary_size": len(self.vectorizer.vocabulary_),
            "n_components": components,
            "explained_variance": float(self.svd.explained_variance_ratio_.sum()),
        }
        self.k = min(self.k, len(frame))
        return self

    def _scores(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, len(self.ids)), dtype=np.float32)
        texts = [_clean_query(text) for text in texts]
        matrix = self.vectorizer.transform(texts)
        queries = self.normalizer.transform(self.svd.transform(matrix)).astype(np.float32)
        return np.maximum(queries @ self.embeddings.T, 0.0)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> VacancyFitModel:
        with Path(path).open("rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, cls):
            raise TypeError("The artifact is not a VacancyFitModel")
        return model

    def _top(self, scores: np.ndarray) -> np.ndarray:
        order = np.lexsort((np.arange(len(scores)), -scores))
        selected, company_counts, title_counts = [], {}, {}
        for position in order:
            company = self.companies[position]
            title = _normal_title(self.titles[position])
            if company_counts.get(company, 0) >= self.max_per_company:
                continue
            if title_counts.get(title, 0) >= self.max_per_title:
                continue
            selected.append(int(position))
            company_counts[company] = company_counts.get(company, 0) + 1
            title_counts[title] = title_counts.get(title, 0) + 1
            if len(selected) == self.k:
                break
        return np.asarray(selected, dtype=int)

    def reference(self, original: str) -> VacancyReference:
        scores = self._scores([original])[0]
        top = self._top(scores)
        scorable = bool(len(top) == self.k and scores[top[-1]] > 0)
        maximum = max(float(scores[top[0]]) if len(top) else 0.0, 1e-12)
        jobs = [
            {
                "id": self.ids[position],
                "title": self.titles[position],
                "skills": self.skills[position][:8],
                "remote": self.remote[position],
                "reference_match": float(scores[position] / maximum),
            }
            for position in top
        ]
        return VacancyReference(original, scores, tuple(top.tolist()), jobs, scorable)

    def evaluate(self, reference: VacancyReference, text: str) -> VacancyPreservation:
        if not reference.scorable:
            return VacancyPreservation(np.nan, np.nan, np.nan, np.nan, np.nan, False)
        current = self._scores([text])[0]
        new_top = self._top(current)
        old_top = np.asarray(reference.top_indices, dtype=int)
        maximum = max(float(reference.scores[old_top[0]]), 1e-12)
        relevance = np.clip(reference.scores / maximum, 0.0, 1.0)
        discounts = 1.0 / np.log2(np.arange(2, len(old_top) + 2))
        ideal = float(np.sum(relevance[old_top] * discounts))
        retrieved = current[new_top] > 0
        dcg = float(np.sum(relevance[new_top] * retrieved * discounts[: len(new_top)]))
        ndcg = min(dcg / ideal, 1.0) if ideal > 0 else 0.0

        weights = relevance[old_top] * discounts
        present = np.isin(old_top, new_top) & (current[old_top] > 0)
        weighted_recall = float(weights[present].sum() / weights.sum()) if weights.sum() else 0.0
        ratios = np.divide(
            current[old_top],
            reference.scores[old_top],
            out=np.zeros(len(old_top), dtype=float),
            where=reference.scores[old_top] > 0,
        )
        signal = float(np.sum(weights * np.clip(ratios, 0.0, 1.0)) / weights.sum()) if weights.sum() else 0.0
        positive_new = set(new_top[current[new_top] > 0])
        overlap = float(len(set(old_top) & positive_new) / self.k)
        utility = 0.45 * ndcg + 0.35 * weighted_recall + 0.20 * signal
        return VacancyPreservation(
            float(np.clip(utility, 0.0, 1.0)),
            ndcg,
            weighted_recall,
            signal,
            overlap,
            True,
        )

    def score(self, originals, anonymized):
        """Batch compatibility API used by the research pipeline."""
        if len(originals) != len(anonymized):
            raise ValueError("Original and transformed result counts differ")
        rows = []
        for original, anon in zip(originals, anonymized, strict=True):
            reference = self.reference(original)
            result = self.evaluate(reference, anon)
            rows.append(
                {
                    f"vacancy_overlap@{self.k}": result.overlap,
                    f"vacancy_ndcg@{self.k}": result.ndcg,
                    "vacancy_signal_retention": result.signal_retention,
                    "vacancy_job_fit_utility": result.utility,
                    "vacancy_utility_loss_risk": 1.0 - result.utility if result.scorable else np.nan,
                    "vacancy_scorable": result.scorable,
                    "vacancy_original_boundary_tie": False,
                    "vacancy_anon_empty_scores": bool(result.scorable and result.signal_retention == 0),
                }
            )
        return pd.DataFrame(rows)


def retrieval_sanity_check(model: VacancyFitModel, *, n=500, seed=42) -> dict:
    """Masked-view retrieval diagnostic; not a resume-vacancy ground-truth test."""
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(model.ids), size=min(n, len(model.ids)), replace=False)
    queries = []
    for position in chosen:
        held_view = model.skills[position][::2]
        queries.append(" ".join([model.titles[position], *held_view]))
    scores = model._scores(queries)
    hits, reciprocal_ranks, skill_overlaps = [], [], []
    for row, target in zip(scores, chosen, strict=True):
        row[target] = -np.inf
        order = np.lexsort((np.arange(len(row)), -row))[:10]
        target_title = _normal_title(model.titles[target])
        target_skills = {item.casefold() for item in model.skills[target]}
        relevant = []
        overlaps = []
        for position in order:
            candidate_skills = {item.casefold() for item in model.skills[position]}
            union = target_skills | candidate_skills
            jaccard = len(target_skills & candidate_skills) / len(union) if union else 0.0
            overlaps.append(jaccard)
            relevant.append(_normal_title(model.titles[position]) == target_title or jaccard >= 0.5)
        ranks = [rank for rank, value in enumerate(relevant, start=1) if value]
        hits.append(bool(ranks))
        reciprocal_ranks.append(1.0 / ranks[0] if ranks else 0.0)
        skill_overlaps.append(max(overlaps, default=0.0))
    return {
        "protocol": "masked_title_plus_alternating_skills; self vacancy excluded",
        "status": "internal_sanity_only_no_resume_vacancy_labels",
        "n_queries": len(chosen),
        "derived_relevance": "same_normalized_title_or_skill_jaccard>=0.5",
        "recall_at_10": float(np.mean(hits)),
        "mrr_at_10": float(np.mean(reciprocal_ranks)),
        "mean_best_skill_jaccard_at_10": float(np.mean(skill_overlaps)),
    }


# Backward-compatible research API name. New code should use VacancyFitModel.
VacancyUtility = VacancyFitModel
