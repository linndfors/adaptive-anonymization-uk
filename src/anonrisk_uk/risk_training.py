"""Validation-only model selection for contextual unit-risk estimation.

The functions in this module consume an already generated S2 contribution bank.
They never query the held-out test split. This separation makes expensive label
generation resumable and keeps model selection independent of final evaluation.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .adaptive_text import UnitRiskModel
from .risk_model import ranking_quality

DEFAULT_MODEL_GRID = [
    {"max_iter": 100, "learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 10, "l2_regularization": 0.0},
    {"max_iter": 200, "learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 20, "l2_regularization": 0.0},
    {"max_iter": 200, "learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 0.1},
    {"max_iter": 300, "learning_rate": 0.03, "max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 1.0},
]


@dataclass
class RiskTrainingResult:
    """All outputs needed to persist and audit a selected risk model."""

    model: UnitRiskModel
    hyperparameter_search: pd.DataFrame
    learning_curve: pd.DataFrame
    validation_units: pd.DataFrame
    validation_summary: pd.DataFrame
    selection: dict


def _validate_tables(features: pd.DataFrame, labels: pd.DataFrame, split: str) -> None:
    required = {"person", "pos", "unit", "label", "target_delta"}
    missing = required - set(labels)
    if missing:
        raise ValueError(f"{split} labels are missing columns: {sorted(missing)}")
    if len(features) != len(labels) or not len(labels):
        raise ValueError(f"{split} features and labels must be non-empty and row-aligned")
    if features.isna().any().any() or not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError(f"{split} features contain non-finite values")
    if not np.isfinite(labels.target_delta.to_numpy(dtype=float)).all():
        raise ValueError(f"{split} targets contain non-finite values")


def person_balanced_weights(labels: pd.DataFrame) -> np.ndarray:
    """Give each person equal total weight regardless of their number of units."""
    counts = labels.groupby("person").person.transform("size").to_numpy(dtype=float)
    return 1.0 / counts


def _prediction_metrics(labels: pd.DataFrame, predictions: np.ndarray) -> dict:
    frame = labels.copy()
    frame["prediction"] = predictions
    weights = person_balanced_weights(frame)
    metrics = ranking_quality(frame, "target_delta", "prediction")
    metrics["weighted_mae"] = float(
        np.average(
            np.abs(predictions - frame.target_delta.to_numpy(dtype=float)),
            weights=weights,
        )
    )
    metrics["mae"] = float(np.mean(np.abs(predictions - frame.target_delta.to_numpy(dtype=float))))
    return metrics


def _normalise_grid(grid: Iterable[dict] | None) -> list[dict]:
    candidates = list(grid or DEFAULT_MODEL_GRID)
    if not candidates:
        raise ValueError("risk_model_grid must contain at least one candidate")
    allowed = {"max_iter", "learning_rate", "max_leaf_nodes", "min_samples_leaf", "l2_regularization"}
    normalised = []
    for candidate in candidates:
        unknown = set(candidate) - allowed
        if unknown:
            raise ValueError(f"Unknown risk-model parameters: {sorted(unknown)}")
        normalised.append({**DEFAULT_MODEL_GRID[0], **candidate})
    return normalised


def _selection_order(table: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric == "weighted_mae":
        return table.sort_values(
            ["weighted_mae", "within_doc_spearman", "precision@1", "complexity", "candidate_id"],
            ascending=[True, False, False, True, True],
            na_position="last",
        )
    if metric == "within_doc_spearman":
        return table.sort_values(
            ["within_doc_spearman", "precision@1", "weighted_mae", "complexity", "candidate_id"],
            ascending=[False, False, True, True, True],
            na_position="last",
        )
    raise ValueError("risk_selection_metric must be 'weighted_mae' or 'within_doc_spearman'")


def nested_person_subsets(
    labels: pd.DataFrame, requested_sizes: Iterable[int | None], seed: int
) -> list[tuple[str, np.ndarray]]:
    """Return deterministic nested row subsets without splitting a person."""
    people = np.asarray(sorted(labels.person.astype(str).unique()))
    people = people[np.random.default_rng(seed).permutation(len(people))]
    resolved = []
    for requested in requested_sizes:
        size = len(people) if requested is None else min(int(requested), len(people))
        if size < 1:
            raise ValueError("risk_learning_curve_persons values must be positive or null")
        if size not in [item[0] for item in resolved]:
            selected = set(people[:size])
            rows = np.flatnonzero(labels.person.astype(str).isin(selected).to_numpy())
            resolved.append((size, rows))
    return [("all" if size == len(people) else str(size), rows) for size, rows in resolved]


def fit_risk_model(
    train_features: pd.DataFrame,
    train_labels: pd.DataFrame,
    validation_features: pd.DataFrame,
    validation_labels: pd.DataFrame,
    *,
    seed: int,
    model_grid: Iterable[dict] | None = None,
    context_weights: Iterable[float] = (0.0, 0.25, 0.5, 1.0),
    selection_metric: str = "within_doc_spearman",
    learning_curve_persons: Iterable[int | None] = (50, 100, 200, 400, None),
) -> RiskTrainingResult:
    """Tune on validation data, refit on all train people, and report diagnostics."""
    _validate_tables(train_features, train_labels, "train")
    _validate_tables(validation_features, validation_labels, "validation")
    if list(train_features.columns) != list(validation_features.columns):
        raise ValueError("Train and validation feature schemas differ")
    if set(train_labels.person.astype(str)) & set(validation_labels.person.astype(str)):
        raise ValueError("Train and validation people overlap")

    context_weights = [float(value) for value in context_weights]
    if not context_weights or any(not 0.0 <= value <= 1.0 for value in context_weights):
        raise ValueError("risk_context_weights must contain values between 0 and 1")
    candidates = _normalise_grid(model_grid)
    train_weights = person_balanced_weights(train_labels)
    search_rows = []
    fitted_models = {}
    for candidate_id, parameters in enumerate(candidates):
        model = UnitRiskModel(**parameters).fit(
            train_features,
            train_labels.target_delta.to_numpy(dtype=float),
            seed=seed,
            sample_weight=train_weights,
        )
        fitted_models[candidate_id] = model
        for context_weight in context_weights:
            model.context_weight = context_weight
            metrics = _prediction_metrics(validation_labels, model.predict(validation_features))
            search_rows.append(
                {
                    "candidate_id": candidate_id,
                    **parameters,
                    "context_weight": context_weight,
                    "complexity": parameters["max_iter"] * parameters["max_leaf_nodes"],
                    **metrics,
                }
            )

    search = pd.DataFrame(search_rows)
    chosen = _selection_order(search, selection_metric).iloc[0]
    selected_id = int(chosen.candidate_id)
    selected_parameters = candidates[selected_id]
    model = fitted_models[selected_id]
    model.context_weight = float(chosen.context_weight)

    curve_rows = []
    for label, rows in nested_person_subsets(train_labels, learning_curve_persons, seed):
        subset_labels = train_labels.iloc[rows].reset_index(drop=True)
        subset_features = train_features.iloc[rows].reset_index(drop=True)
        subset_model = UnitRiskModel(**selected_parameters).fit(
            subset_features,
            subset_labels.target_delta.to_numpy(dtype=float),
            seed=seed,
            sample_weight=person_balanced_weights(subset_labels),
        )
        subset_model.context_weight = model.context_weight
        metrics = _prediction_metrics(validation_labels, subset_model.predict(validation_features))
        curve_rows.append(
            {
                "train_persons": subset_labels.person.nunique(),
                "train_queries": subset_labels.pos.nunique(),
                "train_units": len(subset_labels),
                "subset": label,
                **metrics,
            }
        )

    validation_units = validation_labels.copy()
    validation_units["pred_category"] = validation_units.label.map(model.class_prior).fillna(0.0)
    validation_units["pred_model"] = model.predict(validation_features)
    summaries = []
    for predictor in ["pred_category", "pred_model"]:
        summaries.append(
            {
                "predictor": predictor,
                **_prediction_metrics(validation_units, validation_units[predictor].to_numpy(dtype=float)),
            }
        )
    selection = {
        "selection_split": "validation",
        "selection_metric": selection_metric,
        "candidate_id": selected_id,
        "context_weight": model.context_weight,
        "model_parameters": selected_parameters,
        "n_train_people": int(train_labels.person.nunique()),
        "n_train_queries": int(train_labels.pos.nunique()),
        "n_train_units": len(train_labels),
        "n_validation_people": int(validation_labels.person.nunique()),
        "n_validation_queries": int(validation_labels.pos.nunique()),
        "n_validation_units": len(validation_labels),
    }
    return RiskTrainingResult(
        model=model,
        hyperparameter_search=search,
        learning_curve=pd.DataFrame(curve_rows),
        validation_units=validation_units,
        validation_summary=pd.DataFrame(summaries),
        selection=selection,
    )
