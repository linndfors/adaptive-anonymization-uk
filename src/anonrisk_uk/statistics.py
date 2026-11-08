"""Person-clustered paired bootstrap and curve comparison without extrapolation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def cluster_weights(clusters, n=1000, seed=42):
    """Resample people and give all records of one person equal multiplicity."""
    codes, persons = pd.factorize(np.asarray(clusters), sort=False)
    if not len(persons) or np.any(codes < 0) or n <= 0:
        raise ValueError("Bootstrap requires non-empty known clusters and n > 0")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(persons), size=(n, len(persons)))
    multiplicities = np.zeros((n, len(persons)), dtype=float)
    np.add.at(multiplicities, (np.arange(n)[:, None], draws), 1)
    return multiplicities[:, codes]


def interval(values, alpha=0.05):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    return tuple(float(v) for v in np.quantile(values, [alpha / 2, 1 - alpha / 2]))


def cluster_mean_ci(values, clusters, n=1000, seed=42):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) != len(clusters):
        raise ValueError("Values and clusters must be one-dimensional and have equal length")
    weights = cluster_weights(clusters, n, seed)
    valid = np.isfinite(values)
    denominator = weights[:, valid].sum(axis=1)
    estimates = np.divide(weights[:, valid] @ values[valid], denominator, out=np.full(n, np.nan), where=denominator > 0)
    return interval(estimates)


def metric_draws(frame, task_classes=None, weights=None):
    """Treat each weight row as one resample and keep macro-F1 classes fixed."""
    weights = np.ones((1, len(frame))) if weights is None else np.asarray(weights, dtype=float)
    if weights.ndim != 2 or weights.shape[1] != len(frame):
        raise ValueError("The weight matrix does not match the evaluation rows")
    out = {}
    for column in [c for c in frame if c.startswith("recall@") or c in {"rr", "retention"}]:
        values = frame[column].to_numpy(dtype=float)
        valid = np.isfinite(values)
        denom = weights[:, valid].sum(axis=1)
        out["mrr" if column == "rr" else column] = np.divide(
            weights[:, valid] @ values[valid], denom, out=np.full(len(weights), np.nan), where=denom > 0
        )
    for task, classes in (task_classes or {}).items():
        classes = list(classes)
        y, pred = frame[f"{task}_true"].to_numpy(), frame[f"{task}_pred"].to_numpy()
        valid = np.array([v in classes and p in classes for v, p in zip(y, pred)])
        w = weights[:, valid]
        y, pred = y[valid], pred[valid]
        truth = np.column_stack([y == c for c in classes]).astype(float)
        guess = np.column_stack([pred == c for c in classes]).astype(float)
        tp = w @ (truth * guess)
        denom = w @ (truth + guess)
        f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0).mean(axis=1)
        n_valid = w.sum(axis=1)
        f1[n_valid == 0] = np.nan
        out[f"{task}_macroF1"] = f1
        out[f"{task}_acc"] = np.divide(
            w @ (y == pred).astype(float), n_valid, out=np.full(len(w), np.nan), where=n_valid > 0
        )
    return out


def metric_intervals(frame, task_classes=None, n=1000, seed=42):
    draws = metric_draws(frame, task_classes, cluster_weights(frame.person, n, seed))
    points = metric_draws(frame, task_classes)
    rows = []
    for metric, values in draws.items():
        lo, hi = interval(values)
        valid = np.ones(len(frame), dtype=bool)
        for task in task_classes or {}:
            if metric.startswith(task + "_"):
                valid = frame[f"{task}_true"].notna() & frame[f"{task}_pred"].notna()
        rows.append(
            {
                "metric": metric,
                "estimate": points[metric][0],
                "ci_low": lo,
                "ci_high": hi,
                "n_resumes": int(np.sum(valid)),
                "n_persons": frame.loc[valid, "person"].nunique(),
                "n_boot_valid": int(np.isfinite(values).sum()),
            }
        )
    return pd.DataFrame(rows)


def paired_differences(candidate, reference, task_classes=None, n=1000, seed=42):
    """Return candidate-minus-reference deltas using paired identity resampling."""
    if (
        not candidate[["pos", "person"]]
        .reset_index(drop=True)
        .equals(reference[["pos", "person"]].reset_index(drop=True))
    ):
        raise ValueError("Paired comparison requires the same resumes in the same order")
    weights = cluster_weights(candidate.person, n, seed)
    a, b = metric_draws(candidate, task_classes, weights), metric_draws(reference, task_classes, weights)
    pa, pb = metric_draws(candidate, task_classes), metric_draws(reference, task_classes)
    rows = []
    for metric in a:
        lo, hi = interval(a[metric] - b[metric])
        valid = np.ones(len(candidate), dtype=bool)
        for task in task_classes or {}:
            if metric.startswith(task + "_"):
                valid = (
                    candidate[f"{task}_true"].notna()
                    & reference[f"{task}_true"].notna()
                    & candidate[f"{task}_pred"].notna()
                    & reference[f"{task}_pred"].notna()
                )
        rows.append(
            {
                "metric": metric,
                "delta": pa[metric][0] - pb[metric][0],
                "ci_low": lo,
                "ci_high": hi,
                "n_resumes": int(np.sum(valid)),
                "n_persons": candidate.loc[valid, "person"].nunique(),
                "n_boot_valid": int(np.isfinite(a[metric] - b[metric]).sum()),
            }
        )
    return pd.DataFrame(rows)


def interpolate_utility(frame, target_risk, metric, risk="recall@1"):
    """Linearly interpolate within the observed range and return NaN outside it."""
    d = frame[[risk, metric]].replace([np.inf, -np.inf], np.nan).dropna()
    d = d.groupby(risk, as_index=False)[metric].max().sort_values(risk)
    if d.empty:
        return {"value": np.nan, "status": "no_data", "risk_min": np.nan, "risk_max": np.nan}
    lo, hi = float(d[risk].min()), float(d[risk].max())
    if not np.isfinite(target_risk) or target_risk < lo or target_risk > hi:
        return {"value": np.nan, "status": "out_of_range", "risk_min": lo, "risk_max": hi}
    return {
        "value": float(np.interp(target_risk, d[risk], d[metric])),
        "status": "observed" if (d[risk] == target_risk).any() else "interpolated",
        "risk_min": lo,
        "risk_max": hi,
    }
