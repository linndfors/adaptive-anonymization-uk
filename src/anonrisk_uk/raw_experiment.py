"""Evaluate the raw-text prototype without modifying legacy notebook results."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .e3jsi import gold_spans
from .pipeline import span_role


def training_group_prior(ablation, split):
    """Validate label provenance before aggregation and reject test contributions."""
    required = {"pos", "group", "mode", "delta_S2"}
    if not required <= set(ablation):
        raise ValueError(f"Missing ablation columns: {required - set(ablation)}")
    positions = ablation.pos.to_numpy()
    if not np.issubdtype(positions.dtype, np.integer) or np.any(positions < 0) or np.any(positions >= len(split)):
        raise ValueError("Ablation positions do not match the stored split")
    if not split.iloc[positions]["split"].eq("train").all():
        raise ValueError("Risk priors must be derived from training contributions only")
    if (
        "person" in ablation
        and "person_cluster" in split
        and not np.array_equal(
            ablation.person.astype(str).to_numpy(), split.iloc[positions].person_cluster.astype(str).to_numpy()
        )
    ):
        raise ValueError("Training-ablation identities do not match the split")
    if not ablation["mode"].eq("addback").all() or not np.isfinite(ablation.delta_S2).all():
        raise ValueError("Finite training S2 add-back ablation values are required")
    if ablation.duplicated(["pos", "group"]).any():
        raise ValueError("Duplicate training labels for the same position/group")
    # Average signed contributions before clipping a negative class mean.
    return ablation.groupby("group").delta_S2.mean().clip(lower=0).to_dict()


def leakage_details(doc, decisions, mode):
    """Measure transformed-span coverage in original-text coordinates.

    This is residual gold-character coverage, not a guarantee that a fact can or
    cannot be reconstructed. Repeated mentions are counted separately.
    """
    rows = []
    modified = [d for d in decisions if d.action != "keep" and d.replacement != d.text]
    for s in gold_spans(doc):
        covered = set()
        for d in modified:
            covered.update(range(max(s.start, d.span.start), min(s.end, d.span.end)))
        fraction = len(covered) / (s.end - s.start)
        rows.append(
            {
                "doc_id": str(doc["source_row_id"]),
                "mode": mode,
                "label": s.label,
                "role": span_role(s),
                "source": s.source,
                "start": s.start,
                "end": s.end,
                "n_chars": s.end - s.start,
                "changed_chars": len(covered),
                "changed_coverage": fraction,
                "fully_unmodified": fraction == 0,
                "residual_chars": fraction < 1,
                "leak_under_50pct": fraction < 0.5,
            }
        )
    return rows


def leakage_summary(frame):
    rows = []
    for mode, part in frame.groupby("mode", sort=False):
        for role, subset in [("ALL", part), *list(part.groupby("role", sort=False))]:
            rows.append(
                {
                    "mode": mode,
                    "role": role,
                    "n_entities": len(subset),
                    "n_docs": subset.doc_id.nunique(),
                    "fully_unmodified_share": subset.fully_unmodified.mean(),
                    "residual_entity_share": subset.residual_chars.mean(),
                    "leak_under_50pct": subset.leak_under_50pct.mean(),
                    "changed_char_coverage": subset.changed_chars.sum() / subset.n_chars.sum(),
                }
            )
    return pd.DataFrame(rows)


def raw_queries(pop, positions, anonymizer=None, exclude=()):
    """Pass complete rendered originals through detection and transformation."""
    texts = [pop.codec.render(pop.extracted[k], {}, exclude=set(exclude)) for k in positions]
    if anonymizer is None:
        return texts
    return [anonymizer.anonymize(text)[0] for text in texts]


def evaluate_raw_policy(pop, positions, anonymizer, retention_vectorizer, tasks=(), policy=None):
    """Evaluate S2 against a fixed pool and utility labels from originals."""
    from .attack import ranking_result
    from .statistics import metric_draws

    positions = np.asarray(positions, dtype=int)
    if not len(positions):
        raise ValueError("No linked queries are available for S2 evaluation")
    extracted = [pop.extracted[k] for k in positions]
    original = raw_queries(pop, positions)
    if policy is not None:
        queries = [pop.codec.render(ex, policy) for ex in extracted]
    else:
        queries = raw_queries(pop, positions, anonymizer)
    targets, excludes = pop.targets_and_excludes(positions, "S2")
    result = ranking_result(pop.index, queries, targets, excludes, batch_size=32)[0]
    detail = result.to_frame().assign(pos=positions, person=pop.clusters[positions])
    A, B = retention_vectorizer.transform(original), retention_vectorizer.transform(queries)
    detail["retention"] = np.asarray(A.multiply(B).sum(axis=1)).ravel()
    for task in tasks:
        y = task.labels(extracted)
        valid = np.array([v in task.classes_ for v in y])
        prediction = np.full(len(positions), None, dtype=object)
        if valid.any():
            if policy is not None:
                utility_texts = [pop.codec.render(ex, policy, exclude=task.exclude) for ex in extracted]
            else:
                utility_texts = raw_queries(pop, positions, anonymizer, exclude=task.exclude)
            prediction[valid] = task.model.predict([utility_texts[i] for i in np.flatnonzero(valid)])
        detail[f"{task.name}_true"] = [v if keep else None for v, keep in zip(y, valid)]
        detail[f"{task.name}_pred"] = prediction
    classes = {task.name: sorted(task.classes_) for task in tasks}
    summary = {key: value[0] for key, value in metric_draws(detail, classes).items()}
    summary.update(n_resumes=len(detail), n_persons=detail.person.nunique())
    for task in tasks:
        summary[f"{task.name}_n"] = int(detail[f"{task.name}_true"].notna().sum())
    return summary, detail


def select_operating_point(validation, target_recall):
    """Select maximum retention among feasible validation points.

    If no point is feasible, select minimum Recall@1 and report that the target
    was not met.
    """
    if not 0 <= target_recall <= 1 or validation.empty:
        raise ValueError("Validation scores and target_recall in [0, 1] are required")
    if "split" not in validation or not validation["split"].eq("val").all():
        raise ValueError("The operating point must be selected on validation data, not test data")
    if not np.isfinite(validation[["recall@1", "retention"]].to_numpy()).all():
        raise ValueError("Validation metrics are invalid")
    feasible = validation[validation["recall@1"] <= target_recall]
    if len(feasible):
        row = feasible.sort_values(
            ["retention", "recall@1", "tau_mask", "tau_generalize"], ascending=[False, True, True, True]
        ).iloc[0]
        return row, "target_met_on_validation"
    row = validation.sort_values(
        ["recall@1", "retention", "tau_mask", "tau_generalize"], ascending=[True, False, True, True]
    ).iloc[0]
    return row, "target_unmet_on_validation"
