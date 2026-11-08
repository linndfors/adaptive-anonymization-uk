"""Training and evaluation of explicit entity decisions, separate from execution."""

from __future__ import annotations

import copy
import hashlib
import json
import pickle
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from tqdm.auto import tqdm

from .action_learning import lexical_retention
from .action_research import POLICIES, context
from .action_space import semantic_retention
from .attack import TfidfIndex, ranking_result
from .entity_policy import (
    EntityPolicyModel,
    evaluate_rankers,
    oracle_rows,
    proxy_tracking,
    release_sequential,
    release_with_policy,
    sequential_oracle_rows,
    validate_policy,
)
from .policy_execution import VERSION, ToolRouter, execute_fixed
from .statistics import metric_draws, metric_intervals, paired_differences


def make_router(bundle, config):
    llm = None
    if config.get("llm_provider") == "openai":
        from .policy_semantics import LazySemanticTool

        llm = LazySemanticTool(config)
    return ToolRouter(bundle["codec"], config, llm=llm)


OBJECTIVES = {
    "v2_mixed": "0.5*S2_effect + 0.5*S1_effect - cost_weight*utility_loss; effects averaged over current and isolated contexts",
    "v3_s2_least_destructive": (
        "S2_effect only (mean of current and isolated contexts); among feasible changes with gain > noise_margin "
        "and gain >= relative_tolerance*best, the lowest severity, then the higher gain; else KEEP. "
        "objective column = S2_effect - cost_weight*severity (for regret only)"
    ),
    "sequential_s2_margin": (
        "greedy trajectory from mandatory protection; per step and entity, select_least_destructive on the drop of "
        "the normalised S2 margin (true other resume vs best other person) in the current state; the labelled "
        "change with the best gain per severity is applied; context = step"
    ),
}
POLICY_MODES = ("independent", "sequential")


def policy_mode(config):
    mode = config.get("policy_mode", "independent")
    if mode not in POLICY_MODES:
        raise ValueError(f"Unknown policy_mode: {mode!r}")
    return mode


def proxy_k(config):
    """Neighbour pseudo-target count when the release proxy is in use, else None."""
    return config.get("neighbor_k", 3) if config.get("plan_check", "self_exposure") == "neighbor_proxy" else None


def run_labels(config):
    from .research import _checkpoint, _save_json

    objective = config.get("label_objective", "v2_mixed")
    sequential = policy_mode(config) == "sequential"
    if sequential:
        objective = "sequential_s2_margin"
    floors = {
        "max_units": config.get("supervision_units_per_doc", 24),
        "utility_floor": config["action_lexical_floor"],
        "content_floor": config["action_content_floor"],
        "cost_weight": config["action_cost_weight"],
        "skill_floor": config.get("action_skill_floor", 0.5),
        "noise_margin": config.get("label_noise_margin", 0.01),
        "relative_tolerance": config.get("label_relative_tolerance", 0.8),
    }

    out, cc, bundle, pop, vec, positions, docs, _ = context(config, ["train", "val"])
    router = make_router(bundle, config)
    counts = {}
    for split in ["train", "val"]:
        examples, effects, paths, oracle_counts = [], [], [], {}
        for pos in tqdm(positions[split], desc=f"Counterfactual action supervision: {split}"):
            if sequential:
                rows, outcomes, path = _checkpoint(
                    cc,
                    "sequential_supervision",
                    f"{split}_{pos}",
                    lambda pos=int(pos): sequential_oracle_rows(
                        pop,
                        pos,
                        docs[pos],
                        router,
                        vec,
                        **floors,
                        rr_target=config.get("oracle_rr_target", 0.5),
                        max_steps=config.get("oracle_max_steps", 8),
                        neighbor_k=config.get("neighbor_k", 3),
                    ),
                )
                examples.append(rows)
                effects.append(outcomes)
                paths.append(path)
                continue
            rows, outcomes = _checkpoint(
                cc,
                "entity_supervision",
                f"{split}_{pos}",
                lambda pos=int(pos): oracle_rows(
                    pop,
                    pos,
                    docs[pos],
                    router,
                    vec,
                    max_units=config.get("supervision_units_per_doc", 24),
                    utility_floor=config["action_lexical_floor"],
                    content_floor=config["action_content_floor"],
                    cost_weight=config["action_cost_weight"],
                    skill_floor=config.get("action_skill_floor", 0.5),
                    objective=objective,
                    noise_margin=config.get("label_noise_margin", 0.01),
                    relative_tolerance=config.get("label_relative_tolerance", 0.8),
                    neighbor_k=proxy_k(config),
                ),
            )
            examples.append(rows)
            effects.append(outcomes)
        table = pd.concat(examples, ignore_index=True)
        outcomes = pd.concat(effects, ignore_index=True)
        table.to_csv(out / f"entity_{split}.csv", index=False)
        outcomes.to_csv(out / f"action_outcomes_{split}.csv", index=False)
        if sequential:
            trajectories = pd.concat(paths, ignore_index=True)
            trajectories.to_csv(out / f"oracle_trajectories_{split}.csv", index=False)
            ends = trajectories[trajectories.stop_reason.notna()]
            oracle_counts = {
                "oracle_stops": ends.stop_reason.value_counts().to_dict(),
                "oracle_steps_mean": float(ends.step.mean()),
                "oracle_final_s2_rr_mean": float(ends.s2_rr.mean()),
            }
        name = "training" if split == "train" else "validation"
        table[["pos", "person", "unit", "context", "features_json"]].to_csv(
            out / f"unit_{name}_features.csv", index=False
        )
        table.drop(columns="features_json").to_csv(out / f"unit_{name}_labels.csv", index=False)
        counts[split] = {
            "documents": len(positions[split]),
            "people": table.person.nunique(),
            "entity_contexts": len(table),
            "outcomes": len(outcomes),
            "decisions": table.target_decision.value_counts().to_dict(),
            "risk_effect_rate": float(table.risk_effect.mean()),
        }
        counts[split].update(oracle_counts)
    _save_json(
        out / "risk_label_manifest.json",
        {
            "approach": VERSION,
            "splits": counts,
            "target": "best feasible measured action/level, with KEEP winning ties",
            "label_objective": objective,
            "objective": OBJECTIVES[objective],
            "noise_margin": config.get("label_noise_margin", 0.01),
            "relative_tolerance": config.get("label_relative_tolerance", 0.8),
            "generalization_catalog": config.get("generalization_catalog", "regex"),
            "policy_mode": policy_mode(config),
            **(
                {
                    "contexts": "greedy oracle step (0 = mandatory only)",
                    "oracle_rr_target": config.get("oracle_rr_target", 0.5),
                    "oracle_max_steps": config.get("oracle_max_steps", 8),
                    "neighbor_k": config.get("neighbor_k", 3),
                }
                if sequential
                else {
                    "contexts": ["mandatory_only", "generalized"],
                    "entity_isolation": "other quasi-identifiers masked to expose redundant contributions",
                }
            ),
            "test_used_for_labels": False,
        },
    )
    print(json.dumps(counts, ensure_ascii=False, indent=2), flush=True)


def run_training(config):
    from .research import _save_json

    out = Path(config["out"]) / "pipeline"
    train, val = [pd.read_csv(out / f"entity_{split}.csv", dtype={"person": str}) for split in ("train", "val")]
    outcomes = pd.read_csv(out / "action_outcomes_val.csv")
    if set(train.person) & set(val.person):
        raise ValueError("Training/validation identities overlap")
    if val.person.nunique() < config.get("min_validation_people", 1):
        raise ValueError("Not enough independent validation people; increase the bounded population sample")
    # Two fits, three small confidence calibrations each; no test selection.
    candidates, reports = [], []
    for balanced in [False, True]:
        model = EntityPolicyModel(max_iter=config.get("policy_iterations", 60), balanced=balanced, seed=config["seed"])
        model.fit(train)
        for threshold in [0.0, 0.6, 0.8]:
            calibrated = copy.copy(model)
            calibrated.keep_threshold = threshold
            calibrated.parameters = {**model.parameters, "keep_threshold": threshold}
            metrics, predicted = validate_policy(calibrated, val, outcomes)
            metrics.update(candidate_id=len(candidates), balanced=balanced, keep_threshold=threshold)
            candidates.append((calibrated, metrics, predicted))
            reports.append(metrics)
    model, metrics, predicted = min(
        candidates, key=lambda item: (item[1]["person_balanced_regret"], -item[1]["macro_f1"])
    )
    risk_eval = evaluate_rankers(model, train, val, noise_margin=config.get("label_noise_margin", 0.01))
    if config.get("release_ranker", "risk_head") == "validated":
        model.ranker = risk_eval["ranker"]
    else:
        risk_eval.update(
            validated_choice=risk_eval["chosen_ranker"],
            chosen_ranker="risk_head",
            ranker={"name": "risk_head"},
            reason="release_ranker=risk_head in config; evaluation only",
        )
    risk_eval["release_ranker_config"] = config.get("release_ranker", "risk_head")
    _save_json(out / "risk_eval.json", risk_eval)
    if "neighbor_rr_gain" in outcomes:
        proxy_eval = proxy_tracking(outcomes, noise_margin=config.get("label_noise_margin", 0.01), seed=config["seed"])
        proxy_eval["split"] = "validation"
        _save_json(out / "proxy_eval.json", proxy_eval)
        print(json.dumps(proxy_eval["results"], indent=2), flush=True)
    status = (
        "validation_benefit_observed"
        if metrics["person_balanced_regret"] < metrics["keep_regret"] - 1e-6
        else "benefit_not_demonstrated"
    )
    pd.DataFrame(reports).to_csv(out / "policy_model_validation.csv", index=False)
    pd.DataFrame(reports).to_csv(out / "risk_model_validation.csv", index=False)
    predicted.to_csv(out / "policy_validation_predictions.csv", index=False)
    pd.DataFrame(
        classification_report(
            predicted.target_decision, predicted.predicted_decision, output_dict=True, zero_division=0
        )
    ).T.to_csv(out / "policy_class_report.csv")
    _save_json(
        out / "risk_crossfit_audit.json",
        {
            "method": "GroupKFold by person",
            "folds": model.risk_crossfit_groups,
            "out_of_fold_risk_mae": model.oof_risk_mae,
            "risk_prediction_is_policy_feature": True,
        },
    )
    pd.DataFrame(
        [
            {
                "train_people": train.person.nunique(),
                "entity_contexts": len(train),
                "note": "bounded fit; no full-data learning-curve search",
            }
        ]
    ).to_csv(out / "risk_learning_curve.csv", index=False)
    _save_json(
        out / "risk_model_selection.json",
        {
            "approach": VERSION,
            "status": status,
            "model": "multiclass entity-action classifier + risk regression head",
            "selected_parameters": model.parameters,
            "validation": metrics,
            "train_people": train.person.nunique(),
            "validation_people": val.person.nunique(),
            "decisions_seen": train.target_decision.value_counts().to_dict(),
            "rare_training_classes": train.target_decision.value_counts()[lambda s: s < 10].to_dict(),
            "risk_conditioning": "person-group out-of-fold risk estimates for policy training; final risk head at inference",
            "release_ranker": risk_eval["chosen_ranker"],
            "input_contract": "entity, document state and action capabilities only; no tool/replacement/outcome features",
            "test_used": False,
            "training_repeated_on_validation": False,
        },
    )
    with (out / "unit_risk_model.pkl").open("wb") as handle:
        pickle.dump(model, handle)
    print(pd.DataFrame(reports).to_string(index=False), flush=True)
    print("Model status:", status, flush=True)
    print("Release ranker:", risk_eval["chosen_ranker"], "-", risk_eval["reason"], flush=True)


def run_pipeline(config):
    from .research import _checkpoint, _save_json

    started = time.monotonic()
    out, cc, bundle, pop, vec, positions, docs, _ = context(config, ["val", "test"])
    router = make_router(bundle, config)
    model_path = out / "unit_risk_model.pkl"
    with model_path.open("rb") as handle:
        model = pickle.load(handle)
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    cc["_checkpoint_signatures"] = {
        "entity_release": hashlib.sha256((cc["_checkpoint_signature"] + model_hash).encode()).hexdigest()[:24]
    }
    settings = {
        "max_steps": config.get("max_steps", 6),
        "utility_floor": config["action_lexical_floor"],
        "content_floor": config["action_content_floor"],
        "min_self_gain": config.get("action_min_self_gain", 0.01),
        "self_rr_target": config.get("action_self_rr_target", 0.5),
        "skill_floor": config.get("action_skill_floor", 0.5),
    }
    release = release_with_policy
    if policy_mode(config) == "sequential":
        release = release_sequential
        settings["neighbor_k"] = config.get("neighbor_k", 3)
        settings["continue_on_keep"] = config.get("sequential_continue_on_keep", False)
    if proxy_k(config):
        settings.update(
            plan_check="neighbor_proxy",
            neighbor_k=proxy_k(config),
            min_self_gain=config.get("plan_min_proxy_gain", 0.01),
        )
    indexes = {"BM25": pop.index}
    if config.get("use_char_tfidf", True):
        indexes["char-TF-IDF"] = TfidfIndex(max_features=20000).fit(bundle["texts"])
    # Optionally select the operational edit budget on validation outcomes of the
    # complete policy. Entity-label agreement alone is not end-to-end privacy validation.
    select_budget = config.get("select_edit_budget", True)
    calibration = []
    targets, excludes = pop.targets_and_excludes(positions["val"], "S2")
    for budget in config.get("policy_edit_budgets", [settings["max_steps"]]) if select_budget else []:
        queries, retains = [], []
        for pos in positions["val"]:
            pos = int(pos)
            text, _, _, _ = release(
                docs[pos], router, model, pop.index, vec, pos, **{**settings, "max_steps": int(budget)}
            )
            queries.append(text)
            retains.append(float(lexical_retention(vec, docs[pos].text, [text])[0]))
        recalls = {
            name: float(ranking_result(index, queries, targets, excludes)[0].recalls[1].mean())
            for name, index in indexes.items()
        }
        calibration.append(
            {
                "max_steps": int(budget),
                "worst_recall": max(recalls.values()),
                "retention": float(np.mean(retains)),
                **{f"recall_{name}": value for name, value in recalls.items()},
            }
        )
    reachable = [row for row in calibration if row["worst_recall"] <= config["target_validation_recall"]]
    if select_budget:
        chosen = (
            max(reachable, key=lambda row: (row["retention"], -row["max_steps"]))
            if reachable
            else min(calibration, key=lambda row: (row["worst_recall"], -row["retention"], row["max_steps"]))
        )
        settings["max_steps"] = chosen["max_steps"]
        pd.DataFrame(calibration).to_csv(out / "policy_operating_validation.csv", index=False)
    frames, intervals, traces, decisions, operations, available, stores, releases = [], [], [], [], [], [], {}, {}
    test_summary = None
    for split in ["val", "test"]:
        summary = []
        for policy in tqdm(POLICIES, desc=f"Decision then execution: {split}"):
            queries, contents, reasons, thresholds = [], [], [], []
            for pos in positions[split]:
                pos = int(pos)
                doc = docs[pos]
                if policy == "adaptive_model":
                    text, actions, trace, audit = _checkpoint(
                        cc,
                        "entity_release",
                        f"{split}_{pos}",
                        lambda pos=pos, doc=doc: release(doc, router, model, pop.index, vec, pos, **settings),
                    )
                    reasons.append(trace[-1]["stop_reason"])
                    thresholds.append(trace[-1]["threshold_met"])
                    if split == "test":
                        traces.extend({**row, "pos": pos, "policy": policy} for row in trace)
                else:
                    text, actions, audit = execute_fixed(doc, router, policy)
                queries.append(text)
                contents.append(semantic_retention(doc, actions))
                if split == "test":
                    decisions.extend({**row, "pos": pos, "policy": policy} for row in audit)
                    operations.extend(
                        {
                            "pos": pos,
                            "policy": policy,
                            "unit": i,
                            "entity_type": doc.units[i].label,
                            "group": doc.units[i].group,
                            **asdict(a),
                        }
                        for i, a in actions.items()
                    )
            keep = [
                float(lexical_retention(vec, docs[int(pos)].text, [text])[0])
                for pos, text in zip(positions[split], queries)
            ]
            if split == "test":
                releases[policy] = queries
            targets, excludes = pop.targets_and_excludes(positions[split], "S2")
            for attacker, index in indexes.items():
                scored = ranking_result(index, queries, targets, excludes, batch_size=16)[0]
                frame = scored.to_frame().assign(
                    pos=positions[split],
                    person=pop.clusters[positions[split]],
                    retention=keep,
                    content_retention=contents,
                    policy=policy,
                    attacker=attacker,
                )
                if policy == "adaptive_model":
                    frame = frame.assign(stop_reason=reasons, proxy_threshold_met=thresholds)
                    if split == "val":
                        pd.DataFrame({"n_resumes": frame.groupby("stop_reason").size()}).to_csv(
                            out / "validation_stop_reasons.csv"
                        )
                row = {metric: values[0] for metric, values in metric_draws(frame).items()}
                row.update(
                    policy=policy,
                    attacker=attacker,
                    scenario="S2",
                    cohort="all",
                    n_resumes=len(frame),
                    n_persons=frame.person.nunique(),
                    content_retention=float(np.mean(contents)),
                )
                summary.append(row)
                if split == "test":
                    frames.append(frame)
                    stores[attacker, policy] = frame
                    intervals.append(
                        metric_intervals(frame, n=config.get("n_bootstrap", 100), seed=config["seed"]).assign(
                            policy=policy, attacker=attacker
                        )
                    )
        table = pd.DataFrame(summary)
        if split == "val":
            table.to_csv(out / "validation_tradeoffs.csv", index=False)
            table.to_csv(out / "validation_grid.csv", index=False)
            adaptive = table[table.policy.eq("adaptive_model")]
            _save_json(
                out / "operating_point.json",
                {
                    "settings": settings,
                    "selection": "validation_only_edit_budget"
                    if select_budget
                    else "none: all non-keep decisions, utility verifier, max_steps resource cap (null = none)",
                    "validation_worst_recall": float(adaptive["recall@1"].max()),
                    "target": config["target_validation_recall"],
                    "status": "target_met"
                    if adaptive["recall@1"].max() <= config["target_validation_recall"]
                    else "target_unmet",
                },
            )
        else:
            test_summary = table
    for pos in positions["test"]:
        doc = docs[int(pos)]
        for i, unit in enumerate(doc.units):
            for key in router.available(doc, i):
                # Export CAPABILITIES, not pre-generated replacement candidates.
                available.append(
                    {
                        "pos": int(pos),
                        "unit": i,
                        "entity_type": unit.label,
                        "group": unit.group,
                        "original": doc.text[unit.start : unit.end],
                        "decision": key,
                        "operation": key.split(":")[0],
                        "replacement": "generated only after decision",
                        "tool": "resolved after decision",
                        "severity": np.nan,
                    }
                )
    pairs = []
    for attacker in indexes:
        for baseline in [name for name in POLICIES if name != "adaptive_model"]:
            pairs.append(
                paired_differences(
                    stores[attacker, "adaptive_model"],
                    stores[attacker, baseline],
                    n=config.get("n_bootstrap", 100),
                    seed=config["seed"],
                ).assign(attacker=attacker, policy="adaptive_model", reference=baseline)
            )
    test_summary.to_csv(out / "test_summary.csv", index=False)
    pd.concat(frames, ignore_index=True).to_csv(out / "test_per_query.csv", index=False)
    pd.concat(intervals, ignore_index=True).to_csv(out / "test_intervals.csv", index=False)
    pd.concat(pairs, ignore_index=True).to_csv(out / "paired_differences.csv", index=False)
    pd.DataFrame(traces).to_csv(out / "release_traces.csv", index=False)
    pd.DataFrame(decisions).to_csv(out / "entity_decisions.csv", index=False)
    pd.DataFrame(operations).to_csv(out / "release_operations.csv", index=False)
    pd.DataFrame(available).to_csv(out / "available_actions.csv", index=False)
    final = pd.DataFrame(traces)
    final[final.stop_reason.notna()].groupby("stop_reason").size().rename("n_resumes").reset_index().to_csv(
        out / "stop_reasons.csv", index=False
    )
    with (out / "release_queries.pkl").open("wb") as handle:
        pickle.dump({"positions": positions["test"], "queries": releases}, handle)
    with (out / "retention_vectorizer.pkl").open("wb") as handle:
        pickle.dump(vec, handle)
    _save_json(
        out / "policy_specifications.json",
        {
            "L0_none": "keep",
            "L1_scrub_text": "mandatory masks only",
            "L2_generalize": "mandatory + lowest available range/generalization level",
            "L3_strong": "mandatory + broadest generalization or mask",
            "mask_detected": "mask all non-context entities",
            "adaptive_model": "learned action/level, then independent tool routing and verifier",
        },
    )
    selection = json.loads((out / "risk_model_selection.json").read_text())
    _save_json(
        out / "manifest.json",
        {
            "approach": VERSION,
            "experiment_role": "development_diagnostic",
            "config": config,
            "training": selection,
            "n_val_queries": len(positions["val"]),
            "n_test_queries": len(positions["test"]),
            "test_people": len(set(pop.clusters[positions["test"]])),
            "test_used_for_model_selection": False,
            "decision_before_execution": True,
            "replacement_or_tool_features": False,
            "llm_provider": config.get("llm_provider", "none"),
            "elapsed_seconds": time.monotonic() - started,
            "limitations": [
                "empirical counterfactual action labels, not human action gold",
                "declared action-cost utility proxy",
                "bounded development dataset",
                "no guarantee of superiority or anonymity",
            ],
        },
    )
    print(test_summary.to_string(index=False), flush=True)
