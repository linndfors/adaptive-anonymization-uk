"""Small, reproducible v6 experiments using one shared typed action inventory."""

from __future__ import annotations

import hashlib
import json
import pickle
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error
from tqdm.auto import tqdm

from .action_learning import (
    action_training_rows,
    choose_actions,
    feature_records,
    lexical_retention,
    train_action_model,
)
from .action_space import VERSION, action_bank, fixed_actions, render, semantic_retention, typed_document
from .attack import BM25Index, TfidfIndex, ranking_result
from .experiment import Population
from .statistics import metric_draws, metric_intervals, paired_differences

POLICIES = ["L0_none", "L1_scrub_text", "L2_generalize", "L3_strong", "adaptive_model", "mask_detected"]
POLICY_SPECS = {
    "L0_none": "Keep the original representation, including identifiers (evaluation reference).",
    "L1_scrub_text": "Mask direct and sensitive units; keep quasi identifiers.",
    "L2_generalize": "Mandatory masks + first available semantic/range generalization; keep unmapped quasi units.",
    "L3_strong": "Mandatory masks + broadest generalization; mask quasi units with no generalization.",
    "adaptive_model": "Learned entity/action values, utility constraints and deployment-only self-linkage verification.",
    "mask_detected": "Mask every non-context unit using its specific type.",
}


def context(config, splits):
    from .research import _checkpoint, _fingerprint, _load_bundle, _pick, _save_json, detector_set

    if config.get("approach") not in {"typed_actions_v1", "entity_decision_v2"}:
        raise ValueError("This workflow requires a typed action or entity-decision profile")
    out = Path(config["out"]) / "pipeline"
    out.mkdir(parents=True, exist_ok=True)
    calibration = Path(config["out"]) / "detection/selected_detector.json"
    selected = json.loads(calibration.read_text(encoding="utf-8"))
    if selected.get("status") != "ready" or not set(config["release_detectors"]) <= set(selected["detectors"]):
        raise ValueError("Run detector calibration for this profile before action learning")
    signature = {
        "config": config,
        "code": {p.name: _fingerprint(p) for p in Path(__file__).parent.glob("*.py")},
        "inputs": {name: _fingerprint(config[name]) for name in ("workua_path", "e3_path")},
        "calibration": _fingerprint(calibration),
        "dependencies": {name: version(name) for name in ("numpy", "scikit-learn", "spacy", "transformers")},
    }
    if config.get("generalization_catalog", "regex") != "regex":
        from .category_taxonomy import MAP_PATH

        signature["inputs"]["category_map"] = _fingerprint(MAP_PATH)
    if config.get("llm_provider") == "openai":
        from .action_tools import provider_identity

        signature["llm_identity"] = provider_identity(config)
    namespace = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:24]
    checkpoint_config = {**config, "resume_checkpoints": True, "_checkpoint_signature": namespace}
    _save_json(out / "action_signature.json", signature)
    bundle = _load_bundle(config)
    texts = bundle["texts"]
    index = BM25Index(max_features=config.get("bm25_max_features", 20000)).fit(texts)
    equal_texts = {}
    for pos, text in enumerate(texts):
        equal_texts.setdefault(text, []).append(pos)
    index.exact_text_targets = {pos: matches for matches in equal_texts.values() for pos in matches}
    pop = Population(bundle["ids"], bundle["extracted"], bundle["clusters"], bundle["codec"], index, texts)
    train_texts = [text for text, split in zip(texts, bundle["split"].split) if split == "train"]
    vectorizer = TfidfVectorizer(max_features=20000, sublinear_tf=True).fit(train_texts)
    linked = pop.linked_positions()
    positions = {}
    for split in splits:
        eligible = [
            k
            for k in linked
            if bundle["split"].split.iloc[k] == split and len(texts[k]) <= config.get("max_document_chars", 5000)
        ]
        if config.get("approach") == "entity_decision_v2":
            by_person = {}
            for k in np.random.default_rng(config["seed"]).permutation(eligible):
                by_person.setdefault(str(bundle["clusters"][k]), []).append(int(k))
            chosen = []
            person_order = np.random.default_rng(config["seed"]).permutation(sorted(by_person))
            for offset in range(config.get("queries_per_person", 2)):
                for person in person_order:
                    if offset < len(by_person[person]) and len(chosen) < config["n_" + split]:
                        chosen.append(by_person[person][offset])
            positions[split] = np.asarray(chosen, dtype=int)
        else:
            positions[split] = _pick(eligible, config["n_" + split], config["seed"])
        if not len(positions[split]):
            raise ValueError(f"No eligible {split} identities; increase the bounded pool_sample")
    if "test" in splits:
        train_people = set(bundle["split"].loc[bundle["split"].split.eq("train"), "person_cluster"])
        if train_people & set(bundle["clusters"][positions["test"]]):
            raise ValueError("Train/test identity overlap")
    detector, _ = detector_set(config)
    docs, banks = {}, {}
    for pos in tqdm(np.concatenate(list(positions.values())), desc="Typed documents and shared actions"):

        def prepare(pos=pos):
            doc = typed_document(
                bundle["extracted"][pos],
                bundle["codec"],
                detector,
                include_residual=config.get("include_residual", True),
            )
            bank = {} if config.get("approach") == "entity_decision_v2" else action_bank(doc, bundle["codec"])
            if config.get("llm_provider") == "openai" and config.get("approach") != "entity_decision_v2":
                from .action_tools import add_openai_candidates

                bank = add_openai_candidates(doc, bank, config)
            return doc, bank

        docs[int(pos)], banks[int(pos)] = _checkpoint(checkpoint_config, "typed_documents", str(pos), prepare)
    return out, checkpoint_config, bundle, pop, vectorizer, positions, docs, banks


def run_labels(config):
    from .research import _checkpoint, _save_json

    out, cc, _bundle, pop, vec, positions, docs, banks = context(config, ["train", "val"])
    counts = {}
    for split in ["train", "val"]:
        blocks = []
        for pos in tqdm(positions[split], desc=f"Action outcomes: {split}"):
            rows = _checkpoint(
                cc,
                "action_labels",
                f"{split}_{pos}",
                lambda pos=pos: action_training_rows(
                    pop, [pos], docs, banks, vec, max_rows=config.get("max_action_rows_per_document", 400)
                ),
            )
            blocks.append(rows)
        table = pd.concat(blocks, ignore_index=True)
        table.to_csv(out / f"action_{split}.csv", index=False)
        # Keep the six-stage runner's artifact contract, with explicit action data.
        name = "training" if split == "train" else "validation"
        table.drop(columns=["target_gain", "s2_gain", "self_gain"]).to_csv(
            out / f"unit_{name}_features.csv", index=False
        )
        table[["pos", "person", "unit", "action", "context", "target_gain", "s2_gain", "self_gain"]].to_csv(
            out / f"unit_{name}_labels.csv", index=False
        )
        counts[split] = {
            "documents": len(positions[split]),
            "people": table.person.nunique(),
            "action_rows": len(table),
        }
    _save_json(
        out / "risk_label_manifest.json",
        {
            "approach": VERSION,
            "target": "action-specific counterfactual gain",
            "contexts": ["mandatory_only", "L2_generalized"],
            "splits": counts,
        },
    )
    print(counts, flush=True)
    return counts


def run_training(config):
    from .research import _save_json

    out = Path(config["out"]) / "pipeline"
    train, val = [pd.read_csv(out / f"action_{split}.csv") for split in ("train", "val")]
    if set(train.person.astype(str)) & set(val.person.astype(str)):
        raise ValueError("Action model train/validation identities overlap")
    model = train_action_model(train, seed=config["seed"], max_iter=config.get("action_model_iterations", 60))
    prediction = model.predict(feature_records(val))
    validation = pd.DataFrame(
        [
            {
                "predictor": "action_value_model",
                "mae": mean_absolute_error(val.target_gain, prediction),
                "n_action_rows": len(val),
                "n_people": val.person.nunique(),
            },
            {
                "predictor": "zero_gain",
                "mae": mean_absolute_error(val.target_gain, np.zeros(len(val))),
                "n_action_rows": len(val),
                "n_people": val.person.nunique(),
            },
        ]
    )
    validation.to_csv(out / "risk_model_validation.csv", index=False)
    val.assign(predicted_gain=prediction).to_csv(out / "action_validation_predictions.csv", index=False)
    pd.DataFrame(
        [
            {
                "train_people": train.person.nunique(),
                "train_action_rows": len(train),
                "note": "single bounded fit; no learning-curve search",
            }
        ]
    ).to_csv(out / "risk_learning_curve.csv", index=False)
    _save_json(
        out / "risk_model_selection.json",
        {
            "model": "HistGradientBoosting action-value regression",
            "target": "0.5*S2_RR_reduction + 0.5*continuous_S1_exposure_reduction",
            "selection": "fixed bounded baseline",
            "train_people": train.person.nunique(),
            "action_space": VERSION,
            "validation_used_for_fit": False,
        },
    )
    with (out / "unit_risk_model.pkl").open("wb") as handle:
        pickle.dump(model, handle)
    print(validation.to_string(index=False), flush=True)
    return validation


def run_pipeline(config):
    from .research import _checkpoint, _save_json, run_vacancy_evaluation

    started = time.monotonic()
    out, cc, bundle, pop, vec, positions, docs, banks = context(config, ["val", "test"])
    with (out / "unit_risk_model.pkl").open("rb") as handle:
        model = pickle.load(handle)
    model_digest = hashlib.sha256((out / "unit_risk_model.pkl").read_bytes()).hexdigest()
    cc["_checkpoint_signatures"] = {
        "action_release": hashlib.sha256((cc["_checkpoint_signature"] + model_digest).encode()).hexdigest()[:24]
    }
    settings = {
        "utility_floor": config.get("action_lexical_floor", 0.65),
        "semantic_floor": config.get("action_content_floor", 0.8),
        "max_steps": config.get("max_steps", 6),
        "candidate_limit": config.get("candidate_limit", 8),
        "min_self_gain": config.get("action_min_self_gain", 0.01),
        "cost_weight": config.get("action_cost_weight", 0.2),
        "self_rr_target": config.get("action_self_rr_target", 0.5),
    }
    indexes = {"BM25": pop.index}
    if config.get("use_char_tfidf", True):
        indexes["char-TF-IDF"] = TfidfIndex(max_features=20000).fit(bundle["texts"])
    all_test, test_traces, operations, available, summaries, intervals, stores = [], [], [], [], [], [], {}
    saved_queries = {}
    for split in ["val", "test"]:
        summaries_split = []
        for policy in tqdm(POLICIES, desc=f"Shared-action policies: {split}"):
            queries, lexical, meaning = [], [], []
            policy_traces = []
            for pos in positions[split]:
                pos = int(pos)
                doc, bank = docs[pos], banks[pos]
                if policy == "adaptive_model":
                    text, actions, trace = _checkpoint(
                        cc,
                        "action_release",
                        f"{split}_{pos}",
                        lambda pos=pos, doc=doc, bank=bank: choose_actions(
                            doc, bank, model, pop.index, vec, pos, **settings
                        ),
                    )
                    if split == "test":
                        test_traces.extend({**row, "pos": pos, "policy": policy} for row in trace)
                    policy_traces.append(trace[-1])
                else:
                    actions = fixed_actions(doc, bank, policy)
                    text = render(doc, actions)
                queries.append(text)
                lexical.append(float(lexical_retention(vec, doc.text, [text])[0]))
                meaning.append(semantic_retention(doc, actions))
                if split == "test":
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
            if split == "test":
                saved_queries[policy] = queries
            targets, excludes = pop.targets_and_excludes(positions[split], "S2")
            for name, index in indexes.items():
                result = ranking_result(index, queries, targets, excludes, batch_size=16)[0]
                frame = result.to_frame().assign(
                    pos=positions[split],
                    person=pop.clusters[positions[split]],
                    retention=lexical,
                    content_retention=meaning,
                    policy=policy,
                    attacker=name,
                )
                if policy == "adaptive_model":
                    frame["stop_reason"] = [row["stop_reason"] for row in policy_traces]
                    frame["proxy_threshold_met"] = [row["threshold_met"] for row in policy_traces]
                row = {key: values[0] for key, values in metric_draws(frame).items()}
                row.update(
                    policy=policy,
                    attacker=name,
                    scenario="S2",
                    cohort="all",
                    n_resumes=len(frame),
                    n_persons=frame.person.nunique(),
                    content_retention=float(np.mean(meaning)),
                )
                summaries_split.append(row)
                if split == "test":
                    all_test.append(frame)
                    stores[(name, policy)] = frame
                    intervals.append(
                        metric_intervals(frame, n=config.get("n_bootstrap", 100), seed=config["seed"]).assign(
                            policy=policy, attacker=name
                        )
                    )
        table = pd.DataFrame(summaries_split)
        if split == "val":
            table.to_csv(out / "validation_grid.csv", index=False)
            table.to_csv(out / "validation_tradeoffs.csv", index=False)
            adaptive = table[table.policy.eq("adaptive_model")]
            _save_json(
                out / "operating_point.json",
                {
                    "settings": settings,
                    "selection": "fixed_predeclared_no_grid",
                    "validation_worst_recall": float(adaptive["recall@1"].max()),
                    "target": config.get("target_validation_recall", 0.2),
                    "status": "ready"
                    if adaptive["recall@1"].max() <= config.get("target_validation_recall", 0.2)
                    else "target_unmet_on_validation",
                },
            )
        else:
            summaries = summaries_split
    for pos in positions["test"]:
        for i, unit in enumerate(docs[int(pos)].units):
            for action in banks[int(pos)][i]:
                available.append(
                    {
                        "pos": int(pos),
                        "unit": i,
                        "entity_type": unit.label,
                        "role": unit.role,
                        "group": unit.group,
                        "original": docs[int(pos)].text[unit.start : unit.end],
                        **asdict(action),
                    }
                )
    pairs = []
    for attacker in indexes:
        for baseline in [p for p in POLICIES if p != "adaptive_model"]:
            pairs.append(
                paired_differences(
                    stores[(attacker, "adaptive_model")],
                    stores[(attacker, baseline)],
                    n=config.get("n_bootstrap", 100),
                    seed=config["seed"],
                ).assign(policy="adaptive_model", reference=baseline, attacker=attacker)
            )
    pd.DataFrame(summaries).to_csv(out / "test_summary.csv", index=False)
    pd.concat(all_test, ignore_index=True).to_csv(out / "test_per_query.csv", index=False)
    pd.concat(intervals, ignore_index=True).to_csv(out / "test_intervals.csv", index=False)
    pd.concat(pairs, ignore_index=True).to_csv(out / "paired_differences.csv", index=False)
    pd.DataFrame(test_traces).to_csv(out / "release_traces.csv", index=False)
    pd.DataFrame(operations).to_csv(out / "release_operations.csv", index=False)
    pd.DataFrame(available).to_csv(out / "available_actions.csv", index=False)
    traces = pd.DataFrame(test_traces)
    traces[traces.stop_reason.notna()].groupby("stop_reason").size().rename("n_resumes").reset_index().to_csv(
        out / "stop_reasons.csv", index=False
    )
    with (out / "release_queries.pkl").open("wb") as handle:
        pickle.dump({"positions": positions["test"], "queries": saved_queries}, handle)
    with (out / "retention_vectorizer.pkl").open("wb") as handle:
        pickle.dump(vec, handle)
    _save_json(out / "policy_specifications.json", POLICY_SPECS)
    _save_json(
        out / "manifest.json",
        {
            "approach": VERSION,
            "experiment_role": "development_diagnostic",
            "config": config,
            "action_tools_shared_between_policies": True,
            "n_train_queries": json.loads((out / "risk_label_manifest.json").read_text())["splits"]["train"][
                "documents"
            ],
            "n_val_queries": len(positions["val"]),
            "n_test_queries": len(positions["test"]),
            "test_identities": len(set(pop.clusters[positions["test"]])),
            "controller": "BM25_S1_exact_text_group",
            "evaluation_attackers": list(indexes),
            "llm_provider": config.get("llm_provider", "none"),
            "content_retention": "field-balanced declared action-cost proxy; not a semantic quality measurement",
            "limitations": [
                "small development sample",
                "partial lexical taxonomy",
                "no independent semantic evaluator",
            ],
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    if config.get("vacancies_path"):
        run_vacancy_evaluation(config, bundle=bundle)
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)
    return pd.DataFrame(summaries)
