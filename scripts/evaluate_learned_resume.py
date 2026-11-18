"""Evaluate frozen regression and search controllers on the same person cohort.

The regressor receives no reference pool. Pool access is restricted to actual
attack scoring and the explicitly labelled search baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from evaluate_adaptive import replay_pool, used_ids
from group_policy_eval import holdout_pool, person_bootstrap
from threadpoolctl import threadpool_limits

from anonrisk_uk.action_learning import lexical_retention
from anonrisk_uk.attack import BM25Index, TfidfIndex, ranking_result
from anonrisk_uk.experiment import Population
from anonrisk_uk.learned_resume import anonymize_learned
from anonrisk_uk.learned_resume_risk import VERSION
from anonrisk_uk.policy import fixed_policies
from anonrisk_uk.resume_adaptive import NeighbourProxy, options, render, search
from anonrisk_uk.resume_parser import parse_resume
from anonrisk_uk.resume_resources import load_resume_resources

ROOT = Path(__file__).resolve().parents[1]


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    with args.model.open('rb') as handle:
        estimator = pickle.load(handle)
    if args.reuse_pool:
        if args.purpose != 'regression':
            raise ValueError('An already-seen pool must be labelled regression')
        ids, extracted, people = replay_pool(args.reuse_pool)
        excluded = None
    else:
        used, _ = used_ids(ROOT)
        ids, extracted, people, excluded = holdout_pool(ROOT / 'data/raw/all_resumes.csv', used,
                                                       args.pool_size, args.seed)
        assert not set(map(str, ids)) & used
    texts = [estimator.codec.render(ex, {}) for ex in extracted]
    indexes = {'bm25': BM25Index(max_features=20000).fit(texts),
               'char': TfidfIndex(max_features=60000).fit(texts)}
    people = np.asarray(people)
    pop = Population(ids, extracted, people, estimator.codec, indexes['bm25'], texts)
    linked = pop.linked_positions()
    eligible = [p for p in sorted(set(people[linked]))
                if all(0 < len(texts[k]) <= 20000 for k in np.flatnonzero(people == p))]
    selected = set(np.random.default_rng(args.seed).permutation(eligible)[:args.max_persons])
    queries = [int(k) for k in linked if people[k] in selected]
    if len(selected) < 2:
        raise ValueError('Not enough new linked people for evaluation')
    resources = load_resume_resources(args.resources_config)
    standalone = SimpleNamespace(config={'adaptive': {'utility_floor': .55}}, risk_estimator=estimator,
                                 detector=resources.detector, background_texts=[], feature_index=None)
    structure_path = args.model.with_name('structure_model.pkl')
    structure = None
    if structure_path.exists():
        with structure_path.open('rb') as handle:
            structure = pickle.load(handle)
    protocol = {'method': VERSION, 'purpose': args.purpose, 'queries': len(queries), 'persons': len(selected),
                'pool_size': len(ids), 'seed': args.seed, 'selected': 'regression',
                'excluded_historical_person_clusters': excluded, 'utility_floor': .55,
                'learned_target': estimator.target, 'model_sha256': hashlib.sha256(args.model.read_bytes()).hexdigest(),
                'prediction_access': 'standalone learned artifact only; no live pool, no attack outcomes',
                'selection': 'all model and policy selection on development people only; no test selection',
                'scope': 'bounded person-level diagnostic, not a powered confirmatory experiment',
                'parser': 'shared parser and codec for regression and search comparison',
                'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in [Path(__file__), ROOT / 'src/anonrisk_uk/learned_resume.py',
                                            ROOT / 'src/anonrisk_uk/learned_resume_risk.py']}}
    (args.out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    pd.DataFrame({'id': ids}).to_csv(args.out / 'pool_ids.csv', index=False)
    outputs = {key: [] for key in ['L0', 'historical_L2', 'historical_L3', 'matched_direct_only',
                                   'matched_all_quasi', 'search_v3_3', 'regression']}
    if structure:
        outputs['regression_structure_only'] = []
    records = []
    policies = fixed_policies()
    with (args.out / 'releases.jsonl').open('x') as handle:
        for number, k in enumerate(queries, 1):
            doc = parse_resume(texts[k], estimator.codec, resources.detector)
            learned = anonymize_learned(standalone, texts[k])
            outputs['regression'].append(learned.releases['adaptive'].text)
            outputs['L0'].append(texts[k])
            for name, policy in [('historical_L2', 'L2_generalize'), ('historical_L3', 'L3_strong')]:
                outputs[name].append(estimator.codec.render(extracted[k], policies[policy]))
            outputs['matched_direct_only'].append(render(doc, {}))
            outputs['matched_all_quasi'].append(render(doc, {u.group: 1 for u in doc.units if len(options(doc, u)) > 1}))
            proxy = NeighbourProxy(indexes, texts, texts[k])
            actions, _ = search(doc, proxy, estimator.retention_vectorizer, utility_floor=.55, target=0.)
            outputs['search_v3_3'].append(render(doc, actions))
            if structure:
                control = SimpleNamespace(config={'adaptive': {'utility_floor': .55}}, risk_estimator=structure,
                                          detector=resources.detector, background_texts=[], feature_index=None)
                outputs['regression_structure_only'].append(anonymize_learned(control, texts[k]).releases['adaptive'].text)
            prediction = learned.diagnostics['metrics']['risk_after']
            records.append({'id': str(ids[k]), 'person': str(people[k]),
                            'predicted_risk': prediction, 'status': learned.diagnostics['stop_reason']})
            handle.write(json.dumps({'id': str(ids[k]), 'person': str(people[k]), 'original': texts[k],
                                     'outputs': {name: values[-1] for name, values in outputs.items()}},
                                    ensure_ascii=False) + '\n')
            print(f'evaluated {number}/{len(queries)}', flush=True)
    frame = pd.DataFrame(records)
    targets, excludes = pop.targets_and_excludes(queries, 'S2')
    summary = []
    for name, values in outputs.items():
        row = {'method': name}
        all_rr = []
        for attacker, index in indexes.items():
            actual = ranking_result(index, values, targets, excludes)[0]
            frame[f'{name}_{attacker}_r1'] = actual.recalls[1]
            row[f'{attacker}_recall_at_1'] = float(actual.recalls[1].mean())
            row[f'{attacker}_mrr'] = float(actual.reciprocal_ranks.mean())
            all_rr.append(actual.reciprocal_ranks)
        frame[f'{name}_actual_worst_rr'] = np.maximum.reduce(all_rr)
        retention = [float(lexical_retention(estimator.retention_vectorizer, texts[k], [value])[0])
                     for k, value in zip(queries, values)]
        frame[f'{name}_lexical'] = retention
        row['lexical_retention'] = float(np.mean(retention))
        summary.append(row)
    paired = {}
    for control in ['historical_L2', 'search_v3_3']:
        for metric in ['bm25_r1', 'char_r1', 'lexical']:
            paired[f'vs_{control}_{metric}'] = person_bootstrap(frame[f'regression_{metric}'] - frame[f'{control}_{metric}'],
                                                              frame.person)
    report = {**protocol, 'summary': summary, 'paired_differences_95ci': paired,
              'regression_mae_on_releases': float(np.abs(frame.predicted_risk - frame.regression_actual_worst_rr).mean()),
              'statuses': frame.status.value_counts().to_dict(), 'live_pool_used_for_prediction': False}
    (args.out / 'report.json').write_text(json.dumps(report, indent=2))
    pd.DataFrame(summary).to_csv(args.out / 'summary.csv', index=False)
    frame.to_csv(args.out / 'per_query.csv', index=False)
    print(pd.DataFrame(summary).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--resources-config', default='configs/baselines/adaptive_resume_v3_3.json')
    parser.add_argument('--purpose', choices=['test', 'regression'], required=True)
    parser.add_argument('--reuse-pool', type=Path)
    parser.add_argument('--pool-size', type=int, default=8000)
    parser.add_argument('--max-persons', type=int, default=60)
    parser.add_argument('--seed', type=int, default=20261109)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args)


if __name__ == '__main__':
    main()
