"""Fixed anonymization baselines and per-document adaptive policies.

Adaptive policies apply transformations until operational S1 risk falls below a
threshold. Candidate order may come from a category prior, contextual model, or
greedy proxy search. Final privacy is evaluated with ground-truth S2 links.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .attack import ranking_result
from .experiment import Population, attack_result, render_queries
from .workua import FREE_TEXT_GROUPS, GROUPS, OP_STRENGTH, SENSITIVE_GROUPS, nonempty_groups

# Groups without a meaningful generalization move directly to removal.
NO_GENERALIZE = {"driver_license", "veteran", "employment", "exp_industry"}


def fixed_policies() -> dict[str, dict]:
    qi = [g for g in GROUPS if g not in FREE_TEXT_GROUPS | SENSITIVE_GROUPS]
    L1 = {g: "generalize" for g in FREE_TEXT_GROUPS}  # Scrub residual free-text PII.
    L2 = {**L1, **{g: "generalize" for g in qi if g not in NO_GENERALIZE}, **{g: "remove" for g in SENSITIVE_GROUPS}}
    L3 = {
        **L2,
        **{
            g: "remove"
            for g in [
                "age",
                "city",
                "exp_company",
                "exp_city",
                "edu_institution",
                "edu_years",
                "edu_city",
                "add_education",
                "exp_resp",
                "additional_info",
            ]
        },
    }
    return {"L0_none": {}, "L1_scrub_text": L1, "L2_generalize": L2, "L3_strong": L3}


def next_op(group: str, current: str) -> str | None:
    if current == "keep":
        return "remove" if group in NO_GENERALIZE else "generalize"
    if current == "generalize":
        return "remove"
    return None


def steps_from_order(order: list[str]) -> list[tuple[str, str]]:
    """Generalize groups in order, then remove them in the same order."""
    s1 = [(g, "generalize") for g in order if g not in NO_GENERALIZE]
    s2 = [(g, "remove") for g in order]
    return s1 + s2


@dataclass
class ProxyAssessment:
    names: list[str]
    rr: np.ndarray  # Controllers by queries
    recalls1: np.ndarray
    margins: np.ndarray
    scores: np.ndarray

    @property
    def risk(self):
        return self.rr.max(axis=0)

    def meets(self, tau, require_margin=True):
        return (self.risk <= tau) & ((self.margins.max(axis=0) <= 0) | (tau >= 1) | (not require_margin))

    def to_frame(self):
        out = {"proxy_rr": self.risk}
        for i, name in enumerate(self.names):
            out[f"rr_{name}"] = self.rr[i]
            out[f"recall1_{name}"] = self.recalls1[i]
            out[f"margin_{name}"] = self.margins[i]
        return pd.DataFrame(out)


class ProxyController:
    """Take worst-case expected reciprocal rank across self-linkage proxies."""

    def __init__(self, populations: dict[str, Population], require_margin=True):
        if not populations:
            raise ValueError("At least one proxy attacker is required")
        first = next(iter(populations.values()))
        if any(p.ids != first.ids for p in populations.values()):
            raise ValueError("All proxy attackers must use the same ordered identifiers")
        self.populations, self.require_margin = populations, require_margin
        self.population = first

    def assess(self, positions, policies, batch_size=256):
        rr, recalls, margins, scores = [], [], [], []
        for pop in self.populations.values():
            tg, ex = pop.targets_and_excludes(positions, "S1")
            result, margin, score = ranking_result(
                pop.index, render_queries(pop, positions, policies), tg, ex, batch_size
            )
            rr.append(result.reciprocal_ranks)
            recalls.append(result.recalls[1])
            margins.append(margin)
            scores.append(score)
        return ProxyAssessment(
            list(self.populations), np.vstack(rr), np.vstack(recalls), np.vstack(margins), np.vstack(scores)
        )


def _controller(pop):
    return pop if isinstance(pop, ProxyController) else ProxyController({"S1": pop})


def ordered_adaptive(pop, positions, orders, tau, base_policy=None, batch_size=256):
    return ordered_adaptive_multi(pop, positions, orders, [tau], base_policy, batch_size)[tau]


def ordered_adaptive_multi(
    pop, positions, orders: list[list[str]], taus: list[float], base_policy: dict | None = None, batch_size: int = 256
) -> dict[float, list[dict]]:
    """Evaluate several thresholds in one attack pass for a risk-utility curve."""
    base_policy = base_policy or {}
    controller = _controller(pop)
    pop = controller.population
    if len(positions) != len(orders) or any(not 0 < t <= 1 for t in taus):
        raise ValueError("Candidate orders must match queries and thresholds must be in (0, 1]")
    q_pos, q_pol, cand = [], [], []
    for k, order in zip(positions, orders):
        present = set(nonempty_groups(pop.extracted[k]))
        pol = dict(base_policy)
        prefixes = [dict(pol)]
        for g, op in steps_from_order([g for g in order if g in present]):
            if OP_STRENGTH[op] > OP_STRENGTH.get(pol.get(g, "keep"), 0):
                pol[g] = op
                prefixes.append(dict(pol))
        cand.append(prefixes)
        q_pos += [k] * len(prefixes)
        q_pol += prefixes
    assessment = controller.assess(q_pos, q_pol, batch_size)
    attained = {t: assessment.meets(t, controller.require_margin) for t in taus}
    out = {t: [] for t in taus}
    i = 0
    for prefixes in cand:
        i += len(prefixes)
        for t in taus:
            ok = np.where(attained[t][i - len(prefixes) : i])[0]
            out[t].append(prefixes[int(ok[0])] if len(ok) else prefixes[-1])
    return out


def greedy_adaptive(
    pop,
    k: int,
    tau: float,
    util_proxy=None,
    lam: float = 0.05,
    base_policy: dict | None = None,
    max_steps: int | None = None,
) -> tuple[dict, list]:
    """Greedy proxy-ensemble search that continues across score plateaus."""
    if not 0 < tau <= 1 or lam <= 0:
        raise ValueError("tau must be in (0, 1] and lam must be positive")
    controller = _controller(pop)
    pop = controller.population
    ex = pop.extracted[k]
    pol = dict(base_policy or {})
    present = nonempty_groups(ex)
    max_steps = 2 * len(present) if max_steps is None else max_steps
    cur = controller.assess([k], [pol])
    scales = np.maximum(np.abs(cur.scores[:, 0]), 1e-9)
    cur_loss = util_proxy.loss(ex, [pol])[0] if util_proxy else 0.0

    def record(step, group, op, reason):
        return {
            "step": step,
            "group": group,
            "operation": op,
            "proxy_rr": cur.risk[0],
            "utility_loss": cur_loss,
            "met_threshold": bool(cur.meets(tau, controller.require_margin)[0]),
            "reason": reason,
            **{f"rr_{name}": cur.rr[i, 0] for i, name in enumerate(cur.names)},
        }

    trace = [record(0, None, None, "initial")]
    for _ in range(max_steps):
        if cur.meets(tau, controller.require_margin)[0]:
            break
        cands = []
        for g in present:
            op = next_op(g, pol.get(g, "keep"))
            if op:
                p = dict(pol)
                p[g] = op
                cands.append((g, op, p))
        if not cands:
            break
        new = controller.assess([k] * len(cands), [c[2] for c in cands])
        losses = util_proxy.loss(ex, [c[2] for c in cands]) if util_proxy else np.zeros(len(cands))
        active = (cur.rr[:, 0] > tau) | (cur.margins[:, 0] > 0)
        if not active.any():
            active[:] = True
        margin_gain = ((cur.margins[active] - new.margins[active]) / scales[active, None]).min(axis=0)
        gain = cur.risk[0] - new.risk + 0.25 * margin_gain
        reason = "risk_or_margin"
        if gain.max() <= 1e-9:
            gain = ((cur.scores[active] - new.scores[active]) / scales[active, None]).min(axis=0)
            reason = "target_score"
        cost = np.maximum(losses - cur_loss, 0) + lam
        if gain.max() <= 1e-9:
            # Continue escalation on a plateau instead of returning an unprotected document.
            score = -cost
            reason = "plateau_escalation"
        else:
            score = np.where(gain > 1e-9, gain / cost, -np.inf)
        j = int(np.argmax(score))
        g, op, pol = cands[j]
        cur = ProxyAssessment(
            new.names,
            new.rr[:, j : j + 1],
            new.recalls1[:, j : j + 1],
            new.margins[:, j : j + 1],
            new.scores[:, j : j + 1],
        )
        cur_loss = losses[j]
        trace.append(record(len(trace), g, op, reason))
    trace[-1]["stop_reason"] = "threshold_met" if trace[-1]["met_threshold"] else "actions_or_steps_exhausted"
    return pol, trace


def policy_stats(policies: list[dict]) -> dict:
    n_gen = np.mean([sum(v == "generalize" for v in p.values()) for p in policies])
    n_rem = np.mean([sum(v == "remove" for v in p.values()) for p in policies])
    return {"avg_generalized_groups": float(n_gen), "avg_removed_groups": float(n_rem)}


def evaluate_policy(
    pop: Population,
    positions,
    policies,
    tasks,
    codec,
    extracted_test,
    scenario: str = "S2",
    return_details=False,
    utility_details=None,
):
    from .statistics import metric_draws
    from .utility import retention_scores

    if isinstance(policies, dict):
        policies = [policies] * len(positions)
    result = attack_result(pop, positions, policies, scenario)
    details = result.to_frame().assign(pos=list(positions), person=pop.clusters[list(positions)])
    if utility_details is not None:
        if not details[["pos", "person"]].equals(utility_details[["pos", "person"]]):
            raise ValueError("The utility cache does not match the query set")
        for column in utility_details:
            if column == "retention" or column.endswith(("_true", "_pred")):
                details[column] = utility_details[column]
    else:
        for t in tasks:
            y, pred, valid = t.predictions(codec, extracted_test, policies)
            details[f"{t.name}_true"] = [v if keep else None for v, keep in zip(y, valid)]
            details[f"{t.name}_pred"] = pred
        details["retention"] = retention_scores(codec, extracted_test, policies)
    classes = {t.name: sorted(t.classes_) for t in tasks}
    row = {metric: values[0] for metric, values in metric_draws(details, classes).items()}
    row["median_rank"] = float(np.median(result.ranks))
    for t in tasks:
        row[f"{t.name}_n"] = int(details[f"{t.name}_true"].notna().sum())
    row.update(policy_stats(policies))
    return (row, details) if return_details else row
