"""Canonical adaptive resume policy: v2-style greedy risk/utility group search.

One engine serves text inference, Streamlit, CLI and evaluation. Historical
scripts remain reproducibility controls, not alternative production models.
"""

from __future__ import annotations

import re
from dataclasses import asdict

import numpy as np

from .action_learning import lexical_retention
from .action_space import _parents
from .attack import TfidfIndex
from .category_taxonomy import lookup
from .entity_demo import DemoRelease, EntityDemoResult
from .resume_context import birth_units, job_generalizations, skill_generalization_allowed
from .resume_parser import ADDRESS, EMAIL, PHONE, employer_without_geography, parse_resume
from .resume_parser import VERSION as PARSER_VERSION

VERSION = "adaptive_resume_v3.3"
# Used ONLY by quality measurement. None of these fields is protected from edits.
CORE = {"skills", "title", "exp_position", "exp_resp", "duration"}
MASKS = {
    "PERSON": "[PERSON]", "CONTACT": "[CONTACT]", "EMAIL": "[EMAIL]", "PHONE": "[PHONE]",
    "URL": "[URL]", "ADDRESS": "[ADDRESS]", "ID_NUMBER": "[ID_NUMBER]", "BIRTH_DATE": "[BIRTH_DATE]",
    "HEALTH": "[HEALTH]",
}
GENERAL = {
    "EMPLOYER": "[КОМПАНІЯ]", "ORGANIZATION": "[ОРГАНІЗАЦІЯ]", "LOCATION": "[НАСЕЛЕНИЙ ПУНКТ]",
    "EDUCATION_INSTITUTION": "[ЗАКЛАД ОСВІТИ]", "EDUCATION_PERIOD": "[ПЕРІОД НАВЧАННЯ]",
    "AGE": "[ВІК]", "SALARY": "[ЗАРПЛАТА]",
}


class DirectIdentifierError(ValueError):
    """A release failed an independent deterministic direct-identifier check."""


def check_direct_identifiers(text):
    """Check known direct patterns, independently of the search gate.

    Passing these rules is not proof of exhaustive detection. Do not include raw
    sensitive values in diagnostics printed by the CLI.
    """
    issues = [{"label": u.label, "start": u.start, "end": u.end} for u in birth_units(text)]
    for label, pattern in (("EMAIL", EMAIL), ("PHONE", PHONE), ("ADDRESS", ADDRESS)):
        issues.extend({"label": label, "start": m.start(), "end": m.end()} for m in pattern.finditer(text))
    return {"status": "remaining_direct_patterns" if issues else "no_remaining_direct_patterns", "issues": issues}


def options(doc, u):
    value = doc.text[u.start:u.end]
    if u.role in {"DIRECT", "SENSITIVE"}:
        return [MASKS.get(u.label, f"[{u.label}]")]
    if u.role == "CONTEXT":
        return [value]
    parents = _parents("JOB_TITLE" if u.label == "OCCUPATION_EDU" else u.label, value, "taxonomy+regex")
    specific = parents[0][0] if parents else value
    broad = parents[1][0] if len(parents) > 1 else specific
    job = job_generalizations(value) if u.label == "JOB_TITLE" else None
    if job:
        specific, broad = job
    if u.label == "SKILL" and (not skill_generalization_allowed(doc, u)
                              or re.search(r"[-‑–]", value) and not lookup("SKILL", value)):
        # Do not insert a nominative category after an unmodelled governor or
        # replace the technology prefix of a compound word. Masking is still available.
        specific = broad = value
    if u.label == "EMPLOYMENT_PERIOD":
        years = [int(y) for y in re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", value)]
        # Year-only ranges do NOT establish exact duration; keep the approximation explicit.
        if len(years) == 2 and 0 <= years[1] - years[0] <= 70:
            specific = f"[ПЕРІОД РОБОТИ: приблизно {years[1] - years[0]} р.]"
        # Ongoing/incomplete periods can be masked; no duration is invented.
        broad = "[ПЕРІОД РОБОТИ]"
    if u.label in GENERAL:
        specific = broad = GENERAL[u.label]
    if u.label == "LANGUAGE":
        specific = broad = re.sub(r"\s*\([^)]*\)", "", value)
    # Suppression removes the identifying value, not the grammatical slot.
    # An empty replacement created "з  до ." and comma-only skills lists.
    # Level 4 remains an alias for compatibility with recorded policies.
    placeholder = f"[{u.label}]"
    return [value, specific, broad, placeholder, placeholder]


def planned_edits(doc, policy):
    """Produce one auditable edit plan, including category merging inside skill lists.

    Entity offsets stay intact for detection metrics. render_* offsets include
    a removed duplicate's preceding separator and are used by both the executor
    and highlighting. Categories never merge across lines, prose or contacts.
    """
    rows, seen, previous = [], set(), None
    for u in doc.units:
        choices = options(doc, u)
        level = min(policy.get(u.group, 0), len(choices) - 1)
        original, replacement = doc.text[u.start:u.end], choices[level]
        geographic_only = u.source == "employer_phrase" and level == 0 and policy.get("city", 0) > 0
        if geographic_only:
            replacement = employer_without_geography(original)
        same_list = (previous is not None and previous.group == u.group == "skills"
                     and re.fullmatch(r"[ \t]*(?:[,;]|та|і|й|and|&)[ \t]*",
                                      doc.text[previous.end:u.start], re.IGNORECASE) is not None)
        if not same_list:
            seen.clear()
        applied = replacement != original
        decision = ("mask" if u.role in {"DIRECT", "SENSITIVE"} else
                    ("keep", "generalize", "generalize_broad", "mask", "remove")[level] if applied else "keep")
        if geographic_only and applied:
            decision = "generalize_location"
        row = {"replacement": replacement, "applied": applied, "decision": decision,
               "status": "applied" if applied else "unchanged", "tool": VERSION}
        if u.group == "skills" and level > 0 and applied and replacement:
            key = " ".join(replacement.casefold().split())
            if key in seen:
                row.update(replacement="", decision="mask_merge" if level >= 3 else "generalize_merge",
                           merged_into=replacement,
                           render_start=previous.end, render_end=u.end,
                           render_text=doc.text[previous.end:u.end])
            seen.add(key)
        rows.append(row)
        previous = u
    for i, u in enumerate(doc.units):
        if u.label == "SKILL_HEADER" and any(v.group == "skills" and v.source == u.source and rows[j]["applied"]
                                              for j, v in enumerate(doc.units)):
            rows[i].update(replacement="Навички: ", applied=True, decision="surface_rephrase", status="applied")
    return rows


def render(doc, policy):
    chunks, cursor = [], 0
    for u, edit in zip(doc.units, planned_edits(doc, policy)):
        start, end = edit.get("render_start", u.start), edit.get("render_end", u.end)
        chunks.extend((doc.text[cursor:start], edit["replacement"]))
        cursor = end
    chunks.append(doc.text[cursor:])
    return "".join(chunks)


class NeighbourProxy:
    """Freeze top-k pseudo-targets before transformations, exclude exact duplicates.

    Margins are relative similarity gaps, not probabilities. Background indexes
    never refit on each submitted resume. An OOV input is unassessed, not safe.
    """

    def __init__(self, indexes, background, original, k=3):
        self.indexes = indexes
        self.excluded = np.array([i for i, t in enumerate(background) if t == original], dtype=int)
        self.targets = {}
        self.cache = {}
        self.assessable = True
        for name, index in indexes.items():
            s = index.scores([original])[0].copy()
            s[self.excluded] = -np.inf
            available = np.flatnonzero(np.isfinite(s))
            if len(available) <= k or (len(available) and s[available].max() <= 0):
                self.assessable = False
            self.targets[name] = available[np.argsort(-s[available], kind="stable")[:k]]

    def margins(self, texts):
        if not self.assessable:
            return None
        missing = list(dict.fromkeys(t for t in texts if t not in self.cache))
        if not missing:
            return np.array([self.cache[t] for t in texts])
        columns = []
        for name, index in self.indexes.items():
            scores = index.scores(missing).copy()
            scores[:, self.excluded] = -np.inf
            targets = self.targets[name]
            best = scores[:, targets].max(axis=1)
            scores[:, targets] = -np.inf
            other = scores.max(axis=1)
            columns.append((best - other) / np.maximum(np.maximum(np.abs(best), np.abs(other)), 1e-9))
        self.cache.update(zip(missing, np.column_stack(columns)))
        return np.array([self.cache[t] for t in texts])


def search(doc, proxy, retvec, *, target=0.0, utility_floor=0.25):
    if not -1 <= target <= 1 or not 0 <= utility_floor <= 1:
        raise ValueError("Invalid search thresholds")
    policy, trace = {}, []
    output = render(doc, policy)  # Mandatory direct/sensitive masks BEFORE the stopping gate.
    assessable = proxy.assessable and retvec.transform([doc.text]).nnz > 0
    if not assessable:
        # Conservative, clearly labelled fallback when search has no evidence.
        policy = {u.group: 1 for u in doc.units if len(options(doc, u)) > 1}
        return policy, [{"stop_reason": "unassessed_conservative_fallback"}]
    current = proxy.margins([output])[0]
    retention = float(lexical_retention(retvec, doc.text, [output])[0])
    groups = sorted({u.group for u in doc.units if len(options(doc, u)) > 1})
    while np.max(current) > target:
        candidates = []
        texts = []
        for group in groups:
            # Skip ineffective intermediate levels, e.g. an unknown skill with
            # no defensible category. Masking remains a real explicit candidate.
            for level in range(policy.get(group, 0) + 1, 5):
                candidate = {**policy, group: level}
                changed = render(doc, candidate)
                if changed != output:
                    candidates.append((group, level))
                    texts.append(changed)
                    break
        if not texts:
            break
        margins = proxy.margins(texts)
        utility = lexical_retention(retvec, doc.text, texts)
        gain = current.sum() - margins.sum(axis=1)
        # Same summed-margin objective as v2; the previous per-attacker veto
        # blocked useful paths. Both attackers must still meet the final target.
        valid = (gain > 1e-8) & (utility >= utility_floor)
        if not valid.any():
            break
        merit = gain / (np.maximum(0.0, retention - utility) + 0.01)
        j = int(np.argmax(np.where(valid, merit, -np.inf)))
        group, level = candidates[j]
        policy[group] = level
        output = texts[j]
        current, retention = margins[j], float(utility[j])
        trace.append({"group": group, "level": level,
                      "operation": ("keep", "generalize", "generalize_broad", "mask", "remove")[level],
                      "proxy_margins": current.tolist(), "retention": retention})
    trace.append({"stop_reason": "proxy_target_met" if np.max(current) <= target else "target_unmet",
                  "proxy_margins": current.tolist(), "retention": retention})
    return policy, trace


def release_metrics(doc, release, proxy):
    """Describe the actual release; neighbour margins are NOT identity probabilities.

    Keep the original, mandatory-mask and final phases separate so an early
    mandatory change is not incorrectly attributed to the adaptive search.
    Missing attack evidence and absent skill annotations remain unavailable.
    """
    margins = proxy.margins([doc.text, render(doc, {}), release.text]) if proxy.assessable else None
    skills = [edit for edit in release.edits if edit["group"] == "skills"]
    kept = sum(not edit["applied"] for edit in skills)
    return {
        "risk_before": float(margins[0].max()) if margins is not None else None,
        "risk_after_direct_masking": float(margins[1].max()) if margins is not None else None,
        "risk_after": float(margins[2].max()) if margins is not None else None,
        "risk_target": 0.0,
        "risk_by_attacker": [
            {"attacker": name, "before": float(margins[0, i]),
             "after_direct_masking": float(margins[1, i]), "after": float(margins[2, i])}
            for i, name in enumerate(proxy.indexes)
        ] if margins is not None else [],
        "lexical_retention": release.retention,
        "skills_retention": kept / len(skills) if skills else None,
        "skills_kept": kept, "skills_total": len(skills),
        "skills_generalized": sum(edit.get("decision", "").startswith("generalize") for edit in skills),
        "skills_masked": sum(edit.get("decision", "").startswith(("mask", "remove")) for edit in skills),
    }


def compare_resume(resources, text, *, extractor=None, indexes=None, document=None, proxy=None):
    doc = document if document is not None else parse_resume(text, resources.codec, resources.detector, extractor=extractor)
    if doc.text != text:
        raise ValueError("Document must describe the submitted original text")
    if indexes is None:
        # Cached on the locally locked demo resource; both attackers share the same reference pool.
        if not hasattr(resources, "resume_indexes"):
            resources.resume_indexes = {
                "bm25": resources.feature_index,
                "char_tfidf": TfidfIndex(max_features=60_000).fit(resources.background_texts),
            }
        indexes = resources.resume_indexes
    proxy = proxy if proxy is not None else NeighbourProxy(indexes, resources.background_texts, text)
    settings = resources.config.get("adaptive", {})
    floor = float(settings.get("utility_floor", .25))
    target = float(settings.get("target", 0.0))
    policy, trace = search(doc, proxy, resources.retention_vectorizer, utility_floor=floor, target=target)
    all_groups = {u.group: 1 for u in doc.units if len(options(doc, u)) > 1}
    # Matched-tool controls. These are intentionally NOT labelled historical L2/L3.
    methods = {"adaptive_guarded": (policy, trace), "direct_only": ({}, []),
               "quasi_generalized": (all_groups, [])}
    entities = [{"unit": i, "text": text[u.start:u.end], **asdict(u)} for i, u in enumerate(doc.units)]
    releases = {}
    in_vocabulary = resources.retention_vectorizer.transform([text]).nnz > 0
    for name, (actions, steps) in methods.items():
        output = render(doc, actions)
        edits = []
        for row, edit in zip(entities, planned_edits(doc, actions)):
            edits.append({**row, **edit})
        ret = float(lexical_retention(resources.retention_vectorizer, text, [output])[0]) if in_vocabulary else None
        releases[name] = DemoRelease(output, edits, ret, steps)
    unresolved = [e["text"] for e in entities if e["label"] in {"OTHER_QUASI", "DATE_TIME", "DEMOGRAPHIC"}]
    return EntityDemoResult(text, entities, releases, "guarded_api" if extractor else "guarded_local", {
        "method_version": VERSION, "parser_version": PARSER_VERSION,
        "extractor_model": extractor.model if extractor else None,
        "rejected_extractions": getattr(extractor, "rejected_entities", 0),
        "stop_reason": trace[-1]["stop_reason"], "proxy_assessable": proxy.assessable,
        "unresolved_types": unresolved, "pool_size": len(resources.background_texts),
        "target": target, "utility_floor": floor,
        "metrics": release_metrics(doc, releases["adaptive_guarded"], proxy),
        "policy": policy,
        "direct_check": check_direct_identifiers(releases["adaptive_guarded"].text),
    })


def anonymize_resume(resources, text, *, extractor=None):
    """The single production entry point. Fixed policies are research controls only."""
    result = compare_resume(resources, text, extractor=extractor)
    if result.diagnostics["direct_check"]["issues"]:
        raise DirectIdentifierError("Known direct-identifier patterns remain in the proposed release")
    result.releases = {"adaptive": result.releases["adaptive_guarded"]}
    result.mode = "adaptive"
    return result
