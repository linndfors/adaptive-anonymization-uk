"""Typed entities and a shared, versioned action inventory for the v6 workflow.

Policies choose candidates; tools only construct replacements. No frequency-based
deletion, semantic inference from first words, or policy-specific action tools.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from . import category_taxonomy
from .adaptive_text import make_document
from .field_typing import LOCATION_GROUPS, employer_parent, provenance_type
from .pipeline import generalize_span
from .workua import FREE_TEXT_GROUPS, GROUP_LABELS, norm

VERSION = "typed_actions_v1"
# "regex": the explicit PARENTS catalog only; "taxonomy+regex": the cached closed
# category taxonomy first, PARENTS as fallback. The action's tool names the source.
CATALOGS = ("regex", "taxonomy+regex")
TYPES = {
    "title": "JOB_TITLE",
    "exp_position": "JOB_TITLE",
    "skills": "SKILL",
    "edu_faculty": "EDUCATION_FIELD",
    "exp_industry": "INDUSTRY",
    "edu_institution": "EDUCATION_INSTITUTION",
    "add_education": "EDUCATION_COURSE",
    "languages": "LANGUAGE",
    "age": "AGE",
    "salary": "SALARY",
    "exp_company": "EMPLOYER",
    "exp_dates": "EMPLOYMENT_PERIOD",
    "edu_years": "EDUCATION_PERIOD",
}

# Explicit lexical coverage, not a learned ontology. Unknown values stay unknown.
# Specific pattern -> defensible parent category -> broader category.
PARENTS = {
    "SKILL": [
        (
            r"\b(?:excel|outlook|word|powerpoint)\b|\b(?:ms\s+office|microsoft\s+office|office\s*365)\b",
            "офісні програми",
            "цифрові навички",
        ),
        (r"\b(?:wordpress|drupal|joomla)\b", "системи керування вебконтентом", "вебтехнології"),
        (r"битрикс|bitrix|salesforce|crm", "CRM-системи", "корпоративні інформаційні системи"),
        (r"\b1\s*[сc]\b|\bbaf\b|\bbas\b|\bsap\b|\berp\b", "облікові системи", "корпоративні інформаційні системи"),
        (
            r"python|java|javascript|typescript|c\+\+|c#|php|ruby",
            "програмування",
            "розроблення програмного забезпечення",
        ),
        (r"sql|postgres|mysql|oracle|mongodb", "робота з базами даних", "керування даними"),
        (
            r"ремонт.*авто|технічн.*стан|спец.*технік|вантажн|кранів|кран|екслуатац",
            "обслуговування техніки",
            "технічні навички",
        ),
        (r"економічн.*об[гґ]рун|бюджет|фінанс.*аналіз", "економічний аналіз", "аналітичні навички"),
        (r"управлін|керуван|керівниц|лідер|менеджмент", "управління", "організаційні навички"),
    ],
    "JOB_TITLE": [
        (
            r"(?:директор|начальник|керівник|руководител).*(?:фінанс|финанс|економ|эконом|планир|планув)",
            "керівник фінансово-економічного напряму",
            "керівник функціонального напряму",
        ),
        (r"транспорт|механ[іи]к|виробнич.*баз", "фахівець із транспорту та техніки", "технічний фахівець"),
        (r"бухгалтер|accountant", "фахівець з бухгалтерського обліку", "фінансовий фахівець"),
        (
            r"програм[іи]ст|розробник|разработчик|developer|[іи]нженер.*програм",
            "розробник програмного забезпечення",
            "ІТ-фахівець",
        ),
        (r"продаж|sales|торгов", "фахівець із продажів", "комерційний фахівець"),
        (r"маркет|реклам|marketing", "фахівець із маркетингу", "комерційний фахівець"),
        (r"вчител|учител|виклада|преподавател|педагог", "педагогічний працівник", "фахівець у сфері освіти"),
        (r"кадр|персонал|рекрутер|\bhr\b", "фахівець з управління персоналом", "фахівець з організації роботи"),
    ],
    "EDUCATION_FIELD": [
        (r"комп[’'ьюю]|[іи]нформат|програм", "інформаційні технології", "технічна освіта"),
        (r"ф[іи]нанс|облік|уч[её]т|[еэ]коном|менеджмент", "економіка та управління", "суспільні науки"),
    ],
}


@dataclass(frozen=True)
class Action:
    key: str
    operation: str
    replacement: str
    tool: str
    severity: float
    level: int = 0


def typed_document(extracted, codec, detector, *, exclude=(), include_residual=True):
    """Keep coarse detector labels for uncertain free text; trust structured schema.

    Recognize only known skill terms in free text. Do not guess that every coarse
    OCCUPATION_EDU span is a job title, nor relabel person/contact spans as skills.
    """
    skill_rx = re.compile(
        r"(?<!\w)(?:Python|JavaScript|TypeScript|SQL|Excel|MS Office|Bitrix24|Битрикс24|1С)(?!\w)", re.IGNORECASE
    )

    def detect(text):
        from .e3jsi import Span

        spans = list(detector(text))
        skills = [Span(m.start(), m.end(), "SKILL", "QUASI", "skill_lexicon") for m in skill_rx.finditer(text)]
        for skill in skills:
            overlaps = [s for s in spans if s.start < skill.end and skill.start < s.end]
            if not any(s.role in {"DIRECT", "SENSITIVE"} for s in overlaps):
                spans = [s for s in spans if not (s.start < skill.end and skill.start < s.end)]
                spans.append(skill)
        return spans

    doc = make_document(extracted, codec, detect, exclude=exclude, include_residual=include_residual)
    return replace(
        doc,
        units=[
            replace(u, label=provenance_type(u.source), generalized=None)
            if u.group in LOCATION_GROUPS
            else replace(u, label=TYPES.get(u.group, u.label), generalized=None)
            if u.group not in FREE_TEXT_GROUPS
            else replace(u, generalized=None)
            for u in doc.units
        ],
    )


def check_catalog(catalog):
    if catalog not in CATALOGS:
        raise ValueError(f"Unknown generalization_catalog: {catalog!r}; expected one of {CATALOGS}")
    if catalog != "regex" and not category_taxonomy.MAP_PATH.is_file():
        raise FileNotFoundError(
            f"generalization_catalog={catalog!r} needs {category_taxonomy.MAP_PATH}; "
            "build it with scripts/build_category_taxonomy.py or use 'regex'"
        )
    return catalog


def _parents(label, text, catalog="regex"):
    if catalog != "regex" and label in category_taxonomy.TAXONOMY:
        found = category_taxonomy.lookup(label, text)
        if found:
            return found
    for pattern, parent, broad in PARENTS.get(label, []):
        if re.search(pattern, text, re.IGNORECASE):
            if label == "JOB_TITLE" and re.search(r"керівник|начальник|директор", text, re.IGNORECASE):
                parent = parent.replace("фахівець", "керівник напряму", 1)
            return [(parent, "semantic_catalog", 0.2), (broad, "semantic_catalog", 0.4)]
    return []


def action_options(unit, value, codec, catalog="regex"):
    """Construct once; every policy receives exactly these same candidates."""
    options = [Action("keep", "keep", value, "identity", 0)]
    if unit.role == "CONTEXT":
        return options
    replacements = []
    if unit.role == "QUASI":
        if unit.group in {"age", "salary", "exp_dates", "edu_years", "exp_company", "edu_institution"}:
            replacements = [("; ".join(codec.generalize(unit.group, [value])), "field_rules", 0.2)]
            if unit.group == "age":
                age = int(value)
                replacements.append((f"{age // 20 * 20}–{age // 20 * 20 + 19}", "age_range", 0.4))
            elif unit.group == "salary":
                salary = int(re.sub(r"\D", "", value))
                if salary >= 100_000:
                    broad = "понад 50 тис. грн"
                else:
                    lo = salary // 50_000 * 50_000
                    broad = f"{lo // 1000}–{(lo + 50_000) // 1000} тис. грн"
                replacements.append((broad, "salary_range", 0.4))
        elif unit.label in PARENTS or catalog != "regex" and unit.label in category_taxonomy.TAXONOMY:
            replacements = _parents(unit.label, value, catalog)
        elif unit.label == "EMPLOYER_TYPE":
            replacements = [(employer_parent(value), "employer_type_category", 0.2)]
        elif unit.label == "EMPLOYER" and unit.group in LOCATION_GROUPS:
            # A named employer in a location field: generalize to its venue/legal-form parent.
            replacements = [(employer_parent(value), "named_employer_category", 0.2)]
        else:
            result = generalize_span(unit.label, value)
            if result:
                replacements = [(result, "span_rules", 0.25)]
    seen = {norm(value)}
    for replacement, tool, severity in replacements:
        if replacement and norm(replacement) not in seen:
            seen.add(norm(replacement))
            level = sum(action.operation == "generalize" for action in options) + 1
            options.append(Action(f"generalize_{level}", "generalize", replacement, tool, severity, level))
    options.extend(
        [
            Action("mask", "mask", f"[{unit.label}]", "typed_placeholder", 0.95),
            Action("remove", "remove", "", "deletion", 1.0),
        ]
    )
    return options


def action_bank(doc, codec):
    return {i: action_options(unit, doc.text[unit.start : unit.end], codec) for i, unit in enumerate(doc.units)}


# make_document layout: "Label: item; item. Label: item." Removal must not leave "; ;" or an empty label.
_LABELS = "|".join(re.escape(label) for label in GROUP_LABELS.values())
_HEAD = re.compile(rf"(?:^|\. )(?:{_LABELS}): \Z")
_TAIL = re.compile(rf"[.!?…]*(?:(; )|\.(?= (?:{_LABELS}): |\Z))")


def _item_start(text, s):
    head = _HEAD.search(text, max(0, s - 64), s)
    if head:
        return "head", head.start() + head.group(0).startswith(".")
    if text.endswith("; ", 0, s):
        return "sep", s - 2
    return ("bos" if not text[:s].strip() else None), s


def _joins(text, region, unit):
    gap = text[region[1] : unit.start]
    if re.fullmatch(r"[\s,]*", gap):
        return True
    starts_item = _item_start(text, region[0])[0] is not None
    if re.fullmatch(r"[\s,.!?:…]*", gap):  # keep the full stop of a sentence that is partly kept
        return starts_item
    # Merge across an item separator only when a whole item disappears with it.
    return bool(re.fullmatch(r"[.!?…]*; ", gap)) and (starts_item or _TAIL.match(text, unit.end) is not None)


def _widen(text, s, e, lo, hi):
    """Extend a removed span over the separator, label or space it would leave dangling."""
    kind, at = _item_start(text, s)
    tail = _TAIL.match(text, e)
    if kind in {"head", "sep"} and tail:
        if tail.group(1):  # another item follows
            return (s - 2, tail.end() - 2) if kind == "sep" else (s, tail.end())
        if kind == "sep":
            return s - 2, tail.end() - 1
        # No item left: drop the label too, exactly as make_document omits an empty field.
        return at, tail.end() + (at == 0 and text.startswith(" ", tail.end()))
    if kind:
        after = text[e:hi]
        return s, e + len(after) - len(after.lstrip(",.!?:;… \t\n"))
    before, after = text[lo:s], text[e:hi]
    left, right = len(before) - len(before.rstrip()), len(after) - len(after.lstrip())
    if left and right:
        return (s - left, e) if "\n" in after[:right] else (s, e + right)
    if left and (e == len(text) or text[e] in ",.!?:;…)"):
        return s - left, e
    return s, e


def render(doc, actions):
    text, edits, removing = doc.text, [], False
    for i, unit in enumerate(doc.units):
        action = actions.get(i)
        if action is None or action.operation == "keep":
            removing = False
        elif action.operation != "remove":
            edits.append([unit.start, unit.end, action.replacement])
            removing = False
        elif removing and _joins(text, edits[-1], unit):
            edits[-1][1] = unit.end
        else:
            edits.append([unit.start, unit.end, None])
            removing = True
    chunks, cursor = [], 0
    for k, (start, end, replacement) in enumerate(edits):
        if replacement is None:
            hi = edits[k + 1][0] if k + 1 < len(edits) else len(text)
            start, end = _widen(text, start, end, cursor, hi)
            replacement = ""
        start = max(start, cursor)
        chunks.extend([text[cursor:start], replacement])
        cursor = max(cursor, end)
    return "".join(chunks) + text[cursor:]


def mandatory(doc, bank):
    return {
        i: next(a for a in bank[i] if a.operation == "mask")
        for i, u in enumerate(doc.units)
        if u.role in {"DIRECT", "SENSITIVE"}
    }


def fixed_actions(doc, bank, policy):
    if policy == "L0_none":
        return {}
    actions = mandatory(doc, bank)
    if policy == "L1_scrub_text":
        return actions
    if policy not in {"L2_generalize", "L3_strong", "mask_detected"}:
        raise ValueError(f"Unknown fixed policy: {policy}")
    for i, unit in enumerate(doc.units):
        if unit.role != "QUASI":
            continue
        generalizations = [a for a in bank[i] if a.operation == "generalize"]
        if policy == "L2_generalize":
            if generalizations:
                actions[i] = generalizations[0]
        elif policy == "L3_strong" and generalizations:
            actions[i] = generalizations[-1]
        else:
            actions[i] = next(a for a in bank[i] if a.operation == "mask")
    return actions


def semantic_retention(doc, actions):
    """Explicit action-cost proxy, not a semantic model or recruiter evaluation.

    Average within fields before averaging fields, so a long skills list cannot
    outweigh the loss of the job title. Never count mandatory PII as useful content.
    """
    fields = {}
    for i, unit in enumerate(doc.units):
        if unit.role in {"DIRECT", "SENSITIVE"}:
            continue
        fields.setdefault(unit.group, []).append(1 - actions[i].severity if i in actions else 1.0)
    return sum(sum(v) / len(v) for v in fields.values()) / max(1, len(fields))
