"""Conservative field hints for the local text demo, with exact source offsets.

These rules are an inference adapter, not a trained parser. Unrecognised prose
is left outside the action space and must not be mistaken for verified safe text.
"""

from __future__ import annotations

import re
from dataclasses import replace

from .adaptive_text import ReleaseDocument, TextUnit
from .policy_execution import ToolRouter

VERSION = "resume_fields_v1"
CORE_GROUPS = {"skills", "title", "exp_position"}
SKILLS = (
    "Python", "JavaScript", "TypeScript", "Java", "C++", "C#", "PHP", "SQL", "PostgreSQL",
    "MySQL", "MongoDB", "Redis", "Excel", "MS Office", "Microsoft Office", "Word", "PowerPoint",
    "Bitrix24", "Битрикс24", "1С", "BAS", "SAP", "React", "Angular", "Vue", "Node.js",
    "Django", "Docker", "Kubernetes", "Linux", "Git", "Figma", "Photoshop", "AutoCAD",
)
SKILL_RX = re.compile(
    r"(?<!\w)(?:" + "|".join(re.escape(v) for v in sorted(SKILLS, key=len, reverse=True)) + r")(?!\w)",
    re.IGNORECASE,
)
SKILL_HEADER = re.compile(r"(?im)^\s*(?:навички|ключові навички|технології|стек|skills|tech stack)\s*:\s*")
FIELD_HEADER = re.compile(
    r"(?im)^\s*(?P<kind>посада|бажана посада|position|job title|зарплата|очікувана зарплата|salary)\s*:\s*"
)
WORK_CONTEXT = re.compile(r"працю|работ|компан|роботодав|работодат|employ|worked|company|experience", re.IGNORECASE)
EDU_CONTEXT = re.compile(r"освіт|навча|університет|коледж|образован|университет|education|university", re.IGNORECASE)
PERIOD = re.compile(r"(?:19|20)\d{2}\s*(?:[–—-]|до|to)\s*(?:(?:19|20)\d{2}|дотепер|present)", re.IGNORECASE)


def _overlap(a, b):
    return a.start < b.end and b.start < a.end


def field_document(text: str, detected: ReleaseDocument) -> ReleaseDocument:
    """Use explicit headers and NER context; direct/sensitive spans always win."""
    protected = [u for u in detected.units if u.role in {"DIRECT", "SENSITIVE"}]
    candidates = []
    for header in SKILL_HEADER.finditer(text):
        end = text.find("\n", header.end())
        end = len(text) if end < 0 else end
        for item in re.finditer(r"[^,;•\n]+", text[header.end():end]):
            value = item.group().strip().strip(". ")
            if not value:
                continue
            start = header.end() + item.start() + item.group().index(value)
            candidates.append(TextUnit(start, start + len(value), "SKILL", "QUASI", "skills", "resume_header"))
    for match in SKILL_RX.finditer(text):
        candidates.append(TextUnit(match.start(), match.end(), "SKILL", "QUASI", "skills", "skill_lexicon"))
    for header in FIELD_HEADER.finditer(text):
        end = text.find("\n", header.end())
        end = len(text) if end < 0 else end
        value = text[header.end():end].strip().strip(". ")
        if not value:
            continue
        start = header.end() + text[header.end():end].index(value)
        if re.search(r"зарплат|salary", header["kind"], re.IGNORECASE):
            match = re.search(r"\d[\d \u00a0]*", value)
            if match and not value[match.end():].lstrip().startswith(("тис", "k", "K", "–", "-")):
                candidates.append(TextUnit(
                    start + match.start(), start + match.end(), "SALARY", "QUASI", "salary", "resume_header"
                ))
        else:
            candidates.append(TextUnit(start, start + len(value), "JOB_TITLE", "QUASI", "title", "resume_header"))
    for match in PERIOD.finditer(text):
        a, b = text.rfind("\n", 0, match.start()) + 1, text.find("\n", match.end())
        line = text[a:b if b >= 0 else len(text)]
        group = "exp_dates" if WORK_CONTEXT.search(line) else "edu_years" if EDU_CONTEXT.search(line) else None
        if group:
            start, end = match.start(), match.end()
            prefix = re.search(r"\b(?:з|із|від|с|from)\s+$", text[:start], re.IGNORECASE)
            if prefix:
                start = prefix.start()
            suffix = re.match(r"\s*(?:роках|років|роки|року|рр\.)(?!\w)", text[end:], re.IGNORECASE)
            if suffix:
                end += suffix.end()
            candidates.append(TextUnit(start, end, "EMPLOYMENT_PERIOD" if group == "exp_dates"
                                       else "EDUCATION_PERIOD", "QUASI", group, "resume_context"))
    for unit in detected.units:
        if unit in protected or unit.source == "residual_text":
            continue
        a, b = text.rfind("\n", 0, unit.start) + 1, text.find("\n", unit.end)
        line = text[a:b if b >= 0 else len(text)]
        if unit.label == "ORGANIZATION" and EDU_CONTEXT.search(line) and not WORK_CONTEXT.search(line):
            unit = replace(unit, label="EDUCATION_INSTITUTION", group="edu_institution", source="resume_context")
        elif unit.label == "ORGANIZATION" and WORK_CONTEXT.search(line):
            unit = replace(unit, label="EMPLOYER", group="exp_company", source="resume_context")
        elif unit.label == "DATE_TIME" and WORK_CONTEXT.search(line) and re.search(r"(?:19|20)\d{2}", text[unit.start:unit.end]):
            unit = replace(unit, label="EMPLOYMENT_PERIOD", group="exp_dates", source="resume_context")
        candidates.append(unit)
    selected = list(protected)
    for unit in candidates:
        if not any(_overlap(unit, other) for other in selected):
            selected.append(unit)
    return ReleaseDocument(text, sorted(selected, key=lambda u: (u.start, u.end)))


class FieldToolRouter(ToolRouter):
    """Demo-only surface realization for inferred fields; research tools stay intact."""

    def execute(self, doc, decision):
        action = super().execute(doc, decision)
        unit = doc.units[decision.unit]
        if action.operation != "generalize":
            return action
        value, before = doc.text[unit.start:unit.end], doc.text[:unit.start]
        if unit.group == "exp_dates":
            years = [int(y) for y in re.findall(r"(?:19|20)\d{2}", value)]
            decades = sorted({y // 10 * 10 for y in years})
            if decades:
                period = " — ".join(f"{y}-х" for y in decades)
                if re.match(r"(?:з|із|від|с|from)\s", value, re.IGNORECASE):
                    replacement = f"у {period} роках"
                elif re.search(r"\b[ув]\s+$", before, re.IGNORECASE):
                    replacement = f"{period} роках"
                else:
                    replacement = f"{period} років"
                action = replace(action, replacement=replacement, tool="period_decades_context")
        elif unit.group == "exp_company":
            if re.search(r"\bкомпанії\s+$", before, re.IGNORECASE):
                action = replace(action, replacement="без уточнення назви", tool="employer_category_context")
            elif re.search(r"\b[ув]\s+$", before, re.IGNORECASE):
                replacement = "банківській установі" if "банк" in value.casefold() else "організації"
                action = replace(action, replacement=replacement, tool="employer_category_context")
        return action


class ContentGuardRouter(FieldToolRouter):
    """Keep professional core values or generalize them; no masking/deletion.

    Identifiers in a skills/title header retain their detected role and remain
    mandatory masks. This restriction changes the inference policy explicitly.
    """

    def available(self, doc, unit_id):
        options = super().available(doc, unit_id)
        unit = doc.units[unit_id]
        if unit.role == "QUASI" and unit.group in CORE_GROUPS:
            return [key for key in options if key == "keep" or key.startswith("generalize:")]
        return options
