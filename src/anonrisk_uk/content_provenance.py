"""Post-hoc content views anchored to source fields and verified edit provenance.

This module never chooses an anonymization action. It avoids reparsing field
headers after those headers have themselves been transformed. Field boundaries
are available to the evaluator from the original WorkUA serialization, not to
the deployed action predictor as hidden labels.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .content_evaluation import PROFESSIONAL_GROUPS
from .resume_adaptive import GENERAL, MASKS, planned_edits
from .resume_parser import FIELD_TYPES, FIELDS
from .taxonomy import ROLE_OF_CLASS
from .workua import GROUP_LABELS, GROUPS

KNOWN_LABELS = ({value[1:-1] for value in [*GENERAL.values(), *MASKS.values()]} | set(FIELDS)
                | set(FIELD_TYPES.values()) | {group.upper() for group in GROUPS}
                | set(ROLE_OF_CLASS) | {"YEAR", "NUMBER", "ПЕРІОД РОБОТИ", "QUASI_IDENT"})


def remove_mask_labels(text):
    """Remove exact generated masks, retaining literals and duration payloads."""
    def replace(match):
        label = match[1]
        if label.startswith("ПЕРІОД РОБОТИ:"):
            return label.split(":", 1)[1].strip()
        return " " if label in KNOWN_LABELS else match[0]
    return re.sub(r"\[([^\]\n]+)\]", replace, text)


def content_text(text):
    """Normalize a content view after removing only known service markers."""
    value = remove_mask_labels(text)
    return re.sub(r"\s+", " ", value).strip(" .;,\n")


@dataclass(frozen=True)
class SourceField:
    group: str
    start: int
    end: int


def serialized_fields(extracted):
    """Reconstruct the exact public serialization and its original value spans."""
    parts, fields, cursor = [], [], 0
    for group in GROUPS:
        values = extracted.get(group, [])
        if not values:
            continue
        prefix = (" " if parts else "") + GROUP_LABELS[group] + ": "
        value = "; ".join(values)
        start = cursor + len(prefix)
        parts.extend([prefix, value, "."])
        fields.append(SourceField(group, start, start + len(value)))
        cursor = start + len(value) + 1
    return "".join(parts), fields


def field_view(extracted):
    values = [content_text("; ".join(extracted.get(group, []))) for group in GROUPS
              if group in PROFESSIONAL_GROUPS]
    return ". ".join(value for value in values if re.search(r"\w", value))


def operation_plan(doc, policy):
    return [{"start": edit.get("render_start", unit.start),
             "end": edit.get("render_end", unit.end), "replacement": edit["replacement"]}
            for unit, edit in zip(doc.units, planned_edits(doc, policy))]


def project_operations(source, fields, operations, *, expected_output=None):
    """Project an exact renderer plan into original professional value spans.

    Replacements crossing multiple field bodies are not assigned a guessed
    semantic origin. A word-bearing cross-field replacement touching a
    professional field makes that record explicitly unscorable. Plain masks
    and deletion carry no content and need no ambiguous attribution.
    """
    values = {field.group: [] for field in fields if field.group in PROFESSIONAL_GROUPS}
    output, ambiguities, cursor = [], [], 0

    def copied(start, end):
        for field in fields:
            if field.group not in values:
                continue
            a, b = max(start, field.start), min(end, field.end)
            if a < b:
                values[field.group].append(source[a:b])

    for operation in operations:
        start, end, replacement = operation["start"], operation["end"], operation["replacement"]
        if not isinstance(start, int) or not isinstance(end, int) or not cursor <= start <= end <= len(source):
            raise ValueError("Edit provenance overlaps or lies outside the source")
        output.extend([source[cursor:start], replacement])
        copied(cursor, start)
        if replacement == source[start:end]:
            copied(start, end)
        else:
            touched = [field for field in fields if max(start, field.start) < min(end, field.end)]
            professional = [field for field in touched if field.group in values]
            if len(touched) == 1 and professional:
                values[professional[0].group].append(replacement)
            elif professional and re.search(r"\w", content_text(replacement)):
                ambiguities.append({"start": start, "end": end,
                                    "groups": [field.group for field in touched]})
        cursor = end
    output.append(source[cursor:])
    copied(cursor, len(source))
    actual = "".join(output)
    if expected_output is not None and actual != expected_output:
        raise ValueError("Recorded edit provenance does not reproduce the frozen release")
    cleaned = [content_text("".join(values[group])) for group in GROUPS if group in values]
    view = ". ".join(value for value in cleaned if re.search(r"\w", value))
    return {"text": view, "scorable": not ambiguities, "ambiguous_cross_field_edits": ambiguities,
            "release_matches": expected_output is None or actual == expected_output}
