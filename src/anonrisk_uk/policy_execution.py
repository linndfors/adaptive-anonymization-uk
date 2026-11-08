"""Decision contracts and tool routing. Execution happens AFTER policy selection."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .action_space import Action, _parents, action_options, check_catalog, render
from .adaptive_text import linked_units
from .pipeline import generalize_span

VERSION = "entity_decision_v2"
CLASSES = ("keep", "range:1", "range:2", "generalize:1", "generalize:2", "mask", "remove")
NUMERIC_GROUPS = {"age", "salary"}
PROFESSIONAL_TYPES = {"SKILL", "JOB_TITLE", "EDUCATION_FIELD", "INDUSTRY"}
EMPLOYER_LABELS = {"EMPLOYER_TYPE", "EMPLOYER"}


@dataclass(frozen=True)
class Decision:
    unit: int
    action: str
    level: int = 0
    risk_score: float = 0.0
    confidence: float = 1.0
    reason: str = "fixed_policy"

    @property
    def key(self):
        return f"{self.action}:{self.level}" if self.action in {"range", "generalize"} else self.action

    @classmethod
    def from_key(cls, unit, key, **kwargs):
        operation, _, level = key.partition(":")
        if key not in CLASSES:
            raise ValueError(f"Unsupported anonymization decision: {key}")
        return cls(unit, operation, int(level) if level else 0, **kwargs)


class UnsupportedDecision(ValueError):
    pass


class ToolRouter:
    """Resolve action/type compatibility independently of the learned policy.

    Routing is deliberately deterministic: numeric intervals for numeric fields,
    a known taxonomy parent for professional entities, LLM only as the configured
    semantic fallback. Unavailable execution never silently becomes a mask.
    """

    def __init__(self, codec, config=None, llm=None):
        self.codec = codec
        self.config = config or {}
        self.llm = llm
        self.catalog = check_catalog(self.config.get("generalization_catalog", "regex"))
        self.events = []

    def available(self, doc, unit_id):
        """Capability metadata only. No replacement generation or network calls."""
        unit = doc.units[unit_id]
        if unit.role == "CONTEXT":
            return ["keep"]
        if unit.role in {"DIRECT", "SENSITIVE"}:
            return ["mask"]
        result = ["keep", "mask", "remove"]
        value = doc.text[unit.start : unit.end]
        if unit.group in NUMERIC_GROUPS:
            result[1:1] = ["range:1", "range:2"]
        elif unit.label in PROFESSIONAL_TYPES:
            parents = _parents(unit.label, value, self.catalog)
            if parents:
                result[1:1] = [f"generalize:{i + 1}" for i in range(len(parents))]
            elif self.llm is not None:
                result.insert(1, "generalize:1")
        elif (
            unit.group in {"exp_dates", "edu_years", "exp_company", "edu_institution"} or unit.label in EMPLOYER_LABELS
        ):
            # EMPLOYER_TYPE and a named EMPLOYER in a location field execute through
            # action_options: venue/legal form -> parent category.
            result.insert(1, "generalize:1")
        return result

    def execute(self, doc, decision):
        # Identity is also needed by the explicit L0 evaluation reference. The
        # adaptive workflow enforces mandatory masks outside the learned policy.
        if decision.action != "keep" and decision.key not in self.available(doc, decision.unit):
            raise UnsupportedDecision(f"{decision.key} is unavailable for {doc.units[decision.unit].label}")
        unit = doc.units[decision.unit]
        value = doc.text[unit.start : unit.end]
        if decision.action == "keep":
            action = Action("keep", "keep", value, "identity", 0)
        elif decision.action in {"mask", "remove"}:
            action = Action(
                decision.key,
                decision.action,
                f"[{unit.label}]" if decision.action == "mask" else "",
                "typed_placeholder" if decision.action == "mask" else "deletion",
                0.95 if decision.action == "mask" else 1.0,
            )
        elif decision.action == "generalize" and unit.group == "exp_dates":
            years = [int(year) for year in re.findall(r"(?:19|20)\d{2}", value)]
            if not years:
                raise UnsupportedDecision("No valid employment period to generalize")
            decades = sorted({year // 10 * 10 for year in years})
            replacement = (
                f"від {decades[0]}-х років"
                if "дотепер" in value.lower()
                else " — ".join(f"{year}-ті" for year in decades) + " роки"
            )
            action = Action(decision.key, "generalize", replacement, "period_decades", 0.2, decision.level)
        elif decision.action == "generalize" and unit.group == "exp_company":
            replacement = generalize_span("ORGANIZATION", value) or "роботодавець"
            action = Action(decision.key, "generalize", replacement, "employer_category", 0.2, decision.level)
        else:
            # The replacement tool is invoked here, only after a Decision exists.
            options = [a for a in action_options(unit, value, self.codec, self.catalog) if a.operation == "generalize"]
            if options and decision.level <= len(options):
                chosen = options[decision.level - 1]
                tool = "numeric_range" if decision.action == "range" else chosen.tool
                replacement = chosen.replacement
                if unit.group == "salary" and replacement.startswith("понад "):
                    replacement = "від " + replacement[len("понад ") :]
                action = Action(decision.key, decision.action, replacement, tool, chosen.severity, decision.level)
            elif self.llm is not None and decision.action == "generalize" and decision.level == 1:
                replacement, tool = self.llm(doc, decision)
                if not replacement or replacement.casefold() == value.casefold():
                    raise UnsupportedDecision("Semantic tool did not return a usable broader category")
                action = Action(decision.key, "generalize", replacement, tool, 0.3, 1)
            else:
                raise UnsupportedDecision("Selected tool cannot implement this level")
        self.events.append({"unit": decision.unit, "decision": decision.key, "tool": action.tool})
        return action

    def apply(self, doc, current, decision):
        result = dict(current)
        for unit_id in linked_units(doc, decision.unit):
            actual = Decision.from_key(
                unit_id,
                decision.key,
                risk_score=decision.risk_score,
                confidence=decision.confidence,
                reason=decision.reason,
            )
            action = self.execute(doc, actual)
            if action.operation == "keep":
                result.pop(unit_id, None)
            else:
                result[unit_id] = action
        return result


def protected_actions(doc, router):
    actions = {}
    for i, unit in enumerate(doc.units):
        if unit.role in {"DIRECT", "SENSITIVE"}:
            actions = router.apply(doc, actions, Decision(i, "mask", reason="mandatory_protection"))
    return actions


def fixed_decisions(doc, router, policy):
    if policy not in {"L0_none", "L1_scrub_text", "L2_generalize", "L3_strong", "mask_detected"}:
        raise ValueError(policy)
    decisions = []
    for i, unit in enumerate(doc.units):
        key = "keep"
        if policy != "L0_none" and unit.role in {"DIRECT", "SENSITIVE"}:
            key = "mask"
        elif unit.role == "QUASI" and policy not in {"L0_none", "L1_scrub_text"}:
            options = [key for key in router.available(doc, i) if key.startswith(("range:", "generalize:"))]
            if policy == "L2_generalize":
                key = options[0] if options else "keep"
            elif policy == "L3_strong":
                key = options[-1] if options else "mask"
            elif policy == "mask_detected":
                key = "mask"
        decisions.append(Decision.from_key(i, key))
    return decisions


def execute_fixed(doc, router, policy):
    actions, audit = {}, []
    for decision in fixed_decisions(doc, router, policy):
        try:
            actions = router.apply(doc, actions, decision)
            tool = actions[decision.unit].tool if decision.unit in actions else "identity"
            audit.append({"unit": decision.unit, "decision": decision.key, "tool": tool, "status": "executed"})
        except UnsupportedDecision as exc:
            audit.append({"unit": decision.unit, "decision": decision.key, "tool": "unavailable", "status": str(exc)})
    return render(doc, actions), actions, audit
