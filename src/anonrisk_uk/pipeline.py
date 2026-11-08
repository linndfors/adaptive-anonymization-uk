"""Legacy raw-text prototype from detection to adaptive transformation.

The heuristic fragment score combines a class prior from WorkUA ablation with a
population-rarity multiplier. It is not an S2 probability. The current learned
unit-risk workflow is implemented in ``adaptive_text`` and ``risk_training``.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .e3jsi import Span, regex_detect
from .taxonomy import ROLE_OF_CLASS
from .workua import norm

# Detector-class to WorkUA-group mapping for priors and value frequencies
CLASS_TO_GROUPS = {
    "LOCATION": ["city", "exp_city", "edu_city"],
    "ORGANIZATION": ["exp_company", "edu_institution"],
    "OCCUPATION_EDU": ["title", "exp_position", "edu_faculty"],
    "DATE_TIME": ["exp_dates", "edu_years"],
    "DEMOGRAPHIC": ["age"],
    "FINANCIAL": ["salary"],
}
# Fallback priors used when no empirical table is supplied
DEFAULT_PRIOR = {
    "PERSON": 1.0,
    "CONTACT": 1.0,
    "ID_NUMBER": 1.0,
    "ADDRESS": 1.0,
    "HEALTH": 0.8,
    "ORGANIZATION": 0.5,
    "DATE_TIME": 0.5,
    "DEMOGRAPHIC": 0.4,
    "LOCATION": 0.3,
    "OCCUPATION_EDU": 0.2,
    "FINANCIAL": 0.2,
    "OTHER_QUASI": 0.3,
    "UNMAPPED": 0.3,
}
MASK = {
    "PERSON": "[PERSON]",
    "CONTACT": "[CONTACT]",
    "ID_NUMBER": "[ID_NUMBER]",
    "ADDRESS": "[ADDRESS]",
    "HEALTH": "[HEALTH_INFORMATION]",
    "ORGANIZATION": "[ORGANIZATION]",
    "DATE_TIME": "[DATE]",
    "DEMOGRAPHIC": "[AGE]",
    "LOCATION": "[LOCATION]",
    "OCCUPATION_EDU": "[OCCUPATION]",
    "FINANCIAL": "[AMOUNT]",
    "OTHER_QUASI": "[OTHER]",
    "UNMAPPED": "[OTHER]",
}


def money_value(text: str):
    """Parse one integer amount without merging ranges or decimals."""
    value = re.sub(r"\s+", " ", text.lower().strip())
    currency = "грн"
    for pattern, label in [(r"[$]|usd|долар", "USD"), (r"[€]|eur|євро", "EUR"), (r"[£]|gbp|фунт", "GBP")]:
        if re.search(pattern, value):
            currency = label
    value = re.sub(r"грн\.?|гривень|гривні|uah|usd|eur|gbp|доларів|євро|[$€£₴]", "", value).strip()
    if not re.fullmatch(r"\d+(?: \d{3})*", value):
        return None
    return int(re.sub(r"\s", "", value)), currency


def canonical_values(cls: str, text: str) -> set[str]:
    """Return canonical frequency keys for dates, ages, money, or text.

    Date ranges contribute individual years. Currencies not represented by
    WorkUA remain unknown.
    """
    if cls == "DATE_TIME":
        return set(re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", text))
    if cls == "DEMOGRAPHIC":
        m = re.fullmatch(r"\s*(\d{1,2})\s*(?:років|роки|рік|річний|річна)?\s*", norm(text))
        return {str(int(m[1]))} if m and 14 <= int(m[1]) <= 90 else set()
    if cls == "FINANCIAL":
        parsed = money_value(text)
        return {f"{parsed[0]} {parsed[1]}"} if parsed else set()
    return {norm(text)} if norm(text) else set()


class SpanFrequencyReference:
    """Count train documents containing a canonical class value.

    Values are unioned across groups within a document to prevent double counting.
    """

    def __init__(self):
        self.freq = {cls: Counter() for cls in CLASS_TO_GROUPS}
        self.n_documents = 0

    def fit(self, extracted):
        self.freq = {cls: Counter() for cls in CLASS_TO_GROUPS}
        self.n_documents = 0
        for ex in extracted:
            self.n_documents += 1
            for cls, groups in CLASS_TO_GROUPS.items():
                values = {key for g in groups for value in ex.get(g, []) for key in canonical_values(cls, value)}
                self.freq[cls].update(values)
        return self

    def frequency(self, cls, value):
        keys = canonical_values(cls, value)
        # The rarest known range component determines its frequency.
        return min((self.freq.get(cls, {}).get(key, 0) for key in keys), default=0)


def span_role(span):
    role = ROLE_OF_CLASS.get(span.label, span.role)
    return span.role if span.role in {"DIRECT", "SENSITIVE"} else role


def select_spans(text, spans):
    """Give direct and sensitive spans priority over longer quasi-identifiers."""
    spans = set(spans)
    for s in spans:
        if not (0 <= s.start < s.end <= len(text)):
            raise ValueError(f"Invalid span boundaries: {s}")
    selected = []
    protected = sorted(
        (s for s in spans if span_role(s) in {"DIRECT", "SENSITIVE"}),
        key=lambda s: (s.start, -s.end, s.label, s.source),
    )
    for s in protected:
        if selected and s.start < selected[-1].end:
            previous = selected.pop()
            selected.append(
                Span(previous.start, max(previous.end, s.end), previous.label, span_role(previous), "merged_protected")
            )
        else:
            selected.append(s)
    others = sorted(
        (s for s in spans if span_role(s) not in {"DIRECT", "SENSITIVE"}),
        key=lambda s: (-(s.end - s.start), s.start, s.label, s.source),
    )
    for s in others:
        if all(s.end <= o.start or s.start >= o.end for o in selected):
            selected.append(s)
    return sorted(selected, key=lambda s: s.start)


def generalize_span(cls: str, text: str) -> str | None:
    """Return a meaningful generalization or ``None`` when none is available."""
    result = None
    if cls == "DATE_TIME":
        years = sorted(canonical_values(cls, text))
        result = "–".join(years) if years else None
    elif cls == "DEMOGRAPHIC":
        values = canonical_values(cls, text)
        if values:
            lo = int(next(iter(values))) // 10 * 10
            result = f"{lo}–{lo + 9} років"
    elif cls == "FINANCIAL":
        parsed = money_value(text)
        if parsed:
            x, currency = parsed
            mag = 10 ** max(0, len(str(x)) - 1)
            result = f"близько {x // mag * mag} {currency}"
    elif cls == "ORGANIZATION":
        low = text.lower()
        for k, v in [
            ("банк", "банк"),
            ("лікарн", "лікарня"),
            ("клінік", "клініка"),
            ("суд", "суд"),
            ("університет", "університет"),
            ("школ", "школа"),
            ("страхов", "страхова компанія"),
        ]:
            if k in low:
                result = v
                break
    return result if result is not None and norm(result) != norm(text) else None


@dataclass
class Decision:
    span: Span
    text: str
    risk: float
    rarity: float
    action: str
    replacement: str
    reason: str


class TextAnonymizer:
    def __init__(
        self,
        detectors=None,
        class_prior: dict | None = None,
        codec=None,
        tau_mask: float = 0.6,
        tau_generalize: float = 0.3,
        rarity_ref: int = 100,
        frequency_reference=None,
    ):
        """Configure detectors, class priors, and a fitted frequency reference."""
        if not 0 <= tau_generalize <= tau_mask <= 1 or rarity_ref <= 1:
            raise ValueError("Expected 0 <= tau_generalize <= tau_mask <= 1 and rarity_ref > 1")
        if any(not math.isfinite(float(v)) or not 0 <= float(v) <= 1 for v in (class_prior or {}).values()):
            raise ValueError("Prior scores must be finite numbers in [0, 1]")
        self.detectors = [regex_detect] if detectors is None else detectors
        self.frequency_reference = frequency_reference
        self.prior = {**DEFAULT_PRIOR, **(class_prior or {})}
        self.codec, self.tau_mask, self.tau_gen, self.ref = codec, tau_mask, tau_generalize, rarity_ref

    @staticmethod
    def prior_from_groups(group_delta: dict[str, float]) -> dict[str, float]:
        """Convert mean group add-back contributions to normalized class priors."""
        group_delta = {g: max(0.0, float(v)) for g, v in group_delta.items() if math.isfinite(float(v))}
        top = max(group_delta.values(), default=0.0) or 1.0
        out = {}
        for cls, groups in CLASS_TO_GROUPS.items():
            vals = [group_delta[g] for g in groups if g in group_delta]
            if vals:
                out[cls] = max(vals) / top
        return out

    def rarity(self, cls: str, value: str) -> float:
        """Return one for unseen values and approach zero for common values."""
        if self.frequency_reference is not None:
            f = self.frequency_reference.frequency(cls, value)
        elif self.codec is not None and cls in CLASS_TO_GROUPS:
            # Legacy compatibility: old counters do not canonicalize complete dates.
            keys = canonical_values(cls, value)
            f = max((self.codec.population_frequency(g, key) for g in CLASS_TO_GROUPS[cls] for key in keys), default=0)
        else:
            return 1.0
        return 1.0 - min(1.0, math.log1p(f) / math.log1p(self.ref))

    def analyze(self, text: str) -> list[Decision]:
        spans = select_spans(text, [s for detector in self.detectors for s in detector(text)])
        decisions = []
        for sp in spans:
            val = text[sp.start : sp.end]
            role = span_role(sp)
            rar = self.rarity(sp.label, val)
            if role in ("DIRECT", "SENSITIVE"):
                risk, action, reason = 1.0, "mask", f"{role.lower()}: always redacted"
            elif role == "CONTEXT":
                risk, action, reason = 0.0, "keep", "context: retained"
            else:
                risk = self.prior.get(sp.label, 0.3) * (0.5 + 0.5 * rar)
                gen = generalize_span(sp.label, val)
                if risk >= self.tau_mask or (risk >= self.tau_gen and gen is None):
                    action = "mask"
                elif risk >= self.tau_gen:
                    action = "generalize"
                else:
                    action = "keep"
                reason = f"class prior {self.prior.get(sp.label, 0.3):.2f}, value rarity {rar:.2f} -> risk {risk:.2f}"
            rep = {
                "mask": MASK.get(sp.label, "[REDACTED]"),
                "keep": val,
                "generalize": generalize_span(sp.label, val) or MASK.get(sp.label, "[REDACTED]"),
            }[action]
            decisions.append(Decision(sp, val, risk, rar, action, rep, reason))
        return decisions

    def anonymize(self, text: str) -> tuple[str, list[Decision]]:
        decisions = self.analyze(text)
        out, last = [], 0
        for d in sorted(decisions, key=lambda d: d.span.start):
            out.append(text[last : d.span.start])
            out.append(d.replacement)
            last = d.span.end
        out.append(text[last:])
        return "".join(out), decisions

    @staticmethod
    def report(decisions: list[Decision]):
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "fragment": d.text,
                    "class": d.span.label,
                    "risk": round(d.risk, 2),
                    "rarity": round(d.rarity, 2),
                    "action": d.action,
                    "replacement": d.replacement,
                    "reason": d.reason,
                }
                for d in decisions
            ],
            columns=["fragment", "class", "risk", "rarity", "action", "replacement", "reason"],
        )
