"""Taxonomy of identifying features.

E3-JSI contains hundreds of sparse open labels, so evaluation maps them to a
small set of coarse classes and four roles: DIRECT, QUASI, SENSITIVE, and
CONTEXT. Rules are ordered and the first matching rule wins. Export and manually
review the mapping table before a final run.
"""

from __future__ import annotations

import re
from collections import Counter

import pandas as pd

# Tuples contain coarse class, role, and a regex over lower-case open labels.
# Exact exceptions are evaluated before regex rules.
EXACT: dict[str, tuple[str, str]] = {
    "blood type": ("HEALTH", "SENSITIVE"),
    "field of study": ("OCCUPATION_EDU", "QUASI"),
    "identity document": ("OTHER_QUASI", "QUASI"),
    "account information": ("ID_NUMBER", "DIRECT"),
    "account detail": ("ID_NUMBER", "DIRECT"),
    "credit card": ("ID_NUMBER", "DIRECT"),
    "social media": ("CONTACT", "DIRECT"),
    "doctor": ("PERSON", "DIRECT"),
    "financial information": ("FINANCIAL", "QUASI"),
    "transaction": ("FINANCIAL", "QUASI"),
    "reason for visit": ("HEALTH", "SENSITIVE"),
    "academic activity": ("OCCUPATION_EDU", "QUASI"),
    "city name": ("LOCATION", "QUASI"),
    "case name": ("OTHER_QUASI", "QUASI"),
    "event name": ("OTHER_QUASI", "QUASI"),
    "personal information": ("PERSON", "DIRECT"),
    "major": ("OCCUPATION_EDU", "QUASI"),
    "crime": ("OTHER_QUASI", "QUASI"),
}

RULES: list[tuple[str, str, str]] = [
    # Evaluate CONTEXT first so "financial term" does not become FINANCIAL.
    (
        "CONTEXT",
        "CONTEXT",
        (
            r"\b(term|concept|topic|category|attribute|object|item|field|subject|"
            r"requirement|feature|action|process|strategy|context|advice|recommendation|"
            r"type|data type|domain|benefit|system|service|product|platform|market|"
            r"technology|instrument|index|argument|remedy|matter|issue|purpose|concern|"
            r"status|group|people group|class name|application|brand|method|vulnerability)\b"
        ),
    ),
    # Match organization names before the PERSON rule.
    (
        "ORGANIZATION",
        "QUASI",
        (
            r"(organization|bank|company|hospital|department|branch|business|hotel|"
            r"court|venue) name"
        ),
    ),
    # Role words may appear as secondary labels such as patient or employer.
    (
        "PERSON",
        "DIRECT",
        (
            r"^(patient|defendant|plaintiff|client|customer|account holder|student|employee|"
            r"applicant|litigant|recipient|sender|witness|complainant|individual|investor|"
            r"driver|officer|prosecutor|nurse|professor|relative|wife|husband|children|"
            r"testifier|signer|contract party|salutation|cfo|financial advisor|financial analyst|"
            r"certified financial planner|transaction initiator|transaction recipient|"
            r"law enforcement official)$"
        ),
    ),
    (
        "ORGANIZATION",
        "QUASI",
        (
            r"^(clinic|employer|former employer|corporation|workplace|office|school|"
            r"government office|law enforcement agency|volunteering place)$"
        ),
    ),
    ("CONTACT", "DIRECT", r"^landline$"),
    ("HEALTH", "SENSITIVE", r"^(surgery|imaging|biomarker|prescription drug|scan|test)$"),
    ("LOCATION", "QUASI", r"^(residence|area|mountains|tourist attraction)$"),
    ("DATE_TIME", "QUASI", r"^(month|temporal expression|deadline)$"),
    # Direct identifiers
    (
        "CONTACT",
        "DIRECT",
        (
            r"(e-?mail|phone|fax|contact number|mobile|username|social media handle|"
            r"ip address|contact information|contact$|url|website)"
        ),
    ),
    (
        "ID_NUMBER",
        "DIRECT",
        (
            r"(number|\bid\b|identifier|iban|cpf|cnpj|cvv|passport|license plate|"
            r"registration|serial|card last|account ending|policy number|case id|"
            r"transaction id|patient id|medical record)"
        ),
    ),
    ("ADDRESS", "DIRECT", r"(address|street|postal code|zip ?code|zipcode)"),
    (
        "PERSON",
        "DIRECT",
        (
            r"(person|name$|^name|patient name|doctor name|physician|attorney|lawyer|judge|"
            r"agent name|advisor name|personal name|family member|dependent|relation|"
            r"honorific|bank employee)"
        ),
    ),
    # Sensitive attributes
    (
        "HEALTH",
        "SENSITIVE",
        (
            r"(medical|disease|condition|diagnos|symptom|medication|treatment|"
            r"blood|health|dosage|procedure|test result|dietary|patient age)"
        ),
    ),
    # Quasi-identifiers
    ("DEMOGRAPHIC", "QUASI", r"(\bage\b|gender|nationality|birthplace|marital|hobby)"),
    ("DATE_TIME", "QUASI", r"(date|time|year|duration|period|days|span|range|expir)"),
    (
        "LOCATION",
        "QUASI",
        (
            r"(location|city|country|state|airport|landmark|destination|park|"
            r"branch location|venue|region|birthplace)"
        ),
    ),
    (
        "FINANCIAL",
        "QUASI",
        (
            r"(amount|money|salary|income|\bcost\b|balance|\blimit\b|\bvalue\b|\brate\b|\bscore\b|"
            r"percentage|currency|debt|payment|account type|credit card brand|"
            r"credit card type|loan|investment|insurance type|insurance plan)"
        ),
    ),
    (
        "OCCUPATION_EDU",
        "QUASI",
        (
            r"(job|occupation|profession|position|title|role|specializ|specialty|"
            r"degree|qualification|field of study|discipline|education|gpa|skill|"
            r"certification|experience|professional|industry|sector)"
        ),
    ),
    (
        "ORGANIZATION",
        "QUASI",
        (
            r"(organization|company|bank|institution|hospital|court|insurance|"
            r"university|facility|firm|provider|department|division|branch|"
            r"business|merchant|airline|hotel|issuer|authority|\bentity\b|law firm)"
        ),
    ),
    (
        "OTHER_QUASI",
        "QUASI",
        (
            r"(vehicle|car |car$|flight|ticket|reservation|event|incident|device|"
            r"equipment|invention|document|violation|legal|case|charge|criminal)"
        ),
    ),
]
_COMPILED = [(c, r, re.compile(p)) for c, r, p in RULES]

COARSE_CLASSES = sorted({c for c, _, _ in RULES})
ROLE_OF_CLASS = {c: r for c, r, _ in RULES}

# Natural-language class descriptions used by GLiNER
GLINER_LABELS = {
    "PERSON": "person name",
    "CONTACT": "phone number or email or username",
    "ID_NUMBER": "identification or account number",
    "ADDRESS": "street address",
    "HEALTH": "medical condition or treatment",
    "DEMOGRAPHIC": "age or gender or nationality",
    "DATE_TIME": "date",
    "LOCATION": "location",
    "FINANCIAL": "monetary amount",
    "OCCUPATION_EDU": "job title or education",
    "ORGANIZATION": "organization",
    "OTHER_QUASI": "vehicle or ticket or case identifier",
}


def map_label(label: str) -> tuple[str, str]:
    """Map one open label to a coarse class and role."""
    lab = label.lower().strip()
    if lab in EXACT:
        return EXACT[lab]
    for coarse, role, rx in _COMPILED:
        if rx.search(lab):
            return coarse, role
    return "UNMAPPED", "QUASI"  # Conservatively treat unknown labels as quasi-identifiers.


def map_types(types: list[str]) -> tuple[str, str]:
    """Map a multi-label E3-JSI entity using the most protective role."""
    order = {"DIRECT": 0, "SENSITIVE": 1, "QUASI": 2, "CONTEXT": 3}
    mapped = [map_label(t) for t in types]
    # A person-role word does not turn an explicitly labeled organization into a person.
    if any(m[0] == "ORGANIZATION" for m in mapped):
        role_rx = next(rx for c, r, rx in _COMPILED if c == "PERSON" and rx.pattern.startswith("^("))
        mapped = [
            m for t, m in zip(types, mapped) if not (m[0] == "PERSON" and role_rx.search(t.lower().strip()))
        ] or mapped
    known = [m for m in mapped if m[0] != "UNMAPPED"]
    mapped = known or mapped or [("UNMAPPED", "QUASI")]  # Use unmapped only when no known label exists.
    mapped.sort(key=lambda cr: (order[cr[1]], cr[0] == "UNMAPPED"))
    return mapped[0]


def export_mapping(labels: Counter) -> pd.DataFrame:
    rows = [
        {"label": l, "count": n, "coarse": map_label(l)[0], "role": map_label(l)[1]} for l, n in labels.most_common()
    ]
    return pd.DataFrame(rows)
