"""Utilities for the translated Ukrainian E3-JSI corpus.

The module loads and validates annotations, repairs repeated mentions, creates a
fixed split, exports GLiNER records, and evaluates entity detection from a
privacy perspective. Entity leak rate and character recall are reported because
uncovered personal-data characters matter more than exact type classification.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .taxonomy import GLINER_LABELS, ROLE_OF_CLASS, map_types


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------
def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def validate_offsets(docs: list[dict]) -> list[dict]:
    bad = []
    for d in docs:
        for e in d["entities_uk"]:
            if d["text_uk"][e["start"] : e["end"]] != e["entity"]:
                bad.append({"source_row_id": d["source_row_id"], **e})
    return bad


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    label: str  # Coarse taxonomy class
    role: str = "QUASI"
    source: str = "gold"


def gold_spans(doc: dict, include_context: bool = False) -> list[Span]:
    out = []
    for e in doc["entities_uk"]:
        coarse, role = map_types(e.get("types", []))
        if role == "CONTEXT" and not include_context:
            continue
        out.append(Span(e["start"], e["end"], coarse, role, "auto" if e.get("auto_added") else "gold"))
    return out


# ---------------------------------------------------------------------------
# Repair repeated name mentions lost during placeholder-based translation. Such
# omissions penalise correct detections and create real leaks in gold-based
# anonymization.
# ---------------------------------------------------------------------------
_CAP_TOKEN = re.compile(r"[А-ЯІЇЄҐA-Z][\w'’\-]{2,}")
_HONORIFICS = {"Д-р", "Пан", "Пані", "Містер", "Місіс", "Доктор", "Mr", "Mrs", "Dr"}


def find_unannotated_mentions(doc: dict) -> list[dict]:
    text = doc["text_uk"]
    covered = [(e["start"], e["end"]) for e in doc["entities_uk"]]
    found, seen = [], set()
    for e in doc["entities_uk"]:
        coarse, _ = map_types(e.get("types", []))
        if coarse != "PERSON":
            continue
        for tok in _CAP_TOKEN.findall(e["entity"]):
            if tok in _HONORIFICS:
                continue
            stem = tok[: max(4, len(tok) - 2)]  # Approximate Ukrainian inflection
            for m in re.finditer(r"\b" + re.escape(stem) + r"\w*", text):
                s, t = m.start(), m.end()
                if any(a <= s < b for a, b in covered) or (s, t) in seen:
                    continue
                seen.add((s, t))
                found.append(
                    {
                        "entity": text[s:t],
                        "start": s,
                        "end": t,
                        "types": ["person"],
                        "auto_added": True,
                        "placeholder": None,
                        "source_of_repair": e["entity"],
                    }
                )
    return found


def repair_docs(docs: list[dict]) -> tuple[list[dict], pd.DataFrame]:
    """Return repaired document copies and an audit table of inserted mentions."""
    repaired, log = [], []
    for d in docs:
        add = find_unannotated_mentions(d)
        nd = dict(d)
        nd["entities_uk"] = sorted(d["entities_uk"] + add, key=lambda e: e["start"])
        repaired.append(nd)
        for a in add:
            log.append(
                {
                    "source_row_id": d["source_row_id"],
                    "added": a["entity"],
                    "from": a["source_of_repair"],
                    "start": a["start"],
                }
            )
    return repaired, pd.DataFrame(log)


# ---------------------------------------------------------------------------
# Fixed domain-stratified split
# ---------------------------------------------------------------------------
def make_split(docs: list[dict], seed: int = 42, test: float = 0.2, val: float = 0.1) -> dict:
    from sklearn.model_selection import train_test_split

    ids = [d["source_row_id"] for d in docs]
    dom = [d["domain"] for d in docs]
    tr, te = train_test_split(ids, test_size=test, random_state=seed, stratify=dom)
    dom_tr = [d["domain"] for d in docs if d["source_row_id"] in set(tr)]
    tr, va = train_test_split(tr, test_size=val / (1 - test), random_state=seed, stratify=dom_tr)
    return {**{i: "train" for i in tr}, **{i: "val" for i in va}, **{i: "test" for i in te}}


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------
REGEX_PATTERNS: list[tuple[str, str]] = [
    ("CONTACT", r"[\w.+\-]+@[\w\-]+(?:\.[\w\-]+)+"),
    (
        "CONTACT",
        r"(?<!\w)\+?38[\s\-]?\(?0[\s\-]?\d{2}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}(?!\w)",
    ),  # +380 67 123 45 67
    ("CONTACT", r"(?<!\w)\(?0\d{2}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}(?!\w)"),  # 067 123 45 67
    ("CONTACT", r"https?://\S+|www\.\S+"),
    ("CONTACT", r"@[A-Za-z0-9_]{3,}"),
    ("CONTACT", r"(?<!\w)(?:\+?\d{1,3}[\s\-]?)?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}(?!\w)"),
    ("CONTACT", r"(?<!\w)\(?\d{3}\)?[\s\-]\d{3}[\s\-]\d{4}(?!\w)"),
    ("ID_NUMBER", r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b"),  # IBAN
    ("ID_NUMBER", r"(?<!\d)(?:\d[ \-]?){13,19}(?!\d)"),  # Payment-card number
    ("ID_NUMBER", r"\b\d{3}[\-\.]\d{2,3}[\-\.]\d{3,4}(?:[\-\.]\d{2})?\b"),  # SSN or CPF-like identifier
    ("ID_NUMBER", r"\b[A-ZА-ЯІЇЄ]{1,3}[\-\s]?\d{5,10}\b"),  # Passport or case number
    ("ID_NUMBER", r"(?<![\d.])\d{6,}(?![\d.])"),
    ("DATE_TIME", r"\b\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4}\b"),
    (
        "DATE_TIME",
        (
            r"\b\d{1,2}\s+(?:січня|лютого|березня|квітня|травня|червня|липня|серпня|"
            r"вересня|жовтня|листопада|грудня)(?:\s+\d{4})?(?:\s*р(?:оку|\.)?)?"
        ),
    ),
    ("DATE_TIME", r"\b(?:19|20)\d{2}\s*(?:р(?:оку|\.)|року)\b"),
    ("ADDRESS", r"\b(?:вул\.|вулиця|просп\.|проспект|бульв\.|пл\.)\s*[^,.;]{2,40}?\s*\d+[а-яА-Я]?"),
    ("DEMOGRAPHIC", r"\b\d{1,3}[\s\-]?(?:річн\w*|років|роки|рік)\b"),
    ("FINANCIAL", r"(?:\$|€|£|₴)\s?\d[\d\s,.]*|\d[\d\s,.]*\s?(?:грн|гривень|доларів|євро|USD|EUR|UAH)\b"),
]
_REGEX = [(lab, re.compile(p)) for lab, p in REGEX_PATTERNS]


def regex_detect(text: str) -> list[Span]:
    spans = []
    for lab, rx in _REGEX:
        for m in rx.finditer(text):
            if m.end() - m.start() >= 3:
                spans.append(Span(m.start(), m.end(), lab, source="regex"))
    return resolve_overlaps(spans)


def resolve_overlaps(spans: list[Span]) -> list[Span]:
    """Greedily retain longer spans when detections overlap."""
    out = []
    for s in sorted(spans, key=lambda s: (-(s.end - s.start), s.start)):
        if all(s.end <= o.start or s.start >= o.end for o in out):
            out.append(s)
    return sorted(out, key=lambda s: s.start)


class GlinerDetector:
    """Wrap GLiNER and map predictions to the project's coarse taxonomy.

    ``E3-JSI/gliner-multi-pii-domains-v1`` was trained on the English source
    corpus. Its score on this translation is contaminated and cannot be reported
    as zero-shot evidence.
    """

    def __init__(
        self,
        model_name: str = "urchade/gliner_multi_pii-v1",
        labels: dict[str, str] | None = None,
        threshold: float = 0.4,
        device: str | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
    ):
        from gliner import GLiNER

        if local_files_only:
            # GLiNER 0.2.22 does not forward this flag to its encoder tokenizer.
            from unittest.mock import patch

            with patch("transformers.utils.hub._is_offline_mode", True):
                self.model = GLiNER.from_pretrained(model_name, local_files_only=True, revision=revision)
        else:
            self.model = GLiNER.from_pretrained(model_name, revision=revision)
        if device:
            self.model = self.model.to(device)
        self.labels = labels or GLINER_LABELS
        self.inv = {v: k for k, v in self.labels.items()}
        self.threshold = threshold

    def detect(self, text: str) -> list[Span]:
        from .detectors import text_windows, validate_spans

        spans = []
        for offset, chunk in text_windows(text):
            ents = self.model.predict_entities(chunk, list(self.labels.values()), threshold=self.threshold)
            for e in ents:
                label = self.inv.get(e["label"], "OTHER_QUASI")
                spans.append(
                    Span(offset + e["start"], offset + e["end"], label, ROLE_OF_CLASS.get(label, "QUASI"), "gliner")
                )
        return validate_spans(text, spans)


def combine(*span_lists: list[Span]) -> list[Span]:
    return resolve_overlaps([s for sl in span_lists for s in sl])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def _overlap(a: Span, b: Span) -> int:
    return max(0, min(a.end, b.end) - max(a.start, b.start))


def evaluate_detection(gold: dict, pred: dict, by_label: bool = True) -> dict:
    """Return summary and per-class metrics for ``{doc_id: [Span]}`` mappings.

    Strict matching requires exact boundaries and class. Overlap matching ignores
    class and accepts any intersection. A gold entity leaks when predictions
    cover less than half of its characters.
    """
    st = Counter()
    per = defaultdict(Counter)
    char_gold = char_cov = 0
    for doc_id, G in gold.items():
        P = pred.get(doc_id, [])
        gset = {(g.start, g.end, g.label) for g in G}
        pset = {(p.start, p.end, p.label) for p in P}
        st["strict_tp"] += len(gset & pset)
        st["n_gold"] += len(G)
        st["n_pred"] += len(P)
        st["overlap_tp_pred"] += sum(any(_overlap(p, g) for g in G) for p in P)
        for g in G:
            cov = set()
            for p in P:
                cov.update(range(max(g.start, p.start), min(g.end, p.end)))
            frac = len(cov) / max(1, g.end - g.start)
            char_gold += g.end - g.start
            char_cov += len(cov)
            hit = frac > 0
            leak = frac < 0.5
            st["overlap_tp_gold"] += hit
            st["leaked"] += leak
            per[g.label]["n"] += 1
            per[g.label]["detected"] += hit
            per[g.label]["leaked"] += leak
            per[g.label]["strict"] += (g.start, g.end, g.label) in pset
    f = lambda p, r: 2 * p * r / (p + r) if p + r else 0.0
    sp, sr = st["strict_tp"] / max(1, st["n_pred"]), st["strict_tp"] / max(1, st["n_gold"])
    op, orr = st["overlap_tp_pred"] / max(1, st["n_pred"]), st["overlap_tp_gold"] / max(1, st["n_gold"])
    summary = {
        "strict_P": sp,
        "strict_R": sr,
        "strict_F1": f(sp, sr),
        "overlap_P": op,
        "overlap_R": orr,
        "overlap_F1": f(op, orr),
        "entity_leak_rate": st["leaked"] / max(1, st["n_gold"]),
        "char_recall": char_cov / max(1, char_gold),
        "n_gold": st["n_gold"],
        "n_pred": st["n_pred"],
    }
    table = pd.DataFrame(
        [
            {
                "class": k,
                "n": v["n"],
                "recall_overlap": v["detected"] / v["n"],
                "recall_strict": v["strict"] / v["n"],
                "leak_rate": v["leaked"] / v["n"],
            }
            for k, v in per.items()
        ]
    ).sort_values("n", ascending=False)
    return {"summary": summary, "per_class": table.reset_index(drop=True)}


# ---------------------------------------------------------------------------
# GLiNER fine-tuning export; callers must provide train or validation data only
# ---------------------------------------------------------------------------
def to_gliner_records(docs: list[dict], use_coarse: bool = True) -> list[dict]:
    """Convert documents to GLiNER tokens and inclusive token-span labels."""
    out = []
    for d in docs:
        text = d["text_uk"]
        toks = [(m.start(), m.end()) for m in re.finditer(r"\w+(?:[\-'’]\w+)*|[^\w\s]", text)]
        ner = []
        for sp in gold_spans(d):
            idx = [i for i, (a, b) in enumerate(toks) if a < sp.end and b > sp.start]
            if idx:
                lab = GLINER_LABELS.get(sp.label, sp.label.lower()) if use_coarse else sp.label
                ner.append([idx[0], idx[-1], lab])
        out.append({"source_row_id": d["source_row_id"], "tokenized_text": [text[a:b] for a, b in toks], "ner": ner})
    return out
