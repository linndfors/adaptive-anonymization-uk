"""Load WorkUA resumes, extract fields, transform text, and resolve person links.

``other_resumes`` provides same-person ground truth for record-linkage attacks,
which the synthetic E3-JSI corpus does not provide. Risk is therefore measured on
WorkUA, while E3-JSI supports entity-detection evaluation. The ``id``,
``creation_date``, and ``other_resumes`` fields never enter rendered text.
"""

from __future__ import annotations

import ast
import io
import json
import math
import re
import tokenize
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .field_typing import settlement_keys

GROUP_LABELS: dict[str, str] = {
    "title": "Бажана посада",
    "age": "Вік",
    "city": "Місто",
    "salary": "Бажана зарплата",
    "employment": "Зайнятість",
    "driver_license": "Водійське посвідчення",
    "veteran": "Статус ветерана",
    "disability": "Інвалідність",
    "exp_position": "Посади",
    "exp_company": "Компанії",
    "exp_dates": "Періоди роботи",
    "exp_city": "Міста роботи",
    "exp_industry": "Галузі",
    "exp_resp": "Обов'язки",
    "languages": "Мови",
    "skills": "Навички",
    "edu_institution": "Заклади освіти",
    "edu_faculty": "Спеціальності",
    "edu_years": "Роки навчання",
    "edu_city": "Міста навчання",
    "add_education": "Додаткова освіта",
    "additional_info": "Додаткова інформація",
}
GROUPS = list(GROUP_LABELS)
SENSITIVE_GROUPS = {"veteran", "disability"}
FREE_TEXT_GROUPS = {"exp_resp", "additional_info"}
MASK_TOKEN = {g: "[" + g.upper() + "]" for g in GROUPS}

# Operations are ordered by strength; adaptive policies only move to the right.
OPS = ("keep", "generalize", "mask", "remove")
OP_STRENGTH = {op: i for i, op in enumerate(OPS)}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
LIST_COLUMNS = ("other_resumes", "work_experiences", "languages", "skills", "educations", "additional_educations")
TEXT_COLUMNS = (
    "title",
    "city",
    "employment_type",
    "work_location_preference",
    "creation_date",
    "disability",
    "additional_info",
)


def _missing(v) -> bool:
    return v is None or v is pd.NA or (isinstance(v, (float, np.floating)) and math.isnan(v))


def _identifier(v) -> str | None:
    value = _s(v)
    # CSV or JSON may represent an integer ID as 123.0; preserve leading zeroes.
    return re.sub(r"^(\d+)\.0+$", r"\1", value) if value else None


def _boolean(v) -> bool:
    if _missing(v) or (isinstance(v, str) and v.strip().lower() in {"", "none", "null", "nan"}):
        return False
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, float, np.number)) and v in (0, 1):
        return bool(v)
    if isinstance(v, str) and v.strip().lower() in {"true", "false", "1", "0"}:
        return v.strip().lower() in {"true", "1"}
    raise ValueError("Expected a Boolean value (True/False or 1/0)")


def load_workua(path: str | Path | None = None, sample: int | None = None, seed: int = 42) -> pd.DataFrame:
    """Load WorkUA from Hugging Face or a supported local tabular format.

    Nested CSV fields may contain JSON or Python literals; ``eval`` is never used.
    Sampling preserves complete person clusters. Cleaning diagnostics are stored
    in ``df.attrs['load_report']``.
    """
    if sample is not None and (isinstance(sample, bool) or not isinstance(sample, (int, np.integer)) or sample <= 0):
        raise ValueError("sample must be a positive integer or None")
    if path is None:
        from datasets import load_dataset

        df = load_dataset("KSE-RESEARCH-Group/Work_UA_resumes", split="train").to_pandas()
    else:
        path = Path(path).expanduser()
        suffix = path.suffix.lower()
        if suffix == ".csv":
            # Disable automatic NA parsing so values such as the skill "NA" remain text.
            df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        elif suffix == ".parquet":
            df = pd.read_parquet(path)
        elif suffix in {".json", ".jsonl"}:
            df = pd.read_json(path, lines=suffix == ".jsonl", dtype={"id": str})
        else:
            raise ValueError(f"Unsupported Work_UA format: {suffix}")
    report = {"source_rows": len(df)}
    index_columns = [c for c in df.columns if not str(c).strip() or str(c).startswith("Unnamed:")]
    df = df.drop(columns=index_columns).copy()
    report["dropped_index_columns"] = len(index_columns)
    missing = {"id", "title", "other_resumes"} - set(df.columns)
    if missing:
        raise ValueError(f"Work_UA is missing required columns: {', '.join(sorted(missing))}")
    df["id"] = df["id"].map(_identifier)
    report["dropped_missing_ids"] = int(df["id"].isna().sum())
    df = df[df["id"].notna()].copy()
    report["dropped_duplicate_ids"] = int(df["id"].duplicated().sum())
    df = df.drop_duplicates("id").reset_index(drop=True)
    if df.empty:
        raise ValueError("Work_UA contains no resumes with a valid id")
    for column in LIST_COLUMNS:
        if column not in df:
            df[column] = [[] for _ in range(len(df))]
        parsed = []
        for row_number, value in enumerate(df[column]):
            try:
                values = _as_list(value, strict=True)
                item_type = str if column == "skills" else dict
                if any(not isinstance(item, item_type) for item in values):
                    raise ValueError("Invalid nested-list item type")
                if column == "other_resumes":
                    values = [dict(item, resume_id=_identifier(item.get("resume_id"))) for item in values]
                parsed.append(values)
            except (ValueError, SyntaxError, TypeError) as error:
                raise ValueError(f"Invalid field {column}, record {row_number + 1}") from error
        df[column] = parsed
    for column in TEXT_COLUMNS:
        df[column] = df[column].map(_s) if column in df else None
    for column in ("age", "desired_salary"):
        values = df[column].map(_s) if column in df else pd.Series(None, index=df.index, dtype=object)
        numeric = pd.to_numeric(values, errors="coerce").astype(float)
        valid = np.isfinite(numeric) & (numeric % 1 == 0)
        valid &= numeric.between(14, 90) if column == "age" else numeric > 0
        report[f"invalid_{column}"] = int((values.notna() & ~valid).sum())
        df[column] = numeric.where(valid, np.nan)
    for column in ("driver_license", "veteran"):
        if column not in df:
            df[column] = False
        try:
            df[column] = df[column].map(_boolean)
        except ValueError as error:
            raise ValueError(f"Invalid Boolean field: {column}") from error
    report["clean_rows"] = len(df)
    if sample is not None and sample < len(df):
        clusters = build_person_clusters(df)
        members = df.groupby(clusters, sort=False).indices
        cluster_ids = np.random.default_rng(seed).permutation(list(members))
        positions = []
        for person in cluster_ids:
            group = members[person]
            if len(positions) + len(group) <= sample:
                positions.extend(group)
            if len(positions) == sample:
                break
        if not positions:
            raise ValueError("sample is too small for a complete identity cluster; increase it")
        df = df.iloc[sorted(positions)].reset_index(drop=True)
    report["sampled_rows"] = len(df)
    df.attrs["load_report"] = report
    return df


def _numpy_list_literal(value: str) -> str:
    """Restore separators in a NumPy array's CSV representation, outside strings.

    Pandas exports ndarray cells as ``[{'a': 1} {'a': 2}]`` or
    ``['Python' 'SQL']``. Tokenization protects punctuation inside quoted text
    and prevents Python's implicit concatenation from merging string items.
    Only the outer list is repaired; malformed nested values still fail.
    """
    tokens = []
    stack = []
    previous = None
    atoms = {tokenize.STRING, tokenize.NUMBER, tokenize.NAME}
    for token in tokenize.generate_tokens(io.StringIO(value).readline):
        if token.type in {tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER}:
            continue
        ends_item = previous is not None and (previous.type in atoms or previous.string in ("]", "}", ")"))
        starts_item = token.type in atoms or token.string in ("[", "{", "(", "+", "-")
        if stack == ["["] and ends_item and starts_item:
            tokens.append((tokenize.OP, ","))
        tokens.append((token.type, token.string))
        if token.type == tokenize.OP:
            if token.string in ("[", "{", "("):
                stack.append(token.string)
            elif token.string in ("]", "}", ")") and stack:
                stack.pop()
        previous = token
    return tokenize.untokenize(tokens)


def _as_list(v, strict: bool = False) -> list:
    if _missing(v):
        return []
    if isinstance(v, str):
        v = v.strip()
        if not v or v.lower() in {"none", "null", "nan"}:
            return []
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            try:
                v = ast.literal_eval(_numpy_list_literal(v))
            except (ValueError, SyntaxError, tokenize.TokenError, IndentationError):
                if strict:
                    raise ValueError("Expected a JSON array or Python list") from None
                return [v]
    if _missing(v):
        return []
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if strict and not isinstance(v, (list, tuple)):
        raise ValueError("Expected a list")
    return list(v) if isinstance(v, (list, tuple)) else [v]


def _s(v) -> str | None:
    if _missing(v):
        return None
    v = str(v).strip()
    return v if v and v.lower() not in {"none", "null", "nan"} else None


def norm(v: str) -> str:
    return re.sub(r"\s+", " ", str(v).lower()).strip(" .;,:-")


# ---------------------------------------------------------------------------
# Same-person resume links using union-find over other_resumes
# ---------------------------------------------------------------------------
def build_person_clusters(df: pd.DataFrame) -> pd.Series:
    ids = set(df["id"])
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for rid, other in zip(df["id"], df["other_resumes"]):
        for o in _as_list(other):
            oid = _identifier(o.get("resume_id")) if isinstance(o, dict) else None
            if oid in ids and oid != rid:
                ra, rb = find(rid), find(oid)
                if ra != rb:
                    parent[ra] = rb
    return df["id"].map(find)


def linkage_report(df: pd.DataFrame, clusters: pd.Series) -> dict:
    referenced = [
        _identifier(o.get("resume_id"))
        for v in df["other_resumes"]
        for o in _as_list(v)
        if isinstance(o, dict) and _identifier(o.get("resume_id")) is not None
    ]
    ids = set(df["id"])
    sizes = clusters.value_counts()
    return {
        "resumes": len(df),
        "resumes_with_other_resumes": int(sum(bool(_as_list(v)) for v in df["other_resumes"])),
        "referenced_ids": len(set(referenced)),
        "referenced_ids_present_in_dataset": len(set(referenced) & ids),
        "persons_with_2plus_resumes": int((sizes >= 2).sum()),
        "resumes_in_linked_clusters": int(sizes[sizes >= 2].sum()),
    }


# ---------------------------------------------------------------------------
# Field extraction
# ---------------------------------------------------------------------------
def extract(rec: dict) -> dict[str, list[str]]:
    exps = [e for e in _as_list(rec.get("work_experiences")) if isinstance(e, dict)]
    edus = [e for e in _as_list(rec.get("educations")) if isinstance(e, dict)]
    out: dict[str, list[str]] = {g: [] for g in GROUPS}
    if _s(rec.get("title")):
        out["title"] = [_s(rec["title"])]
    age = rec.get("age")
    if age is not None and not (isinstance(age, float) and math.isnan(age)) and 14 <= int(age) <= 90:
        out["age"] = [str(int(age))]
    if _s(rec.get("city")):
        out["city"] = [_s(rec["city"])]
    sal = rec.get("desired_salary")
    if sal is not None and not (isinstance(sal, float) and math.isnan(sal)):
        out["salary"] = [f"{int(sal)} грн"]
    out["employment"] = [x for x in (_s(rec.get("employment_type")), _s(rec.get("work_location_preference"))) if x]
    if rec.get("driver_license") is True:
        out["driver_license"] = ["так"]
    if rec.get("veteran") is True:
        out["veteran"] = ["так"]
    if _s(rec.get("disability")):
        out["disability"] = [_s(rec["disability"])]
    out["exp_position"] = [p for p in (_s(e.get("position")) for e in exps) if p]
    out["exp_company"] = [p for p in (_s(e.get("company")) for e in exps) if p]
    out["exp_dates"] = [
        f"{(_s(e.get('start_date')) or '?')[:7]}–{(_s(e.get('end_date')) or 'дотепер')[:7]}"
        for e in exps
        if _s(e.get("start_date"))
    ]
    out["exp_city"] = [p for p in (_s(e.get("city")) for e in exps) if p]
    out["exp_industry"] = [p for p in (_s(e.get("industry")) for e in exps) if p]
    out["exp_resp"] = [p for p in (_s(e.get("responsibilities")) for e in exps) if p]
    out["languages"] = [
        f"{_s(l.get('language'))} ({_s(l.get('level')) or '-'})"
        for l in _as_list(rec.get("languages"))
        if isinstance(l, dict) and _s(l.get("language"))
    ]
    out["skills"] = [s for s in (_s(x) for x in _as_list(rec.get("skills"))) if s]
    out["edu_institution"] = [p for p in (_s(e.get("institution")) for e in edus) if p]
    out["edu_faculty"] = [p for p in (_s(e.get("faculty")) for e in edus) if p]
    out["edu_years"] = [
        f"{e.get('start_year') or '?'}–{e.get('end_year') or '?'}"
        for e in edus
        if e.get("start_year") or e.get("end_year")
    ]
    out["edu_city"] = [p for p in (_s(e.get("city")) for e in edus) if p]
    out["add_education"] = [
        p
        for p in (_s(e.get("institution")) for e in _as_list(rec.get("additional_educations")) if isinstance(e, dict))
        if p
    ]
    if _s(rec.get("additional_info")):
        out["additional_info"] = [_s(rec["additional_info"])]
    # Repeated values within one group add no information.
    return {g: list(dict.fromkeys(v)) for g, v in out.items()}


# ---------------------------------------------------------------------------
# Generalization fitted on a reference population such as the train split
# ---------------------------------------------------------------------------
_FREE_TEXT_PII = [
    (re.compile(r"\[(?:FIRST_NAME|LAST_NAME|FULL_NAME|NAME)\]\s*[А-ЯІЇЄҐA-Z][\w'’\-]+"), "[PERSON]"),
    (re.compile(r"[\w.+\-]+@[\w\-]+\.[\w.\-]+"), "[EMAIL]"),
    (re.compile(r"https?://\S+|www\.\S+|t\.me/\S+"), "[URL]"),
    (re.compile(r"(?<!\w)\+?\d[\d\s()\-]{8,}\d"), "[PHONE]"),
    (re.compile(r"\b(?:19|20)\d{2}\b"), "[YEAR]"),
    (re.compile(r"\b\d{3,}\b"), "[NUMBER]"),
]
_INSTITUTION_TYPES = [
    ("університет", "університет"),
    ("академі", "академія"),
    ("інститут", "інститут"),
    ("коледж", "коледж"),
    ("технікум", "технікум"),
    ("училищ", "училище"),
    ("ліцей", "ліцей"),
    ("птн", "профтехучилище"),
    ("пту", "профтехучилище"),
    ("школ", "школа"),
    ("университет", "університет"),
    ("институт", "інститут"),
    ("техникум", "технікум"),
    ("university", "університет"),
    ("college", "коледж"),
]
_SALARY_BINS = [0, 10_000, 15_000, 20_000, 25_000, 30_000, 40_000, 60_000, 100_000, 10**9]


def scrub_free_text(text: str) -> str:
    for rx, rep in _FREE_TEXT_PII:
        text = rx.sub(rep, text)
    return text


@dataclass
class ResumeCodec:
    """Transform extracted fields into text under a ``{group: operation}`` policy."""

    min_freq: int = 20  # Local suppression threshold for rare values
    freq: dict[str, Counter] = field(default_factory=dict)
    settlements: set[str] = field(default_factory=set)  # From the selected top-level city

    def fit(self, extracted: list[dict]) -> ResumeCodec:
        self.freq = {g: Counter() for g in GROUPS}
        self.settlements = settlement_keys(v for ex in extracted for v in ex.get("city", []))
        for ex in extracted:
            for g, vals in ex.items():
                forms = {norm(v) for v in vals}
                if g in ("title", "exp_position"):  # Frequencies of occupation heads
                    forms.update(norm(self._head(v)) for v in vals)
                    forms.update(norm(v.split()[0]) for v in vals if v.split())
                if g == "languages":  # Language-name frequencies without levels
                    forms.update(norm("lang::" + v.split(" (")[0]) for v in vals)
                self.freq[g].update(forms - {""})
        return self

    # Helpers
    @staticmethod
    def _head(v: str) -> str:
        return re.split(r"[,/;(]", v)[0].strip()

    def _frequent(self, g: str, v: str) -> bool:
        return self.freq[g][norm(v)] >= self.min_freq

    def population_frequency(self, g: str, v: str) -> int:
        return self.freq[g][norm(v)]

    def generalize(self, g: str, vals: list[str]) -> list[str]:
        if not vals:
            return []
        if g == "age":
            lo = int(vals[0]) // 10 * 10
            return [f"{lo}–{lo + 9}"]
        if g == "salary":
            x = int(re.sub(r"\D", "", vals[0]) or 0)
            i = int(np.searchsorted(_SALARY_BINS, x, side="right")) - 1
            lo, hi = _SALARY_BINS[i], _SALARY_BINS[min(i + 1, len(_SALARY_BINS) - 1)]
            return [f"{lo // 1000}–{hi // 1000} тис. грн" if hi < 10**9 else f"понад {lo // 1000} тис. грн"]
        if g == "exp_dates":
            total = 0.0
            for v in vals:
                m = re.match(r"(\d{4})-?(\d{2})?–(\d{4}|дотепер)", v)
                if m:
                    end = 2025 if m.group(3) == "дотепер" else int(m.group(3))
                    total += max(0, end - int(m.group(1)))
            return [f"загальний стаж ≈ {round(total)} років"]
        if g == "edu_years":
            ends = [int(y) for v in vals for y in re.findall(r"(?:19|20)\d{2}", v.split("–")[-1])]
            return [f"закінчення у {max(ends) // 10 * 10}-х"] if ends else []
        if g == "edu_institution":
            out = []
            for v in vals:
                low = v.lower()
                out.append(next((t for k, t in _INSTITUTION_TYPES if k in low), "заклад освіти"))
            return list(dict.fromkeys(out))
        if g == "exp_company":
            n = len(vals)
            return [f"{n} місце роботи" if n == 1 else f"{n} місця роботи" if n < 5 else f"{n} місць роботи"]
        if g in FREE_TEXT_GROUPS:
            return [scrub_free_text(v) for v in vals]
        if g == "disability":
            return ["так"]
        if g in ("driver_license", "veteran", "employment", "exp_industry"):
            return vals  # Low-information fields remain unchanged.
        if g == "languages":
            names = dict.fromkeys(v.split(" (")[0] for v in vals)  # Remove proficiency levels.
            return [n for n in names if self._frequent(g, "lang::" + n)]
        if g in ("title", "exp_position"):
            out = []
            for v in vals:
                for cand in (v, self._head(v), v.split()[0] if v.split() else ""):
                    if cand and self._frequent(g, cand):
                        out.append(cand)
                        break
            return list(dict.fromkeys(out))
        if g in ("city", "exp_city", "edu_city"):
            return list(dict.fromkeys(v if self._frequent(g, v) else "інший населений пункт" for v in vals))
        # Suppress rare skills, faculties, and additional education values.
        return [v for v in vals if self._frequent(g, v)]

    def transform(self, ex: dict, policy: dict[str, str]) -> dict[str, list[str]]:
        out = {}
        for g in GROUPS:
            vals, op = ex.get(g, []), policy.get(g, "keep")
            if op == "keep" or not vals:
                out[g] = vals
            elif op == "generalize":
                out[g] = self.generalize(g, vals)
            elif op == "mask":
                out[g] = [MASK_TOKEN[g]]
            elif op == "remove":
                out[g] = []
            else:
                raise ValueError(op)
        return out

    def render(self, ex: dict, policy: dict[str, str] | None = None, exclude: set[str] = frozenset()) -> str:
        t = self.transform(ex, policy or {})
        parts = [f"{GROUP_LABELS[g]}: {'; '.join(v)}." for g, v in t.items() if v and g not in exclude]
        return " ".join(parts)


def nonempty_groups(ex: dict) -> list[str]:
    return [g for g in GROUPS if ex.get(g)]
