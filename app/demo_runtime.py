"""UI policies and optional retrieval assessment, independent of model decisions."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from anonrisk_uk.action_learning import lexical_retention
from anonrisk_uk.attack import BM25Index, TfidfIndex, ranking_result
from anonrisk_uk.content_provenance import remove_mask_labels
from anonrisk_uk.entity_demo import DemoRelease
from anonrisk_uk.resume_adaptive import (
    DirectIdentifierError,
    anonymize_resume,
    check_direct_identifiers,
    options,
    planned_edits,
    render,
)
from anonrisk_uk.resume_parser import HEADERS, parse_resume
from anonrisk_uk.workua import GROUP_LABELS, LIST_COLUMNS, ResumeCodec, _as_list, extract


@dataclass(frozen=True)
class Method:
    key: str
    title: str
    description: str
    level: int | None


METHODS = (
    Method('adaptive', 'Адаптивна модель', 'Навчений регресор обирає зміни з урахуванням втрати інформації.', None),
    Method('direct', 'Лише прямі дані (L1)', 'Приховує виявлені прямі ідентифікатори та чутливі дані.', 0),
    Method('generalize', 'Узагальнення (L2)',
           'Застосовує перший рівень узагальнення до всіх доступних квазіідентифікаторів.', 1),
    Method('strong', 'Сильне узагальнення (L3)', 'Застосовує ширші категорії; деталі без узагальнення приховує.', 2),
    Method('mask', 'Маскування деталей', 'Приховує всі виявлені квазіідентифікатори.', 3),
)
METHOD_BY_KEY = {method.key: method for method in METHODS}
MAX_CORPUS_BYTES = 50 * 1024 * 1024
MAX_RECORDS = 6000
MAX_RECORD_CHARS = 20_000
MAX_CORPUS_CHARS = 20_000_000


@dataclass
class MethodResult:
    original: str
    method: str
    release: DemoRelease


def run_method(resources, text, method):
    spec = METHOD_BY_KEY[method]
    if spec.level is None:
        result = anonymize_resume(resources, text)
        return MethodResult(text, method, result.releases['adaptive'])
    doc = parse_resume(text, resources.codec, resources.detector)
    policy = {}
    for i, unit in enumerate(doc.units):
        if unit.role != 'QUASI' or spec.level == 0:
            continue
        choices = options(doc, unit)
        level = spec.level
        if method == 'strong' and choices[min(2, len(choices) - 1)] == text[unit.start:unit.end]:
            level = 3
        policy[f'@{i}'] = min(level, len(choices) - 1)
    output = render(doc, policy)
    if check_direct_identifiers(output)['issues']:
        raise DirectIdentifierError('Known direct identifiers remain')
    entities = [{'unit': i, 'text': text[unit.start:unit.end], **asdict(unit)}
                for i, unit in enumerate(doc.units)]
    edits = [{**entity, **edit, 'decision_source': 'fixed_policy'}
             for entity, edit in zip(entities, planned_edits(doc, policy))]
    retention = float(lexical_retention(resources.retention_vectorizer, text, [output])[0])
    return MethodResult(text, method, DemoRelease(output, edits, retention, []))


def normalize_text(text):
    return ' '.join(unicodedata.normalize('NFC', text).casefold().split())


_SEARCH_FIELDS = (
    *GROUP_LABELS.values(), 'посада', 'компанія', 'роботодавець', 'навички', 'технології',
    'досвід', 'досвід роботи', 'період роботи', 'період навчання', 'попередній досвід',
    'місце роботи', 'освіта', 'спеціальність', 'контакти', 'адреса', 'телефон',
    'email', 'e-mail', 'phone', 'contact', 'contacts', 'skills', 'experience',
    'education', 'job title', 'position', 'address',
)
_SEARCH_FIELD_PATTERN = re.compile(
    r'(?<!\w)(?:' + '|'.join([*(re.escape(value) for value in _SEARCH_FIELDS), *HEADERS.values()])
    + r')[ \t]*:[ \t]*', re.IGNORECASE)
_SEARCH_SECTION_PATTERN = re.compile(
    r'^[ \t]*(?:про мене|досвід роботи|навички|проєкти|проекти|освіта|контакти|'
    r'додаткова інформація|about me|work experience|skills|projects|education|contacts)[ \t]*:?[ \t]*$',
    re.IGNORECASE | re.MULTILINE)


def search_text(text):
    """Use content, not field headings or empty formatting slots, for UI attacks.

    Apply the same public, deterministic rules to the pool and both query states.
    Entity values and narrative remain searchable; mask labels and headings do not.
    """
    content = _SEARCH_SECTION_PATTERN.sub(' ', remove_mask_labels(text))
    content = _SEARCH_FIELD_PATTERN.sub(' ', content)
    tokens = [token.rstrip('./-') for token in re.findall(r"(?u)\.?[\w][\w'’ʼ+#.\-/]*", content)]
    return normalize_text(' '.join(tokens))


@dataclass(frozen=True)
class CorpusRecord:
    key: str
    label: str
    text: str
    person: str = ''


@dataclass(frozen=True)
class Corpus:
    records: tuple[CorpusRecord, ...]
    fingerprint: str
    duplicates: int = 0
    skipped: int = 0
    limited: bool = False


def add_demo_target(corpus, key, text):
    """Add one explicitly synthetic S2 target to a session-only corpus view."""
    target = f'demo:{key}'
    fingerprint = hashlib.sha256(json.dumps([corpus.fingerprint, target, text], ensure_ascii=False).encode()).hexdigest()
    record = CorpusRecord(target, 'Пов’язаний демонстраційний профіль', text, target)
    return Corpus((*corpus.records, record), fingerprint, corpus.duplicates, corpus.skipped, corpus.limited), target


def upload_digest(files):
    digest = hashlib.sha256()
    total = 0
    for name, payload in files:
        total += len(payload)
        if total > MAX_CORPUS_BYTES:
            raise ValueError('Загальний розмір корпусу має бути не більше 50 МБ.')
        digest.update(json.dumps([name, len(payload)], ensure_ascii=False).encode())
        digest.update(payload)
    return digest.hexdigest()


def _string(value):
    if isinstance(value, float) and not math.isfinite(value):
        return ''
    return str(value).strip() if isinstance(value, (str, int, float)) and not isinstance(value, bool) else ''


def _record_text(row):
    lowered = {str(key).strip().casefold(): value for key, value in row.items() if key is not None}
    for key in ('text', 'resume_text', 'content', 'resume', 'текст', 'резюме'):
        if key in lowered:
            value = lowered[key]
            if not isinstance(value, str):
                raise ValueError('Поле тексту має містити текстовий рядок.')
            return value
    if 'title' in row and any(key in row for key in ('work_experiences', 'educations', 'skills')):
        value = dict(row)
        try:
            for key in LIST_COLUMNS:
                if key in value:
                    value[key] = _as_list(value[key], strict=True)
            for key in ('age', 'desired_salary'):
                value[key] = int(float(value[key])) if _string(value.get(key)) else None
            for key in ('driver_license', 'veteran'):
                value[key] = value.get(key) is True or _string(value.get(key)).lower() == 'true'
            return ResumeCodec().render(extract(value), {})
        except (ValueError, TypeError) as exc:
            raise ValueError('Некоректні числові або спискові поля Work.ua.') from exc
    raise ValueError('У CSV/JSON потрібне поле text або структуровані поля Work.ua.')


def _rows(name, payload):
    try:
        text = payload.decode('utf-8-sig')
    except UnicodeDecodeError as exc:
        raise ValueError('Корпус має бути в кодуванні UTF-8.') from exc
    suffix = Path(name).suffix.lower()
    if suffix == '.txt':
        yield {'text': text, 'id': Path(name).stem}
    elif suffix == '.csv':
        try:
            dialect = csv.Sniffer().sniff(text[:8192], delimiters=',;\t')
        except csv.Error:
            dialect = csv.excel
        yield from csv.DictReader(io.StringIO(text), dialect=dialect)
    elif suffix in {'.json', '.jsonl'}:
        if suffix == '.json':
            value = json.loads(text)
            if isinstance(value, dict):
                value = value.get('records', [value])
            if not isinstance(value, list):
                raise ValueError('JSON має містити список записів.')
            yield from value
        else:
            for line in text.splitlines():
                if line.strip():
                    yield json.loads(line)
    else:
        raise ValueError('Підтримуються CSV, JSON, JSONL і TXT.')


def parse_corpus(files):
    """Read text records only; uploaded files never execute code or write to disk."""
    fingerprint = upload_digest(files)
    records, seen = [], set()
    duplicates = skipped = characters = 0
    limited = False
    try:
        for file_index, (name, payload) in enumerate(files):
            for row_index, row in enumerate(_rows(name, payload), 1):
                if not isinstance(row, dict):
                    raise TypeError('Кожен запис корпусу має бути об’єктом із полем text.')
                row = {str(key).strip().casefold(): value for key, value in row.items() if key is not None}
                text = _record_text(row).strip()
                normalized = normalize_text(text)
                person = _string(row.get('person_id', row.get('person')))
                duplicate_key = (normalized, person)
                if not normalized:
                    skipped += 1
                    continue
                if duplicate_key in seen:
                    duplicates += 1
                    continue
                if len(text) > MAX_RECORD_CHARS:
                    raise ValueError('Окреме резюме в корпусі перевищує 20 000 символів.')
                if len(records) == MAX_RECORDS:
                    limited = True
                    break
                characters += len(text)
                if characters > MAX_CORPUS_CHARS:
                    raise ValueError('Забагато тексту: зменште корпус до 20 мільйонів символів.')
                seen.add(duplicate_key)
                identifier = _string(row.get('id', row.get('resume_id'))) or str(row_index)
                label = Path(name).name if Path(name).suffix.lower() == '.txt' else f'{Path(name).name} · {identifier}'
                records.append(CorpusRecord(f'{file_index}:{row_index}', label[:180], text, person))
            if limited:
                break
    except (json.JSONDecodeError, csv.Error, OverflowError, TypeError) as exc:
        raise ValueError('Не вдалося прочитати корпус. Перевірте формат файлу.') from exc
    if not records:
        raise ValueError('У корпусі немає непорожніх резюме.')
    return Corpus(tuple(records), fingerprint, duplicates, skipped, limited)


@dataclass(frozen=True)
class CorpusAssessment:
    kind: str
    before: float | None
    after: float | None
    rows: tuple[dict, ...] = ()
    excluded: int = 0
    candidates: int = 0
    note: str = ''


class CorpusIndex:
    """Session-owned, fixed retrieval indexes; never passed to the action model."""

    def __init__(self, corpus, original=None):
        self.corpus = corpus
        self.original = original
        texts = [search_text(record.text) for record in corpus.records]
        self.normalized = tuple(normalize_text(remove_mask_labels(record.text)) for record in corpus.records)
        self.excluded = 0
        self.target_position = None
        if original is not None:
            query = remove_mask_labels(original)
            normalized = normalize_text(query)
            self.excluded = self.normalized.count(normalized)
            texts = [text for text, value in zip(texts, self.normalized) if value != normalized]
            self.target_position = len(texts)
            texts.append(search_text(original))
        try:
            self.indexes = {'BM25': BM25Index(max_features=30_000).fit(texts),
                            'Символьний TF-IDF': TfidfIndex(min_df=1, max_features=60_000).fit(texts)}
        except ValueError as exc:
            raise ValueError('У корпусі немає достатніх текстових ознак для пошуку.') from exc

    def self_assessment(self, original, output):
        if self.original != original:
            return CorpusIndex(self.corpus, original).self_assessment(original, output)
        count = len(self.corpus.records) - self.excluded + 1
        if count < 2:
            return CorpusAssessment('unassessed', None, None, excluded=self.excluded, candidates=count,
                                    note='Для пошукової атаки потрібне хоча б одне інше резюме.')
        return self.ranked_assessment(original, output, [self.target_position], [], 'self', self.excluded, count)

    def ranked_assessment(self, original, output, targets, excluded, kind, duplicates, count):
        texts = [search_text(text) for text in (original, output)]
        if self.indexes['Символьний TF-IDF'].scores(texts[:1]).max() <= 0:
            return CorpusAssessment('unassessed', None, None, excluded=duplicates, candidates=count,
                                    note='Вхідний текст не має пошукових ознак у цьому індексі.')
        rows = []
        for name, index in self.indexes.items():
            ranked, margins, best_scores = ranking_result(index, texts, [targets, targets], exclude=[excluded, excluded])
            competitor = np.maximum(0., best_scores - margins)
            gap = margins / (np.abs(best_scores) + np.abs(competitor) + 1e-9)
            rows.append({'Атака': name, 'Ранг до': float(ranked.ranks[0]), 'Ранг після': float(ranked.ranks[1]),
                         'RR до': float(ranked.reciprocal_ranks[0]), 'RR після': float(ranked.reciprocal_ranks[1]),
                         'Recall@1 до': float(ranked.recalls[1][0]),
                         'Recall@1 після': float(ranked.recalls[1][1]),
                         'Бал цілі до': float(best_scores[0]), 'Бал цілі після': float(best_scores[1]),
                         'Бал конкурента до': float(competitor[0]), 'Бал конкурента після': float(competitor[1]),
                         'Відрив до': float(gap[0]), 'Відрив після': float(gap[1])})
        return CorpusAssessment(kind, max(row['RR до'] for row in rows),
                                max(row['RR після'] for row in rows), tuple(rows), duplicates, count)

    def assess(self, original, output, target_key=None):
        if target_key is None:
            return self.self_assessment(original, output)
        if self.original is not None:
            return CorpusIndex(self.corpus).assess(original, output, target_key)
        query = normalize_text(remove_mask_labels(original))
        excluded = np.array([i for i, text in enumerate(self.normalized) if text == query], dtype=int)
        count = len(self.corpus.records) - len(excluded)
        if count < 2:
            return CorpusAssessment('unassessed', None, None, excluded=len(excluded), candidates=count,
                                    note='Потрібно щонайменше два записи, відмінні від оригіналу.')
        selected = next((record for record in self.corpus.records if record.key == target_key), None)
        if selected is None:
            raise ValueError('Пов’язаний запис більше не належить поточному корпусу.')
        targets = [i for i, record in enumerate(self.corpus.records)
                   if (record.person == selected.person if selected.person else record.key == target_key)
                   and i not in excluded]
        if not targets:
            return CorpusAssessment('unassessed', None, None, excluded=len(excluded), candidates=count,
                                    note='Обраний запис є копією оригіналу. Оберіть інше резюме цієї людини.')
        if len(targets) == count:
            return CorpusAssessment('unassessed', None, None, excluded=len(excluded), candidates=count,
                                    note='Для оцінювання потрібні також резюме інших людей.')
        return self.ranked_assessment(original, output, targets, excluded, 'linked', len(excluded), count)
