"""Grounded sentence context and grammatical alternatives for resume phrases."""

import re
from dataclasses import replace

from .adaptive_text import ReleaseDocument, TextUnit

MONTHS = (
    'січня|лютого|березня|квітня|травня|червня|липня|серпня|вересня|жовтня|листопада|грудня|'
    'января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря'
)
YEAR = r'(?:19|20)\d{2}'
FULL_DATE = (rf'(?:\d{{1,2}}[ \t]+(?:{MONTHS})[ \t]+{YEAR}'
             rf'|\d{{1,2}}[./-]\d{{1,2}}[./-]{YEAR}|{YEAR}[-/]\d{{1,2}}[-/]\d{{1,2}})')
BIRTH_PREFIX = (r'\b(?:народився|народилася|народилась|народжений|народжена|дата[ \t]+народження|'
                r'родился|родилась|дата[ \t]+рождения|born(?:[ \t]+on)?|date[ \t]+of[ \t]+birth|DOB)')
BIRTH_DATE = re.compile(
    rf'{BIRTH_PREFIX}[ \t]*(?:[:—-][ \t]*)?(?:[ув][ \t]+)?'
    rf'(?P<date>{FULL_DATE}|{YEAR})(?!\d)(?:[ \t]+(?:року|года)|[ \t]*[рг]\.)?', re.IGNORECASE,
)
BIRTH_DATE_AFTER_PLACE = re.compile(
    rf'{BIRTH_PREFIX}[ \t]+(?:[ув]|в[ \t]+місті)[ \t]+[^\n.!?]{{1,70}}?[ \t,]+'
    rf'(?P<date>{FULL_DATE})(?!\d)(?:[ \t]+(?:року|года))?', re.IGNORECASE,
)
BIRTH_DATE_SUFFIX = re.compile(rf'(?P<date>{FULL_DATE})[ \t]+р\.[ \t]*н\.', re.IGNORECASE)
AGE = re.compile(r'\b(?:мені|мне|age)[ \t]*:?[ \t]*(?P<age>\d{1,2}[ \t]+(?:років|роки|рік|лет|years?))(?!\w)',
                 re.IGNORECASE)
TECHNOLOGIES = (
    'Python', 'JavaScript', 'TypeScript', 'Java', 'C++', 'C#', '.NET', 'Node.js',
    'Rust', 'Go', 'PHP', 'Ruby', 'Kotlin', 'Swift', 'Scala', 'Dart', 'React', 'SQL', 'DevOps',
)
ROLE_FORMS = {
    'розробник': ('розробник програмного забезпечення', 'ІТ-фахівець'),
    'розробником': ('розробником програмного забезпечення', 'ІТ-фахівцем'),
    'розробника': ('розробника програмного забезпечення', 'ІТ-фахівця'),
    'розробнику': ('розробнику програмного забезпечення', 'ІТ-фахівцю'),
    'розробниця': ('розробниця програмного забезпечення', 'ІТ-фахівчиня'),
    'розробницею': ('розробницею програмного забезпечення', 'ІТ-фахівчинею'),
    'розробниці': ('розробниці програмного забезпечення', 'ІТ-фахівчині'),
    'розробницю': ('розробницю програмного забезпечення', 'ІТ-фахівчиню'),
    'програміст': ('розробник програмного забезпечення', 'ІТ-фахівець'),
    'програмістом': ('розробником програмного забезпечення', 'ІТ-фахівцем'),
    'програмістка': ('розробниця програмного забезпечення', 'ІТ-фахівчиня'),
    'програмісткою': ('розробницею програмного забезпечення', 'ІТ-фахівчинею'),
    'інженер': ('інженер програмного забезпечення', 'ІТ-фахівець'),
    'інженером': ('інженером програмного забезпечення', 'ІТ-фахівцем'),
    'інженерка': ('інженерка програмного забезпечення', 'ІТ-фахівчиня'),
    'інженеркою': ('інженеркою програмного забезпечення', 'ІТ-фахівчинею'),
    'developer': ('software developer', 'IT specialist'),
}
TECHNICAL_JOB = re.compile(
    r'(?<![\w+\-‑–])(?:' + '|'.join(re.escape(v) for v in sorted(TECHNOLOGIES, key=len, reverse=True))
    + r')[-‑–](?P<role>' + '|'.join(sorted(ROLE_FORMS, key=len, reverse=True)) + r')(?!\w)', re.IGNORECASE,
)
SENTENCE_END = re.compile(r'[.!?](?=\s|$)|[\r\n]+')
SKILL_STATEMENT = re.compile(
    r'(?i)[ \t]*(?P<header>(?:(?:я|також|добре|вільно|впевнено)[ \t]+){0,3}'
    r'(?:володію|знаю|опанував|опанувала|маю[ \t]+досвід[ \t]+роботи[ \t]+з)[ \t]+)'
)
LIST_SEPARATOR = re.compile(r'[,;]|[ \t]+(?:та|і|й|and|&)[ \t]+', re.IGNORECASE)
WORK = re.compile(r'працю|работ|worked|employ|компан|company', re.IGNORECASE)
STUDY = re.compile(r'навча|навчив|закінчив|закінчила|учил|education|studied', re.IGNORECASE)
SCHOOL = re.compile(r'університет|університеті|академ|коледж|university|college', re.IGNORECASE)


def skill_item_ranges(text, start, end):
    cursor = start
    for separator in [*LIST_SEPARATOR.finditer(text, start, end), None]:
        stop = separator.start() if separator else end
        a, b = cursor, stop
        while a < b and text[a].isspace():
            a += 1
        while b > a and (text[b - 1].isspace() or text[b - 1] == '.'):
            b -= 1
        if a < b:
            yield a, b
        cursor = separator.end() if separator else end


def sentence_ranges(text):
    start = 0
    for boundary in SENTENCE_END.finditer(text):
        if start < boundary.start():
            yield start, boundary.start()
        start = boundary.end()
    if start < len(text):
        yield start, len(text)


def sentence_context(text, start, end):
    left = max((m.end() for m in SENTENCE_END.finditer(text, 0, start)), default=0)
    boundary = SENTENCE_END.search(text, end)
    return text[left:boundary.start() if boundary else len(text)]


def birth_units(text):
    units = []
    for pattern in (BIRTH_DATE, BIRTH_DATE_AFTER_PLACE, BIRTH_DATE_SUFFIX):
        for match in pattern.finditer(text):
            start, end = match.span('date')
            # The suffix is part of the exact source date, not a free-standing word.
            if pattern is not BIRTH_DATE_SUFFIX:
                end = match.end()
            units.append(TextUnit(start, end, 'BIRTH_DATE', 'DIRECT', 'birth_date', 'birth_context'))
    return units


def context_units(text):
    units = birth_units(text)
    for match in AGE.finditer(text):
        units.append(TextUnit(*match.span('age'), 'AGE', 'QUASI', 'age', 'age_context'))
    for match in TECHNICAL_JOB.finditer(text):
        context = sentence_context(text, match.start(), match.end())
        units.append(TextUnit(match.start(), match.end(), 'JOB_TITLE', 'QUASI',
                              'exp_position' if WORK.search(context) else 'title', 'technical_job_phrase'))
    for start, end in sentence_ranges(text):
        statement = SKILL_STATEMENT.match(text[start:end])
        if not statement:
            continue
        body_start = start + statement.end()
        source = f'skill_statement:{start + statement.start("header")}'
        items = [TextUnit(a, b, 'SKILL', 'QUASI', 'skills', source)
                 for a, b in skill_item_ranges(text, body_start, end)]
        if items:
            units.append(TextUnit(start + statement.start('header'), body_start,
                                  'SKILL_HEADER', 'CONTEXT', 'skills_header', source))
            units.extend(items)
    return units


def refine_sentence_types(text, units):
    result = []
    for u in units:
        context = sentence_context(text, u.start, u.end)
        if u.label in {'ORGANIZATION', 'EMPLOYER', 'EDUCATION_INSTITUTION'}:
            if STUDY.search(context) or SCHOOL.search(text[u.start:u.end]) and not WORK.search(context):
                u = replace(u, label='EDUCATION_INSTITUTION', group='edu_institution', source='sentence_context')
            elif WORK.search(context):
                u = replace(u, label='EMPLOYER', group='exp_company', source='sentence_context')
        elif u.label in {'DATE_TIME', 'EMPLOYMENT_PERIOD', 'EDUCATION_PERIOD'}:
            value = text[u.start:u.end]
            if re.search(r'(?:19|20)\d{2}', value):
                if STUDY.search(context):
                    u = replace(u, label='EDUCATION_PERIOD', group='edu_years', source='sentence_context')
                elif WORK.search(context):
                    u = replace(u, label='EMPLOYMENT_PERIOD', group='exp_dates', source='sentence_context')
                else:
                    u = replace(u, label='DATE_TIME', group='dates', source='sentence_context')
        result.append(u)
    return result


def overlay_context(doc):
    """Specific contextual phrases outrank coarse fragments, never real contacts."""
    contextual = context_units(doc.text)
    births = [u for u in contextual if u.label == 'BIRTH_DATE']
    protected = [u for u in doc.units if u.role in {'DIRECT', 'SENSITIVE'}]
    selected = []
    for u in sorted([*births, *protected], key=lambda u: (u.label != 'BIRTH_DATE', -(u.end - u.start))):
        overlaps = [v for v in selected if v.start < u.end and u.start < v.end]
        if not overlaps:
            selected.append(u)
        elif any(u.start < v.start or u.end > v.end for v in overlaps):
            first = overlaps[0]
            selected = [v for v in selected if v not in overlaps]
            selected.append(replace(first, start=min(u.start, *(v.start for v in overlaps)),
                                    end=max(u.end, *(v.end for v in overlaps))))
    # Repair lexical skill fragments inside productive compound words. Their
    # full technical-role counterpart below takes priority when recognized.
    base = []
    word_char = re.compile(r"[\w+#'’ʼ\-‑–]")
    for u in doc.units:
        if u.label == 'SKILL' and (u.start > 0 and doc.text[u.start - 1] in '-‑–'
                                  or u.end < len(doc.text) and doc.text[u.end] in '-‑–'):
            a, b = u.start, u.end
            while a > 0 and word_char.fullmatch(doc.text[a - 1]):
                a -= 1
            while b < len(doc.text) and word_char.fullmatch(doc.text[b]):
                b += 1
            u = replace(u, start=a, end=b)
        base.append(u)
    contextual = [replace(u, group=next((v.group for v in doc.units
                                        if v.label == 'JOB_TITLE' and v.start <= u.start and u.end <= v.end), u.group))
                  if u.source == 'technical_job_phrase' else u for u in contextual]
    candidates = [u for u in contextual if u.label != 'BIRTH_DATE'] + [u for u in base if u not in protected]
    for u in candidates:
        if not any(v.start < u.end and u.start < v.end for v in selected):
            selected.append(u)
    return ReleaseDocument(doc.text, sorted(selected, key=lambda u: u.start))


def job_generalizations(value):
    match = TECHNICAL_JOB.fullmatch(value)
    return ROLE_FORMS[match['role'].casefold()] if match else None


def skill_generalization_allowed(doc, u):
    if u.source.startswith('skill_statement:') or u.source == 'public_field_header':
        return True
    value = doc.text[u.start:u.end]
    if doc.text.strip().strip('. ') == value:
        return True
    line = doc.text[doc.text.rfind('\n', 0, u.start) + 1:u.start]
    if re.match(r'(?i)[ \t]*(?:(?:ключові|інші|технічні)[ \t]+)?(?:навички|технології|досвід|стек|skills|tech stack)[ \t]*:', line):
        return True
    # A bare comma-separated skill list is safe; a governed prose phrase is not.
    if all(v.group == 'skills' for v in doc.units):
        gaps, cursor = [], 0
        for v in doc.units:
            gaps.append(doc.text[cursor:v.start]); cursor = v.end
        gaps.append(doc.text[cursor:])
        return all(re.fullmatch(r'[\s,;.]*', gap) for gap in gaps)
    return False
