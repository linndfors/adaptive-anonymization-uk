"""Larger compositional synthetic parser test; no API or adaptive-search confounding."""

import argparse
import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from evaluate_resume_quality import decode, measure
from threadpoolctl import threadpool_limits

from anonrisk_uk.resume_adaptive import render
from anonrisk_uk.resume_parser import parse_resume
from anonrisk_uk.resume_resources import load_resume_resources

FIRST = ['Олена', 'Тарас', 'Ірина', 'Дмитро', 'Софія', 'Андрій', 'Катерина', 'Юлія', 'Михайло', 'Марія']
LAST = ['Мельник', 'Коваль', 'Дорош', 'Гончар', 'Бойко', 'Левченко', 'Синиця', 'Гнатюк', 'Мороз', 'Козак']
JOBS = [
    ('Розробник', ['Python', 'Rust', 'PostgreSQL', 'Docker'], 'Розробляв сервіси та інтеграції'),
    ('Бухгалтер', ['Excel', 'BAS', 'фінансовий аналіз'], 'Готував звітність та нараховував зарплату'),
    ('Електромонтажник', ['читання схем', 'монтаж щитів', 'мультиметр'], 'Виконував монтаж електромереж'),
    ('Вчитель', ['англійська C1', 'підготовка до НМТ', 'робота з дітьми'], 'Готував навчальні матеріали'),
    ('Кондитер', ['HACCP', 'декорування тортів', 'випікання без глютену'], 'Виконував замовлення клієнтів'),
    ('Інженер', ['KiCad', 'Zephyr RTOS', 'VHDL', 'пайка BGA'], 'Розробляв плати та тестував прототипи'),
    ('Дизайнер', ['Figma', 'типографіка', 'прототипування'], 'Розробляв інтерфейси та макети'),
    ('Логіст', ['SAP', 'планування маршрутів', 'складський облік'], 'Керував відвантаженнями товарів'),
]


def cases(count, seed):
    rng = random.Random(seed)
    for i in range(count):
        name = rng.choice(FIRST) + ' ' + rng.choice(LAST)
        job, skills, responsibility = rng.choice(JOBS)
        skills = rng.sample(skills, len(skills))
        number, flat = rng.randint(1, 250), rng.randint(1, 99)
        street = rng.choice(['Зелена', 'Садова', 'Квіткова', 'Шевченка', 'Сонячна'])
        address = rng.choice([f'вул. {street}, {number}, кв. {flat}',
                              f'вулиця {street} {number}-Б',
                              f'проспект {street} {number}А, квартира {flat}',
                              f'пров. {street}, буд. {number}, кв. {flat}'])
        city = rng.choice(['Київ', 'Львів', 'Одеса', 'Дніпро', 'Полтава', 'Тернопіль'])
        phone = f'+380 {rng.choice([50, 63, 67, 93, 97])} {rng.randint(100,999)} {rng.randint(10,99)} {rng.randint(10,99)}'
        email = f'candidate{i}@example.org'
        skill_text = ', '.join(f'[[SKILL|{s}]]' for s in skills)
        if i % 2:
            annotated = (f'ПІБ: [[PERSON|{name}]]\nПосада: [[JOB_TITLE|{job}]]\n'
                         f'Адреса: [[LOCATION|{city}]], [[ADDRESS|{address}]].\n'
                         f'Навички: {skill_text}.\nОбов’язки: [[RESPONSIBILITY|{responsibility}]].\n'
                         f'Email: [[EMAIL|{email}]]. Телефон: [[PHONE|{phone}]].')
        else:
            annotated = (f'Мене звати [[PERSON|{name}]].\nБажана посада: [[JOB_TITLE|{job}]].\n'
                         f'Проживаю у місті [[LOCATION|{city}]]: [[ADDRESS|{address}]].\n'
                         f'Технології: {skill_text}.\n[[RESPONSIBILITY|{responsibility}]].\n'
                         f'Контакт: [[EMAIL|{email}]], [[PHONE|{phone}]].')
        text, gold = decode(annotated)
        yield {'id': f'synthetic_{i:04d}', 'text': text, 'gold': gold}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=256)
    parser.add_argument('--seed', type=int, default=20261113)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    rows = []
    with threadpool_limits(limits=1):
        resources = load_resume_resources()
        with (args.out / 'cases.jsonl').open('w') as handle:
            for i, case in enumerate(cases(args.count, args.seed)):
                doc = parse_resume(case['text'], resources.codec, resources.detector)
                edits = [{'start': u.start, 'end': u.end, 'text': doc.text[u.start:u.end],
                          'label': u.label, 'role': u.role, 'applied': u.role in {'DIRECT', 'SENSITIVE'}}
                         for u in doc.units]
                release = SimpleNamespace(text=render(doc, {}), edits=edits, retention=None)
                rows.append({'id': case['id'], **measure(case['gold'], release)})
                handle.write(json.dumps(case, ensure_ascii=False) + '\n')
                if i % 32 == 0:
                    print(f'{i + 1}/{args.count}', flush=True)
    frame = pd.DataFrame(rows)
    sums = frame.sum(numeric_only=True)
    report = {'n': len(frame), 'seed': args.seed, 'kind': 'compositional synthetic parser test; not natural-resume gold',
              'direct_char_recall': sums.masked_direct_chars / sums.direct_chars,
              'direct_mask_precision': sums.masked_direct_chars / sums.masked_chars,
              'direct_entity_leak_rate': sums.leaked_direct_entities / sums.direct_entities,
              'exact_type_recall': sums.exact_typed_entities / sums.gold_entities,
              'parser_sha256': hashlib.sha256(Path('src/anonrisk_uk/resume_parser.py').read_bytes()).hexdigest()}
    frame.to_csv(args.out / 'per_case.csv', index=False)
    (args.out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
