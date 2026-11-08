"""Generate synthetic records that follow the Work_UA_resumes schema.

The generator is intended only for offline pipeline checks. Metrics calculated
from these records are not research results. The data emulate linked resumes,
shared employment and education histories, imbalanced cities, rare companies
and skills, and free text containing residual PII.
"""

from __future__ import annotations

import random

import pandas as pd

CITIES = (
    ["Київ"] * 30
    + ["Львів"] * 10
    + ["Харків"] * 8
    + ["Одеса"] * 8
    + ["Дніпро"] * 8
    + [
        "Запоріжжя",
        "Вінниця",
        "Полтава",
        "Черкаси",
        "Луцьк",
        "Тернопіль",
        "Ужгород",
        "Чернівці",
        "Житомир",
        "Рівне",
        "Суми",
    ]
    * 2
    + ["Жовті Води", "Калуш", "Дрогобич", "Кременчук", "Мукачево", "Бориспіль", "Ковель", "Умань"]
)
PROFS = {
    "Менеджер з продажу": (
        "Оптова торгівля",
        ["Ведення CRM", "Переговори", "Продаж B2B", "Холодні дзвінки", "MS Excel"],
    ),
    "Бухгалтер": ("Консалтинг, бухгалтерія, аудит", ["1С", "M.E.Doc", "Податковий облік", "MS Excel", "Клієнт-Банк"]),
    "Водій": ("Транспорт, логістика", ["Водійські права кат. B", "Знання доріг міста", "Водійські права кат. C"]),
    "Python-розробник": ("IT", ["Python", "Django", "FastAPI", "PostgreSQL", "Docker", "Git", "Redis"]),
    "Продавець-консультант": ("Роздрібна торгівля", ["Комунікабельність", "Контроль цінників", "Робота з касою"]),
    "Адміністратор": ("Послуги для населення та бізнесу", ["Ведення документації", "MS Office", "Комунікабельність"]),
    "Лікар-терапевт": ("Медицина, охорона здоров'я, аптеки", ["Ведення медичної документації", "Діагностика"]),
    "Юрист": ("Юриспруденція", ["Договірна робота", "Судова практика", "Корпоративне право"]),
    "Електрик": ("Енергетика", ["Електромонтаж", "Група допуску", "Ремонт обладнання"]),
    "Маркетолог": ("Маркетинг, реклама, PR", ["SMM", "Google Ads", "Таргетована реклама", "Копірайтинг"]),
}
RARE_SKILLS = [
    "Autodesk Maya",
    "ZBrush",
    "Unreal Engine",
    "QuickBooks",
    "Битрикс24",
    "Rigging",
    "GSAP",
    "Доктор Елекс",
    "SolidWorks",
    "Екслуатація кранів",
    "IFRS",
    "Tableau",
]
UNIS = [
    "Київський національний університет імені Тараса Шевченка",
    "НУ «Львівська політехніка»",
    "Харківський національний економічний університет",
    "Одеський національний університет",
    "Дніпровський технічний коледж",
    "Луцький національний технічний університет",
    "Запорізький медичний фаховий коледж",
    "Національний авіаційний університет",
    "Вінницький торговельно-економічний інститут",
    "СПТУ 33",
    "Технікум харчової промисловості",
]
FACULTIES = [
    "Фінанси та кредит",
    "Комп'ютерні науки",
    "Менеджмент",
    "Право",
    "Лікувальна справа",
    "Маркетинг",
    "Електротехніка",
    "Облік і аудит",
    "Психологія",
    "Філологія",
]
PREFIX = ["Нова", "Галич", "Укр", "Мега", "Тех", "Агро", "Буд", "Фарм", "Лого", "Вест", "Сіті", "Еко"]
SUFFIX = ["трейд", "сервіс", "груп", "лайн", "маркет", "профі", "інвест", "плюс", "буд", "софт"]
NAMES = ["Анна", "Олена", "Ірина", "Марія", "Олег", "Андрій", "Тарас", "Юлія", "Дмитро", "Наталія"]


def _company(rng):
    return (
        f"ТОВ «{rng.choice(PREFIX)}{rng.choice(SUFFIX)}»"
        if rng.random() < 0.85
        else rng.choice(["Нова пошта", "Rozetka", "АТБ", "Епіцентр", "ПриватБанк"])
    )


def _person(rng, pid):
    prof = rng.choice(list(PROFS))
    industry, skills = PROFS[prof]
    city = rng.choice(CITIES)
    age = rng.randint(18, 64)
    n_exp = rng.choice([0, 1, 1, 2, 2, 3])
    year = 2025 - rng.randint(0, 3)
    exps = []
    for _ in range(n_exp):
        dur = rng.randint(1, 6)
        start = year - dur
        exps.append(
            {
                "position": prof if rng.random() < 0.7 else rng.choice(list(PROFS)),
                "start_date": f"{start}-{rng.randint(1, 12):02d}-01",
                "end_date": None if not exps else f"{year}-{rng.randint(1, 12):02d}-01",
                "company": _company(rng),
                "city": city if rng.random() < 0.7 else "",
                "industry": industry,
                "responsibilities": None
                if rng.random() < 0.6
                else f"Робота з клієнтами, звітність, {rng.choice(skills).lower()}",
            }
        )
        year = start
    edus = (
        []
        if rng.random() < 0.35
        else [
            {
                "institution": rng.choice(UNIS),
                "faculty": rng.choice(FACULTIES),
                "city": city if rng.random() < 0.6 else None,
                "level": rng.choice(["Вища", "Середня спеціальна"]),
                "start_year": 2025 - age + 17,
                "end_year": 2025 - age + 21,
            }
        ]
    )
    sk = rng.sample(skills, k=min(len(skills), rng.randint(1, len(skills))))
    if rng.random() < 0.3:
        sk.append(rng.choice(RARE_SKILLS))
    info = None
    if rng.random() < 0.15:
        info = f"[FIRST_NAME] {rng.choice(NAMES)}. Про себе: відповідальна, маю досвід {rng.randint(2, 15)} років."
    return {
        "prof": prof,
        "city": city,
        "age": age if rng.random() < 0.75 else None,
        "exps": exps,
        "edus": edus,
        "skills": sk,
        "info": info,
        "salary": int(rng.choice([8, 12, 15, 20, 25, 30, 35, 40, 60, 80]) * 1000),
        "veteran": rng.random() < 0.08,
        "disability": None if rng.random() > 0.03 else "Третя група",
        "lang": [{"language": "Українська", "level": "рідна"}]
        + (
            [{"language": "Англійська", "level": rng.choice(["середній", "початковий", "вище середнього"])}]
            if rng.random() < 0.5
            else []
        )
        + ([{"language": "Румунська", "level": "середній"}] if rng.random() < 0.02 else []),
    }


def _resume(rng, rid, p, variant: int):
    """Create a resume variant with changed title/skills and shared work history."""
    title = p["prof"] if variant == 0 else rng.choice([p["prof"] + ", помічник", rng.choice(list(PROFS))])
    skills = p["skills"] if variant == 0 else rng.sample(p["skills"], k=max(1, len(p["skills"]) - 1))
    exps = p["exps"] if variant == 0 or rng.random() < 0.8 else p["exps"][:1]
    return {
        "id": str(rid),
        "title": title,
        "age": p["age"],
        "city": p["city"],
        "desired_salary": p["salary"] if variant == 0 else int(p["salary"] * rng.choice([0.8, 1, 1.2])),
        "employment_type": rng.choice([None, "повна", "неповна", "повна, неповна"]),
        "work_location_preference": rng.choice([None, None, "Дистанційно"]),
        "driver_license": rng.random() < 0.35,
        "creation_date": "2025-08-01",
        "other_resumes": [],
        "veteran": p["veteran"],
        "disability": p["disability"],
        "work_experiences": exps,
        "languages": p["lang"],
        "skills": skills,
        "educations": p["edus"],
        "additional_educations": [],
        "additional_info": p["info"],
    }


def make_mock_workua(n_persons: int = 6000, multi_share: float = 0.3, seed: int = 0) -> pd.DataFrame:
    rng = random.Random(seed)
    rows, rid = [], 10_000_000
    for pid in range(n_persons):
        p = _person(rng, pid)
        k = 1 if rng.random() > multi_share else rng.choice([2, 2, 3])
        group = []
        for v in range(k):
            rid += rng.randint(1, 50)
            group.append(_resume(rng, rid, p, v))
        for r in group:
            r["other_resumes"] = [{"title": o["title"], "resume_id": o["id"]} for o in group if o is not r]
        rows.extend(group)
    return pd.DataFrame(rows)
