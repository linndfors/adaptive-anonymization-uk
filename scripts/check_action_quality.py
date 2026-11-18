"""Deterministic synthetic challenge diagnostics, separate from real-corpus metrics.

Cases are newly authored compositional checks, not independent human evaluation.
They must not be copied into training and later advertised as an untouched test.
The rules-only detector intentionally exposes missing statistical NER coverage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from anonrisk_uk.category_taxonomy import TAXONOMY
from anonrisk_uk.e3jsi import regex_detect
from anonrisk_uk.resume_adaptive import check_direct_identifiers, render
from anonrisk_uk.resume_parser import parse_resume
from anonrisk_uk.workua import ResumeCodec


@dataclass(frozen=True)
class Challenge:
    name: str
    text: str
    mandatory: tuple = ()
    work_years: tuple = ()
    forbidden: tuple = ()


def challenges():
    """Synthetic author-created cases; no real resume text or external API."""
    return [
        Challenge("role_and_contact", "ПІБ: Марія Яворська\nПрацюю Rust-розробницею з 2021 року. "
                  "Пошта: maria.synthetic@example.com.",
                  (("Марія Яворська", "PERSON"), ("maria.synthetic@example.com", "EMAIL")), ("2021",),
                  ("мови програмування-розробницею",)),
        Challenge("alternative_role", "Працював Java/Scala-розробником у 2018–2022 роках. "
                  "Навички: Java, Scala, PostgreSQL.", work_years=("2018", "2022"),
                  forbidden=("Java/розробником", "мови програмування-розробником")),
        Challenge("bullet_skills", "Навички: Python • TypeScript • SQL • PostgreSQL."),
        Challenge("conjoined_skills", "Володію Python, JavaScript та PostgreSQL. "
                  "Пишу інтеграційні тести з 2020 року.", work_years=("2020",),
                  forbidden=("Володію мови програмування",)),
        Challenge("birth_then_work", "Дата народження: 14.02.1994\nПрацюю інженеркою з 2019 року. "
                  "Телефон: +380 67 234 56 78.",
                  (("14.02.1994", "BIRTH_DATE"), ("+380 67 234 56 78", "PHONE")), ("2019",)),
        Challenge("birth_year_only", "Народилася у 1997 році. Працюю з 2021 року.",
                  (("1997 році", "BIRTH_DATE"),), ("2021",), ("[BIRTH_DATE] році",)),
        Challenge("street_apartment", "Адреса: вул. Сонячна, 27, кв. 8\nНавички: Excel, SQL.",
                  (("вул. Сонячна, 27, кв. 8", "ADDRESS"),)),
        Challenge("avenue_address", "Проживаю: проспект Миру, 42\nПрацюю бухгалтеркою з 2016 року.",
                  (("проспект Миру, 42", "ADDRESS"),), ("2016",)),
        Challenge("english_address", "Address: 54 Willow Street, apt 7\nSkills: Docker, AWS.",
                  (("54 Willow Street, apt 7", "ADDRESS"),)),
        Challenge("contact_inside_skills", "Навички: Python, demo.resume@example.com, SQL.",
                  (("demo.resume@example.com", "EMAIL"),)),
        Challenge("person_inside_skills", "ПІБ: Ірина Верес\nНавички: MS Office, Excel, Outlook Express.",
                  (("Ірина Верес", "PERSON"),)),
        Challenge("company_compound", "З 2023 року працюю у тернопільській IT-компанії «Синтетичні рішення». "
                  "Навички: Go, Redis.", work_years=("2023",),
                  forbidden=("[КОМПАНІЯ]компанії", "[КОМПАНІЯ]-компанії")),
        Challenge("study_work_clauses", "Закінчила університет Дельта, а з 2022 року працюю у VectorLabs. "
                  "Контакт: delta.synthetic@example.com.",
                  (("delta.synthetic@example.com", "EMAIL"),), ("2022",)),
        Challenge("move_between_places", "У 2015 році переїхав з Коломиї до Луцька. "
                  "Працюю в логістиці з 2017 року.", work_years=("2015", "2017")),
        Challenge("retouching_domain", "Навички: Beauty.ретушь, Photoshop, Lightroom. "
                  "З 2020 року редагую студійні фотографії.", work_years=("2020",),
                  forbidden=("б'юті-послуги",)),
        Challenge("ambiguous_express", "Навички: MS Office, Express, Excel. "
                  "Готувала звітність у 2019–2021 роках.", work_years=("2019", "2021"),
                  forbidden=("серверне розроблення та API",)),
        Challenge("office_vs_framework", "Навички: Express.js, Node.js, SQL. "
                  "З 2024 року розробляю серверні застосунки.", work_years=("2024",)),
        Challenge("repeated_contact", "Пошта: repeated.synthetic@example.com\n"
                  "Для рекомендацій: repeated.synthetic@example.com.",
                  (("repeated.synthetic@example.com", "EMAIL"),)),
        Challenge("birth_and_calendar_date", "Народився 19 серпня 1993 року. "
                  "Працюю з 19 серпня 2020 року. Навички: Java, SQL.",
                  (("19 серпня 1993 року", "BIRTH_DATE"),), ("2020",)),
        Challenge("free_prose_name", "Мене звати Олексій Калиновський. Працюю аналітиком з 2022 року. "
                  "Контакт: oleksii.synthetic@example.com.",
                  (("Олексій Калиновський", "PERSON"), ("oleksii.synthetic@example.com", "EMAIL")),
                  ("2022",)),
    ]


def surface_issues(output, case):
    issues = []
    if re.search(r"\b(?:з|із)[ \t]+(?:до|у|в)(?=[ \t.!?])", output, re.IGNORECASE):
        issues.append("empty_location_slots")
    if re.search(r"\[[^]]+\][A-Za-zА-Яа-яІЇЄҐіїєґ]", output):
        issues.append("mask_concatenated_with_word")
    if re.search(r"(?:мови програмування|цифрові навички|робота з базами даних)[-‑–]"
                 r"(?:розроб|інженер|програміст)", output, re.IGNORECASE):
        issues.append("category_inserted_inside_professional_compound")
    for value in case.forbidden:
        if value in output:
            issues.append("forbidden_surface:" + value)
    for spec in TAXONOMY.values():
        for name in {*(c[0] for c in spec["categories"].values()), *spec["domains"].values()}:
            if re.search(re.escape(name) + r"[ \t]*(?:[,;•]|та|і|й|and)[ \t]*" + re.escape(name), output):
                issues.append("adjacent_duplicate_category:" + name)
    return sorted(set(issues))


def model_diagnostics(release, case):
    """Separate selected date/context actions from accidental professional rewrites."""
    output = release.releases["adaptive"].text
    edits = [edit for edit in release.releases["adaptive"].edits if edit["applied"]]
    year_events = []
    for year in case.work_years:
        if year in output:
            continue
        matches = list(re.finditer(re.escape(year), case.text))
        changed = [edit for edit in edits if any(edit["start"] < m.end() and m.start() < edit["end"]
                                               for m in matches)]
        if any(edit["group"] in {"exp_dates", "edu_years", "dates"}
               or edit["label"] in {"DATE_TIME", "EMPLOYMENT_PERIOD", "EDUCATION_PERIOD"} for edit in changed):
            reason = "selected_date_action"
        elif changed and all(edit["group"] in {"additional_info", "exp_resp"} for edit in changed):
            reason = "selected_context_action"
        else:
            reason = "unintended_or_unexplained"
        year_events.append({"year": year, "reason": reason})
    return {"remaining_expected_direct_values": sum(value in output for value, _ in case.mandatory),
            "surface_issues": surface_issues(output, case),
            "remaining_known_direct_patterns": len(check_direct_identifiers(output)["issues"]),
            "changed_fragments": len(edits),
            "mandatory_changed_fragments": sum(e["role"] in {"DIRECT", "SENSITIVE"} for e in edits),
            "optional_changed_fragments": sum(e["role"] not in {"DIRECT", "SENSITIVE"} for e in edits),
            "year_changes": year_events,
            "unintended_year_losses": sum(event["reason"] == "unintended_or_unexplained" for event in year_events),
            "steps": len(release.releases["adaptive"].trace) - 1,
            "live_reference_pool_used": release.diagnostics["live_reference_pool_used"]}


def expected_spans(case):
    for value, label in case.mandatory:
        for match in re.finditer(re.escape(value), case.text):
            yield match.start(), match.end(), label


def run(args):
    estimator, detector = None, regex_detect
    codec = ResumeCodec().fit([{"skills": ["Python", "SQL", "Excel", "Photoshop"]}])
    if args.model:
        from anonrisk_uk.action_resume import ActionResumeRegressor

        with args.model.open("rb") as handle:
            estimator = pickle.load(handle)
        if not isinstance(estimator, ActionResumeRegressor):
            raise ValueError("--model must be a fitted action-value model")
        codec = estimator.codec
    if args.detector == "configured":
        from anonrisk_uk.resume_resources import load_resume_resources

        detector = load_resume_resources(args.config).detector
    records = []
    for case in challenges():
        doc = parse_resume(case.text, codec, detector)
        expected, covered, typed, entities, full_entities = 0, 0, 0, 0, 0
        for start, end, label in expected_spans(case):
            positions = {i for i in range(start, end) if not case.text[i].isspace()}
            found = {i for u in doc.units if u.role in {"DIRECT", "SENSITIVE"}
                     for i in range(u.start, u.end)}
            typed_found = {i for u in doc.units if u.label == label for i in range(u.start, u.end)}
            expected += len(positions)
            covered += len(positions & found)
            typed += len(positions & typed_found)
            entities += 1
            full_entities += positions <= found
        control_outputs = {"mandatory": render(doc, {}),
                           "professional_generalization": render(doc, {"skills": 1, "exp_position": 1}),
                           "quasi_masking": render(doc, {u.group: 3 for u in doc.units if u.role == "QUASI"})}
        row = {"case": case.name, "expected_direct_characters": expected, "covered_direct_characters": covered,
               "correct_type_characters": typed, "expected_entities": entities,
               "fully_covered_entities": full_entities,
               "control_surface_issues": {name: surface_issues(text, case) for name, text in control_outputs.items()},
               "unselected_work_years_preserved": all(y in control_outputs["professional_generalization"]
                                                     for y in case.work_years)}
        if estimator is not None:
            from anonrisk_uk.action_resume import anonymize_action

            resources = SimpleNamespace(config={"adaptive": {}}, risk_estimator=estimator, detector=detector)
            release = anonymize_action(resources, case.text)
            row["model"] = model_diagnostics(release, case)
        records.append(row)
    denominator = sum(r["expected_direct_characters"] for r in records)
    report = {"purpose": "newly authored synthetic diagnostic; not a real-corpus test or human review",
              "case_usage": "synthetic development/challenge checks; some prompted renderer fixes; "
                            "excluded from supervised training; not independent validation",
              "detector": args.detector, "model_supplied": estimator is not None, "cases": len(records),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "implementation_sha256": {
                  str(path.relative_to(Path(__file__).resolve().parents[1])):
                      hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in [Path(__file__).resolve().parents[1] / "src/anonrisk_uk" / name for name in
                               ["resume_parser.py", "resume_context.py", "resume_fields.py", "resume_adaptive.py",
                                "category_taxonomy.py"]]},
              "parser": {"direct_character_recall": sum(r["covered_direct_characters"] for r in records)
                         / denominator if denominator else None,
                         "typed_character_recall": sum(r["correct_type_characters"] for r in records)
                         / denominator if denominator else None,
                         "fully_covered_entities": sum(r["fully_covered_entities"] for r in records),
                         "expected_entities": sum(r["expected_entities"] for r in records)},
              "control_cases_with_surface_issues": sum(any(r["control_surface_issues"].values()) for r in records),
              "unselected_work_year_failures": sum(not r["unselected_work_years_preserved"] for r in records),
              "records": records}
    if estimator is not None:
        report["model_sha256"] = hashlib.sha256(args.model.read_bytes()).hexdigest()
        report["complete_pipeline"] = {
            "cases_with_expected_direct_leaks": sum(r["model"]["remaining_expected_direct_values"] > 0 for r in records),
            "cases_with_surface_issues": sum(bool(r["model"]["surface_issues"]) for r in records),
            "unintended_year_losses": sum(r["model"]["unintended_year_losses"] for r in records),
            "intentional_year_changes": sum(event["reason"] != "unintended_or_unexplained"
                                            for r in records for event in r["model"]["year_changes"]),
            "total_changed_fragments": sum(r["model"]["changed_fragments"] for r in records),
            "mandatory_changed_fragments": sum(r["model"]["mandatory_changed_fragments"] for r in records),
            "optional_changed_fragments": sum(r["model"]["optional_changed_fragments"] for r in records),
            "uses_live_reference_pool": any(r["model"]["live_reference_pool_used"] for r in records),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--detector", choices=["rules", "configured"], default="rules")
    parser.add_argument("--config", type=Path, default=Path("configs/adaptive_resume.json"))
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
