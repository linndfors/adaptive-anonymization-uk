"""One resume, one release, exact before/after highlighting."""

from __future__ import annotations

import html
import json
import threading
from pathlib import Path

import pandas as pd
import streamlit as st
from threadpoolctl import threadpool_limits

from anonrisk_uk.demo_highlights import highlighted_pair
from anonrisk_uk.resume_adaptive import VERSION, DirectIdentifierError, anonymize_resume
from anonrisk_uk.resume_resources import DEFAULT_CONFIG, load_resume_resources, resource_paths

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = {
    "Розробник": (
        "Я Андрій Мельник, розробник програмного забезпечення з Києва.\n"
        "Досвід: Python, JavaScript, TypeScript, SQL, PostgreSQL.\n"
        "Працював у компанії SoftServe у 2020–2024 роках.\n"
        "Розробляв вебсервіси та інтеграції з базами даних.\n"
        "Контакт: andrii@example.com, +380 50 123 45 67."
    ),
    "Адреса й контакти": (
        "ПІБ: Олена Коваль\nПосада: Бухгалтерка\n"
        "Адреса: Львів, вул. Зелена, 15, кв. 8.\n"
        "Навички: Excel, BAS, фінансовий аналіз.\n"
        "Працювала у ПриватБанку з 2018 до 2023 року.\n"
        "Телефон: +380 67 123 45 67. Email: olena@example.com."
    ),
    "Переїзд і робота": (
        "У 2019 році переїхала з Дрогобича до Львова.\n"
        "З січня 2024 року працюю фінансовою аналітикинею у львівській IT-компанії SoftServe."
    ),
    "Незвичні навички": (
        "ПІБ: Марія Дорош\nПосада: Backend engineer\n"
        "Навички: Rust, KiCad, Zephyr RTOS, Docker, аналіз вимог.\n"
        "Розробляла інтеграції та тестувала прототипи.\n"
        "Компанія: Bright Apps\nКонтакт: maria@example.net"
    ),
}
LABELS = {
    "SKILL_HEADER": "Форма переліку навичок",
    "SKILL": "Навичка", "JOB_TITLE": "Посада", "RESPONSIBILITY": "Обов’язки",
    "OCCUPATION_EDU": "Професійний фрагмент", "DATE_TIME": "Дата / час",
    "CONTEXT_TEXT": "Інший текст", "INDUSTRY": "Галузь",
    "PERSON": "Ім’я", "EMAIL": "Email", "PHONE": "Телефон", "CONTACT": "Контакт",
    "ADDRESS": "Адреса", "URL": "Посилання", "EMPLOYER": "Роботодавець",
    "ORGANIZATION": "Організація", "LOCATION": "Місто / місце",
    "EMPLOYMENT_PERIOD": "Період роботи", "EDUCATION_PERIOD": "Період навчання",
    "EDUCATION_INSTITUTION": "Заклад освіти", "ID_NUMBER": "Номер документа",
    "BIRTH_DATE": "Дата народження", "HEALTH": "Дані про здоров’я", "AGE": "Вік", "SALARY": "Зарплата",
}
GROUPS = {
    "city": "Місце проживання", "exp_company": "Роботодавець", "exp_dates": "Період роботи",
    "skills": "Навички", "title": "Бажана посада", "exp_position": "Посади",
    "exp_resp": "Обов’язки", "additional_info": "Інший текст",
    "edu_institution": "Заклад освіти", "edu_years": "Період навчання", "salary": "Зарплата", "age": "Вік",
}


@st.cache_resource(show_spinner=False)
def resources_for(path, version):
    return load_resume_resources(path), threading.Lock()


def pane(markup):
    # st.html avoids Markdown interpreting user newlines, indentation or backticks.
    st.html('<div style="white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.8;'
            'border:1px solid #94a3b8;border-radius:10px;padding:18px;height:385px;overflow:auto">' + markup + "</div>")


def use_example():
    name = st.session_state["resume_example"]
    if name in EXAMPLES:
        st.session_state["resume_input"] = EXAMPLES[name]
        st.session_state.pop("upload_error", None)



def use_file():
    uploaded = st.session_state.get('resume_file')
    if uploaded is None:
        st.session_state.pop('upload_error', None)
        return
    try:
        text = uploaded.getvalue().decode('utf-8-sig')
        if len(text) > 20_000:
            raise ValueError
        st.session_state['resume_input'] = text
        st.session_state.pop('upload_error', None)
    except (UnicodeDecodeError, ValueError):
        st.session_state['upload_error'] = 'Потрібен текстовий файл UTF-8 до 20 000 символів.'


def show_quality():
    paths = [ROOT / 'results/adaptive_resume_v3.3/fresh_report.json',
             ROOT / 'results/adaptive_resume_v3.2/regression_report.json',
             ROOT / 'results/adaptive_resume_v3/final_report.json']
    path = next((p for p in paths if p.exists()), paths[-1])
    if not path.exists():
        st.write('Збережений звіт відсутній.')
        return
    report = json.loads(path.read_text())
    if report['method'] != VERSION:
        st.caption(f"Це звіт попередньої версії. Для поточної {VERSION} повну оцінку ще не повторено.")
    st.caption(f"Оцінка {report['method']}: {report['queries']} резюме, {report['persons']} людей, "
               f"фон {report['pool_size']} резюме.")
    if report.get('purpose') == 'regression':
        st.caption('Повторна перевірка після виправлень на раніше використаному тестовому корпусі; параметри не підбиралися.')
    elif report['method'] != VERSION:
        st.caption('Це попередня повна оцінка v3, до останніх виправлень формування тексту.')
    else:
        st.caption('Тест на нових людях із зафіксованими параметрами; оцінки не використовувалися для підбору моделі.')
    names = {'historical_L2': 'L2', report['selected']: report['method']}
    rows = [{'Метод': names[r['method']], 'BM25 Recall@1 ↓': f"{r['bm25_recall_at_1']:.1%}",
             'Char-TFIDF Recall@1 ↓': f"{r['char_recall_at_1']:.1%}", 'Лексична схожість ↑': f"{r['lexical_retention']:.1%}"}
            for r in report['summary'] if r['method'] in names]
    semantic_path = ROOT / 'results/adaptive_resume_v3.3/content_fresh_report.json'
    if semantic_path.exists():
        semantic = json.loads(semantic_path.read_text())
        if semantic['method'] == report['method'] and semantic['queries'] == report['queries']:
            values = {r['method']: r['professional_cosine'] for r in semantic['summary']}
            for row, name in zip(rows, [r['method'] for r in report['summary'] if r['method'] in names]):
                value = values.get(name)
                row['Професійна семантична схожість ↑'] = f'{value:.3f}' if value is not None else 'Немає оцінки'
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption('Recall@1 — частка успішних зв’язувань з іншим резюме тієї самої людини; менше краще. '
               'Семантична схожість — косинус ембедингів професійних полів, не відсоток збережених фактів. '
               'Це показники тестового набору, а не ймовірність для цього резюме.')


def main():
    st.set_page_config(page_title='Анонімізація резюме', page_icon='🔐', layout='wide')
    if st.session_state.get('text_demo_schema') != 10:
        st.session_state.pop('text_demo_result', None)
        st.session_state['text_demo_schema'] = 10
    st.title('Анонімізація резюме')
    config_path = DEFAULT_CONFIG
    try:
        model_settings, config, paths = resource_paths(config_path)
        if not all(path.is_file() for path in paths.values()):
            raise FileNotFoundError
    except (FileNotFoundError, ValueError, KeyError):
        st.error('Не знайдено локальних ресурсів моделі. Перевірте configs/adaptive_resume.json.')
        return
    with st.expander('Завантажити файл або приклад'):
        upload_column, example_column = st.columns(2)
        with upload_column:
            st.file_uploader('Файл .txt', type=['txt'], key='resume_file', on_change=use_file)
        with example_column:
            st.selectbox('Приклад', ['Власний текст', *EXAMPLES], key='resume_example', on_change=use_example)
        if st.session_state.get('upload_error'):
            st.error(st.session_state['upload_error'])
    left, right = st.columns(2)
    with left:
        st.subheader('Ваше резюме')
        if 'resume_input' not in st.session_state:
            st.session_state['resume_input'] = ''
        text = st.text_area('Текст резюме', key='resume_input', height=405, max_chars=20_000,
                            label_visibility='collapsed', placeholder='Вставте текст резюме…')
        run = st.button('Анонімізувати', type='primary', disabled=not text.strip())
        st.caption('Текст обробляється локально.')
    if run:
        st.session_state.pop('text_demo_result', None)
        try:
            files = [config_path, *paths.values(), ROOT / 'data/taxonomy/category_map.json']
            version = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in files if p.exists())
            with st.spinner('Обробляю резюме…'):
                resources, lock = resources_for(str(config_path), version)
                with lock, threadpool_limits(limits=config.get('cpu_threads', 1)):
                    result = anonymize_resume(resources, text)
            st.session_state['text_demo_result'] = (str(config_path), result)
        except DirectIdentifierError:
            st.error('Результат не пройшов перевірку прямих ідентифікаторів. Його не видано як анонімізований текст.')
        except Exception as exc:  # noqa: BLE001 - do not expose sensitive provider details
            st.error(f'Обробка не завершилась ({type(exc).__name__}). Перевірте локальні ресурси проєкту.')
    saved = st.session_state.get('text_demo_result')
    if not saved:
        with right:
            st.subheader('Результат')
            st.info('Вставте резюме та натисніть «Анонімізувати».')
        return
    path, result = saved
    if (result.original != text or path != str(config_path)
            or result.diagnostics['utility_floor'] != model_settings['adaptive']['utility_floor']):
        with right:
            st.info('Текст змінився. Натисніть «Анонімізувати» повторно.')
        return
    release = result.releases['adaptive']
    changes = [e for e in release.edits if e['applied']]
    metrics = result.diagnostics['metrics']
    try:
        before, after = highlighted_pair(result.original, release.text, release.edits)
    except ValueError:
        before, after = html.escape(result.original), html.escape(release.text)
    with right:
        st.subheader('Результат')
        pane(after)
        st.download_button('Завантажити .txt', release.text, 'resume_draft.txt', 'text/plain')
        st.caption('Зеленим позначено заміни. Це чернетка для перевірки.')
    a, b = st.columns(2)
    risk, utility = metrics['risk_after'], metrics['lexical_retention']
    a.metric('Пошуковий індекс ризику', f'{risk:.3f}' if risk is not None else 'Не оцінено',
             help='Внутрішній показник зв’язування з корпусом, не ймовірність. '
                  'Від’ємне значення не доводить відсутності персональних даних. Менше краще; ціль ≤ 0.')
    b.metric('Лексична схожість', f'{utility:.1%}' if utility is not None else 'Не оцінено',
             help='Проксі utility: схожість слів з оригіналом. Не вимірює граматику чи збереження змісту.')
    status = result.diagnostics.get('stop_reason')
    if status == 'target_unmet':
        st.warning('Поріг пошукового ризику не досягнуто. Перевірте результат перед використанням.')
    elif status == 'unassessed_conservative_fallback':
        st.warning('Ризик не вдалося оцінити. Перевірте результат перед використанням.')
    with st.expander('Заміни, алгоритм і метрики якості'):
        changed_tab, algorithm_tab, quality_tab = st.tabs(['Заміни', 'Алгоритм', 'Якість на тестах'])
        with changed_tab:
            if changes:
                st.dataframe(pd.DataFrame([
                    {'Тип': LABELS.get(e['label'], e['label']), 'Було': e['text'],
                     'Стало': f"Об’єднано: {e['merged_into']}" if e.get('merged_into') else e['replacement']}
                    for e in changes
                ]), hide_index=True)
            else:
                st.write('Заміни відсутні; це не означає відсутності персональних даних.')
            st.caption('Рожевим — змінені фрагменти оригіналу.')
            pane(before)
        with algorithm_tab:
            steps = [step for step in release.trace if 'group' in step]
            if steps:
                st.dataframe(pd.DataFrame([
                    {'Крок': i, 'Поле': GROUPS.get(step['group'], step['group']),
                     'Дія': step.get('operation'), 'BM25': step['proxy_margins'][0],
                     'Char-TFIDF': step['proxy_margins'][1], 'Utility': step['retention']}
                    for i, step in enumerate(steps, 1)
                ]), hide_index=True)
            else:
                st.write('Додаткових адаптивних кроків не обрано.')
            st.dataframe(pd.DataFrame(metrics['risk_by_attacker']), hide_index=True)
            st.caption(f"Модель: {result.diagnostics['method_version']}. "
                       f"Навички: дослівно {metrics['skills_kept']}, узагальнено {metrics['skills_generalized']}, "
                       f"приховано {metrics['skills_masked']}. Пошуковий індекс не є ймовірністю деанонімізації.")
        with quality_tab:
            show_quality()


if __name__ == '__main__':
    main()
