"""One resume, one release, exact before/after highlighting."""

from __future__ import annotations

import html
import threading
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from threadpoolctl import threadpool_limits

from anonrisk_uk.demo_highlights import highlighted_pair
from anonrisk_uk.resume_adaptive import DirectIdentifierError, anonymize_resume, options, render
from anonrisk_uk.resume_parser import parse_resume
from anonrisk_uk.resume_resources import DEFAULT_CONFIG, load_resume_resources, resource_paths

try:
    from app.demo_examples import DEFAULT_EXAMPLE, DEMO_EXAMPLES, EXAMPLE_BY_KEY
except ModuleNotFoundError:
    from demo_examples import DEFAULT_EXAMPLE, DEMO_EXAMPLES, EXAMPLE_BY_KEY

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = {example.title: example.text for example in DEMO_EXAMPLES}
SHIELD = '<span class="demo-shield" aria-hidden="true">✓</span>'
LABELS = {
    "SKILL_HEADER": "Форма переліку навичок",
    "SKILL": "Навичка", "JOB_TITLE": "Посада", "RESPONSIBILITY": "Обов’язки",
    "CONTEXT_TEXT": "Текстовий фрагмент", "EXPERIENCE_DURATION": "Тривалість досвіду",
    "OCCUPATION_EDU": "Професійний фрагмент", "DATE_TIME": "Дата / час",
    "INDUSTRY": "Галузь",
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
    st.html('<div class="demo-output">' + markup + '</div>')


def use_example():
    example = EXAMPLE_BY_KEY.get(st.session_state.get('resume_example'))
    if example:
        st.session_state['resume_input'] = example.text
        st.session_state.pop('text_demo_result', None)
        st.session_state.pop("upload_error", None)


def input_changed():
    st.session_state['resume_example'] = None
    st.session_state.pop('text_demo_result', None)


def clear_input():
    st.session_state['resume_input'] = ''
    st.session_state['resume_example'] = None
    st.session_state['upload_generation'] = st.session_state.get('upload_generation', 0) + 1
    st.session_state.pop('text_demo_result', None)
    st.session_state.pop('upload_error', None)


def use_file(key='resume_file'):
    uploaded = st.session_state.get(key)
    if uploaded is None:
        st.session_state.pop('upload_error', None)
        return
    try:
        text = uploaded.getvalue().decode('utf-8-sig')
        if len(text) > 20_000:
            raise ValueError
        st.session_state['resume_input'] = text
        st.session_state['resume_example'] = None
        st.session_state.pop('text_demo_result', None)
        st.session_state.pop('upload_error', None)
    except (UnicodeDecodeError, ValueError):
        st.session_state['upload_error'] = 'Потрібен текстовий файл UTF-8 до 20 000 символів.'


def panel_heading(number, title):
    st.html(f'<div class="demo-panel-heading"><span class="demo-panel-number">{number}</span>'
            f'<h2>{html.escape(title)}</h2></div>')


def initialize_state():
    if st.session_state.get('text_demo_schema') != 13:
        st.session_state.pop('text_demo_result', None)
        st.session_state['text_demo_schema'] = 13
    if 'resume_input' not in st.session_state:
        st.session_state['resume_input'] = DEFAULT_EXAMPLE.text
        st.session_state['resume_example'] = DEFAULT_EXAMPLE.key
    elif st.session_state.get('resume_example') not in EXAMPLE_BY_KEY:
        st.session_state['resume_example'] = None


def empty_result():
    st.html('<div class="demo-empty"><div><div class="demo-empty-icon">' + SHIELD + '</div>'
            '<h3>Тут буде ваше резюме</h3>'
            '<p>Натисніть «Анонімізувати». Заміни з’являться на зеленому тлі, '
            'а решту тексту можна буде одразу перечитати.</p></div></div>')


def result_is_current(saved, text, version, floor):
    return (saved is not None and saved['result'].original == text and saved['version'] == version
            and saved['result'].diagnostics['utility_floor'] == floor)


def show_details(saved, before, changes):
    result = saved['result']
    release = result.releases['adaptive']
    with st.expander(f'Що змінилося · {len(changes)} фрагментів'):
        if changes:
            st.dataframe(pd.DataFrame([
                {'Що': LABELS.get(edit['label'], edit['label']), 'Було': edit['text'],
                 'Стало': f"Об’єднано: {edit['merged_into']}" if edit.get('merged_into') else edit['replacement']}
                for edit in changes
            ]), hide_index=True, height=min(420, 38 * (len(changes) + 1)))
        else:
            st.write('Модель не обрала замін. Перегляньте текст вручну.')
        with st.expander('Побачити фрагменти в оригіналі'):
            st.caption('Рожевим позначено лише ті фрагменти, які справді змінилися.')
            pane(before)
    with st.expander('Порівняти з рівномірним узагальненням'):
        st.caption('Ті самі засоби замін, але фіксоване правило першого рівня для всіх квазіідентифікаторів. '
                   'Це демонстраційне порівняння збереження тексту; приватність оцінюється окремо на корпусі.')
        adaptive_column, fixed_column = st.columns(2)
        with adaptive_column:
            st.markdown('**Адаптивний вибір**')
            pane(html.escape(release.text))
        with fixed_column:
            st.markdown('**Рівномірне узагальнення**')
            pane(html.escape(saved['uniform']))
    with st.expander('Як модель обирала дії'):
        steps = [step for step in release.trace if 'group' in step]
        operations = {'generalize': 'Узагальнити', 'generalize_broad': 'Узагальнити ширше',
                      'mask': 'Приховати', 'keep': 'Залишити'}
        if steps:
            rows = [{'Крок': index, 'Поле': GROUPS.get(step['group'], step['group']),
                     'Дія': operations.get(step.get('operation'), step.get('operation')),
                     'Очікувана користь дії': step.get('predicted_gain')}
                    for index, step in enumerate(steps, 1)]
            st.dataframe(pd.DataFrame(rows), hide_index=True)
        else:
            st.write('Виконано обов’язкові заміни. Додаткових адаптивних кроків не обрано.')
        st.caption('Навчений регресор прогнозує ефект можливих дій. Контролер враховує втрату інформації '
                   'й обирає наступну заміну. Прогноз користі дії не є ймовірністю деанонімізації.')


def main():
    st.set_page_config(page_title='Анонімізація резюме', page_icon='🔐', layout='wide')
    st.html('<style>' + (ROOT / 'app/demo.css').read_text() + '</style>')
    initialize_state()
    st.html('<div class="demo-hero"><div><h1>Адаптивна анонімізація резюме</h1>'
            '<p>Вставте текст — модель обере, які фрагменти приховати чи узагальнити, '
            'враховуючи збереження професійного змісту.</p></div>')
    st.html('<div class="demo-examples-label">Спробуйте на прикладі</div>')
    st.pills('Демонстраційний сценарій', list(EXAMPLE_BY_KEY), key='resume_example',
             format_func=lambda key: EXAMPLE_BY_KEY[key].title,
             on_change=use_example, label_visibility='collapsed')
    example = EXAMPLE_BY_KEY.get(st.session_state.get('resume_example'))
    if example and st.session_state['resume_input'] == example.text:
        st.html('<div class="demo-example-note"><span class="demo-synthetic">Вигаданий приклад</span>'
                f'<strong>{html.escape(example.subtitle)}</strong><span>{html.escape(example.focus)}</span></div>')
    else:
        st.html('<div class="demo-example-note">Ваш текст · приклади вище можна редагувати</div>')
    ready, version, paths, config, model_settings = True, (), {}, {}, {}
    try:
        model_settings, config, paths = resource_paths(DEFAULT_CONFIG)
        if not all(path.is_file() for path in paths.values()):
            raise FileNotFoundError
        files = [DEFAULT_CONFIG, ROOT / model_settings['resources_config'], *paths.values(),
                 ROOT / 'data/taxonomy/category_map.json']
        version = tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in files if path.exists())
    except (FileNotFoundError, ValueError, KeyError):
        ready = False
        st.error('Локальні ресурси моделі ще не підготовлено. Інструкція запуску — у README проєкту.')
    left, right = st.columns(2, gap='large')
    with left, st.container(border=True, key='input_card'):
        panel_heading('01', 'Ваше резюме')
        st.html('<div class="demo-editor-meta">Вставте текст, відредагуйте приклад або завантажте .txt</div>')
        text = st.text_area('Текст резюме', key='resume_input', height=400, max_chars=20_000,
                            on_change=input_changed, label_visibility='collapsed', placeholder='Вставте текст резюме…')
        upload_column, clear_column, count_column = st.columns([2, 1.4, 1.6])
        with upload_column, st.popover('Завантажити .txt', icon=':material/upload_file:'):
            upload_key = f"resume_file_{st.session_state.get('upload_generation', 0)}"
            st.file_uploader('Текстовий файл UTF-8', type=['txt'], key=upload_key,
                             on_change=use_file, args=(upload_key,), help='До 20 000 символів.')
        with clear_column:
            st.button('Очистити', on_click=clear_input)
        with count_column:
            st.caption(f'{len(text.split())} слів · {len(text):,} символів'.replace(',', ' '))
        if st.session_state.get('upload_error'):
            st.error(st.session_state['upload_error'])
        run = st.button('Анонімізувати', type='primary', width='stretch',
                        icon=':material/auto_fix_high:', disabled=not ready or not text.strip())
    if run:
        st.session_state.pop('text_demo_result', None)
        started = time.perf_counter()
        try:
            with st.spinner('Модель обирає перетворення…'):
                resources, lock = resources_for(str(DEFAULT_CONFIG), version)
                with lock, threadpool_limits(limits=config.get('cpu_threads', 1)):
                    result = anonymize_resume(resources, text)
                    doc = parse_resume(text, resources.codec, resources.detector)
                    uniform = render(doc, {unit.group: 1 for unit in doc.units if len(options(doc, unit)) > 1})
            st.session_state['text_demo_result'] = {'version': version, 'result': result, 'uniform': uniform,
                                                    'elapsed': time.perf_counter() - started}
        except DirectIdentifierError:
            st.error('Перевірка виявила залишки прямих ідентифікаторів. Результат потребує доопрацювання.')
        except Exception as exc:  # noqa: BLE001 - keep user text and provider details private
            st.error(f'Обробка не завершилась ({type(exc).__name__}). Перевірте локальні ресурси проєкту.')
    saved = st.session_state.get('text_demo_result')
    current = ready and result_is_current(saved, text, version, model_settings['adaptive']['utility_floor'])
    before, changes = '', []
    with right, st.container(border=True, key='output_card'):
        panel_heading('02', 'Анонімізований текст')
        if not current:
            st.html('<div class="demo-editor-meta">Результат з’явиться після оброблення</div>')
            empty_result()
            st.html('<div class="demo-result-note">Зеленим буде позначено фактичні заміни.</div>')
        else:
            result = saved['result']
            release = result.releases['adaptive']
            changes = [edit for edit in release.edits if edit['applied']]
            labelled_edits = [{**edit, 'label': LABELS.get(edit['label'], edit['label'])} for edit in release.edits]
            try:
                before, after = highlighted_pair(result.original, release.text, labelled_edits)
            except ValueError:
                before, after = html.escape(result.original), html.escape(release.text)
            status_column, toggle_column = st.columns([1, 1])
            with status_column:
                st.html('<div class="demo-output-toolbar"><span class="dot"></span>'
                        f"Готово за {saved['elapsed']:.1f} с</div>")
            with toggle_column:
                highlights = st.toggle('Підсвічувати заміни', value=True, key='highlight_changes')
            pane(after if highlights else html.escape(release.text))
            first, second = st.columns(2)
            first.metric('Змінено фрагментів', len(changes))
            retention = result.diagnostics['metrics']['lexical_retention']
            second.metric('Лексична схожість', f'{retention:.2f}' if retention is not None else '—',
                          help='Косинусна схожість слів, від 0 до 1. Це не частка збережених фактів і не оцінка приватності.')
            st.download_button('Завантажити результат', release.text, 'resume_anonymized.txt', 'text/plain',
                               icon=':material/download:', width='stretch')
    if current:
        st.html('<div class="demo-detail-title">Детальніше про цей результат</div>')
        show_details(saved, before, changes)
    with st.expander('Як працює модель'):
        st.html('<div class="demo-method">'
                '<div class="demo-method-card"><strong>1 · Визначення фрагментів</strong>'
                '<p>Розпізнаються особисті дані, посади, навички, організації та інші деталі резюме.</p></div>'
                '<div class="demo-method-card"><strong>2 · Вибір перетворень</strong>'
                '<p>Навчений регресор оцінює можливі дії. Контролер обирає їх з урахуванням втрати інформації.</p></div>'
                '<div class="demo-method-card"><strong>3 · Перевірка результату</strong>'
                '<p>Ви отримуєте текст і точний перелік замін. Інші резюме для цього не завантажуються.</p></div>'
                '</div>')
        st.caption('Приклади авторські й синтетичні. Вони демонструють поведінку алгоритму; '
                   'оцінка якості проводиться на окремій корпусній вибірці. '
                   'Модель не дає гарантії анонімності довільного тексту.')


if __name__ == '__main__':
    main()
