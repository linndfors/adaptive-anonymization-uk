"""Interactive anonymization with selectable policies and optional corpus assessment."""

from __future__ import annotations

import csv
import html
import io
import threading
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from threadpoolctl import threadpool_limits

from anonrisk_uk.demo_highlights import highlighted_pair
from anonrisk_uk.resume_adaptive import DirectIdentifierError
from anonrisk_uk.resume_resources import DEFAULT_CONFIG, load_resume_resources, resource_paths
from anonrisk_uk.workua import GROUP_LABELS

try:
    from app.demo_examples import DEFAULT_EXAMPLE, DEMO_EXAMPLES, EXAMPLE_BY_KEY
    from app.demo_runtime import (
        METHOD_BY_KEY,
        METHODS,
        CorpusIndex,
        add_demo_target,
        parse_corpus,
        run_method,
        upload_digest,
    )
except ModuleNotFoundError:
    from demo_examples import DEFAULT_EXAMPLE, DEMO_EXAMPLES, EXAMPLE_BY_KEY
    from demo_runtime import (
        METHOD_BY_KEY,
        METHODS,
        CorpusIndex,
        add_demo_target,
        parse_corpus,
        run_method,
        upload_digest,
    )

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / 'data/corpus/default.jsonl'
LABELS = {
    'SKILL_HEADER': 'Перелік навичок', 'SKILL': 'Навичка', 'JOB_TITLE': 'Посада', 'RESPONSIBILITY': 'Обов’язки',
    'CONTEXT_TEXT': 'Фрагмент тексту', 'EXPERIENCE_DURATION': 'Тривалість досвіду',
    'OCCUPATION_EDU': 'Професійний фрагмент', 'DATE_TIME': 'Дата', 'INDUSTRY': 'Галузь',
    'PERSON': 'Ім’я', 'EMAIL': 'Email', 'PHONE': 'Телефон', 'CONTACT': 'Контакт',
    'ADDRESS': 'Адреса', 'URL': 'Посилання', 'EMPLOYER': 'Роботодавець', 'ORGANIZATION': 'Організація',
    'LOCATION': 'Місце', 'EMPLOYMENT_PERIOD': 'Період роботи', 'EDUCATION_PERIOD': 'Період навчання',
    'EDUCATION_INSTITUTION': 'Заклад освіти', 'ID_NUMBER': 'Номер документа',
    'BIRTH_DATE': 'Дата народження', 'HEALTH': 'Дані про здоров’я', 'AGE': 'Вік', 'SALARY': 'Зарплата',
}
GROUPS = {**GROUP_LABELS, 'dates': 'Дати', 'skills': 'Навички', 'title': 'Бажана посада',
          'exp_position': 'Посади', 'additional_info': 'Інший текст'}


@st.cache_resource(show_spinner=False)
def resources_for(path, version):
    return load_resume_resources(path), threading.Lock()


def pane(markup):
    st.html('<div class="demo-output">' + markup + '</div>')


def invalidate_result():
    st.session_state.pop('text_demo_result', None)
    st.session_state.pop('method_comparison', None)


def use_example():
    example = EXAMPLE_BY_KEY.get(st.session_state.get('resume_example'))
    if example:
        st.session_state['resume_input'] = example.text
        st.session_state.pop('corpus_target', None)
        st.session_state.pop('upload_error', None)
        if example.related_text:
            clear_corpus()
            st.session_state['use_corpus'] = True
            st.session_state['corpus_source'] = 'default'
        invalidate_result()


def input_changed():
    st.session_state['resume_example'] = None
    st.session_state.pop('corpus_target', None)
    invalidate_result()


def clear_input():
    st.session_state['resume_input'] = ''
    st.session_state['resume_example'] = None
    st.session_state.pop('corpus_target', None)
    st.session_state['upload_generation'] = st.session_state.get('upload_generation', 0) + 1
    st.session_state.pop('upload_error', None)
    invalidate_result()


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
        st.session_state.pop('corpus_target', None)
        st.session_state.pop('upload_error', None)
        invalidate_result()
    except (UnicodeDecodeError, ValueError):
        st.session_state['upload_error'] = 'Потрібен текстовий файл UTF-8 до 20 000 символів.'


def initialize_state():
    if st.session_state.get('text_demo_schema') != 25:
        invalidate_result()
        st.session_state.pop('corpus_index', None)
        st.session_state.pop('corpus_target', None)
        st.session_state['text_demo_schema'] = 25
    if 'resume_input' not in st.session_state:
        st.session_state['resume_input'] = DEFAULT_EXAMPLE.text
        st.session_state['resume_example'] = DEFAULT_EXAMPLE.key
        if DEFAULT_EXAMPLE.related_text:
            st.session_state['use_corpus'] = True
    elif st.session_state.get('resume_example') not in EXAMPLE_BY_KEY:
        if st.session_state.get('resume_example') is not None:
            st.session_state['resume_example'] = DEFAULT_EXAMPLE.key
            use_example()
    else:
        selected = EXAMPLE_BY_KEY[st.session_state['resume_example']]
        if st.session_state['resume_input'] != selected.text:
            use_example()


def panel_heading(number, title):
    st.html(f'<div class="demo-panel-heading"><span class="demo-panel-number">{number}</span>'
            f'<h2>{html.escape(title)}</h2></div>')


def sample_csv():
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=['id', 'text'])
    writer.writeheader()
    writer.writerows({'id': example.key, 'text': example.text} for example in DEMO_EXAMPLES)
    return stream.getvalue().encode('utf-8-sig')


@st.cache_data(show_spinner=False, max_entries=2)
def default_corpus_for(path, version, demo_csv=None):
    source = Path(path)
    if version:
        corpus = parse_corpus(((source.name, source.read_bytes()),))
        return corpus, 'Стандартний'
    corpus = parse_corpus((('demo.csv', demo_csv if demo_csv is not None else sample_csv()),))
    return corpus, 'Демо'


def clear_corpus():
    for key in ('loaded_corpus', 'corpus_index', 'corpus_target'):
        st.session_state.pop(key, None)


def activate_corpus(corpus):
    previous = st.session_state.get('loaded_corpus')
    if previous is None or previous.fingerprint != corpus.fingerprint:
        clear_corpus()
        st.session_state['loaded_corpus'] = corpus


def corpus_controls():
    if not st.session_state.get('use_corpus', False):
        clear_corpus()
        st.session_state.pop('corpus_source', None)
        return None, None
    source = st.session_state.get('corpus_source', 'default')
    corpus = None
    default_error = None
    default_label = 'Стандартний'
    if source == 'default':
        try:
            version = (DEFAULT_CORPUS.stat().st_mtime_ns, DEFAULT_CORPUS.stat().st_size) if DEFAULT_CORPUS.is_file() else ()
            corpus, default_label = default_corpus_for(str(DEFAULT_CORPUS), version, None if version else sample_csv())
            activate_corpus(corpus)
        except (OSError, ValueError):
            clear_corpus()
            default_error = 'Не вдалося прочитати стандартний корпус. Перевірте файл або оберіть власний.'
    previous = st.session_state.get('loaded_corpus')
    label = default_label if source == 'default' else 'Власний'
    title = f'Корпус · {label.lower()}' + (f' · {len(previous.records)} резюме' if previous else '')
    with st.expander(title, expanded=previous is None):
        st.radio('Джерело корпусу', ['default', 'upload'], horizontal=True,
                 format_func=lambda key: 'Стандартний' if key == 'default' else 'Власний',
                 key='corpus_source', on_change=clear_corpus, label_visibility='collapsed')
        if source == 'default':
            if default_error:
                st.error(default_error)
                return None, None
            if default_label == 'Демо':
                st.info('Демо-корпус із синтетичних профілів.')
        else:
            uploads = st.file_uploader('Корпус резюме', type=['csv', 'json', 'jsonl', 'txt'],
                                       accept_multiple_files=True, key='corpus_files',
                                       label_visibility='collapsed',
                                       help='CSV/JSON із полем text або структурований CSV Work.ua; один TXT — одне резюме.')
            st.download_button('Зразок CSV', sample_csv(), 'corpus_example.csv', 'text/csv',
                               help='Шість вигаданих прикладів; це зразок формату, а не незалежний тест.')
            if not uploads:
                clear_corpus()
                st.info('Завантажте файли або оберіть стандартний корпус.')
                return None, None
            try:
                files = tuple((uploaded.name, uploaded.getvalue()) for uploaded in uploads)
                fingerprint = upload_digest(files)
                corpus = previous if previous is not None and previous.fingerprint == fingerprint else parse_corpus(files)
                activate_corpus(corpus)
            except ValueError as exc:
                clear_corpus()
                st.error(str(exc))
                return None, None
        notes = []
        if corpus.duplicates:
            notes.append(f'{corpus.duplicates} дублікатів пропущено')
        if corpus.skipped:
            notes.append(f'{corpus.skipped} порожніх записів пропущено')
        if corpus.limited:
            notes.append('взято перші 6000 записів')
        if notes:
            st.write(' · '.join(notes))
        base_corpus = corpus
        demo_target = None
        example = EXAMPLE_BY_KEY.get(st.session_state.get('resume_example'))
        if example and example.related_text and st.session_state.get('resume_input') == example.text:
            corpus, demo_target = add_demo_target(corpus, example.key, example.related_text)
            if 'corpus_target' not in st.session_state:
                st.session_state['corpus_target'] = demo_target
        labels = {record.key: record.label for record in corpus.records}
        target = st.selectbox('Ціль пошукової атаки', [None, *labels], key='corpus_target',
                              format_func=lambda key: 'Оригінал резюме (автоматично)' if key is None else labels[key],
                              help='Для довільного тексту додаємо оригінал до корпусу й шукаємо його за результатом. '
                                   'Для атаки на інше резюме тієї самої людини оберіть пов’язаний запис; '
                                   'person_id об’єднує записи цієї особи.')
        if demo_target is not None and target == demo_target:
            return corpus, target
        return base_corpus, target


def assessment_for(corpus, original, output, target):
    if corpus is None:
        return None
    index = st.session_state.get('corpus_index')
    index_original = original if target is None else None
    if (index is None or index.corpus.fingerprint != corpus.fingerprint
            or index.original != index_original):
        with st.spinner('Готуємо пошуковий індекс…'), threadpool_limits(limits=1):
            index = CorpusIndex(corpus, index_original)
        st.session_state['corpus_index'] = index
    with threadpool_limits(limits=1):
        return index.assess(original, output, target)


def assess_safely(corpus, original, output, target):
    try:
        return assessment_for(corpus, original, output, target)
    except ValueError as exc:
        st.warning(str(exc))
        return None


def corpus_metric(assessment):
    target = 'інше резюме тієї самої людини' if assessment.kind == 'linked' else 'оригінал у корпусі'
    return 'Пошукова вразливість · RR', (f'Шукаємо {target}. Найвищий очікуваний зворотний ранг серед двох атак. '
                                     '1 — ціль перша; без нічиїх 0.2 — п’ята. Менше — слабша атака. '
                                     'Відсоток під значенням — відносна зміна RR. Це не ймовірність деанонімізації.')


def target_similarity(assessment):
    if assessment is None or assessment.kind == 'unassessed':
        return None
    return next(row for row in assessment.rows if row['Атака'] == 'Символьний TF-IDF')


def show_assessment(assessment):
    if assessment is None:
        return
    if assessment.kind == 'unassessed':
        st.info(assessment.note)
        return
    with st.expander('Оцінка за корпусом'):
        if assessment.kind == 'self':
            st.write(f'Пошук оригіналу · {assessment.candidates} кандидатів, включно з оригіналом. '
                     f'Повторів оригіналу об’єднано: {assessment.excluded}.')
        else:
            st.write(f'Пошук іншого резюме тієї самої людини · {assessment.candidates} кандидатів. '
                     f'Точних копій оригіналу виключено: {assessment.excluded}.')
        frame = pd.DataFrame(assessment.rows)
        st.dataframe(frame[['Атака', 'Ранг до', 'Ранг після', 'RR до', 'RR після',
                            'Recall@1 до', 'Recall@1 після']], hide_index=True, width='stretch')
        st.dataframe(frame[['Атака', 'Бал цілі до', 'Бал цілі після', 'Бал конкурента після',
                            'Відрив до', 'Відрив після']], hide_index=True, width='stretch')


def show_details(saved, before, changes):
    result = saved['result']
    with st.expander(f'Що змінилося · {len(changes)} фрагментів'):
        if changes:
            st.dataframe(pd.DataFrame([
                {'Фрагмент': LABELS.get(edit['label'], 'Інші дані'), 'Було': edit['text'],
                 'Стало': f"Об’єднано: {edit['merged_into']}" if edit.get('merged_into') else edit['replacement']}
                for edit in changes
            ]), hide_index=True, width='stretch')
        else:
            st.write('Текст залишився без змін.')
        with st.expander('Виділення в оригіналі'):
            pane(before)
    if result.method == 'adaptive':
        with st.expander('Як модель обирала дії'):
            steps = [step for step in result.release.trace if 'group' in step]
            operations = {'generalize': 'Узагальнити', 'generalize_broad': 'Узагальнити ширше', 'mask': 'Приховати'}
            if steps:
                st.dataframe(pd.DataFrame([
                    {'Крок': i, 'Поле': GROUPS.get(step['group'], 'Інші фрагменти'),
                     'Дія': operations.get(step.get('operation'), 'Змінити'),
                     'Прогноз ефекту': step.get('predicted_gain')}
                    for i, step in enumerate(steps, 1)
                ]), hide_index=True, width='stretch')
            else:
                st.write('Додаткових адаптивних дій не обрано.')


def show_comparison(saved, version, corpus, target):
    with st.expander('Порівняння методів'):
        if st.button('Порівняти всі методи', key='compare_methods'):
            try:
                with st.spinner('Порівнюємо методи…'):
                    resources, lock = resources_for(str(DEFAULT_CONFIG), version)
                    with lock, threadpool_limits(limits=1):
                        st.session_state['method_comparison'] = [
                            saved['result'] if method.key == saved['method'] else
                            run_method(resources, saved['result'].original, method.key) for method in METHODS]
            except Exception as exc:  # noqa: BLE001 -- do not expose resume/provider details
                st.error(f'Порівняння не завершено ({type(exc).__name__}).')
        results = st.session_state.get('method_comparison')
        if results:
            rows = []
            original = saved['result'].original
            for label, value, edits, retention in [
                ('Оригінал (L0)', original, [], 1.0),
                *[(METHOD_BY_KEY[result.method].title, result.release.text, result.release.edits,
                   result.release.retention) for result in results],
            ]:
                row = {'Метод': label, 'Змінено фрагментів': sum(edit['applied'] for edit in edits),
                       'Лексична схожість': retention}
                assessment = assess_safely(corpus, original, value, target)
                if assessment is not None and assessment.kind != 'unassessed':
                    title, _ = corpus_metric(assessment)
                    row[title] = assessment.after
                    row['Збіг із ціллю · TF-IDF'] = target_similarity(assessment)['Бал цілі після']
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')


def main():
    st.set_page_config(page_title='Анонімізація резюме', page_icon='🔐', layout='wide')
    st.html('<style>' + (ROOT / 'app/demo.css').read_text() + '</style>')
    initialize_state()
    st.html('<div class="demo-hero"><div><h1>Анонімізація резюме</h1></div></div>')
    with st.container(key='method_bar'):
        method_column, corpus_column = st.columns([3, 2], vertical_alignment='bottom')
        with method_column:
            method = st.selectbox('Метод', [item.key for item in METHODS], key='resume_method',
                                  format_func=lambda key: METHOD_BY_KEY[key].title, on_change=invalidate_result,
                                  help='Фіксовані політики використовують ті самі засоби замін, що й адаптивна модель.')
        with corpus_column:
            st.toggle('Підключити корпус', key='use_corpus',
                      help='Порівняння пошукової вразливості до та після; вибір замін від корпусу не залежить.')
    corpus, target = corpus_controls()
    st.pills('Приклади', list(EXAMPLE_BY_KEY), key='resume_example',
             format_func=lambda key: EXAMPLE_BY_KEY[key].title, on_change=use_example, label_visibility='collapsed')
    ready, version = True, ()
    try:
        settings, profile, paths = resource_paths(DEFAULT_CONFIG)
        files = [DEFAULT_CONFIG, ROOT / settings['resources_config'], *paths.values(),
                 ROOT / 'data/taxonomy/category_map.json']
        if not all(path.is_file() for path in files):
            raise FileNotFoundError
        version = tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in files)
    except (FileNotFoundError, ValueError, KeyError):
        ready = False
        st.error('Підготуйте локальні ресурси за інструкцією в README.')
    left, right = st.columns(2, gap='large')
    with left, st.container(border=True, key='input_card'):
        panel_heading('01', 'Ваше резюме')
        text = st.text_area('Текст резюме', key='resume_input', height=400, max_chars=20_000,
                            on_change=input_changed, label_visibility='collapsed', placeholder='Вставте текст резюме…')
        upload_column, clear_column = st.columns([2, 1])
        with upload_column, st.popover('Завантажити .txt', icon=':material/upload_file:'):
            upload_key = f"resume_file_{st.session_state.get('upload_generation', 0)}"
            st.file_uploader('Файл UTF-8', type=['txt'], key=upload_key, on_change=use_file, args=(upload_key,))
        with clear_column:
            st.button('Очистити', on_click=clear_input)
        if st.session_state.get('upload_error'):
            st.error(st.session_state['upload_error'])
        enabled = ready and bool(text.strip()) and (not st.session_state.get('use_corpus') or corpus is not None)
        run = st.button('Анонімізувати', type='primary', width='stretch', icon=':material/auto_fix_high:',
                        disabled=not enabled)
    if run:
        invalidate_result()
        started = time.perf_counter()
        try:
            with st.spinner('Обробляємо резюме…'):
                resources, lock = resources_for(str(DEFAULT_CONFIG), version)
                with lock, threadpool_limits(limits=profile.get('cpu_threads', 1)):
                    result = run_method(resources, text, method)
            st.session_state['text_demo_result'] = {'result': result, 'method': method, 'version': version,
                                                    'elapsed': time.perf_counter() - started}
        except DirectIdentifierError:
            st.error('У результаті залишилися прямі ідентифікатори. Потрібна перевірка.')
        except Exception as exc:  # noqa: BLE001 -- do not expose resume/provider details
            st.error(f'Обробка не завершилась ({type(exc).__name__}). Перевірте локальні ресурси.')
    saved = st.session_state.get('text_demo_result')
    current = (ready and saved is not None and saved['result'].original == text
               and saved['method'] == method and saved['version'] == version)
    changes, before = [], ''
    assessment = None
    with right, st.container(border=True, key='output_card'):
        panel_heading('02', 'Результат')
        if not current:
            st.html('<div class="demo-empty"><div><div class="demo-empty-icon">↗</div>'
                    '<h3>Тут з’явиться результат</h3><p>Натисніть «Анонімізувати».</p></div></div>')
        else:
            release = saved['result'].release
            changes = [edit for edit in release.edits if edit['applied']]
            labelled = [{**edit, 'label': LABELS.get(edit['label'], 'Інші дані')} for edit in release.edits]
            try:
                before, after = highlighted_pair(text, release.text, labelled)
            except ValueError:
                before, after = html.escape(text), html.escape(release.text)
            status_column, toggle_column = st.columns([1, 1])
            with status_column:
                st.html('<div class="demo-output-toolbar"><span class="dot"></span>'
                        f"{html.escape(METHOD_BY_KEY[method].title)} · {saved['elapsed']:.1f} с</div>")
            with toggle_column:
                highlights = st.toggle('Виділяти заміни', value=True, key='highlight_changes')
            pane(after if highlights else html.escape(release.text))
            assessment = assess_safely(corpus, text, release.text, target)
            columns = st.columns(3 if assessment is not None and assessment.kind != 'unassessed' else 2)
            columns[0].metric('Змінено фрагментів', len(changes))
            columns[1].metric('Лексична схожість', f'{release.retention:.2f}' if release.retention is not None else '—',
                              help='TF-IDF косинус із вихідним текстом. Це не частка збережених фактів.')
            if len(columns) == 3:
                title, help_text = corpus_metric(assessment)
                delta = None
                if assessment.before > 0 and abs(assessment.after - assessment.before) > .0005:
                    delta = f'{100 * (assessment.after - assessment.before) / assessment.before:+.1f}% RR'
                columns[2].metric(title, f'{assessment.before:.3f} → {assessment.after:.3f}',
                                  delta=delta, delta_color='inverse', help=help_text)
            st.download_button('Завантажити результат', release.text, f'resume_{method}.txt', 'text/plain',
                               icon=':material/download:', width='stretch')
    if current:
        show_assessment(assessment)
        show_details(saved, before, changes)
        show_comparison(saved, version, corpus, target)
    with st.expander('Довідка'):
        st.write('Приклади — відібрані синтетичні пари резюме для ілюстрації роботи моделі, '
                 'а не незалежна тестова вибірка. У них оцінюється пошук іншого профілю тієї самої людини (S2). '
                 'У налаштуваннях корпусу можна обрати пошук самого оригіналу (S1).')
        st.dataframe(pd.DataFrame([{'Метод': item.title, 'Правило': item.description} for item in METHODS]),
                     hide_index=True, width='stretch')
        st.write('RR — найбільший очікуваний зворотний ранг серед двох атак. '
                 'Відсоток під значенням — його відносна зміна, а не ймовірність деанонімізації. '
                 'Бал цілі показує силу збігу; додатний відрив — перевагу над найкращим конкурентом.')
        st.write('Корпус використовується лише для оцінювання. Усі методи мають спільні засоби замін. '
                 'L1–L3 тут працюють із розпізнаними фрагментами та відрізняються від історичних політик '
                 'для структурованих полів. Прогноз моделі стосується окремої дії, а не абсолютного ризику.')


if __name__ == '__main__':
    main()
