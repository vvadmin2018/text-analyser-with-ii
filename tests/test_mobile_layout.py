# -*- coding: utf-8 -*-
"""Вёрстка на узком экране.

Два дефекта, видимых только с телефона:

1. Название «THinkING» разрывалось посреди слова — «G» уезжала на отдельную
   строку. Streamlit задаёт заголовкам overflow-wrap: break-word, а картинка
   призрака высотой 200px сжимала текстовый флекс-элемент. Замерено в живом
   приложении: 2 строки при ширине 390px, 3 при 320px.

2. Таблицы во вкладке «Профили авторов» накладывались колонками. При
   table-layout: fixed и width: 100% восемнадцать колонок получали примерно по
   21px, и числа вида 27.668 не помещались в ячейку.
"""
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def app_source():
    return open(os.path.join(ROOT, 'app.py'), encoding='utf-8').read()


@pytest.fixture(scope="module")
def table_html():
    """Реальная таблица профиля, а не выдуманная разметка."""
    from src import config
    from src.report import build_stats, build_table_html

    matrix = [[0.5 + 0.01 * i] * config.N_FEATURES for i in range(3)]
    return build_table_html('bulichev', build_stats(matrix))


# ---------- заголовок ----------

def test_title_cannot_break_mid_word(app_source):
    """Именно этот разрыв и ронял «G» на отдельную строку."""
    header_css = re.search(r'\.app-header h1 \{\{(.*?)\}\}', app_source, re.S)
    assert header_css, "правила для .app-header h1 не найдены"
    rules = header_css.group(1)
    assert 'white-space: nowrap' in rules
    assert 'word-break: keep-all' in rules
    assert 'overflow-wrap: normal' in rules


def test_title_font_scales_with_viewport(app_source):
    """Без clamp() название на 320px не поместится даже без переносов."""
    assert 'clamp(' in app_source


def test_header_uses_classes_not_inline_styles(app_source):
    """Инлайн-стили нельзя переопределить медиазапросом."""
    assert 'class="app-header"' in app_source
    assert 'class="app-header__ghost"' in app_source
    assert 'style="height:200px' not in app_source


def test_text_block_shrinks_to_content(app_source):
    """С flex-grow блок занимал всю ширину и отгонял картинку к краю экрана.

    Замерено: h1 занимал 800px при названии шириной ~208px, так что призрак
    стоял далеко от текста, хотя gap между ними формально был 24px.
    """
    rules = re.search(r'\.app-header__text \{\{(.*?)\}\}', app_source, re.S)
    assert rules, "правила для .app-header__text не найдены"
    assert 'flex: 0 1 auto' in rules.group(1)
    assert 'flex: 1 1 auto' not in rules.group(1)


def test_ghost_shrinks_on_narrow_screens(app_source):
    """Картинка 200px и была тем, что сжимало текст."""
    assert re.search(r'@media \(max-width: 640px\).*?\.app-header__ghost.*?height: 96px',
                     app_source, re.S)


# ---------- таблица ----------

def test_table_has_minimum_width(table_html):
    """Колонки должны получать реальную ширину, а не делить 390px на 18."""
    assert 'min-width: 880px' in table_html


def test_table_is_wrapped_in_scroll_container(table_html):
    assert 'class="table-scroll"' in table_html
    assert re.search(r'\.table-scroll \{.*?overflow-x: auto', table_html, re.S)


def test_scroll_hint_shown_only_on_narrow_screens(table_html):
    """На десктопе таблица видна целиком, подсказка там лишняя."""
    assert 'Таблицу можно прокручивать вбок' in table_html
    assert re.search(r'\.scroll-hint \{\s*display: none', table_html, re.S)
    assert re.search(r'@media \(max-width: 700px\).*?\.scroll-hint \{\{? ?display: block',
                     table_html, re.S)


def test_table_still_fills_wide_screens(table_html):
    """min-width не должен ломать десктоп: ширина по-прежнему 100%."""
    assert re.search(r'table\.data-table \{.*?width: 100%', table_html, re.S)
