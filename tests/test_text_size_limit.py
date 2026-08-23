# -*- coding: utf-8 -*-
"""Верхний предел на размер анализируемого текста."""
import ast
import os
import re

import pytest

from src import config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def app_source():
    return open(os.path.join(ROOT, 'app.py'), encoding='utf-8').read()


@pytest.fixture(scope="module")
def text_size_mb(app_source):
    """Достаёт функцию из app.py, не поднимая Streamlit."""
    tree = ast.parse(app_source)
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == 'text_size_mb')
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'app.py', 'exec'), ns)
    return ns['text_size_mb']


# ---------- подсчёт размера ----------

@pytest.mark.parametrize("text", ["", None])
def test_empty_text_is_zero(text_size_mb, text):
    assert text_size_mb(text) == 0


def test_whitespace_counts_as_bytes(text_size_mb):
    """Пробелы — такие же байты: предел про фактический объём."""
    assert text_size_mb("   ") == pytest.approx(3 / 1048576)


def test_size_is_measured_in_utf8_bytes(text_size_mb):
    """Кириллица занимает два байта на букву.

    Счёт по символам показал бы вдвое меньше, чем текст весит на самом деле,
    и «3 МБ» в интерфейсе означали бы 6 МБ на диске.
    """
    n = 100_000
    assert text_size_mb("a" * n) == pytest.approx(n / 1048576)
    assert text_size_mb("а" * n) == pytest.approx(2 * n / 1048576)


def test_exact_limit_is_allowed(text_size_mb):
    """Ровно предел — ещё можно, предел проверяется строгим неравенством."""
    at_limit = "a" * (config.MAX_TEXT_MB * 1024 * 1024)
    assert text_size_mb(at_limit) == config.MAX_TEXT_MB
    assert not text_size_mb(at_limit) > config.MAX_TEXT_MB


def test_one_byte_over_limit_is_rejected(text_size_mb):
    over = "a" * (config.MAX_TEXT_MB * 1024 * 1024 + 1)
    assert text_size_mb(over) > config.MAX_TEXT_MB


# ---------- конфигурация ----------

def test_limit_is_configured():
    assert config.MAX_TEXT_MB > 0


def test_text_limit_is_not_below_upload_limit():
    """Файл ограничен отдельно и жёстче: вставить текст можно минуя загрузку.

    Предел на текст ниже, чем на файл, означал бы, что загруженный файл
    нельзя проанализировать — то есть загрузка ведёт в тупик.
    """
    assert config.MAX_TEXT_MB >= config.MAX_UPLOAD_MB


def test_min_below_max():
    max_chars = config.MAX_TEXT_MB * 1024 * 1024
    assert config.MIN_TEXT_LENGTH < max_chars


# ---------- проверка в интерфейсе ----------

def test_analysis_checks_the_limit(app_source):
    """Проверка обязана стоять до разбора, иначе предел ничего не значит."""
    assert 'size_mb > config.MAX_TEXT_MB' in app_source
    body = app_source.split('def run_analysis')[1]
    assert body.index('MAX_TEXT_MB') < body.index('extractor.extract')


def test_counter_shows_the_limit(app_source):
    """Упереться в предел на кнопке, ничего о нём не зная, — плохо."""
    assert 'из {config.MAX_TEXT_MB} МБ' in app_source


def test_megabytes_hidden_for_ordinary_texts(app_source):
    """На обычном тексте мегабайты — лишний шум."""
    assert 'config.MAX_TEXT_MB / 10' in app_source
