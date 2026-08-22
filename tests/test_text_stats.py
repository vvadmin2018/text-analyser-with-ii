# -*- coding: utf-8 -*-
"""Определение языка и общая статистика текста."""
import nltk
import pytest

from src.feature_extractor import FeatureExtractor, Language, detect_language


@pytest.fixture(scope="module", autouse=True)
def nltk_data():
    for package in ('punkt', 'punkt_tab', 'stopwords'):
        nltk.download(package, quiet=True)


@pytest.fixture(scope="module")
def ru():
    return FeatureExtractor(language=Language.RUSSIAN)


PROSE = (
    "Утро выдалось тихое и ясное. Солнце поднималось над рекой.\n\n"
    "Мальчик сидел на берегу. О чём он думал? Никто не знал.\n\n"
    "Вода была неподвижной."
)


# ---------- определение языка ----------

@pytest.mark.parametrize("text,expected", [
    ("Мальчик сидел на берегу и смотрел вдаль.", Language.RUSSIAN),
    ("Міхалка бег па вуліцы і радасна крычаў.", Language.BELARUSIAN),
    ("Маці адчыніла куфар і дастала фатаграфіі.", Language.BELARUSIAN),
    ("Ещё щенок защищал общую площадь.", Language.RUSSIAN),
])
def test_detects_language(text, expected):
    assert detect_language(text) == expected


@pytest.mark.parametrize("text", ["", "   ", None, "Hello world, plain English."])
def test_returns_none_when_undecidable(text):
    """Без отличительных букв честнее не гадать, чем назвать язык наугад."""
    assert detect_language(text) is None


def test_belarusian_specific_letters_outweigh_length():
    """Короткая белорусская фраза не должна проигрывать длине русской."""
    assert detect_language("Ён пайшоў у лес.") == Language.BELARUSIAN


# ---------- статистика ----------

def test_describe_counts(ru):
    stats = ru.describe(PROSE)
    assert stats['chars'] == len(PROSE)
    assert stats['paragraphs'] == 3
    assert stats['sentences'] == 6
    assert stats['words'] > 0
    assert stats['language'] == Language.RUSSIAN


def test_chars_without_spaces_is_smaller(ru):
    stats = ru.describe(PROSE)
    assert 0 < stats['chars_no_spaces'] < stats['chars']


def test_single_block_is_one_paragraph(ru):
    """Текст без пустых строк — один абзац, а не ноль."""
    assert ru.describe("Одно предложение. И второе.")['paragraphs'] == 1


def test_empty_text_is_all_zeros(ru):
    stats = ru.describe("   ")
    assert stats == {'chars': 0, 'chars_no_spaces': 0, 'paragraphs': 0,
                     'sentences': 0, 'words': 0, 'language': None}


def test_word_count_matches_extractor_tokenization(ru):
    """Число слов на экране обязано совпадать с тем, по которому идёт анализ.

    Расхождение выглядит как ошибка, даже когда оба числа по-своему верны,
    поэтому describe() и extract() должны токенизировать одинаково.
    """
    from nltk.tokenize import word_tokenize
    expected = len(ru._content_words(word_tokenize(PROSE)))
    assert ru.describe(PROSE)['words'] == expected


def test_punctuation_is_not_counted_as_words(ru):
    plain = ru.describe("Кот спал крепко")['words']
    punctuated = ru.describe("Кот, спал... — крепко!")['words']
    assert plain == punctuated == 3


def test_describe_does_not_depend_on_extractor_language():
    """Статистика — свойство текста, а не выбранного режима разбора."""
    ru_ex = FeatureExtractor(language=Language.RUSSIAN)
    be_ex = FeatureExtractor(language=Language.BELARUSIAN)
    assert ru_ex.describe(PROSE) == be_ex.describe(PROSE)
