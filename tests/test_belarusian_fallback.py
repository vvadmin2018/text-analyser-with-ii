# -*- coding: utf-8 -*-
"""Суффиксный разбор белорусского текста.

Раньше белорусский разбирался русскими окончаниями и в порядке
«сначала существительные»: инфинитив на -ць уходил в существительные по
конечному -ь, прошедшее время на -ў не совпадало ни с одним правилом и
попадало в существительные веткой else, а прилагательные перехватывались
односимвольными -ая/-ае/-ы раньше своих правил. На корпусе получалось
97% существительных, 1% глаголов, 2% прилагательных — то есть признаки
Сущ/Глаг/Прил переставали различать авторов.
"""
import nltk
import pytest

from src import config
from src.feature_extractor import FeatureExtractor, Language


@pytest.fixture(scope="module", autouse=True)
def nltk_data():
    for package in ('punkt', 'punkt_tab', 'stopwords'):
        nltk.download(package, quiet=True)


@pytest.fixture(scope="module")
def be():
    return FeatureExtractor(language=Language.BELARUSIAN)


def _shares(extractor, text):
    values = dict(zip(config.FEATURE_LIST_SHORT, extractor.extract(text)))
    return values['Сущ'], values['Глаг'], values['Прил']


BE_PROSE = (
    "Міхалка бег па вуліцы і радасна крычаў. Ён любіў хадзіць у стары лес, "
    "дзе высокія дрэвы стаялі цёмнай сцяной.\n\n"
    "Маці адчыніла куфар і дастала пажоўклыя фатаграфіі. Яна доўга глядзела "
    "на маладога хлопца і нічога не казала. Вецер сціхаў, і рабілася ціха."
)


@pytest.mark.parametrize("word,expected", [
    # инфинитивы: -ць / -сці, а не русское -ть
    ("чытаць", "VERB"),
    ("хадзіць", "VERB"),
    ("несці", "VERB"),
    # прошедшее время мужского рода: -ў, а не русское -л
    ("глядзеў", "VERB"),
    ("крычаў", "VERB"),
    ("загінуў", "VERB"),
    # прилагательные не должны перехватываться правилами существительных
    ("вялікая", "ADJ"),
    ("пажоўклыя", "ADJ"),
    ("каляровым", "ADJ"),
    ("маладога", "ADJ"),
    # исключения: -оў/-яў здесь род. мн. существительных, -сьці — наречие
    ("слоў", "NOUN"),
    ("вучняў", "NOUN"),
    ("дзесьці", "NOUN"),
])
def test_belarusian_endings(be, word, expected):
    rules = be._pos_rules()
    tag = next((t for e, t in rules if word.endswith(e)), "NOUN")
    assert tag == expected


def test_rules_sorted_longest_first(be):
    """Порядок обязателен: -ы, стоящее раньше -ыя, съедает прилагательные."""
    lengths = [len(ending) for ending, _ in be._pos_rules()]
    assert lengths == sorted(lengths, reverse=True)


def test_pos_shares_are_plausible(be):
    """Доли частей речи должны попадать в диапазон живой прозы.

    До исправления глаголы и прилагательные давали доли около 0.00-0.02
    на любом белорусском тексте.
    """
    nouns, verbs, adjs = _shares(be, BE_PROSE)
    assert 0.10 < verbs < 0.45, f"доля глаголов вне правдоподобного диапазона: {verbs}"
    assert 0.02 < adjs < 0.30, f"доля прилагательных вне правдоподобного диапазона: {adjs}"
    assert nouns < 0.80, f"почти всё ушло в существительные: {nouns}"


def test_belarusian_reports_degraded_without_stanza(be):
    """Без Stanza приложение обязано сказать, что разбор приблизительный."""
    from src.feature_extractor import STANZA_AVAILABLE
    if STANZA_AVAILABLE and be.stanza_nlp is not None:
        assert be.degraded_reason is None
    else:
        assert be.degraded_reason
        assert "stanza" in be.degraded_reason.lower()
