# -*- coding: utf-8 -*-
"""Инварианты извлечения признаков."""
import nltk
import numpy as np
import pytest

from src import config
from src.feature_extractor import FeatureExtractor, Language


@pytest.fixture(scope="module", autouse=True)
def nltk_data():
    for package in ('punkt', 'punkt_tab', 'stopwords'):
        nltk.download(package, quiet=True)


@pytest.fixture(scope="module")
def extractor():
    return FeatureExtractor(language=Language.RUSSIAN)


PROSE = (
    "Утро выдалось тихое и ясное. Солнце медленно поднималось над рекой, "
    "и вода казалась совсем неподвижной.\n\n"
    "Мальчик сидел на берегу и смотрел вдаль. О чём он думал? Никто не знал. "
    "Может быть, о доме; может быть, о том, что лето скоро закончится...\n\n"
    "— А ты что здесь делаешь? — спросил незнакомец.\n"
    "— Жду, — ответил мальчик."
)


def test_returns_expected_length(extractor):
    features = extractor.extract(PROSE)
    assert len(features) == config.N_FEATURES


def test_no_nan_or_inf(extractor):
    features = extractor.extract(PROSE)
    assert np.all(np.isfinite(features))


def test_shares_are_within_zero_one(extractor):
    """Признаки-доли (?, !, ..., прямая речь, части речи, союзы, предлоги)."""
    features = extractor.extract(PROSE)
    for index in (3, 4, 5, 6, 7, 8, 9, 10, 15, 16):
        assert 0.0 <= features[index] <= 1.0, f"признак {index} вне [0, 1]"


def test_empty_and_tiny_text(extractor):
    assert np.all(extractor.extract("") == 0)
    assert np.all(extractor.extract("Ага.") == 0)


def test_all_direct_speech_does_not_produce_nan(extractor):
    """Сплошной диалог: раньше список длин предложений оставался пустым,
    np.median([]) давал RuntimeWarning и nan, а признаки A1/A2 молча
    превращались в нули."""
    dialogue = "\n".join(["— Ты идёшь? — спросил он."] * 12)
    features = extractor.extract(dialogue)
    assert np.all(np.isfinite(features))
    assert features[0] > 0, "медианная длина предложения не должна обнуляться"


def test_mattr_is_length_stable(extractor):
    """MATTR не должен разъезжаться при удлинении текста тем же стилем —
    ради этого он и заменил RTTR."""
    short = PROSE
    long = PROSE * 6
    assert extractor.extract(short)[7] == pytest.approx(
        extractor.extract(long)[7], abs=0.15)


def test_russian_extractor_is_not_degraded(extractor):
    """pymorphy3 стоит в requirements — для русского деградации быть не должно."""
    assert extractor.degraded_reason is None


def test_belarusian_without_stanza_reports_degradation():
    """Отсутствие Stanza должно быть заявлено явно, а не давать тихие нули."""
    be = FeatureExtractor(language=Language.BELARUSIAN)
    if be.stanza_nlp is None:
        assert be.degraded_reason, "деградация должна быть отражена в degraded_reason"
        features = be.extract("Днём было цёпла і ціха. А ўвечары пайшоў дождж. "
                              "Хлопчык сядзеў каля акна і думаў пра лета.")
        # Ключевое: морфологические признаки не должны быть сплошными нулями
        assert any(features[i] > 0 for i in (8, 9, 10)), \
            "части речи обнулились — это и был молчаливый сбой"
