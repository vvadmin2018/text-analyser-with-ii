# -*- coding: utf-8 -*-
"""Сохранение/загрузка портретов и валидация устаревших pickle-файлов."""
import pickle

import pytest

from main import load_profiles, save_profiles
from src import config
from src.profile_builder import AuthorProfile, TriangularMembership


def _profile(name, n_features):
    profile = AuthorProfile(name)
    profile.features = [TriangularMembership(0.5, 1.0, 1.5) for _ in range(n_features)]
    return profile


@pytest.fixture
def pkl(tmp_path):
    return str(tmp_path / "profiles.pkl")


def test_roundtrip(pkl):
    profiles = {"a": _profile("a", config.N_FEATURES),
                "b": _profile("b", config.N_FEATURES)}
    save_profiles(profiles, pkl)
    loaded = load_profiles(pkl)
    assert loaded is not None
    assert set(loaded) == {"a", "b"}


def test_missing_file_returns_none(tmp_path):
    assert load_profiles(str(tmp_path / "nope.pkl")) is None


def test_one_stale_profile_invalidates_all(pkl):
    """Раньше функция возвращала уцелевшее подмножество, и анализ молча шёл
    по неполному списку авторов."""
    profiles = {"good": _profile("good", config.N_FEATURES),
                "stale": _profile("stale", config.N_FEATURES - 3)}
    save_profiles(profiles, pkl)
    assert load_profiles(pkl) is None


def test_corrupt_file_returns_none(pkl):
    with open(pkl, "wb") as f:
        f.write(b"not a pickle at all")
    assert load_profiles(pkl) is None


def test_empty_mapping_returns_none(pkl):
    with open(pkl, "wb") as f:
        pickle.dump({}, f)
    assert load_profiles(pkl) is None
