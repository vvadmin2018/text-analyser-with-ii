# -*- coding: utf-8 -*-
"""Кэш портретов и переключение языка в config.

load_profiles проверяла только число признаков. Состав авторов не проверялся,
поэтому добавление автора в config не имело никакого эффекта: старый .pkl
проходил проверку и загружался как есть, а новый автор не появлялся ни в
списке, ни в результатах — до ручного «Переобучить».
"""
import pickle

import pytest

from src import config
from main import load_profiles, save_profiles


class _FakeProfile:
    """Портрет с нужным числом признаков и ничем больше."""

    def __init__(self, n_features=None):
        n = config.N_FEATURES if n_features is None else n_features
        self.features = [object()] * n


@pytest.fixture
def profiles_file(tmp_path):
    return str(tmp_path / "authors_profiles_test.pkl")


def _write(path, names, n_features=None):
    with open(path, 'wb') as f:
        pickle.dump({n: _FakeProfile(n_features) for n in names}, f)


def test_missing_file_returns_none(profiles_file):
    assert load_profiles(profiles_file, authors=['a']) is None


def test_matching_author_set_loads(profiles_file):
    _write(profiles_file, ['a', 'b'])
    loaded = load_profiles(profiles_file, authors=['a', 'b'])
    assert loaded is not None
    assert set(loaded) == {'a', 'b'}


def test_author_order_does_not_matter(profiles_file):
    _write(profiles_file, ['a', 'b'])
    assert load_profiles(profiles_file, authors=['b', 'a']) is not None


def test_added_author_invalidates_cache(profiles_file):
    """Ровно тот случай: автор добавлен в config, .pkl остался старым."""
    _write(profiles_file, ['maur', 'bryl'])
    assert load_profiles(profiles_file,
                         authors=['baravikova', 'maur', 'bryl']) is None


def test_removed_author_invalidates_cache(profiles_file):
    _write(profiles_file, ['a', 'b', 'c'])
    assert load_profiles(profiles_file, authors=['a', 'b']) is None


def test_stale_feature_count_still_invalidates(profiles_file):
    _write(profiles_file, ['a'], n_features=config.N_FEATURES - 1)
    assert load_profiles(profiles_file, authors=['a']) is None


def test_without_authors_argument_set_is_not_checked(profiles_file):
    """Обратная совместимость: без authors проверяются только признаки."""
    _write(profiles_file, ['a', 'b'])
    assert load_profiles(profiles_file) is not None


def test_save_then_load_roundtrip(profiles_file):
    save_profiles({'a': _FakeProfile(), 'b': _FakeProfile()}, profiles_file)
    assert load_profiles(profiles_file, authors=['a', 'b']) is not None


# ---------- LANGUAGE_MODE ----------

def test_language_mode_is_supported():
    assert config.LANGUAGE_MODE in config.AUTHORS_BY_LANGUAGE


def test_authors_list_follows_language_mode():
    assert config.AUTHORS_LIST is config.AUTHORS_BY_LANGUAGE[config.LANGUAGE_MODE]
    assert config.PROFILE_FILE == config.PROFILE_FILE_BY_LANGUAGE[config.LANGUAGE_MODE]


def test_language_modes_match_extractor_codes():
    """Ключи должны совпадать с кодами Language, иначе main передаст мусор."""
    from src.feature_extractor import Language
    assert set(config.AUTHORS_BY_LANGUAGE) == {Language.RUSSIAN, Language.BELARUSIAN}


def test_each_language_has_its_own_profile_file():
    files = list(config.PROFILE_FILE_BY_LANGUAGE.values())
    assert len(set(files)) == len(files)
    assert set(config.PROFILE_FILE_BY_LANGUAGE) == set(config.AUTHORS_BY_LANGUAGE)


def test_bad_language_mode_is_rejected():
    with pytest.raises(ValueError, match="LANGUAGE_MODE"):
        config._check_language_mode("ua")


def test_every_author_has_a_label():
    for authors in config.AUTHORS_BY_LANGUAGE.values():
        for name in authors:
            assert name in config.AUTHOR_LABELS, f"нет подписи для {name!r}"


def test_upload_limit_matches_streamlit_config():
    """config.MAX_UPLOAD_MB и server.maxUploadSize должны совпадать."""
    import os
    import re
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        '.streamlit', 'config.toml')
    text = open(path, encoding='utf-8').read()
    match = re.search(r'^\s*maxUploadSize\s*=\s*(\d+)', text, re.M)
    assert match, "maxUploadSize не задан в .streamlit/config.toml"
    assert int(match.group(1)) == config.MAX_UPLOAD_MB
