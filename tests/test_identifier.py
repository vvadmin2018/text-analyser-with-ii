# -*- coding: utf-8 -*-
"""Сравнение профилей и поведение load_profiles."""
import numpy as np
import pytest

from src import config
from src.identifier import identify, is_confident
from src.profile_builder import AuthorProfile, TriangularMembership


def _profile(name, centers):
    profile = AuthorProfile(name)
    profile.features = [TriangularMembership(c * 0.8, c, c * 1.2) for c in centers]
    return profile


@pytest.fixture
def profiles():
    n = config.N_FEATURES
    return {
        "alpha": _profile("alpha", [1.0] * n),
        "beta": _profile("beta", [10.0] * n),
    }


def test_identifies_closest_profile(profiles):
    best, results, details = identify(profiles, np.array([1.0] * config.N_FEATURES))
    assert best == "alpha"
    assert results["alpha"] > results["beta"]
    assert set(details) == {"alpha", "beta"}


def test_details_shape_matches_features(profiles):
    _best, _results, details = identify(profiles, np.array([1.0] * config.N_FEATURES))
    sims, weights, contribs = details["alpha"]
    assert len(sims) == len(weights) == len(contribs) == config.N_FEATURES
    assert contribs == pytest.approx([s * w for s, w in zip(sims, weights)])


def test_perfect_match_scores_one(profiles):
    _best, results, _ = identify(profiles, np.array([1.0] * config.N_FEATURES))
    assert results["alpha"] == pytest.approx(1.0)


def test_empty_profiles_return_none():
    best, results, details = identify({}, np.zeros(config.N_FEATURES))
    assert best is None and results == {} and details == {}


def test_is_confident_uses_config_threshold():
    assert is_confident(config.CONFIDENCE_THRESHOLD)
    assert not is_confident(config.CONFIDENCE_THRESHOLD - 0.01)
    assert is_confident(0.1, threshold=0.05)


def test_profile_without_features_scores_zero():
    empty = AuthorProfile("empty")
    score, details = empty.similarity_with_details(np.zeros(config.N_FEATURES))
    assert score == 0.0
    assert details == ([], [], [])
