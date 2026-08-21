# -*- coding: utf-8 -*-
"""Свойства треугольной функции принадлежности.

Это те самые свойства, на которых ловился реальный дефект: рампа внутри
диапазона опускалась до 0 ровно на границе, а экспоненциальный хвост снаружи
начинался с 0.001 — значение ЗА границей получало μ выше, чем значение НА
границе, и функция была разрывной и немонотонной.
"""
import numpy as np
import pytest

from src.profile_builder import TriangularMembership


@pytest.fixture
def tri():
    return TriangularMembership(6.0, 9.0, 12.0)


def test_peak_at_b(tri):
    assert tri.mu(tri.b) == 1.0


def test_mu_within_unit_range(tri):
    for x in np.linspace(-50, 100, 1501):
        assert 0.0 <= tri.mu(x) <= 1.0


def test_monotone_on_both_sides(tri):
    """μ не убывает слева от b и не возрастает справа."""
    left = np.linspace(tri.b - 60, tri.b, 4001)
    values = [tri.mu(x) for x in left]
    assert all(a <= b + 1e-12 for a, b in zip(values, values[1:]))

    right = np.linspace(tri.b, tri.b + 60, 4001)
    values = [tri.mu(x) for x in right]
    assert all(a >= b - 1e-12 for a, b in zip(values, values[1:]))


@pytest.mark.parametrize("side", ["left", "right"])
def test_continuous_across_soft_edge(tri, side):
    """Ключевая регрессия: на смягчённой границе не должно быть скачка."""
    edge = (tri.a - tri.softening) if side == "left" else (tri.c + tri.softening)
    eps = 1e-6
    inside = tri.mu(edge + eps) if side == "left" else tri.mu(edge - eps)
    outside = tri.mu(edge - eps) if side == "left" else tri.mu(edge + eps)

    assert tri.mu(edge) == pytest.approx(TriangularMembership.EDGE_MU, rel=1e-3)
    assert outside <= tri.mu(edge) <= inside
    assert abs(inside - outside) < 0.01


def test_outliers_stay_distinguishable(tri):
    """Хвост существует ради этого: чем дальше выброс, тем меньше μ.

    Раньше все значения за границей получали одинаковый "пол" 0.001, и два
    совершенно разных по величине выброса выглядели одинаково.
    """
    far = [tri.mu(x) for x in (20, 50, 100, 500)]
    assert all(a > b for a, b in zip(far, far[1:]))
    assert all(v > 0 for v in far)


def test_inside_range_scores_high(tri):
    """Значения внутри [a, c] должны давать заметно высокую принадлежность."""
    for x in (6.5, 7.5, 9.0, 10.5, 11.5):
        assert tri.mu(x) > 0.5


def test_degenerate_range_does_not_crash():
    """a == c — вырожденный случай (все обучающие тексты дали одно значение)."""
    tri = TriangularMembership(3.0, 3.0, 3.0)
    assert tri.mu(3.0) == 1.0
    for x in (-10, 0, 2.9, 3.1, 100):
        assert 0.0 <= tri.mu(x) <= 1.0


def test_zero_anchored_range():
    """Признаки-доли часто прижаты к нулю (a = 0)."""
    tri = TriangularMembership(0.0, 0.05, 0.1)
    assert tri.mu(0.05) == 1.0
    assert tri.mu(0.0) > 0.0
    assert tri.mu(5.0) < tri.mu(0.5)
