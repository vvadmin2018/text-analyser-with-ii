# -*- coding: utf-8 -*-
"""Роза должна оставаться читаемой при любом figsize.

Толщины, кегли и отступ легенды были заданы в пунктах и подобраны под
figsize=(11, 9). При уменьшении фигуры оси сжимались, а маркеры и подписи —
нет: на 5.5x5.5 легенда из семи записей накрывала полярные оси, а маркеры
перекрывали друг друга.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pytest

from src.visualizer import StyleRose

FEATURE_NAMES = ['A', 'B', 'C', 'D', 'E', 'F']

RANGES = {
    'pushkin': [(0.1, 0.5, 0.9)] * len(FEATURE_NAMES),
    'tolstoy': [(0.2, 0.4, 0.8)] * len(FEATURE_NAMES),
    'gogol': [(0.0, 0.6, 1.0)] * len(FEATURE_NAMES),
}
ANON = [0.45] * len(FEATURE_NAMES)


def _build(figsize, authors=None):
    return StyleRose.plot_fuzzy_rose(RANGES, ANON, FEATURE_NAMES,
                                     authors_to_plot=authors, figsize=figsize)


def _boxes(fig):
    """Прямоугольники осей и легенды в долях фигуры."""
    fig.canvas.draw()
    ax = fig.axes[0]
    inv = fig.transFigure.inverted()
    legend = ax.get_legend()
    return (ax.get_window_extent().transformed(inv),
            legend.get_window_extent().transformed(inv))


@pytest.mark.parametrize("figsize", [(11, 9), (7, 7), (5.5, 5.5), (4, 4)])
def test_legend_does_not_cover_axes(figsize):
    fig = _build(figsize)
    try:
        ax_box, legend_box = _boxes(fig)
        assert legend_box.y1 <= ax_box.y0 + 1e-6, (
            f"легенда заходит на оси при figsize={figsize}: "
            f"верх легенды {legend_box.y1:.3f} > низа осей {ax_box.y0:.3f}")
    finally:
        plt.close(fig)


@pytest.mark.parametrize("figsize", [(11, 9), (7, 7), (5.5, 5.5), (4, 4)])
def test_legend_fits_inside_canvas(figsize):
    """Легенда не должна уезжать за нижний край холста."""
    fig = _build(figsize)
    try:
        _, legend_box = _boxes(fig)
        assert legend_box.y0 >= -1e-6, (
            f"легенда обрезается краем холста при figsize={figsize}: "
            f"низ {legend_box.y0:.3f}")
    finally:
        plt.close(fig)


def test_axes_keep_most_of_the_figure():
    """Резерв под легенду не должен съедать график.

    Замер ведётся по факту, поэтому нужен верхний предел: если легенда вдруг
    начнёт занимать половину фигуры, тест это поймает.
    """
    fig = _build((5.5, 5.5))
    try:
        ax_box, _ = _boxes(fig)
        assert ax_box.height > 0.5, (
            f"на оси осталось {ax_box.height:.2f} высоты фигуры")
    finally:
        plt.close(fig)


def test_markers_shrink_with_figure():
    """Маркеры заданы в пунктах, поэтому обязаны масштабироваться сами."""
    big, small = _build((11, 9)), _build((5.5, 5.5))
    try:
        def anon_marker(fig):
            for line in fig.axes[0].get_lines():
                if line.get_label() == 'Анонимный текст':
                    return line.get_markersize()
            raise AssertionError("линия анонимного текста не найдена")

        assert anon_marker(small) < anon_marker(big)
    finally:
        plt.close(big)
        plt.close(small)


def test_single_author_legend_is_narrow():
    """Три записи — не больше двух колонок, иначе легенда шире графика."""
    fig = _build((7, 7), authors=['pushkin'])
    try:
        fig.canvas.draw()
        assert fig.axes[0].get_legend()._ncols <= 2
    finally:
        plt.close(fig)
