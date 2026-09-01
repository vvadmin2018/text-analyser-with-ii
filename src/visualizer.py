# visualizer.py
"""
Единая визуальная тема TH-ink-ING: "бумага и чернила".

Идея оформления: раз система измеряет чернила мысли (стиль автора), то и
графики должны выглядеть как записи в блокноте — тёплый бумажный фон,
тонкая карандашная сетка, и у каждого автора — свой цвет чернил.
Анонимный текст всегда рисуется отдельно: почти чёрные чернила с
акцентом цвета сургучной печати (маркеры), чтобы сразу читался как
"проверяемая" линия на фоне остальных.

Тема существует в двух вариантах — светлом ("бумага") и тёмном ("чернила на
грифельной доске"). Веб-приложение переключает их вместе со своей темой, иначе
белые графики били по глазам на тёмном фоне. Переключение — через use_theme();
все функции модуля читают активную тему в момент отрисовки.
"""
import logging
from dataclasses import dataclass

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from src import config

logger = logging.getLogger(__name__)


# ============================================================
# Палитры: "бумага и чернила" в светлом и тёмном варианте
# ============================================================

@dataclass(frozen=True)
class Theme:
    """Набор цветов одной темы графиков."""
    paper_bg: str        # цвет полотна графика ("лист бумаги")
    figure_bg: str       # цвет фигуры вокруг полотна
    grid: str            # тонкие "карандашные" линии сетки/рамки
    ink_text: str        # основной текст
    ink_text_soft: str   # приглушённый текст (подписи, подзаголовки)
    anon_color: str      # линия анонимного текста
    anon_accent: str     # акцент "сургучной печати" (маркеры анонима)
    series_primary: str  # служебный ряд №1 на столбчатых графиках
    series_second: str   # служебный ряд №2
    neutral: str         # нейтральная заливка ("Остальные" в донате)
    heatmap: tuple       # низ → середина → верх тепловой карты
    lighten_authors: bool  # осветлять ли "чернила" авторов под фон

    @property
    def is_dark(self):
        return self.lighten_authors


PAPER_THEME = Theme(
    paper_bg='#FBF7EE',
    figure_bg='#FFFFFF',
    grid='#DCD2B8',
    ink_text='#2A231C',
    ink_text_soft='#7A6E5C',
    anon_color='#FF6A00',
    anon_accent='#C99A2E',
    series_primary='#1B3B6F',
    series_second='#A63A1F',
    neutral='#B9AE96',
    heatmap=('#7B1E3D', '#F3ECD8', '#2F5233'),
    lighten_authors=False,
)

DARK_THEME = Theme(
    paper_bg='#1A1D24',
    figure_bg='#0E1117',
    grid='#3A3F4B',
    ink_text='#E8E3D8',
    ink_text_soft='#A2998A',
    anon_color='#FF8A3D',
    anon_accent='#FFD166',
    series_primary='#7FB2E5',
    series_second='#E8825C',
    neutral='#6C6558',
    heatmap=('#C2566F', '#2B2F38', '#6FBF7F'),
    lighten_authors=True,
)

# Обратная совместимость: модульные константы светлой темы. На них по-прежнему
# опирается HTML сводной таблицы (src/report.py) — она всегда "лист бумаги".
PAPER_BG = PAPER_THEME.paper_bg
FIGURE_BG = PAPER_THEME.figure_bg
GRID_COLOR = PAPER_THEME.grid
INK_TEXT = PAPER_THEME.ink_text
INK_TEXT_SOFT = PAPER_THEME.ink_text_soft
ANON_COLOR = PAPER_THEME.anon_color
ANON_ACCENT = PAPER_THEME.anon_accent

# Тёплая диаграммная палитра "оттенков чернил" — используется как запасная,
# если для автора нет цвета в config.AUTHOR_COLORS.
FALLBACK_PALETTE = [
    '#7B1E3D',  # бордовые чернила
    '#1B3B6F',  # тёмно-синие чернила
    '#2F5233',  # изумрудные чернила
    '#6B4226',  # сепия
    '#4A2545',  # чернила "баклажан"
    '#0F5C5C',  # бирюзовые чернила
    '#8A5A00',  # янтарные чернила
    '#39476B',  # грифельные чернила
]

# Согласованные цвета зон уверенности (мягче, чем стандартный
# светофор Bootstrap, но сохраняют интуитивную семантику
# "красный/жёлтый/зелёный").
CONF_HIGH = '#2F5233'
CONF_MID = '#8A5A00'
CONF_LOW = '#7B1E3D'

_active_theme = PAPER_THEME


def current_theme():
    """Активная тема графиков."""
    return _active_theme


def use_theme(dark=False):
    """Переключает тему и настраивает rcParams matplotlib под неё."""
    global _active_theme
    theme = DARK_THEME if dark else PAPER_THEME
    _active_theme = theme
    plt.rcParams.update({
        'figure.facecolor': theme.figure_bg,
        'savefig.facecolor': theme.figure_bg,
        'axes.facecolor': theme.paper_bg,
        'axes.edgecolor': theme.grid,
        'axes.labelcolor': theme.ink_text,
        'text.color': theme.ink_text,
        'xtick.color': theme.ink_text_soft,
        'ytick.color': theme.ink_text_soft,
        'font.family': 'DejaVu Sans',
        'font.size': 10.5,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.grid': True,
        'grid.color': theme.grid,
        'grid.linestyle': '--',
        'grid.linewidth': 0.8,
        'grid.alpha': 0.7,
        'legend.frameon': True,
        'legend.framealpha': 0.95,
        'legend.facecolor': theme.figure_bg,
        'legend.edgecolor': theme.grid,
        'legend.fancybox': True,
    })
    return theme


use_theme(dark=False)


def _lighten(hex_color, amount=0.45):
    """Подмешивает белый к цвету. Нужно в тёмной теме: "чернила" авторов
    (#4A2545, #0F5C5C и т.п.) выбраны под бумагу и на тёмном фоне просто
    сливаются с ним."""
    hex_color = hex_color.lstrip('#')
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (int(round(c + (255 - c) * amount)) for c in (r, g, b))
    return f'#{r:02X}{g:02X}{b:02X}'


def _author_color(name, idx=0):
    """Цвет "чернил" автора: сперва config.AUTHOR_COLORS, иначе — запасная
    палитра по индексу (чтобы у новых/безымянных авторов тоже был
    осмысленный, а не случайный цвет). В тёмной теме цвет осветляется."""
    color = config.AUTHOR_COLORS.get(name) or FALLBACK_PALETTE[idx % len(FALLBACK_PALETTE)]
    return _lighten(color) if _active_theme.lighten_authors else color


def _display_name(name):
    """Человекочитаемое имя автора для подписей/легенды."""
    return config.AUTHOR_LABELS.get(name, name)


def _style_spines(ax):
    for spine in ax.spines.values():
        spine.set_color(_active_theme.grid)
        spine.set_linewidth(1.1)


def _glow(linewidth=4.0, color=None):
    """Path effect: мягкий "ореол" цвета бумаги под линией/маркером, чтобы
    линия читалась поверх сетки и других пересекающихся линий."""
    return [pe.Stroke(linewidth=linewidth, foreground=color or _active_theme.paper_bg),
            pe.Normal()]


def _paper_shadow(offset=(1.6, -1.8), alpha=0.20):
    """Path effect: мягкая тень, как от наклеенного на бумагу вырезанного
    столбика/сектора."""
    return [pe.SimplePatchShadow(offset=offset, alpha=alpha, shadow_rgbFace='#000000'),
            pe.Normal()]


class StyleRose:
    """Строит стилевую розу ветров и сопутствующие графики анализа."""

    @staticmethod
    def plot_fuzzy_rose(all_authors_ranges, anon_features, feature_names,
                        authors_to_plot=None, author_colors=None,
                        title="Роза стилевых признаков", figsize=(11, 9)):
        """
        Строит "розу ветров": линия автора = его типичное значение (b),
        закрашенная полоса = диапазон [a, c] треугольной функции
        принадлежности, плюс отдельная линия анонимного текста поверх.

        Нормализация вычисляется по ЕДИНОЙ шкале — минимуму/максимуму
        среди [a, c] ВСЕХ обученных авторов (плюс сам анонимный текст),
        поэтому шкала одной и той же оси не меняется от графика к графику:
        единичный автор, пара с анонимом или "все авторы сразу" — везде
        один и тот же масштаб.

        Args:
            all_authors_ranges: {имя_автора: [(a, b, c), ...]} — диапазоны
                ВСЕХ обученных авторов (даже если рисуем не всех — они
                нужны для вычисления единой шкалы нормализации).
            anon_features: сырой вектор признаков анонимного текста
                (numpy array или список длиной len(feature_names)).
            feature_names: подписи осей.
            authors_to_plot: какие авторы реально рисуются на графике
                (по умолчанию — все из all_authors_ranges).
            author_colors: {имя_автора: hex-цвет}.
        """
        theme = _active_theme
        n_features = len(feature_names)
        angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
        plot_angles = angles + [angles[0]]

        if authors_to_plot is None:
            authors_to_plot = list(all_authors_ranges.keys())

        anon_features = list(anon_features)

        # ===== Валидация входных данных =====
        # Явные, понятные ошибки вместо голого IndexError/KeyError без
        # контекста — это легко может случиться, если, например, "автор.pkl"
        # был сохранён более старой версией кода с другим числом признаков.
        if len(anon_features) != n_features:
            raise ValueError(
                f"plot_fuzzy_rose: длина anon_features ({len(anon_features)}) "
                f"не совпадает с числом осей feature_names ({n_features}). "
                f"Возможно, профили авторов или анонимный текст посчитаны "
                f"устаревшей версией FeatureExtractor — удалите authors_profiles.pkl "
                f"и пересчитайте профили."
            )
        for author_name in authors_to_plot:
            if author_name not in all_authors_ranges:
                raise KeyError(
                    f"plot_fuzzy_rose: автор '{author_name}' не найден в "
                    f"all_authors_ranges (доступны: {list(all_authors_ranges.keys())})."
                )
            n_ranges = len(all_authors_ranges[author_name])
            if n_ranges != n_features:
                raise ValueError(
                    f"plot_fuzzy_rose: у автора '{author_name}' {n_ranges} "
                    f"диапазонов (a,b,c), а осей feature_names — {n_features}. "
                    f"Профиль этого автора, вероятно, посчитан устаревшей версией "
                    f"кода — удалите authors_profiles.pkl и пересчитайте профили."
                )

        # ===== Единая шкала нормализации по каждому признаку =====
        # min/max среди a и c ВСЕХ обученных авторов + сам аноним, чтобы
        # ни автор, ни текст не могли искусственно "упереться" в 1.0.
        global_min, global_max = [], []
        for i in range(n_features):
            vals = [anon_features[i]]
            for ranges in all_authors_ranges.values():
                if i < len(ranges):
                    a, _b, c = ranges[i]
                    vals.append(a)
                    vals.append(c)
            global_min.append(min(vals))
            global_max.append(max(vals))

        def norm(x, i):
            span = global_max[i] - global_min[i]
            if span <= 0:
                return 0.5
            return (x - global_min[i]) / span

        # ===== Масштаб под фактический размер фигуры =====
        # Толщины, кегли и отступ легенды были подобраны под figsize=(11, 9) и
        # заданы в пунктах, то есть в абсолютных единицах. При уменьшении
        # фигуры оси сжимались, а маркеры и подписи — нет, поэтому на 5.5x5.5
        # маркеры перекрывали друг друга, а легенда наезжала на график.
        # Всё, что раньше было константой, теперь домножается на scale.
        base_w, base_h = 11.0, 9.0
        scale = min(figsize[0] / base_w, figsize[1] / base_h)
        # Ниже ~0.55 подписи осей становятся нечитаемыми быстрее, чем график
        # успевает выиграть в компактности, поэтому кегли снижаем мягче линий.
        text_scale = max(0.62, scale ** 0.6)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='polar')
        ax.set_facecolor(theme.paper_bg)

        # ---- Мягкие концентрические "кольца бумаги" для глубины ----
        # Чисто декоративный фон: чередующиеся едва заметные кольца между
        # линиями сетки 0-0.2, 0.4-0.6, 0.8-1.0, как разметка блокнота.
        theta_full = np.linspace(0, 2 * np.pi, 200)
        for r0, r1 in [(0.0, 0.2), (0.4, 0.6), (0.8, 1.0)]:
            ax.fill_between(theta_full, r0, r1, color=theme.grid, alpha=0.14, zorder=0)

        if author_colors is None:
            author_colors = {}

        for idx, author_name in enumerate(authors_to_plot):
            ranges = all_authors_ranges[author_name]
            color = author_colors.get(author_name) or _author_color(author_name, idx)

            b_norm = [norm(ranges[i][1], i) for i in range(n_features)]
            a_norm = [norm(ranges[i][0], i) for i in range(n_features)]
            c_norm = [norm(ranges[i][2], i) for i in range(n_features)]

            # Реальная полоса неопределённости = буквально [a, c] после
            # нормализации (a и c поменяться местами не могут, т.к.
            # a <= b <= c по построению TriangularMembership).
            lower_band = a_norm + [a_norm[0]]
            upper_band = c_norm + [c_norm[0]]
            disp_name = _display_name(author_name)
            ax.fill_between(plot_angles, lower_band, upper_band, color=color,
                            alpha=0.22, zorder=2,
                            label=f'{disp_name} (диапазон a…c)')
            # Тонкий контур границ полосы — иначе край диапазона на бумажном
            # фоне теряется, особенно когда полосы разных авторов пересекаются.
            ax.plot(plot_angles, lower_band, '-', color=color, alpha=0.55,
                    linewidth=0.9 * scale, zorder=2)
            ax.plot(plot_angles, upper_band, '-', color=color, alpha=0.55,
                    linewidth=0.9 * scale, zorder=2)

            b_plot = b_norm + [b_norm[0]]
            b_line, = ax.plot(plot_angles, b_plot, 'o-', linewidth=2.4 * scale,
                              color=color, label=disp_name,
                              markersize=6.5 * scale, zorder=4)
            b_line.set_path_effects(_glow(4.0 * scale))

        # ===== Анонимный текст =====
        anon_norm = [norm(anon_features[i], i) for i in range(n_features)]
        anon_plot = anon_norm + [anon_norm[0]]
        anon_line, = ax.plot(plot_angles, anon_plot, 'D-', linewidth=3.4 * scale,
                             color=theme.anon_color, label='Анонимный текст',
                             markersize=8.5 * scale, markerfacecolor=theme.anon_accent,
                             markeredgecolor=theme.anon_color,
                             markeredgewidth=1.5 * scale, zorder=6)
        anon_line.set_path_effects(_glow(5.4 * scale))

        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'],
                           fontsize=8.5 * text_scale, color=theme.ink_text_soft)
        ax.set_xticks(angles)
        ax.set_xticklabels(feature_names, size=9.5 * text_scale, fontweight='bold',
                           color=theme.ink_text)
        ax.grid(True, linestyle='--', linewidth=0.8 * scale, alpha=0.55,
                color=theme.grid, zorder=1)
        _style_spines(ax)

        plt.title(title + "\n(закрашенная полоса — диапазон [a, c] автора)",
                  size=12.5 * text_scale, fontweight='bold', pad=24 * scale,
                  color=theme.ink_text)

        # Отступ под легенду считается по её фактическому размеру, а не по
        # константе. Раньше здесь стояло bbox_to_anchor=(0.5, -0.22) —
        # значение, подобранное под три строки при figsize=(11, 9). Число
        # строк зависит от количества авторов (каждый даёт две записи: линию и
        # диапазон), а доля высоты, которую занимает строка, — от размера
        # фигуры, поэтому фиксированный отступ верен ровно в одном случае.
        n_entries = 2 * len(authors_to_plot) + 1
        ncol = 1 if n_entries <= 2 else (2 if n_entries <= 4 else 3)
        legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                           ncol=ncol, fontsize=8.8 * text_scale, framealpha=0.95)
        ax.set_aspect('equal', adjustable='box', anchor='C')

        plt.tight_layout()

        # tight_layout не учитывает легенду, вынесенную за пределы осей: она
        # либо обрезается краем холста, либо накладывается на оси. Меряем её
        # после отрисовки и резервируем ровно столько места, сколько нужно.
        fig.canvas.draw()
        legend_h = (legend.get_window_extent()
                    .transformed(fig.transFigure.inverted()).height)
        fig.subplots_adjust(bottom=min(0.45, legend_h + 0.06))
        return fig

    # ============== ДОПОЛНИТЕЛЬНЫЕ ГРАФИКИ ==============

    @staticmethod
    def plot_feature_importance(profile_name, similarities, weights, contributions,
                                feature_names, title=None, figsize=(16, 6)):
        """
        Визуализирует вклад каждого признака в итоговое сходство
        """
        theme = _active_theme
        n_features = len(feature_names)
        x = np.arange(n_features)
        width = 0.35

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        for ax in (ax1, ax2):
            ax.set_facecolor(theme.paper_bg)
            _style_spines(ax)

        bars1 = ax1.bar(x - width / 2, similarities, width, label='μ (принадлежность)',
                        color=theme.series_primary, edgecolor=theme.ink_text,
                        linewidth=0.8, alpha=0.9)
        bars2 = ax1.bar(x + width / 2, contributions, width, label='Вклад (μ × вес)',
                        color=theme.series_second, edgecolor=theme.ink_text,
                        linewidth=0.8, alpha=0.9)
        for b in list(bars1) + list(bars2):
            b.set_path_effects(_paper_shadow())

        ax1.set_xlabel('Признаки', fontsize=12, color=theme.ink_text)
        ax1.set_ylabel('Значение', fontsize=12, color=theme.ink_text)
        ax1.set_title(f'{_display_name(profile_name)}: принадлежность и вклад признаков',
                      fontsize=13.5, fontweight='bold', color=theme.ink_text)
        ax1.set_xticks(x)
        ax1.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=10)
        ax1.legend(fontsize=9.5)
        ax1.grid(True, alpha=0.4, axis='y')
        ax1.set_ylim(0, max(max(similarities), max(contributions)) * 1.2)

        # Добавляем значения на столбцы. Для очень маленьких значений (текст
        # далеко за пределами диапазона автора по этому признаку) двух знаков
        # после запятой недостаточно — они все схлопываются в одинаковый
        # "0.00", будто это ровно ноль. Показываем больше знаков для таких
        # случаев, чтобы было видно, что значения разные (просто маленькие).
        def fmt_value(v):
            return f'{v:.2f}' if v >= 0.01 else f'{v:.4f}'

        for i, (sim, contrib) in enumerate(zip(similarities, contributions)):
            ax1.text(i - width / 2, sim + 0.02, fmt_value(sim),
                     ha='center', va='bottom', fontsize=8, color=theme.ink_text)
            ax1.text(i + width / 2, contrib + 0.02, fmt_value(contrib),
                     ha='center', va='bottom', fontsize=8, color=theme.ink_text)

        # График 2: кольцевая (donut) диаграмма вкладов — как оттиск
        # сургучной печати, разделённый на дольки
        sorted_indices = np.argsort(contributions)[::-1]
        sorted_contribs = [contributions[i] for i in sorted_indices]
        sorted_names = [feature_names[i] for i in sorted_indices]

        top_n = 5
        other_contrib = sum(sorted_contribs[top_n:]) if len(sorted_contribs) > top_n else 0

        if other_contrib > 0:
            plot_contribs = sorted_contribs[:top_n] + [other_contrib]
            plot_names = sorted_names[:top_n] + ['Остальные']
        else:
            plot_contribs = sorted_contribs
            plot_names = sorted_names

        palette = [_lighten(c) for c in FALLBACK_PALETTE] if theme.lighten_authors else FALLBACK_PALETTE
        donut_colors = (list(palette[:top_n]) + [theme.neutral])[:len(plot_contribs)]

        wedges, texts, autotexts = ax2.pie(
            plot_contribs, labels=plot_names, colors=donut_colors,
            autopct='%1.1f%%', startangle=90, pctdistance=0.78,
            textprops={'fontsize': 9.5, 'color': theme.ink_text},
            wedgeprops={'width': 0.42, 'edgecolor': theme.figure_bg, 'linewidth': 2.2}
        )
        for at in autotexts:
            at.set_color(theme.figure_bg)
            at.set_fontweight('bold')
            at.set_fontsize(9)
        for text in texts[:min(top_n, len(texts))]:
            text.set_fontweight('bold')

        # Подпись в центре "бублика" — самый весомый признак
        ax2.text(0, 0, sorted_names[0], ha='center', va='center',
                 fontsize=10.5, fontweight='bold', color=theme.ink_text)

        ax2.set_title(f'{_display_name(profile_name)}: распределение вклада признаков\n(топ-5 признаков)',
                      fontsize=13.5, fontweight='bold', color=theme.ink_text)

        plt.suptitle(title or f'Анализ сходства: {_display_name(profile_name)}',
                     fontsize=15.5, fontweight='bold', y=1.03, color=theme.ink_text)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_authors_comparison(results_dict, title="Сравнение авторов", figsize=(9, 6.5)):
        """
        Строит сравнительную диаграмму для всех авторов
        """
        theme = _active_theme
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor(theme.paper_bg)
        _style_spines(ax)

        authors = list(results_dict.keys())
        authors_disp = [_display_name(a) for a in authors]
        scores = list(results_dict.values())

        colors = [_author_color(a, i) for i, a in enumerate(authors)]

        bars = ax.bar(authors_disp, scores, color=colors, edgecolor=theme.ink_text,
                      linewidth=1.2, alpha=0.92, width=0.6)
        for b in bars:
            b.set_path_effects(_paper_shadow())

        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                    f'{score:.1%}\n({score:.2f})',
                    ha='center', va='bottom', fontweight='bold', fontsize=11,
                    color=theme.ink_text)

        ax.set_ylim(0, 1)
        ax.set_ylabel('Уверенность', fontsize=12, fontweight='bold', color=theme.ink_text)
        ax.set_title(title, fontsize=15.5, fontweight='bold', pad=20, color=theme.ink_text)
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        ax.set_axisbelow(True)

        # Цветные зоны уверенности. Границы берутся из config, чтобы фон
        # графика не разъезжался с порогами, по которым выносится вердикт.
        ax.axhspan(0, config.CONFIDENCE_THRESHOLD, color=CONF_LOW, alpha=0.06, zorder=0)
        ax.axhspan(config.CONFIDENCE_THRESHOLD, config.HIGH_CONFIDENCE_THRESHOLD,
                   color=CONF_MID, alpha=0.06, zorder=0)
        ax.axhspan(config.HIGH_CONFIDENCE_THRESHOLD, 1.0, color=CONF_HIGH, alpha=0.06, zorder=0)

        # Легенды здесь нет намеренно: имена авторов уже стоят подписями по
        # оси X, и легенда просто дублировала бы их вторым списком.

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_feature_heatmap(profiles_data, feature_names, title="Сравнение признаков авторов"):
        """
        Строит тепловую карту признаков для всех авторов
        """
        theme = _active_theme
        authors = list(profiles_data.keys())
        authors_disp = [_display_name(a) for a in authors]
        n_authors = len(authors)
        n_features = len(feature_names)

        data = np.zeros((n_authors, n_features))
        for i, author in enumerate(authors):
            data[i, :] = profiles_data[author]

        fig, ax = plt.subplots(figsize=(14, max(4, n_authors * 0.9 + 1.5)))
        ax.set_facecolor(theme.paper_bg)

        # Тёплая "чернильная" диаграммная шкала: бордовые чернила (низкое
        # значение) → бумага (среднее) → изумрудные чернила (высокое) —
        # согласовано по духу с остальной палитрой темы, вместо
        # стандартной сине-красной цветовой схемы.
        ink_cmap = LinearSegmentedColormap.from_list('ink_diverging', list(theme.heatmap), N=200)

        mesh = ax.pcolormesh(data, cmap=ink_cmap, vmin=0, vmax=1,
                             edgecolors=theme.figure_bg, linewidth=2.2)

        ax.set_xticks(np.arange(n_features) + 0.5)
        ax.set_yticks(np.arange(n_authors) + 0.5)
        ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(authors_disp, fontsize=11, fontweight='bold')
        ax.invert_yaxis()
        ax.tick_params(length=0)

        for i in range(n_authors):
            for j in range(n_features):
                value = data[i, j]
                color = theme.figure_bg if (value > 0.72 or value < 0.28) else theme.ink_text
                ax.text(j + 0.5, i + 0.5, f'{value:.2f}', ha='center', va='center',
                        color=color, fontweight='bold', fontsize=9)

        ax.set_title(title, fontsize=15.5, fontweight='bold', pad=20, color=theme.ink_text)
        ax.set_xlabel('Признаки', fontsize=12, color=theme.ink_text)
        ax.set_ylabel('Авторы', fontsize=12, color=theme.ink_text)

        cbar = plt.colorbar(mesh, ax=ax, label='Значение (нормализованное)', pad=0.02)
        cbar.outline.set_edgecolor(theme.grid)

        plt.tight_layout()
        return fig
