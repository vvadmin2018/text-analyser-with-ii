# visualizer.py
"""
Единая визуальная тема GhostQuill / TH-ink-ING: "бумага и чернила".

Идея оформления: раз система измеряет чернила мысли (стиль автора), то и
графики должны выглядеть как записи в блокноте — тёплый бумажный фон,
тонкая карандашная сетка, и у каждого автора — свой цвет чернил.
Анонимный текст всегда рисуется отдельно: почти чёрные чернила с
акцентом цвета сургучной печати (маркеры), чтобы сразу читался как
"проверяемая" линия на фоне остальных.

Стиль настраивается один раз через rcParams (см. _apply_style) и
применяется одинаково во всех графиках модуля, так что вся отчётность
проекта (веб-приложение и результаты main.py) выглядит согласованно.
"""
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from src import config

# ============================================================
# Единая палитра: "бумага и чернила"
# ============================================================

PAPER_BG = '#FBF7EE'        # цвет полотна графика ("лист бумаги")
FIGURE_BG = '#FFFFFF'       # цвет фигуры вокруг полотна
GRID_COLOR = '#DCD2B8'      # тонкие "карандашные" линии сетки/рамки
INK_TEXT = '#2A231C'        # основной текст — тёмно-коричневые чернила
INK_TEXT_SOFT = '#7A6E5C'   # приглушённый текст (подписи, подзаголовки)
ANON_COLOR = '#171310'      # линия анонимного текста — почти чёрные чернила
ANON_ACCENT = '#C99A2E'     # акцент "сургучной печати" (маркеры анонима)

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

_STYLE_READY = False


def _apply_style():
    """Настраивает rcParams matplotlib под тему проекта. Выполняется один
    раз при импорте модуля — дальше все графики наследуют этот стиль."""
    global _STYLE_READY
    if _STYLE_READY:
        return
    plt.rcParams.update({
        'figure.facecolor': FIGURE_BG,
        'savefig.facecolor': FIGURE_BG,
        'axes.facecolor': PAPER_BG,
        'axes.edgecolor': GRID_COLOR,
        'axes.labelcolor': INK_TEXT,
        'text.color': INK_TEXT,
        'xtick.color': INK_TEXT_SOFT,
        'ytick.color': INK_TEXT_SOFT,
        'font.family': 'DejaVu Sans',
        'font.size': 10.5,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.grid': True,
        'grid.color': GRID_COLOR,
        'grid.linestyle': '--',
        'grid.linewidth': 0.8,
        'grid.alpha': 0.7,
        'legend.frameon': True,
        'legend.framealpha': 0.95,
        'legend.facecolor': FIGURE_BG,
        'legend.edgecolor': GRID_COLOR,
        'legend.fancybox': True,
    })
    _STYLE_READY = True


_apply_style()


def _author_color(name, idx=0):
    """Цвет "чернил" автора: сперва config.AUTHOR_COLORS, иначе — запасная
    палитра по индексу (чтобы у новых/безымянных авторов тоже был
    осмысленный, а не случайный цвет)."""
    color = config.AUTHOR_COLORS.get(name)
    if color:
        return color
    return FALLBACK_PALETTE[idx % len(FALLBACK_PALETTE)]


def _display_name(name):
    """Человекочитаемое имя автора для подписей/легенды."""
    return config.AUTHOR_LABELS.get(name, name)


def _style_spines(ax):
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
        spine.set_linewidth(1.1)


def _glow(linewidth=4.0, color=None):
    """Path effect: мягкий "ореол" цвета бумаги под линией/маркером, чтобы
    линия читалась поверх сетки и других пересекающихся линий."""
    return [pe.Stroke(linewidth=linewidth, foreground=color or PAPER_BG),
            pe.Normal()]


def _paper_shadow(offset=(1.6, -1.8), alpha=0.20):
    """Path effect: мягкая тень, как от наклеенного на бумагу вырезанного
    столбика/сектора."""
    return [pe.SimplePatchShadow(offset=offset, alpha=alpha, shadow_rgbFace='#000000'),
            pe.Normal()]


class StyleRose:
    """Строит стилевую розу ветров и сопутствующие графики анализа."""

    @staticmethod
    def normalize_by_max(authors_profiles, anonymous_vector):
        """
        Нормализует КАЖДЫЙ ПРИЗНАК отдельно делением на максимальное значение
        """
        n_features = len(anonymous_vector)

        # Собираем все значения для каждого признака
        feature_values = [[] for _ in range(n_features)]

        for values in authors_profiles.values():
            for i, val in enumerate(values):
                feature_values[i].append(val)

        for i, val in enumerate(anonymous_vector):
            feature_values[i].append(val)

        print("\n📊 Статистика признаков (нормализация по максимуму):")
        for i in range(n_features):
            max_val = max(feature_values[i])
            if config.LEVEL_LOG == "DEBUG":
                print(f"  Признак {i}: максимум = {max_val:.4f}")

        # Нормализуем делением на максимум
        normalized_profiles = {}

        for name, values in authors_profiles.items():
            norm_values = []
            for i, val in enumerate(values):
                max_val = max(feature_values[i])
                if max_val > 0:
                    norm_val = val / max_val
                else:
                    norm_val = 0
                norm_values.append(norm_val)

            normalized_profiles[name] = norm_values

        # Нормализуем анонимный вектор
        normalized_anon = []
        for i, val in enumerate(anonymous_vector):
            max_val = max(feature_values[i])
            if max_val > 0:
                norm_val = val / max_val
            else:
                norm_val = 0
            normalized_anon.append(norm_val)

        return normalized_profiles, normalized_anon

    @staticmethod
    def plot_max_normalized(authors_profiles, anonymous_vector, feature_names,
                            title="Стилевая роза ветров", figsize=(14, 10)):
        """
        Строит розу ветров с нормализацией по максимальному значению
        """
        norm_profiles, norm_anon = StyleRose.normalize_by_max(
            authors_profiles, anonymous_vector
        )

        n_features = len(feature_names)
        angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
        plot_angles = angles + [angles[0]]

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='polar')
        ax.set_facecolor(PAPER_BG)

        for idx, (author_name, profile_values) in enumerate(norm_profiles.items()):
            color = _author_color(author_name, idx)
            plot_values = profile_values + [profile_values[0]]

            line, = ax.plot(plot_angles, plot_values, 'o-', linewidth=2.5,
                             color=color, label=_display_name(author_name), markersize=7)
            line.set_path_effects(_glow(3.6))
            ax.fill(plot_angles, plot_values, alpha=0.18, color=color)

        plot_anon = norm_anon + [norm_anon[0]]
        anon_line, = ax.plot(plot_angles, plot_anon, 'o-', linewidth=3.4,
                              color=ANON_COLOR, label='Анонимный текст',
                              markersize=9, markerfacecolor=ANON_ACCENT,
                              markeredgecolor=ANON_COLOR, markeredgewidth=1.6,
                              zorder=6)
        anon_line.set_path_effects(_glow(5.0))

        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8.5,
                            color=INK_TEXT_SOFT)
        ax.grid(True, linestyle='--', alpha=0.6, color=GRID_COLOR)
        ax.set_xticks(angles)
        ax.set_xticklabels(feature_names, size=9.5, fontweight='bold', color=INK_TEXT)

        plt.title(title + "\n(каждый признак нормирован на свой максимум)",
                  size=13, fontweight='bold', pad=22, color=INK_TEXT)

        plt.legend(loc='upper right', bbox_to_anchor=(1.32, 1.08),
                   frameon=True, fancybox=True, fontsize=9.5)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_fuzzy_rose(all_authors_ranges, anon_features, feature_names,
                        authors_to_plot=None, author_colors=None,
                        title="Роза стилевых признаков", figsize=(11, 9)):
        """
        Строит "розу ветров": линия автора = его типичное значение (b),
        закрашенная полоса = реальный диапазон [a, c] треугольной функции
        принадлежности (никакой искусственной "дисперсии" — это буквально
        минимум и максимум, увиденные в обучающих текстах автора), плюс
        отдельная линия анонимного текста поверх.

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
            vals = [anon_features[i]] if i < len(anon_features) else [0.0]
            for ranges in all_authors_ranges.values():
                if i < len(ranges):
                    a, b, c = ranges[i]
                    vals.append(a)
                    vals.append(c)
            global_min.append(min(vals))
            global_max.append(max(vals))

        def norm(x, i):
            span = global_max[i] - global_min[i]
            if span <= 0:
                return 0.5
            return (x - global_min[i]) / span

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='polar')
        ax.set_facecolor(PAPER_BG)

        # ---- Мягкие концентрические "кольца бумаги" для глубины ----
        # Чисто декоративный фон: чередующиеся едва заметные кольца между
        # линиями сетки 0-0.2, 0.4-0.6, 0.8-1.0, как разметка блокнота.
        theta_full = np.linspace(0, 2 * np.pi, 200)
        for r0, r1 in [(0.0, 0.2), (0.4, 0.6), (0.8, 1.0)]:
            ax.fill_between(theta_full, r0, r1, color=GRID_COLOR, alpha=0.14, zorder=0)

        default_colors = FALLBACK_PALETTE
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
                     linewidth=0.9, zorder=2)
            ax.plot(plot_angles, upper_band, '-', color=color, alpha=0.55,
                     linewidth=0.9, zorder=2)

            b_plot = b_norm + [b_norm[0]]
            b_line, = ax.plot(plot_angles, b_plot, 'o-', linewidth=2.4, color=color,
                               label=disp_name, markersize=6.5, zorder=4)
            b_line.set_path_effects(_glow(4.0))

        # ===== Анонимный текст =====
        anon_norm = [norm(anon_features[i], i) for i in range(n_features)]
        anon_plot = anon_norm + [anon_norm[0]]
        anon_line, = ax.plot(plot_angles, anon_plot, 'D-', linewidth=3.4, color=ANON_COLOR,
                              label='Анонимный текст', markersize=8.5,
                              markerfacecolor=ANON_ACCENT, markeredgecolor=ANON_COLOR,
                              markeredgewidth=1.5, zorder=6)
        anon_line.set_path_effects(_glow(5.4))

        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8.5,
                            color=INK_TEXT_SOFT)
        ax.set_xticks(angles)
        ax.set_xticklabels(feature_names, size=9.5, fontweight='bold', color=INK_TEXT)
        ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.55, color=GRID_COLOR, zorder=1)
        _style_spines(ax)

        plt.title(title + "\n(закрашенная полоса — реальный диапазон [a, c] автора)",
                  size=12.5, fontweight='bold', pad=24, color=INK_TEXT)

        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.22),
                  ncol=3, fontsize=8.8, framealpha=0.95)
        ax.set_aspect('equal', adjustable='box', anchor='C')

        plt.tight_layout()
        return fig

    # ============== ДОПОЛНИТЕЛЬНЫЕ ГРАФИКИ ==============

    @staticmethod
    def plot_feature_importance(profile_name, similarities, weights, contributions,
                                feature_names, title=None):
        """
        Визуализирует вклад каждого признака в итоговое сходство
        """
        n_features = len(feature_names)
        x = np.arange(n_features)
        width = 0.35

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        for ax in (ax1, ax2):
            ax.set_facecolor(PAPER_BG)
            _style_spines(ax)

        color_mu = _author_color('lermontov', 1)      # синие чернила
        color_contrib = _author_color('saharnov', 6)   # тёплый акцент

        bars1 = ax1.bar(x - width / 2, similarities, width, label='μ (принадлежность)',
                         color=color_mu, edgecolor=INK_TEXT, linewidth=0.8, alpha=0.9)
        bars2 = ax1.bar(x + width / 2, contributions, width, label='Вклад (μ × вес)',
                         color=color_contrib, edgecolor=INK_TEXT, linewidth=0.8, alpha=0.9)
        for b in list(bars1) + list(bars2):
            b.set_path_effects(_paper_shadow())

        ax1.set_xlabel('Признаки', fontsize=12, color=INK_TEXT)
        ax1.set_ylabel('Значение', fontsize=12, color=INK_TEXT)
        ax1.set_title(f'{_display_name(profile_name)}: принадлежность и вклад признаков',
                      fontsize=13.5, fontweight='bold', color=INK_TEXT)
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
                     ha='center', va='bottom', fontsize=8, color=INK_TEXT)
            ax1.text(i + width / 2, contrib + 0.02, fmt_value(contrib),
                     ha='center', va='bottom', fontsize=8, color=INK_TEXT)

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

        donut_colors = (FALLBACK_PALETTE[:top_n] + ['#B9AE96'])[:len(plot_contribs)]

        wedges, texts, autotexts = ax2.pie(
            plot_contribs, labels=plot_names, colors=donut_colors,
            autopct='%1.1f%%', startangle=90, pctdistance=0.78,
            textprops={'fontsize': 9.5, 'color': INK_TEXT},
            wedgeprops={'width': 0.42, 'edgecolor': FIGURE_BG, 'linewidth': 2.2}
        )
        for at in autotexts:
            at.set_color(FIGURE_BG)
            at.set_fontweight('bold')
            at.set_fontsize(9)
        for i, text in enumerate(texts[:min(top_n, len(texts))]):
            text.set_fontweight('bold')

        # Подпись в центре "бублика" — самый весомый признак
        ax2.text(0, 0, sorted_names[0], ha='center', va='center',
                 fontsize=10.5, fontweight='bold', color=INK_TEXT)

        ax2.set_title(f'{_display_name(profile_name)}: распределение вклада признаков\n(топ-5 признаков)',
                      fontsize=13.5, fontweight='bold', color=INK_TEXT)

        plt.suptitle(title or f'Анализ сходства: {_display_name(profile_name)}',
                     fontsize=15.5, fontweight='bold', y=1.03, color=INK_TEXT)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_authors_comparison(results_dict, title="Сравнение авторов", figsize=(9, 6.5)):
        """
        Строит сравнительную диаграмму для всех авторов
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor(PAPER_BG)
        _style_spines(ax)

        authors = list(results_dict.keys())
        authors_disp = [_display_name(a) for a in authors]
        scores = list(results_dict.values())

        colors = [_author_color(a, i) for i, a in enumerate(authors)]

        bars = ax.bar(authors_disp, scores, color=colors, edgecolor=INK_TEXT,
                       linewidth=1.2, alpha=0.92, width=0.6)
        for b in bars:
            b.set_path_effects(_paper_shadow())

        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                    f'{score:.1%}\n({score:.2f})',
                    ha='center', va='bottom', fontweight='bold', fontsize=11,
                    color=INK_TEXT)

        ax.set_ylim(0, 1)
        ax.set_ylabel('Уверенность', fontsize=12, fontweight='bold', color=INK_TEXT)
        ax.set_title(title, fontsize=15.5, fontweight='bold', pad=20, color=INK_TEXT)
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        ax.set_axisbelow(True)

        # Цветные зоны уверенности (согласованы с порогами раскраски столбцов выше)
        ax.axhspan(0, 0.3, color=CONF_LOW, alpha=0.06, zorder=0)
        ax.axhspan(0.3, 0.6, color=CONF_MID, alpha=0.06, zorder=0)
        ax.axhspan(0.6, 1.0, color=CONF_HIGH, alpha=0.06, zorder=0)

        ax.legend(bars, [_display_name(a) for a in authors],
                  loc='lower center', bbox_to_anchor=(0.5, -0.22),
                  ncol=len(authors), fontsize=9, framealpha=0.9)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_feature_heatmap(profiles_data, feature_names, title="Сравнение признаков авторов"):
        """
        Строит тепловую карту признаков для всех авторов
        """
        authors = list(profiles_data.keys())
        authors_disp = [_display_name(a) for a in authors]
        n_authors = len(authors)
        n_features = len(feature_names)

        data = np.zeros((n_authors, n_features))
        for i, author in enumerate(authors):
            data[i, :] = profiles_data[author]

        fig, ax = plt.subplots(figsize=(14, max(4, n_authors * 0.9 + 1.5)))
        ax.set_facecolor(PAPER_BG)

        # Тёплая "чернильная" диаграммная шкала: бордовые чернила (низкое
        # значение) → бумага (среднее) → изумрудные чернила (высокое) —
        # согласовано по духу с остальной палитрой темы, вместо
        # стандартной сине-красной цветовой схемы.
        ink_cmap = LinearSegmentedColormap.from_list(
            'ink_diverging', ['#7B1E3D', '#F3ECD8', '#2F5233'], N=200
        )

        mesh = ax.pcolormesh(data, cmap=ink_cmap, vmin=0, vmax=1,
                              edgecolors=FIGURE_BG, linewidth=2.2)

        ax.set_xticks(np.arange(n_features) + 0.5)
        ax.set_yticks(np.arange(n_authors) + 0.5)
        ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(authors_disp, fontsize=11, fontweight='bold')
        ax.invert_yaxis()
        ax.tick_params(length=0)

        for i in range(n_authors):
            for j in range(n_features):
                value = data[i, j]
                color = FIGURE_BG if (value > 0.72 or value < 0.28) else INK_TEXT
                ax.text(j + 0.5, i + 0.5, f'{value:.2f}', ha='center', va='center',
                        color=color, fontweight='bold', fontsize=9)

        ax.set_title(title, fontsize=15.5, fontweight='bold', pad=20, color=INK_TEXT)
        ax.set_xlabel('Признаки', fontsize=12, color=INK_TEXT)
        ax.set_ylabel('Авторы', fontsize=12, color=INK_TEXT)

        cbar = plt.colorbar(mesh, ax=ax, label='Значение (нормализованное)', pad=0.02)
        cbar.outline.set_edgecolor(GRID_COLOR)

        plt.tight_layout()
        return fig
