# visualizer.py
import matplotlib.pyplot as plt
import numpy as np
from src import config

class StyleRose:
    """Строит стилевую розу ветров"""

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
        print("\n🔄 Нормализация признаков по максимальному значению...")

        # Нормализуем данные
        norm_profiles, norm_anon = StyleRose.normalize_by_max(
            authors_profiles, anonymous_vector
        )

        # Проверяем нормализацию
        print("\n✅ После нормализации:")
        for name, values in norm_profiles.items():
            if config.LEVEL_LOG == "DEBUG":
                print(f"  {name}: от {min(values):.3f} до {max(values):.3f}")

        if config.LEVEL_LOG == "DEBUG":
            print(f"  Аноним: от {min(norm_anon):.3f} до {max(norm_anon):.3f}")

        # Строим график
        n_features = len(feature_names)
        angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='polar')

        # Яркие цвета
        colors = ['#FF4B4B', '#4B7BFF', '#4BFF4B', '#FFB44B', '#B44BFF']

        # Рисуем каждого автора
        for idx, (author_name, profile_values) in enumerate(norm_profiles.items()):
            color = colors[idx % len(colors)]

            plot_angles = angles + [angles[0]]
            plot_values = profile_values + [profile_values[0]]

            ax.plot(plot_angles, plot_values, 'o-', linewidth=2.5,
                    color=color, label=author_name, markersize=8)
            ax.fill(plot_angles, plot_values, alpha=0.25, color=color)

        # Рисуем анонимный текст
        plot_anon = norm_anon + [norm_anon[0]]
        ax.plot(angles + [angles[0]], plot_anon, 'o-', linewidth=4,
                color='black', label='Анонимный текст',
                markersize=10, markerfacecolor='yellow',
                markeredgecolor='black', markeredgewidth=2)

        # Настройки графика
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'])

        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xticks(angles)
        ax.set_xticklabels(feature_names, size=10, fontweight='bold')
        ax.set_facecolor('#f8f9fa')

        plt.title(title + "\n(каждый признак нормирован на свой максимум)",
                  size=14, fontweight='bold', pad=20)

        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0),
                   frameon=True, fancybox=True, shadow=True, fontsize=10)

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

        Ключевое отличие от исходной версии: нормализация вычисляется по
        ЕДИНОЙ шкале — минимуму/максимуму среди [a, c] ВСЕХ обученных
        авторов (плюс сам анонимный текст), а не только по паре
        "автор vs аноним". Раньше при сравнении всего двух значений один
        из них ГАРАНТИРОВАННО получал 1.0 на каждой оси (val / max(val1,
        val2)) — роза выглядела как уверенное совпадение, даже когда
        реальное сходство было низким. Теперь шкала одной и той же оси не
        меняется от графика к графику: единичный автор, пара с анонимом
        или "все авторы сразу" — везде один и тот же масштаб.

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

        default_colors = ['#FF4B4B', '#4B7BFF', '#4BFF4B', '#FFB44B', '#B44BFF',
                           '#FF4BFF', '#00CED1', '#8B4513', '#2E8B57']
        if author_colors is None:
            author_colors = {}

        for idx, author_name in enumerate(authors_to_plot):
            ranges = all_authors_ranges[author_name]
            color = author_colors.get(author_name, default_colors[idx % len(default_colors)])

            b_norm = [norm(ranges[i][1], i) for i in range(n_features)]
            a_norm = [norm(ranges[i][0], i) for i in range(n_features)]
            c_norm = [norm(ranges[i][2], i) for i in range(n_features)]

            # Реальная полоса неопределённости = буквально [a, c] после
            # нормализации (a и c могут поменяться местами не могут, т.к.
            # a <= b <= c по построению TriangularMembership).
            lower_band = a_norm + [a_norm[0]]
            upper_band = c_norm + [c_norm[0]]
            ax.fill_between(plot_angles, lower_band, upper_band, color=color,
                            alpha=0.18, label=f'{author_name} (диапазон a…c)')

            b_plot = b_norm + [b_norm[0]]
            ax.plot(plot_angles, b_plot, 'o-', linewidth=2.3, color=color,
                    label=author_name, markersize=6)

        # ===== Анонимный текст =====
        anon_norm = [norm(anon_features[i], i) for i in range(n_features)]
        anon_plot = anon_norm + [anon_norm[0]]
        ax.plot(plot_angles, anon_plot, 'o-', linewidth=3.5, color='black',
                label='Анонимный текст', markersize=9,
                markerfacecolor='yellow', markeredgecolor='black',
                markeredgewidth=1.8, zorder=5)

        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.set_xticks(angles)
        ax.set_xticklabels(feature_names, size=10, fontweight='bold')
        ax.set_facecolor('#f8f9fa')
        ax.grid(True, linestyle='--', alpha=0.5)

        plt.title(title + "\n(закрашенная полоса — реальный диапазон [a, c] автора)",
                  size=13, fontweight='bold', pad=20)
        plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1),
                   frameon=True, fancybox=True, shadow=False, fontsize=9,
                   framealpha=0.85)
        plt.tight_layout()
        return fig

    # ============== НОВЫЕ МЕТОДЫ ==============

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

        # График 1: Столбчатая диаграмма μ и вкладов
        ax1.bar(x - width / 2, similarities, width, label='μ (принадлежность)',
                color='skyblue', edgecolor='navy', alpha=0.7)
        ax1.bar(x + width / 2, contributions, width, label='Вклад (μ × вес)',
                color='coral', edgecolor='darkred', alpha=0.7)

        ax1.set_xlabel('Признаки', fontsize=12)
        ax1.set_ylabel('Значение', fontsize=12)
        ax1.set_title(f'{profile_name}: Принадлежность и вклад признаков',
                      fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=10)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0, max(max(similarities), max(contributions)) * 1.2)

        # Добавляем значения на столбцы
        for i, (sim, contrib) in enumerate(zip(similarities, contributions)):
            ax1.text(i - width / 2, sim + 0.02, f'{sim:.2f}',
                     ha='center', va='bottom', fontsize=8)
            ax1.text(i + width / 2, contrib + 0.02, f'{contrib:.2f}',
                     ha='center', va='bottom', fontsize=8)

        # График 2: Круговая диаграмма вкладов
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

        colors_pie = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD', '#DDDDDD']

        wedges, texts, autotexts = ax2.pie(plot_contribs, labels=plot_names,
                                           colors=colors_pie[:len(plot_contribs)],
                                           autopct='%1.1f%%', startangle=90,
                                           textprops={'fontsize': 10})

        for i, text in enumerate(texts[:min(top_n, len(texts))]):
            text.set_fontweight('bold')

        ax2.set_title(f'{profile_name}: Распределение вклада признаков\n(топ-5 признаков)',
                      fontsize=14, fontweight='bold')

        plt.suptitle(title or f'Анализ сходства: {profile_name}',
                     fontsize=16, fontweight='bold', y=1.05)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_authors_comparison(results_dict, title="Сравнение авторов"):
        """
        Строит сравнительную диаграмму для всех авторов
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        authors = list(results_dict.keys())
        scores = list(results_dict.values())

        colors = ['#4CAF50' if s > 0.6 else '#FFC107' if s > 0.3 else '#F44336'
                  for s in scores]

        bars = ax.bar(authors, scores, color=colors, edgecolor='black', linewidth=1.5)

        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                    f'{score:.1%}\n({score:.2f})',
                    ha='center', va='bottom', fontweight='bold', fontsize=11)

        ax.set_ylim(0, 1)
        ax.set_ylabel('Уверенность', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)

        # Цветные зоны уверенности (согласованы с порогами раскраски столбцов выше)
        ax.axhspan(0, 0.3, color='red', alpha=0.05, label='Низкая уверенность')
        ax.axhspan(0.3, 0.6, color='orange', alpha=0.05, label='Средняя уверенность')
        ax.axhspan(0.6, 1.0, color='green', alpha=0.05, label='Высокая уверенность')

        ax.legend(loc='upper right')

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_feature_heatmap(profiles_data, feature_names, title="Сравнение признаков авторов"):
        """
        Строит тепловую карту признаков для всех авторов
        """
        from matplotlib.colors import LinearSegmentedColormap

        authors = list(profiles_data.keys())
        n_authors = len(authors)
        n_features = len(feature_names)

        data = np.zeros((n_authors, n_features))
        for i, author in enumerate(authors):
            data[i, :] = profiles_data[author]

        fig, ax = plt.subplots(figsize=(14, 6))

        colors = ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#d9ef8b', '#a6d96a', '#1a9850']
        cmap = LinearSegmentedColormap.from_list('custom', colors, N=100)

        im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=0, vmax=1)

        ax.set_xticks(np.arange(n_features))
        ax.set_yticks(np.arange(n_authors))
        ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(authors, fontsize=11, fontweight='bold')

        for i in range(n_authors):
            for j in range(n_features):
                value = data[i, j]
                if value > 0:
                    color = 'white' if value > 0.5 else 'black'
                    ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                            color=color, fontweight='bold', fontsize=9)

        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Признаки', fontsize=12)
        ax.set_ylabel('Авторы', fontsize=12)

        plt.colorbar(im, ax=ax, label='Значение (нормализованное)')

        plt.tight_layout()
        return fig