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
    def plot_fuzzy_rose(authors_profiles, anonymous_vector, feature_names,
                        profiles_dispersion=None, anonymous_dispersion=None,
                        title="Нечёткая роза ветров", figsize=(14, 10),
                    author_colors=None):

        """
        Строит розу ветров с размытыми секторами, отражающими неопределённость
            Параметры:
            - authors_profiles: словарь {имя_автора: [значения_признаков]}
            - anonymous_vector: список значений для анонимного текста
            - feature_names: названия признаков
            - profiles_dispersion: словарь {имя_автора: [дисперсии_признаков]}
            - anonymous_dispersion: список дисперсий для анонима (не используется)
            - author_colors: словарь {имя_автора: цвет} для индивидуальных цветов
        """

        print("\n🔄 Построение нечёткой розы ветров...")

        # Нормализуем данные
        norm_profiles, norm_anon = StyleRose.normalize_by_max(
            authors_profiles, anonymous_vector
        )

        # Цвета по умолчанию, если не переданы
        if author_colors is None:
            default_colors = ['#FF4B4B', '#4B7BFF', '#4BFF4B', '#FFB44B', '#B44BFF']
            author_colors = {}

            for i, name in enumerate(norm_profiles.keys()):
                author_colors[name] = default_colors[i % len(default_colors)]

        # Обрабатываем дисперсию для АВТОРОВ
        norm_dispersions = {}
        if profiles_dispersion:
            print("\n  Обработка дисперсий для авторов:")
            for name, values in profiles_dispersion.items():
                # Получаем нормализованные значения признаков для этого автора
                profile_values = norm_profiles[name]

                # Создаем нормализованную дисперсию относительно значения признака
                norm_values = []
                for i, (val, disp) in enumerate(zip(profile_values, values)):
                    # Дисперсия как процент от значения (но не больше 0.3 и не меньше 0.05)
                    if val > 0:
                        # Относительная дисперсия: disp / val, но ограничиваем
                        rel_disp = min(0.25, max(disp / val, 0.1)) if val > 0 else 0.01
                    else:
                        rel_disp = 0.01  # значение по умолчанию

                    norm_values.append(rel_disp)

                norm_dispersions[name] = norm_values
                print(f"    {name}: мин={min(norm_values):.3f}, макс={max(norm_values):.3f}")

        n_features = len(feature_names)
        angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='polar')

        # Цвета для авторов
        colors = ['#FF4B4B', '#4B7BFF', '#4BFF4B', '#FFB44B', '#B44BFF']

        # Для каждого автора строим размытый сектор
        for idx, (author_name, profile_values) in enumerate(norm_profiles.items()):
            color = author_colors.get(author_name, f'C{idx}')

            # Получаем дисперсию для автора
            if norm_dispersions and author_name in norm_dispersions:
                dispersion = norm_dispersions[author_name]
            else:
                # Значение по умолчанию - 10% от значения
                dispersion = [min(0.1, max(v * 0.1, 0.05)) for v in profile_values]

            # Создаем верхнюю и нижнюю границы сектора
            plot_angles = angles + [angles[0]]

            # Значения с учетом дисперсии
            upper_values = []
            lower_values = []

            for val, disp in zip(profile_values, dispersion):
                # Дисперсия как абсолютная величина (не процент!)
                upper_values.append(min(1.0, val + disp))
                lower_values.append(max(0.0, val - disp))

            # Замыкаем круги
            upper_values = upper_values + [upper_values[0]]
            lower_values = lower_values + [lower_values[0]]

            # Рисуем размытый сектор
            ax.fill_between(plot_angles, lower_values, upper_values,
                            color=color, alpha=0.2, label=f'{author_name} (вариативность)')

            # Рисуем основную линию
            plot_values = profile_values + [profile_values[0]]
            ax.plot(plot_angles, plot_values, 'o-', linewidth=2.5,
                    color=color, label=author_name, markersize=8)

        # Анонимный текст - только линия
        plot_anon = norm_anon + [norm_anon[0]]
        ax.plot(angles + [angles[0]], plot_anon, 'o-', linewidth=4,
                color='black', label='Анонимный текст',
                markersize=10, markerfacecolor='yellow',
                markeredgecolor='black', markeredgewidth=2)

        # Настройки графика
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'])

        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xticks(angles)
        ax.set_xticklabels(feature_names, size=10, fontweight='bold')
        ax.set_facecolor('#f8f9fa')

        plt.title("\n\n" + title + "\n(размытые сектора показывают вариативность стиля авторов)",
                  size=14, fontweight='bold', pad=20)

        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0),
                   frameon=True, fancybox=True, shadow=True, fontsize=9)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_comparison_old_vs_new(authors_profiles, anonymous_vector, feature_names,
                                   profiles_dispersion=None, anonymous_dispersion=None):
        """
        Сравнивает старую (линейную) и новую (с размытыми секторами) визуализации
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Старая версия (линии)
        norm_profiles, norm_anon = StyleRose.normalize_by_max(authors_profiles, anonymous_vector)
        n_features = len(feature_names)
        angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()

        # Настраиваем первый подграфик
        ax1 = plt.subplot(1, 2, 1, projection='polar')
        colors = ['#FF4B4B', '#4B7BFF', '#4BFF4B']

        for idx, (name, values) in enumerate(norm_profiles.items()):
            plot_vals = values + [values[0]]
            ax1.plot(angles + [angles[0]], plot_vals, 'o-',
                     color=colors[idx], label=name, linewidth=2)

        anon_plot = norm_anon + [norm_anon[0]]
        ax1.plot(angles + [angles[0]], anon_plot, 'o-',
                 color='black', label='Аноним', linewidth=3)

        ax1.set_ylim(0, 1)
        ax1.set_title("Старая версия (только линии)", size=12)
        ax1.set_xticks(angles)
        ax1.set_xticklabels([])
        ax1.grid(True, alpha=0.3)

        # Второй подграфик - новая версия с размытыми секторами
        ax2 = plt.subplot(1, 2, 2, projection='polar')

        if profiles_dispersion is None:
            profiles_dispersion = {}
            for name, values in authors_profiles.items():
                profiles_dispersion[name] = [v * 0.15 for v in values]

        if anonymous_dispersion is None:
            anonymous_dispersion = [v * 0.15 for v in anonymous_vector]

        for idx, (name, values) in enumerate(norm_profiles.items()):
            color = colors[idx % len(colors)]
            disp = profiles_dispersion[name]

            upper = [min(1.0, v + d) for v, d in zip(values, disp)]
            lower = [max(0.0, v - d) for v, d in zip(values, disp)]

            ax2.fill_between(angles + [angles[0]],
                             lower + [lower[0]],
                             upper + [upper[0]],
                             color=color, alpha=0.15)

            ax2.plot(angles + [angles[0]], values + [values[0]], 'o-',
                     color=color, label=name, linewidth=2)

        anon_upper = [min(1.0, v + d) for v, d in zip(norm_anon, anonymous_dispersion)]
        anon_lower = [max(0.0, v - d) for v, d in zip(norm_anon, anonymous_dispersion)]

        ax2.fill_between(angles + [angles[0]],
                         anon_lower + [anon_lower[0]],
                         anon_upper + [anon_upper[0]],
                         color='gray', alpha=0.15)

        ax2.plot(angles + [angles[0]], anon_plot, 'o-',
                 color='black', label='Аноним', linewidth=3)

        ax2.set_ylim(0, 1)
        ax2.set_title("Новая версия (с размытыми секторами)", size=12)
        ax2.set_xticks(angles)
        ax2.set_xticklabels(feature_names, size=8, rotation=45)
        ax2.grid(True, alpha=0.3)

        plt.suptitle("Сравнение визуализаций: линии vs размытые сектора",
                     size=14, fontweight='bold')
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

        #ax.axhline(y=0.3, color='orange', linestyle=':', alpha=0.5, label='Низкая уверенность')
        #ax.axhline(y=0.6, color='green', linestyle=':', alpha=0.5, label='Хорошая уверенность')
        #ax.axhline(y=0.8, color='darkgreen', linestyle=':', alpha=0.5, label='Высокая уверенность')

        # Цветные зоны вместо линий
        #ax.axhspan(0, 0.3, color='red', alpha=0.05, label='Низкая уверенность')
        #ax.axhspan(0.3, 0.6, color='orange', alpha=0.05, label='Средняя уверенность')
        #ax.axhspan(0.6, 1.0, color='green', alpha=0.05, label='Высокая уверенность')

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