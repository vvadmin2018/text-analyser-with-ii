import numpy as np
from src.feature_extractor import FeatureExtractor, Language
from src import config


class TriangularMembership:
    """Треугольная функция принадлежности (смягченная)"""

    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c
        # Добавляем "размытие" границ
        softening = config.SOFTENING
        self.softening = softening * (c - a) if c != a else 0.1

    def mu(self, x):
        # Расширяем границы на величину softening
        a_soft = self.a - self.softening
        c_soft = self.c + self.softening

        if x == self.b:
            return 1.0
        elif x < self.b:
            left_span = self.b - a_soft
            if left_span <= 0 or self.b == self.a:
                return 1.0
            if x >= a_soft:
                # Плавный подъем от a_soft до b
                ratio = (x - a_soft) / left_span
                return ratio ** 0.5  # смягченный подъем (квадратный корень для более высоких значений)
            # За пределами a_soft — раньше здесь был жёсткий "пол" 0.001,
            # ОДИНАКОВЫЙ для любого x, сколь угодно далеко ушедшего влево.
            # Из-за этого два совершенно разных по степени "выброса" текста
            # (например, средняя длина предложения 20 слов и 100 слов при
            # диапазоне автора 6..12) получали ИДЕНТИЧНУЮ оценку 0.001 —
            # на графике оба выглядели как одинаковый "нулевой показатель",
            # хотя реально один текст гораздо ближе к автору, чем другой.
            # Вместо этого продолжаем экспоненциальный спад пропорционально
            # тому, на сколько "мягких зон" (softening) значение ушло за
            # границу — чем дальше, тем меньше μ, но каждое значение
            # получает СВОЙ, различимый результат.
            overshoot = (a_soft - x) / self.softening if self.softening > 0 else 1.0
            return 0.001 * float(np.exp(-overshoot))
        else:
            right_span = c_soft - self.b
            if right_span <= 0 or self.c == self.b:
                return 1.0
            if x <= c_soft:
                # Плавный спуск от b до c_soft
                ratio = (c_soft - x) / right_span
                return ratio ** 0.5  # смягченный спуск (квадратный корень для более высоких значений)
            # Симметричный экспоненциальный "хвост" справа — см. комментарий выше.
            overshoot = (x - c_soft) / self.softening if self.softening > 0 else 1.0
            return 0.001 * float(np.exp(-overshoot))

    def __repr__(self):
        return f"TriangularMembership(a={self.a:.3f}, b={self.b:.3f}, c={self.c:.3f})"


class AuthorProfile:
    """Нечёткий портрет автора"""

    def __init__(self, name):
        self.name = name
        self.features = []  # список функций принадлежности
        self.feature_names = config.FEATURE_LIST

    def build_from_texts(self, texts, language=None):
        """
        texts: список текстов этого автора (каждый текст -> строка)
        language: язык для FeatureExtractor (None = русский)
        """
        import numpy as np
        import pandas as pd

        num_props = config.N_FEATURES

        print(f"\n🖌️  Строим портрет для {self.name}")

        # Собираем все значения по каждому признаку
        all_values = [[] for _ in range(num_props)]
        # Собираем значения по текстам для таблицы (строки - тексты, столбцы - признаки)
        text_features_matrix = []

        extractor = FeatureExtractor(language=language or Language.RUSSIAN)

        for text_idx, text in enumerate(texts):
            print(f"  Обработка текста {text_idx + 1}/{len(texts)}...")

            # проверяем тип text
            if isinstance(text, bytes):
                text = text.decode('utf-8', errors='ignore')
            elif not isinstance(text, str):
                text = str(text)

            try:
                features = extractor.extract(text)
                print(f"    Извлечено признаков: {len(features)}")

                # Проверяем, что признаки извлечены правильно
                if len(features) == num_props:
                    # Сохраняем признаки для таблицы
                    text_features_matrix.append(features.tolist())

                    for i, val in enumerate(features):
                        all_values[i].append(val)
                    print(f"    Значения: {[f'{v:.3f}' for v in features[:5]]}...")
                else:
                    print(f"    ❌ Ожидалось {num_props} признаков, получено {len(features)}")

            except Exception as e:
                print(f"  ❌ Ошибка при извлечении признаков: {e}")

        # Проверяем, что собрали данные
        print(f"\n  Статистика собранных данных:")
        for i in range(num_props):
            print(f"    Признак {i}: {len(all_values[i])} значений")
            if all_values[i]:
                print(f"      min={min(all_values[i]):.3f}, max={max(all_values[i]):.3f}")

        # ===== ПОСТРОЕНИЕ СВОДНОЙ ТАБЛИЦЫ =====
        if text_features_matrix:
            from src.visualizer import _display_name
            print(f"\n  📊 Сводная таблица признаков для автора '{_display_name(self.name)}':")

            # Создаем DataFrame: строки - тексты, столбцы - признаки
            df = pd.DataFrame(text_features_matrix, columns=config.FEATURE_LIST)

            # Добавляем номер текста
            df.insert(0, 'Текст №', range(1, len(df) + 1))

            # Вычисляем среднее и медиану по каждому признаку (для всех текстов)
            mean_values = [round(np.mean(all_values[i]), 3) if all_values[i] else 0 for i in range(num_props)]
            median_values = [round(np.median(all_values[i]), 3) if all_values[i] else 0 for i in range(num_props)]

            # Добавляем строки со средним и медианой
            mean_row = ['Среднее'] + mean_values
            median_row = ['Медиана'] + median_values

            # Форматируем все числовые значения до 3 знаков после запятой
            df_formatted = df.copy()
            for col in config.FEATURE_LIST:
                df_formatted[col] = df_formatted[col].apply(lambda x: round(x, 3))

            # Выводим таблицу
            print("\n  " + "=" * (len(config.FEATURE_LIST) * 12 + 15))
            print(df_formatted.to_string(index=False))
            print("  " + "-" * (len(config.FEATURE_LIST) * 12 + 15))
            print(f"  Среднее:  {'  '.join([f'{v:>10}' for v in mean_values])}")
            print(f"  Медиана:  {'  '.join([f'{v:>10}' for v in median_values])}")
            print("  " + "=" * (len(config.FEATURE_LIST) * 12 + 15))

            # Сохраняем статистику в объекте профиля для дальнейшего использования
            self.feature_stats = {
                'mean': mean_values,
                'median': median_values,
                'text_count': len(texts),
                'dataframe': df_formatted
            }

            # ===== СОХРАНЕНИЕ ТАБЛИЦЫ В OUTPUT =====
            self._save_summary_table(df_formatted, mean_values, median_values)

        # Строим треугольные функции
        self.features = []
        for i in range(num_props):
            values = all_values[i]

            if not values:
                # Если нет данных для признака, создаем функцию с широким диапазоном
                a, b, c = 0.0, 0.5, 1.0
                print(f"  ⚠️ Признак {i}: нет данных, используем значения по умолчанию")
            elif len(values) == 1:
                # Единственный текст — разброс оценить нечем, используем
                # тот же запасной вариант "widen", что и ниже для std==0.
                v = values[0]
                a, c = (0.5 * v, 1.5 * v) if v != 0 else (-0.1, 0.1)
                b = v
            else:
                # Раньше здесь брались буквальные min/max по обучающим
                # текстам автора. При 7-10 текстах на автора это ЖЕСТКО
                # занижает реальный разброс: min/max по маленькой выборке
                # почти всегда уже, чем истинная вариативность автора, а
                # SOFTENING (доля от этого и без того узкого span) не
                # спасает — 0.8×(узкий span) остаётся узким. Именно это
                # раньше "обнуляло" такие стабильные признаки, как средняя
                # длина предложения и MATTR: автор пишет предложения
                # медианной длины то 7, то 9 слов — а любой новый текст с
                # медианой 8.2 воспринимался почти как выброс.
                #
                # Вместо min/max используем толерантный интервал
                # среднее ± k·std (k = config.MEMBERSHIP_STD_MULTIPLIER).
                # Это стандартный статистический приём для маленьких
                # выборок: оценивать не наблюдённый разброс буквально, а
                # правдоподобный разброс "типичного" текста этого автора.
                mean_v = float(np.mean(values))
                std_v = float(np.std(values, ddof=1))  # выборочное std (n-1)

                b = mean_v
                spread = config.MEMBERSHIP_STD_MULTIPLIER * std_v
                a = mean_v - spread
                c = mean_v + spread

                # Защита от вырожденного случая (все тексты дали ровно
                # одно и то же значение признака -> std == 0)
                if spread == 0:
                    a = 0.5 * mean_v if mean_v != 0 else -0.1
                    c = 1.5 * mean_v if mean_v != 0 else 0.1

                # Ни один из 17 признаков не бывает отрицательным (доли,
                # счётчики, дисперсия) — отсекаем нижнюю границу снизу,
                # даже если std "утащил" её в минус.
                a = max(a, 0.0)
                if c <= a:
                    c = a + (0.1 if a == 0 else 0.1 * a)

            # Создаем функцию принадлежности
            self.features.append(TriangularMembership(a, b, c))
            if config.LEVEL_LOG == "DEBUG":
                print(f"    Признак {i}: a={a:.3f}, b={b:.3f}, c={c:.3f}")

        print(f"  ✅ Портрет для {self.name} построен! Всего функций: {len(self.features)}")
        return self.features


    def _make_df_with_stats(self, df_formatted, mean_values, median_values):
        """Достраивает DataFrame строками 'Среднее' и 'Медиана'. Вынесено
        отдельно, чтобы использоваться и при сохранении файлов, и при показе
        таблицы в веб-приложении (get_summary_html) без дублирования кода."""
        import pandas as pd

        df_with_stats = df_formatted.copy()

        mean_row_dict = {'Текст №': 'Среднее'}
        for i, col in enumerate(config.FEATURE_LIST):
            mean_row_dict[col] = mean_values[i]

        median_row_dict = {'Текст №': 'Медиана'}
        for i, col in enumerate(config.FEATURE_LIST):
            median_row_dict[col] = median_values[i]

        mean_row_df = pd.DataFrame([mean_row_dict])
        median_row_df = pd.DataFrame([median_row_dict])
        return pd.concat([df_with_stats, mean_row_df, median_row_df], ignore_index=True)

    def _build_table_html(self, df_with_stats):
        """Строит полный HTML-документ таблицы в теме "бумага и чернила"
        (см. src/visualizer.py). Не пишет на диск — это отдельно делает
        _save_summary_table; get_summary_html() использует этот же метод
        для показа таблицы прямо в Streamlit-приложении."""
        from datetime import datetime
        from src.visualizer import (
            PAPER_BG, GRID_COLOR, INK_TEXT, INK_TEXT_SOFT, ANON_ACCENT,
            _author_color, _display_name,
        )

        author_display = _display_name(self.name)
        author_color = _author_color(self.name)
        n_texts = max(len(df_with_stats) - 2, 0)  # минус строки Среднее/Медиана

        # ===== Компактность: таблица должна умещаться в одну страницу =====
        # При 17 признаках + колонка "Текст №" (18 колонок) полные описательные
        # названия ("Ср. длина предл.", "Доля вопросительных предложений" и
        # т.п.) и 6 знаков после запятой у каждого числа гарантированно
        # выталкивают таблицу за пределы экрана/страницы вправо. Поэтому здесь:
        #   1) заголовки колонок — короткие подписи (те же, что и на графиках,
        #      FEATURE_LIST_SHORT), а не длинные описательные названия;
        #   2) числа округляются до 2 знаков при отображении;
        #   3) table-layout: fixed + width: 100% — таблица физически не может
        #      стать шире родительского контейнера, колонки просто сжимаются.
        short_names = dict(zip(config.FEATURE_LIST, config.FEATURE_LIST_SHORT))
        display_df = df_with_stats.rename(columns=short_names)
        feature_cols = [short_names.get(c, c) for c in config.FEATURE_LIST]

        # Строки "Среднее"/"Медиана" — как заметка золотистыми чернилами на
        # полях: мягкая заливка акцентным цветом и линия сверху, чтобы сразу
        # отличались от построчных значений отдельных текстов.
        def _highlight_stats_row(row):
            if row.iloc[0] in ('Среднее', 'Медиана'):
                return [f'background-color: {ANON_ACCENT}2E; font-weight: 700; '
                        f'border-top: 2px solid {ANON_ACCENT};'] * len(row)
            return [''] * len(row)

        styler = (
            display_df.style
            .format('{:.3f}', subset=feature_cols)
            .apply(_highlight_stats_row, axis=1)
            .set_table_attributes('class="data-table"')
            .hide(axis='index')
            .set_table_styles([
                {'selector': 'table', 'props': [
                    ('table-layout', 'fixed'), ('width', '100%'),
                ]},
                {'selector': 'th', 'props': [
                    ('background-color', author_color), ('color', PAPER_BG),
                    ('padding', '5px 3px'), ('font-weight', '600'),
                    ('font-size', '10.5px'), ('line-height', '1.2'),
                    ('word-break', 'break-word'), ('white-space', 'normal'),
                    ('border', f'1px solid {author_color}'),
                ]},
                {'selector': 'td', 'props': [
                    ('padding', '4px 3px'), ('text-align', 'center'),
                    ('font-size', '11px'),
                    ('border', f'1px solid {GRID_COLOR}'),
                ]},
                {'selector': 'th:first-child, td:first-child', 'props': [
                    ('width', '60px'),
                ]},
                {'selector': 'tbody tr:nth-child(odd)', 'props': [('background-color', PAPER_BG)]},
                {'selector': 'tbody tr:nth-child(even)', 'props': [('background-color', '#F3ECDD')]},
                {'selector': 'tbody tr:hover', 'props': [('background-color', '#EFE3C8')]},
            ], overwrite=False)
        )
        table_html = styler.to_html()

        return f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>Сводная таблица признаков — {author_display}</title>
    <style>
        * {{ box-sizing: border-box; }}
        html, body {{ width: 100%; }}
        body {{
            font-family: Verdana, Geneva, sans-serif;
            background: {PAPER_BG};
            color: {INK_TEXT};
            margin: 0;
            padding: 18px 22px 28px;
        }}
        .sheet {{
            width: 100%;
            max-width: 100%;
            margin: 0 auto;
            background: #FFFFFF;
            border: 1px solid {GRID_COLOR};
            border-radius: 10px;
            padding: 18px 20px 22px;
            box-shadow: 0 6px 18px rgba(42, 35, 28, 0.10);
        }}
        h1 {{
            font-family: Georgia, "Times New Roman", serif;
            font-size: 19px;
            color: {author_color};
            border-bottom: 3px solid {author_color};
            padding-bottom: 8px;
            margin: 0 0 6px;
        }}
        .meta {{ color: {INK_TEXT_SOFT}; font-size: 12.5px; margin: 2px 0; }}
        table.data-table {{
            border-collapse: collapse;
            table-layout: fixed;
            width: 100%;
            margin-top: 16px;
            font-size: 11px;
        }}
        .footer-note {{
            margin-top: 14px;
            font-size: 11.5px;
            color: {INK_TEXT_SOFT};
            font-style: italic;
        }}
    </style>
</head>
<body>
    <div class="sheet">
        <h1>Сводная таблица признаков автора: {author_display}</h1>
        <p class="meta">Текстов в обучающей выборке: {n_texts}</p>
        <p class="meta">Дата генерации: {datetime.now().strftime("%d.%m.%Y %H:%M")}</p>
        {table_html}
        <p class="footer-note">Значения округлены до 3 знаков после запятой.</p>
    </div>
</body>
</html>"""

    def get_summary_html(self):
        """Возвращает готовый HTML сводной таблицы признаков автора — БЕЗ
        сохранения на диск. Предназначен для показа прямо в Streamlit
        (app.py), в дополнение к файлам, которые пишет _save_summary_table.

        Требует, чтобы профиль был построен через build_from_texts() в этом
        же процессе, либо загружен из pickle, сохранённого ПОСЛЕ обучения
        текущей версией кода (self.feature_stats сохраняется как обычный
        атрибут объекта, поэтому переживает pickle). Профили, обученные
        старой версией profile_builder.py (до появления feature_stats),
        такого атрибута не имеют — тогда возвращается None, и вызывающий
        код (app.py) должен предложить переобучить профили.
        """
        stats = getattr(self, 'feature_stats', None)
        if not stats or 'dataframe' not in stats:
            return None
        df_with_stats = self._make_df_with_stats(
            stats['dataframe'], stats['mean'], stats['median']
        )
        return self._build_table_html(df_with_stats)

    def _save_summary_table(self, df_formatted, mean_values, median_values):
        """
        Сохраняет сводную таблицу признаков в output в форматах HTML и TXT.

        Оформление HTML согласовано с темой графиков (см. src/visualizer.py):
        тёплый бумажный фон, тонкая карандашная сетка, а цвет заголовка и
        шапки таблицы — те самые "чернила" автора из config.AUTHOR_COLORS,
        так что таблица и роза ветров одного автора выглядят одним
        комплектом. Имя автора везде отображается через
        config.AUTHOR_LABELS, а не как сырой служебный ключ (self.name).

        Args:
            df_formatted: DataFrame с данными по текстам
            mean_values: список средних значений по признакам
            median_values: список медианных значений по признакам
        """
        import os
        from datetime import datetime
        from src.visualizer import _display_name

        author_display = _display_name(self.name)
        df_with_stats = self._make_df_with_stats(df_formatted, mean_values, median_values)
        html_content = self._build_table_html(df_with_stats)

        # Создаем директорию output если не существует
        output_dir = config.OUTPUT_DIR_MAIN
        os.makedirs(output_dir, exist_ok=True)

        # Создаем поддиректорию с текущей датой и временем
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        author_dir = os.path.join(output_dir, f"{timestamp}_{self.name}")
        os.makedirs(author_dir, exist_ok=True)

        # Сохраняем в HTML
        html_file = os.path.join(author_dir, f"{self.name}_summary.html")
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        # Сохраняем в TXT
        txt_file = os.path.join(author_dir, f"{self.name}_summary.txt")
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(f"Сводная таблица признаков автора: {author_display}\n")
            f.write(f"Количество текстов: {len(df_formatted)}\n")
            f.write(f"Дата генерации: {datetime.now().strftime('%d.%m.%Y %H:%M')}\n")
            f.write("=" * (len(config.FEATURE_LIST) * 12 + 15) + "\n\n")

            # Вывод таблицы в текстовом формате
            f.write(df_with_stats.to_string(index=False))
            f.write("\n\n")

            f.write("=" * (len(config.FEATURE_LIST) * 12 + 15) + "\n")
            f.write("Примечание: все значения округлены до 3 знаков после запятой\n")

        print(f"  💾 Таблица сохранена в:\n     HTML: {html_file}\n     TXT:  {txt_file}")


    def get_weights(self):
        """
        Возвращает веса признаков из конфигурации
        """
        # Используем веса из config.py
        weights = config.DEFAULT_WEIGHTS.copy()

        # Убеждаемся, что весов столько же, сколько признаков
        while len(weights) < len(self.features):
            weights.append(1.0)

        # Обрезаем, если весов слишком много
        return weights[:len(self.features)]

    def similarity(self, text_features, use_weights=True, custom_weights=None):
        """
        Вычисляет сходство с вектором признаков текста

        Args:
            text_features: массив признаков текста
            use_weights: использовать ли веса признаков
            custom_weights: пользовательские веса (если None, используются веса из config)

        Returns:
            float: степень сходства от 0 до 1
        """
        # Проверяем, что есть признаки для сравнения
        if not self.features:
            print(f"  ⚠️ У автора {self.name} нет признаков для сравнения!")
            return 0.0

        # Вычисляем степень принадлежности для каждого признака
        similarities = []
        for i, feat_func in enumerate(self.features):
            if i < len(text_features):
                mu = feat_func.mu(text_features[i])
                similarities.append(mu)
            else:
                similarities.append(0.0)

        if not use_weights:
            return np.mean(similarities)

        # Получаем веса
        if custom_weights is not None:
            weights = custom_weights
        else:
            weights = self.get_weights()

        # Убеждаемся, что у нас достаточно весов
        while len(weights) < len(similarities):
            weights.append(1.0)
        weights = weights[:len(similarities)]

        # Вычисляем взвешенное среднее
        weighted_sum = 0
        total_weight = 0

        if config.LEVEL_LOG == "DEBUG":
            print(f"\n   Взвешенное сходство для {self.name}:")
            print(f"    {'№':<2} {'μ':<6} {'вес':<5} {'вклад':<8}")
            print(f"    {'-' * 25}")

        for i, (mu, weight) in enumerate(zip(similarities, weights)):
            contribution = mu * weight
            weighted_sum += contribution
            total_weight += weight

            if i < config.N_FEATURES and config.LEVEL_LOG == "DEBUG":
                print(f"    {i:<2} {mu:<6.3f} {weight:<5.1f} {contribution:<8.3f}")

        result = weighted_sum / total_weight if total_weight > 0 else 0

        if config.LEVEL_LOG == "DEBUG":
            print(f"    {'-' * 25}")
            print(f"    ИТОГО: {result:.3f}")

        return result

    def similarity_with_details(self, text_features):
        """
        Возвращает сходство и детали (μ, веса, вклады)

        Args:
            text_features: массив признаков текста

        Returns:
            tuple: (сходство, (similarities, weights, contributions))
        """
        # Проверяем, что есть признаки для сравнения
        if not self.features:
            print(f"  ⚠️ У автора {self.name} нет признаков для сравнения!")
            return 0.0, ([], [], [])

        # Вычисляем степень принадлежности
        similarities = []
        for i, feat_func in enumerate(self.features):
            if i < len(text_features):
                mu = feat_func.mu(text_features[i])
                similarities.append(mu)
            else:
                similarities.append(0.0)

        # Получаем веса
        weights = self.get_weights()

        # Вычисляем вклады
        contributions = [s * w for s, w in zip(similarities, weights)]

        # Вычисляем итоговое сходство (взвешенное среднее)
        total = sum(contributions) / sum(weights) if sum(weights) > 0 else 0

        return total, (similarities, weights, contributions)

    def experiment_with_weights(self, text_features, weight_sets):
        """
        Позволяет экспериментировать с разными наборами весов
        """
        results = {}

        # Базовый результат без весов
        base_result = self.similarity(text_features, use_weights=False)
        results['Без весов'] = base_result

        # Результаты с разными весами
        for exp_name, weights in weight_sets.items():
            result = self.similarity(text_features, use_weights=True, custom_weights=weights)
            results[exp_name] = result

        return results
