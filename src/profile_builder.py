# profile_builder.py
"""Нечёткий портрет автора: треугольные функции принадлежности по признакам."""
import logging

import numpy as np

from src import config, report
from src.feature_extractor import FeatureExtractor, Language

logger = logging.getLogger(__name__)


class TriangularMembership:
    """Треугольная функция принадлежности со «смягчёнными» границами."""

    # Значение μ ровно на смягчённой границе [a_soft, c_soft].
    #
    # Раньше рампа внутри диапазона считалась как ratio ** 0.5 и на границе
    # давала РОВНО 0, а экспоненциальный хвост снаружи стартовал с 0.001 —
    # то есть значение чуть ЗА границей получало μ ВЫШЕ, чем значение прямо
    # на границе. Функция была разрывной и немонотонной в самой чувствительной
    # точке. Теперь рампа опускается не до нуля, а до EDGE_MU, и хвост
    # продолжается ровно с этого же уровня: μ непрерывна и монотонна.
    EDGE_MU = 0.001

    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c
        # Расширяем границы на долю от ширины диапазона
        self.softening = config.SOFTENING * (c - a) if c != a else 0.1

    def _ramp(self, ratio):
        """Смягчённый подъём/спуск: EDGE_MU на границе → 1.0 в точке b.

        Квадратный корень делает переход мягче линейного: значения рядом с
        границей штрафуются не так резко.
        """
        ratio = min(max(ratio, 0.0), 1.0)
        return self.EDGE_MU + (1.0 - self.EDGE_MU) * ratio ** 0.5

    def _tail(self, overshoot):
        """Экспоненциальный хвост за смягчённой границей.

        Раньше здесь был жёсткий "пол" 0.001, ОДИНАКОВЫЙ для любого x, сколь
        угодно далеко ушедшего за границу. Из-за этого два совершенно разных
        по степени "выброса" текста (например, средняя длина предложения 20
        слов и 100 слов при диапазоне автора 6..12) получали ИДЕНТИЧНУЮ
        оценку — на графике оба выглядели одинаковым "нулевым показателем".
        Экспонента от числа "мягких зон" за границей даёт каждому значению
        свой, различимый результат.
        """
        return self.EDGE_MU * float(np.exp(-max(overshoot, 0.0)))

    def mu(self, x):
        """Степень принадлежности x диапазону автора, от ~0 до 1."""
        a_soft = self.a - self.softening
        c_soft = self.c + self.softening

        if x == self.b:
            return 1.0

        if x < self.b:
            left_span = self.b - a_soft
            if left_span <= 0:
                return 1.0
            if x >= a_soft:
                return self._ramp((x - a_soft) / left_span)
            return self._tail((a_soft - x) / self.softening if self.softening > 0 else 1.0)

        right_span = c_soft - self.b
        if right_span <= 0:
            return 1.0
        if x <= c_soft:
            return self._ramp((c_soft - x) / right_span)
        return self._tail((x - c_soft) / self.softening if self.softening > 0 else 1.0)

    def __repr__(self):
        return f"TriangularMembership(a={self.a:.3f}, b={self.b:.3f}, c={self.c:.3f})"


class AuthorProfile:
    """Нечёткий портрет автора"""

    def __init__(self, name):
        self.name = name
        self.features = []  # список функций принадлежности
        self.feature_names = config.FEATURE_LIST
        self.feature_stats = None

    # ---------- построение портрета ----------

    def build_from_texts(self, texts, language=None, save_report=True, extractor=None):
        """
        texts: список текстов этого автора (каждый текст -> строка)
        language: язык для FeatureExtractor (None = русский)
        save_report: писать ли сводную таблицу в output/ (веб-приложение
            показывает её на странице и файлы не создаёт)
        extractor: готовый FeatureExtractor. Веб-приложение передаёт свой,
            закэшированный — иначе на каждого автора заново поднимался бы
            MorphAnalyzer (а для белорусского ещё и пайплайн Stanza).
        """
        logger.info("🖌️  Строим портрет для %s", self.name)

        feature_matrix = self._collect_features(texts, language, extractor)

        if feature_matrix:
            self.feature_stats = report.build_stats(feature_matrix)
            report.log_summary_table(self.name, self.feature_stats)
            if save_report:
                report.save_summary_table(self.name, self.feature_stats)

        self._fit_membership(feature_matrix)
        logger.info("  ✅ Портрет для %s построен! Всего функций: %d",
                    self.name, len(self.features))
        return self.features

    def _collect_features(self, texts, language=None, extractor=None):
        """Извлекает признаки из каждого текста. Возвращает матрицу
        (строки — тексты, столбцы — признаки), пропуская сбойные тексты."""
        num_props = config.N_FEATURES
        extractor = extractor or FeatureExtractor(language=language or Language.RUSSIAN)
        feature_matrix = []

        for text_idx, text in enumerate(texts):
            logger.debug("  Обработка текста %d/%d...", text_idx + 1, len(texts))

            if isinstance(text, bytes):
                text = text.decode('utf-8', errors='ignore')
            elif not isinstance(text, str):
                text = str(text)

            try:
                features = extractor.extract(text)
            except Exception as e:
                logger.warning("  ❌ Ошибка при извлечении признаков: %s", e)
                continue

            if len(features) != num_props:
                logger.warning("  ❌ Ожидалось %d признаков, получено %d",
                               num_props, len(features))
                continue

            feature_matrix.append(features.tolist())

        logger.debug("  Собрано текстов с признаками: %d", len(feature_matrix))
        return feature_matrix

    def _fit_membership(self, feature_matrix):
        """Строит треугольные функции принадлежности по матрице признаков."""
        num_props = config.N_FEATURES
        # Транспонируем: values_per_feature[i] — все значения признака i
        values_per_feature = [
            [row[i] for row in feature_matrix] for i in range(num_props)
        ]

        self.features = []
        for i in range(num_props):
            a, b, c = self._membership_params(values_per_feature[i], i)
            self.features.append(TriangularMembership(a, b, c))
            logger.debug("    Признак %d: a=%.3f, b=%.3f, c=%.3f", i, a, b, c)

        return self.features

    @staticmethod
    def _membership_params(values, index):
        """Вычисляет (a, b, c) для одного признака.

        Вместо буквальных min/max по обучающим текстам берётся толерантный
        интервал среднее ± k·std (k = config.MEMBERSHIP_STD_MULTIPLIER). При
        7-10 текстах на автора min/max жёстко занижает реальный разброс: автор
        пишет предложения медианной длины то 7, то 9 слов — а новый текст с
        медианой 8.2 воспринимался почти как выброс.
        """
        if not values:
            logger.warning("  ⚠️ Признак %d: нет данных, используем значения по умолчанию", index)
            return 0.0, 0.5, 1.0

        if len(values) == 1:
            # Единственный текст — разброс оценить нечем.
            v = values[0]
            a, c = (0.5 * v, 1.5 * v) if v != 0 else (-0.1, 0.1)
            return max(a, 0.0), v, c

        mean_v = float(np.mean(values))
        spread = config.MEMBERSHIP_STD_MULTIPLIER * float(np.std(values, ddof=1))
        b = mean_v

        if spread == 0:
            # Все тексты дали ровно одно и то же значение признака.
            a = 0.5 * mean_v if mean_v != 0 else -0.1
            c = 1.5 * mean_v if mean_v != 0 else 0.1
        else:
            a, c = mean_v - spread, mean_v + spread

        # Ни один из 17 признаков не бывает отрицательным (доли, счётчики,
        # дисперсия) — отсекаем нижнюю границу снизу.
        a = max(a, 0.0)
        if c <= a:
            c = a + (0.1 if a == 0 else 0.1 * a)
        return a, b, c

    # ---------- отчётность ----------

    def get_summary_html(self):
        """HTML сводной таблицы признаков автора — БЕЗ сохранения на диск.
        Предназначен для показа прямо в Streamlit (app.py).

        Возвращает None для профилей, обученных версией кода до появления
        feature_stats — тогда вызывающий код предлагает переобучить профили.
        """
        stats = getattr(self, 'feature_stats', None)
        if not stats or 'dataframe' not in stats:
            return None
        return report.build_table_html(self.name, stats)

    # ---------- сравнение с текстом ----------

    def get_weights(self):
        """Веса признаков из конфигурации, подогнанные под их количество."""
        weights = list(config.DEFAULT_WEIGHTS)
        weights += [1.0] * max(0, len(self.features) - len(weights))
        return weights[:len(self.features)]

    def similarity_with_details(self, text_features):
        """
        Возвращает сходство и детали (μ, веса, вклады)

        Args:
            text_features: массив признаков текста

        Returns:
            tuple: (сходство, (similarities, weights, contributions))
        """
        if not self.features:
            logger.warning("  ⚠️ У автора %s нет признаков для сравнения!", self.name)
            return 0.0, ([], [], [])

        similarities = [
            feat_func.mu(text_features[i]) if i < len(text_features) else 0.0
            for i, feat_func in enumerate(self.features)
        ]

        weights = self.get_weights()
        contributions = [s * w for s, w in zip(similarities, weights)]

        total_weight = sum(weights)
        total = sum(contributions) / total_weight if total_weight > 0 else 0.0

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("   Взвешенное сходство для %s:", self.name)
            for i, (mu, weight, contrib) in enumerate(zip(similarities, weights, contributions)):
                logger.debug("    %-2d %-6.3f %-5.1f %-8.3f", i, mu, weight, contrib)
            logger.debug("    ИТОГО: %.3f", total)

        return total, (similarities, weights, contributions)
