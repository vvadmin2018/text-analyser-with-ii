# identifier.py
"""Сравнение вектора признаков со всеми обученными профилями.

Раньше этот модуль содержал класс FuzzyDetective, который нигде не
использовался, а сам цикл "пройтись по профилям и найти лучший" был написан
трижды — в main.py, в app.py и здесь. Теперь он один.
"""
import logging

from src import config

logger = logging.getLogger(__name__)


def identify(profiles, features):
    """Сравнивает признаки текста со всеми профилями.

    Args:
        profiles: {имя автора: AuthorProfile}
        features: вектор признаков анализируемого текста

    Returns:
        tuple: (best_author, results, similarity_details)
        results — {имя: итоговое сходство}
        similarity_details — {имя: (similarities, weights, contributions)},
        тот же формат, что и AuthorProfile.similarity_with_details(),
        удобно передавать напрямую в StyleRose.plot_feature_importance.
        best_author = None, если профилей нет.
    """
    results = {}
    similarity_details = {}

    for name, profile in profiles.items():
        similarity, details = profile.similarity_with_details(features)
        results[name] = similarity
        similarity_details[name] = details
        logger.debug("  Сходство с %s: %.3f", name, similarity)

    best_author = max(results, key=results.get) if results else None
    return best_author, results, similarity_details


def is_confident(score, threshold=None):
    """Достаточна ли уверенность, чтобы называть автора."""
    return score >= (config.CONFIDENCE_THRESHOLD if threshold is None else threshold)
