# main.py - Главный файл запуска исследования
# Запускается командой: python main.py
"""Полный офлайн-прогон: обучение портретов → идентификация → визуализация."""
import logging
import os
import pickle

import matplotlib.pyplot as plt

from src import config, io_utils
from src.feature_extractor import FeatureExtractor, Language
from src.identifier import identify
from src.profile_builder import AuthorProfile
from src.visualizer import StyleRose

logger = logging.getLogger(__name__)


# ============================================
# ЭТАП 1: ПОДГОТОВКА - извлекаем тексты из папок
# ============================================

def build_authors_profiles(authors=None, base_path=None):
    """
    Читает обучающие тексты из папок texts/<автор>/.

    Args:
        authors: список авторов (по умолчанию config.AUTHORS_LIST). Передаётся
            явно веб-приложением — раньше оно подменяло config.AUTHORS_LIST
            глобально прямо перед вызовом, что при двух вкладках/языках в одном
            процессе Streamlit приводило к обучению не на тех авторах.
        base_path: корневая папка с текстами (по умолчанию config.BASE_PATH).

    Returns:
        {имя автора: [текст, ...]} — только авторы, у которых нашлись тексты.
    """
    authors = authors if authors is not None else config.AUTHORS_LIST
    base_path = base_path or config.BASE_PATH

    authors_data = {}
    for author in authors:
        logger.info("📖 Обрабатываем автора: %s", author)
        author_texts = io_utils.load_author_texts(author, base_path)

        if author_texts:
            authors_data[author] = author_texts
            logger.info("  ✅ Получено %d текстов для автора %s", len(author_texts), author)
        else:
            logger.warning("  ❌ Нет текстов для автора %s", author)

    return authors_data


# ============================================
# ЭТАП 2: СОЗДАНИЕ НЕЧЁТКИХ ПОРТРЕТОВ
# ============================================

def create_fuzzy_profiles(authors_data, language=None, save_report=True):
    """По текстам строит треугольные функции принадлежности для каждого автора."""
    profiles = {}
    for author_name, texts in authors_data.items():
        profile = AuthorProfile(author_name)
        profile.build_from_texts(texts, language=language, save_report=save_report)
        profiles[author_name] = profile
    return profiles


# ============================================
# ЭТАП 3: СОХРАНЕНИЕ ПОРТРЕТОВ (чтобы не пересчитывать каждый раз)
# ============================================

def save_profiles(profiles, filename="authors_profiles.pkl"):
    """Сохраняем портреты в файл, чтобы при следующих запусках
    не пересчитывать заново."""
    with open(filename, 'wb') as f:
        pickle.dump(profiles, f)
    logger.info("  💾 Портреты сохранены в %s", filename)


def load_profiles(filename="authors_profiles.pkl"):
    """Загружаем ранее сохранённые портреты.

    Возвращает None, если файла нет или ХОТЯ БЫ ОДИН портрет не соответствует
    текущему числу признаков. Раньше функция возвращала уцелевшее подмножество,
    и анализ молча шёл по неполному списку авторов — ровно тот случай, ради
    которого проверка и писалась.
    """
    if not os.path.exists(filename):
        return None

    try:
        with open(filename, 'rb') as f:
            profiles = pickle.load(f)
    except Exception as e:
        logger.warning("⚠️ Не удалось прочитать %s (%s), портреты будут перестроены",
                       filename, e)
        return None

    if not profiles:
        return None

    for name, profile in profiles.items():
        n_feat = len(getattr(profile, 'features', []))
        if n_feat != config.N_FEATURES:
            logger.warning(
                "  ❌ %s: портрет повреждён или устарел (%d признаков, ожидалось %d) — "
                "все портреты будут перестроены", name, n_feat, config.N_FEATURES)
            return None

    logger.info("📂 Загружены портреты из %s: %s", filename, ", ".join(profiles))
    return profiles


# ============================================
# ЭТАП 4: ИДЕНТИФИКАЦИЯ АНОНИМНОГО ТЕКСТА
# ============================================

def analyze_anonymous_file(profiles, anonymous_file, extractor=None):
    """Определяет автора анонимного текста из файла.

    Returns:
        (best_author, results, anon_features, similarity_details)
        или None, если файл не удалось прочитать.
    """
    logger.info("=" * 60)
    logger.info("🔍 АНАЛИЗ ФАЙЛА: %s", os.path.basename(anonymous_file))
    logger.info("=" * 60)

    anonymous_text = io_utils.read_text_file(anonymous_file)
    if anonymous_text is None:
        logger.warning("❌ Не удалось прочитать %s", anonymous_file)
        return None

    logger.info("  Длина текста: %d символов", len(anonymous_text))

    extractor = extractor or FeatureExtractor()
    anon_features = extractor.extract(anonymous_text)

    if logger.isEnabledFor(logging.DEBUG):
        for name, val in zip(config.FEATURE_LIST, anon_features):
            logger.debug("    %-17s: %.3f", name, val)

    best_author, results, similarity_details = identify(profiles, anon_features)
    for author_name, similarity in results.items():
        logger.info("  Сходство с %s: %.3f", author_name, similarity)

    best_score = results[best_author]
    logger.info("🎯 РЕЗУЛЬТАТ: %s (уверенность %.2f%%)", best_author, best_score * 100)
    if best_score < config.CONFIDENCE_THRESHOLD:
        logger.info("⚠️  Уверенность ниже порога %.0f%% — возможно, автор не из списка",
                    config.CONFIDENCE_THRESHOLD * 100)

    return best_author, results, anon_features, similarity_details


# ============================================
# ЭТАП 5: ВИЗУАЛИЗАЦИЯ
# ============================================

def _save_figure(fig, filename):
    """Сохраняет фигуру и закрывает её (иначе matplotlib копит фигуры в памяти)."""
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("  ✅ %s", filename)


def _normalize_columns(authors_raw_data, n_features):
    """Мин-макс нормализация по каждому признаку для тепловой карты."""
    normalized = {name: [0.0] * n_features for name in authors_raw_data}
    for i in range(n_features):
        col_values = [values[i] for values in authors_raw_data.values()]
        min_val, max_val = min(col_values), max(col_values)
        span = max_val - min_val
        for name, values in authors_raw_data.items():
            normalized[name][i] = (values[i] - min_val) / span if span > 0 else 0.5
    return normalized


def visualize_file_results(profiles, all_authors_ranges, file_basename,
                           results, anon_features, similarity_details):
    """Строит и сохраняет весь набор графиков для одного анонимного файла."""
    feature_names = config.FEATURE_LIST_SHORT
    author_colors = config.AUTHOR_COLORS
    out = config.OUTPUT_DIR

    logger.info("📊 Строим визуализацию для %s...", file_basename)

    # 1. Индивидуальные розы для каждого автора
    for author_name in profiles:
        try:
            fig = StyleRose.plot_fuzzy_rose(
                all_authors_ranges, anon_features, feature_names,
                authors_to_plot=[author_name],
                author_colors={author_name: author_colors.get(author_name,
                                                              author_colors['default'])},
                title=f"{author_name} vs {file_basename}\n"
                      f"(уверенность {results[author_name]:.1%})",
            )
            _save_figure(fig, f'{out}{file_basename}_{author_name}_vs_anon.png')
        except Exception as e:
            logger.warning("  ❌ Ошибка при построении розы для %s: %s", author_name, e)

    # 2. Общая роза (все авторы на одном графике, та же шкала осей)
    try:
        fig = StyleRose.plot_fuzzy_rose(
            all_authors_ranges, anon_features, feature_names,
            author_colors=author_colors,
            title=f"Все авторы vs {file_basename}",
        )
        _save_figure(fig, f'{out}{file_basename}_all_authors.png')
    except Exception as e:
        logger.warning("  ❌ Ошибка при построении общей розы: %s", e)

    # 3. Графики важности признаков для каждого автора
    for author_name, (sims, weights, contribs) in similarity_details.items():
        try:
            fig = StyleRose.plot_feature_importance(
                author_name, sims, weights, contribs, feature_names,
                title=f"{author_name}: {file_basename} "
                      f"(сходство {results[author_name]:.1%})"
            )
            _save_figure(fig, f'{out}{file_basename}_importance_{author_name}.png')
        except Exception as e:
            logger.warning("  ❌ Ошибка при построении графика важности для %s: %s",
                           author_name, e)

    # 4. Сравнительная диаграмма авторов
    try:
        fig = StyleRose.plot_authors_comparison(
            results, title=f"Сравнение уверенности: {file_basename}")
        _save_figure(fig, f'{out}{file_basename}_authors_comparison.png')
    except Exception as e:
        logger.warning("  ❌ Ошибка при построении сравнительной диаграммы: %s", e)

    # 5. Тепловая карта
    try:
        authors_raw_data = {name: [f.b for f in profile.features]
                            for name, profile in profiles.items()}
        authors_raw_data['Аноним'] = list(anon_features)

        fig = StyleRose.plot_feature_heatmap(
            _normalize_columns(authors_raw_data, len(feature_names)), feature_names,
            title=f"Тепловая карта: {file_basename}")
        _save_figure(fig, f'{out}{file_basename}_heatmap.png')
    except Exception as e:
        logger.warning("  ❌ Ошибка при построении тепловой карты: %s", e)


# ============================================
# ГЛАВНАЯ ФУНКЦИЯ - ЗАПУСК ВСЕГО ПРОЦЕССА
# ============================================

def main():
    config.configure_logging()

    logger.info("=" * 60)
    logger.info("🕵️  НЕЧЁТКИЙ ДЕТЕКТИВ - Определение авторства текста")
    logger.info("=" * 60)

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    logger.info("📁 Вывод графики будет в папку: %s", config.OUTPUT_DIR)

    # ШАГ 1: Пробуем загрузить уже готовые портреты
    profiles = load_profiles()

    if profiles is None:
        logger.info("🔄 Не найдены сохранённые портреты. Начинаем обучение...")

        authors_data = build_authors_profiles()
        if not authors_data:
            logger.error("❌ Ошибка: не найдено ни одного текста для обучения!")
            logger.error("   Создайте папку 'texts/' с подпапками авторов и .txt файлами")
            return 1

        profiles = create_fuzzy_profiles(authors_data)
        save_profiles(profiles)
    else:
        logger.info("✅ Используем готовые портреты авторов")

    # ШАГ 2: Получаем все файлы из папки anonim
    logger.info("=" * 60)
    logger.info("🔎 ШАГ 2: АНАЛИЗ ВСЕХ АНОНИМНЫХ ТЕКСТОВ")
    logger.info("=" * 60)

    anonim_folder = os.path.join(config.BASE_PATH, config.ANON_DIR_NAME)
    anonim_files = io_utils.list_txt_files(anonim_folder)

    if not anonim_files:
        logger.error("❌ В папке %s не найдено .txt файлов!", anonim_folder)
        return 1

    logger.info("📂 Найдено файлов для анализа: %d", len(anonim_files))

    # Диапазоны (a, b, c) всех обученных авторов — считаем один раз, они не
    # зависят от конкретного анонимного файла. Нужны для единой шкалы
    # нормализации в StyleRose.plot_fuzzy_rose (см. src/visualizer.py).
    all_authors_ranges = {
        name: [(f.a, f.b, f.c) for f in profile.features]
        for name, profile in profiles.items()
    }

    extractor = FeatureExtractor(language=Language.RUSSIAN)

    for anonymous_file in anonim_files:
        analysis = analyze_anonymous_file(profiles, anonymous_file, extractor)
        if analysis is None:
            continue
        _best_author, results, anon_features, similarity_details = analysis

        file_basename = os.path.splitext(os.path.basename(anonymous_file))[0]
        visualize_file_results(profiles, all_authors_ranges, file_basename,
                               results, anon_features, similarity_details)
        logger.info("✅ Завершён анализ файла: %s", os.path.basename(anonymous_file))

    logger.info("=" * 60)
    logger.info("✅ ИССЛЕДОВАНИЕ ЗАВЕРШЕНО")
    logger.info("📁 Все графики сохранены в папку %s", config.OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
