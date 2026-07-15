# main.py - Главный файл запуска исследования
# Запускается командой: main.py
import numpy as np
import os
import pickle
from src.feature_extractor import FeatureExtractor
from src.profile_builder import AuthorProfile, TriangularMembership
from src.visualizer import StyleRose
import matplotlib.pyplot as plt
from src import config


# ============================================
# ЭТАП 1: ПОДГОТОВКА - извлекаем признаки из текстов
# ============================================

# def prepare_texts_folder():
#    """
#    Организация папок с текстами:

#    texts/
#    ├── pushkin/
#    │   ├── text1.txt
#    │   ├── text2.txt
#    │   └── text3.txt
#    ├── lermontov/
#    │   ├── text1.txt
#    │   └── text2.txt
#    └── anonim/          # сюда кладём анонимные тексты
#        └── secret.txt
#    """
#    pass


# ============================================
# ЭТАП 2: ОБУЧЕНИЕ - извлекаем тексты для построения портретов известных авторов
# ============================================

def build_authors_profiles():
    """
    Читает тексты из папок
    """
    # extractor = FeatureExtractor()
    authors_data = {}

    # Папка с текстами для обучения
    base_path = config.BASE_PATH
    authors = config.AUTHORS_LIST

    for author in authors:
        author_path = os.path.join(base_path, author)
        if not os.path.exists(author_path):
            print(f"Папка {author_path} не найдена, пропускаем")
            continue

        print(f"\n📖 Обрабатываем автора: {author}")
        author_texts = []

        # Читаем все тексты автора из папки
        for filename in os.listdir(author_path):
            if filename.endswith(".txt"):
                filepath = os.path.join(author_path, filename)
                try:
                    # Читаем файл как байты
                    with open(filepath, 'rb') as f:
                        raw_data = f.read()

                    # Пробуем декодировать в строку разными способами
                    text = None
                    for enc in ['utf-8', 'cp1251', 'koi8-r', 'latin-1']:
                        try:
                            text = raw_data.decode(enc)
                            print(f"  - Файл {filename} прочитан в кодировке {enc}")
                            break
                        except UnicodeDecodeError:
                            continue

                    if text is None:
                        print(f"  ❌ Не могу декодировать {filename}")
                        continue

                    # Проверяем, что текст не пустой
                    if len(text.strip()) == 0:
                        print(f"  ⚠️  Файл {filename} пустой, пропускаем")
                        continue

                    print(f"    Длина текста: {len(text)} символов")

                    # СОХРАНЯЕМ ИМЕННО ТЕКСТ, А НЕ ПРИЗНАКИ
                    author_texts.append(text)

                except Exception as e:
                    print(f"  ❌ Ошибка при чтении {filename}: {e}")

        if author_texts:
            # Передаём СПИСОК ТЕКСТОВ
            authors_data[author] = author_texts
            print(f"  ✅ Получено {len(author_texts)} текстов для автора {author}")
        else:
            print(f"  ❌ Нет текстов для автора {author}")

    return authors_data


# ============================================
# ЭТАП 3: СОЗДАНИЕ НЕЧЁТКИХ ПОРТРЕТОВ
# ============================================

def create_fuzzy_profiles(authors_data):
    """
    По признакам строит треугольные функции принадлежности
    """
    profiles = {}

    for author_name, texts_features in authors_data.items():
        print(f"\n🖌️  Строим нечёткий портрет для {author_name}")

        # Создаём профиль автора
        profile = AuthorProfile(author_name)

        # Передаём ВСЕ тексты для построения функций
        profile.build_from_texts(texts_features)

        profiles[author_name] = profile
        print(f"  ✅ Портрет построен (набор треугольных функций)")

    return profiles


# ============================================
# ЭТАП 4: СОХРАНЕНИЕ ПОРТРЕТОВ (чтобы не пересчитывать каждый раз)
# ============================================

def save_profiles(profiles, filename="authors_profiles.pkl"):
    """
    Сохраняем портреты в файл, чтобы при следующих запусках
    не пересчитывать заново
    """
    with open(filename, 'wb') as f:
        pickle.dump(profiles, f)
    print(f"\n  Портреты сохранены в {filename}")


def load_profiles(filename="authors_profiles.pkl"):
    """
    Загружаем ранее сохранённые портреты
    """
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            profiles = pickle.load(f)
        print(f"\n📂 Загружены портреты из {filename}")

        # Проверяем, что портреты корректны
        valid_profiles = {}
        for name, profile in profiles.items():
            n_feat = len(getattr(profile, 'features', []))
            if n_feat == config.N_FEATURES:
                valid_profiles[name] = profile
                print(f"  ✅ {name}: {n_feat} функций")
            else:
                print(f"  ❌ {name}: портрет повреждён или устарел "
                      f"({n_feat} признаков, ожидалось {config.N_FEATURES}), будет перестроен")

        if valid_profiles:
            return valid_profiles
        else:
            print("  Все портреты повреждены, нужно перестроить")
            return None
    return None


# ============================================
# ЭТАП 5: ИДЕНТИФИКАЦИЯ АНОНИМНОГО ТЕКСТА
# ============================================

def identify_anonymous_text(profiles, anonymous_filepath):
    """
    Определяет автора анонимного текста
    """

    print(f"\n🔍 Анализируем анонимный текст: {anonymous_filepath}")

    # Читаем текст
    with open(anonymous_filepath, 'r', encoding='utf-8') as f:
        anonymous_text = f.read()

    print(f"  Длина текста: {len(anonymous_text)} символов")

    # Извлекаем признаки
    extractor = FeatureExtractor()
    anon_features = extractor.extract(anonymous_text)

    print("  Признаки анонимного текста:")
    feature_names = config.FEATURE_LIST
    for i, (name, val) in enumerate(zip(feature_names, anon_features)):
        if config.LEVEL_LOG == "DEBUG":
            print(f"    {name:17}: {val:.3f}")

    results = {}
    similarity_details = {}

    # Сравниваем с каждым автором
    for author_name, profile in profiles.items():
        similarity, details = profile.similarity_with_details(anon_features)
        results[author_name] = similarity
        similarity_details[author_name] = details
        print(f"  Сходство с {author_name}: {similarity:.3f}")

    # Определяем лучшего
    best_author = max(results, key=results.get)
    best_score = results[best_author]

    print(f"\n🎯 РЕЗУЛЬТАТ: {best_author} (уверенность {best_score:.2%})")

    if best_score < 0.5:
        print("⚠️  Но уверенность низкая - возможно, автор не из списка")

    return best_author, results, anon_features, similarity_details


# ============================================
# ЭТАП 6: ВИЗУАЛИЗАЦИЯ - РОЗА ВЕТРОВ
# ============================================
# Реальная визуализация теперь строится через StyleRose.plot_fuzzy_rose()
# прямо в main() (см. ниже). Полоса неопределённости = реальный диапазон
# [a, c] автора, нормализация — по единой шкале всех обученных авторов
# (не по паре "автор vs аноним", как в исходной версии) — см. src/visualizer.py.
#
# Ранее здесь лежали функции visualize_detailed_results(), visualize_results()
# и create_pdf_report() (~320 строк) — они никогда не вызывались из main()
# и дублировали (местами с ошибками, напр. неверный импорт visualizer вместо
# src.visualizer, и пропущенный префикс f у f-строк с именами файлов) логику,
# реализованную ниже. Удалены как мёртвый код.

# ============================================
# ГЛАВНАЯ ФУНКЦИЯ - ЗАПУСК ВСЕГО ПРОЦЕССА
# ============================================

def analyze_anonymous_file(profiles, anonymous_file):
    """
    Анализирует один анонимный файл и возвращает результаты
    """
    print(f"\n{'=' * 60}")
    print(f"🔍 АНАЛИЗ ФАЙЛА: {os.path.basename(anonymous_file)}")
    print(f"{'=' * 60}")

    # Идентифицируем автора
    best_author, results, anon_features, similarity_details = identify_anonymous_text(profiles, anonymous_file)

    # ========== ДИАГНОСТИКА ЗНАЧЕНИЙ ПРИЗНАКОВ ==========
    if config.LEVEL_LOG == "DEBUG":
        print("\n" + "=" * 60)
        print("🔍 ДЕТАЛЬНАЯ ДИАГНОСТИКА ЗНАЧЕНИЙ ПРИЗНАКОВ")
        print("=" * 60)

        # Проверяем каждый профиль
        for name, profile in profiles.items():
            print(f"\n📖 АВТОР: {name}")
            typical_values = []
            for i, feat_func in enumerate(profile.features):
                val = feat_func.b
                typical_values.append(val)
                print(f"    Признак {i:2d}: b = {val:.6f}")

            print(f"  → Диапазон: [{min(typical_values):.6f}, {max(typical_values):.6f}]")
            print(f"  → Среднее: {np.mean(typical_values):.6f}")

        print(f"\n📄 АНОНИМНЫЙ ТЕКСТ:")
        feature_names_diag = [
            'Предл', 'Дисп.', 'Абзац', '?', '!', '...', 'Прям. речь', 'TTR',
            'Сущ', 'Глаг', 'Прил', 'ДлСл', ',',
            '—', ':', 'Союз', 'Предлоги'
        ]
        for i, (name, val) in enumerate(zip(feature_names_diag, anon_features)):
            print(f"    {name:10}: {val:.6f}")

    return best_author, results, anon_features, similarity_details


def main():
    print("=" * 60)
    print("🕵️  НЕЧЁТКИЙ ДЕТЕКТИВ - Определение авторства текста")
    print("=" * 60)

    # Создаём папку output с timestamp если она не существует
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    print(f"\n📁 Вывод графики будет в папку: {config.OUTPUT_DIR}")

    # ШАГ 1: Пробуем загрузить уже готовые портреты
    profiles = load_profiles()

    if profiles is None:
        print("\n🔄 Не найдены сохранённые портреты. Начинаем обучение...")

        # ШАГ 2: Собираем данные из текстов
        authors_data = build_authors_profiles()

        if not authors_data:
            print("❌ Ошибка: не найдено ни одного текста для обучения!")
            print("   Создайте папку 'texts/' с подпапками авторов и .txt файлами")
            return

        # ШАГ 3: Строим нечёткие портреты
        profiles = create_fuzzy_profiles(authors_data)

        # ШАГ 4: Сохраняем портреты
        save_profiles(profiles)
    else:
        print("\n✅ Используем готовые портреты авторов:")
        for name in profiles.keys():
            print(f"   - {name}")

    # ШАГ 5: Получаем все файлы из папки anonim
    print("\n" + "=" * 60)
    print("🔎 ШАГ 2: АНАЛИЗ ВСЕХ АНОНИМНЫХ ТЕКСТОВ")
    print("=" * 60)

    anonim_folder = os.path.join(config.BASE_PATH, "anonim")

    if not os.path.exists(anonim_folder):
        print(f"❌ Папка {anonim_folder} не найдена!")
        return

    # Получаем список всех .txt файлов в папке anonim
    anonim_files = [
        os.path.join(anonim_folder, f)
        for f in os.listdir(anonim_folder)
        if f.endswith('.txt')
    ]

    if not anonim_files:
        print(f"❌ В папке {anonim_folder} не найдено .txt файлов!")
        return

    print(f"\n📂 Найдено файлов для анализа: {len(anonim_files)}")
    for f in anonim_files:
        print(f"   - {os.path.basename(f)}")

    # ШАГ 6: Анализируем каждый файл и строим визуализацию

    # Диапазоны (a, b, c) всех обученных авторов — считаем один раз, они не
    # зависят от конкретного анонимного файла. Нужны для единой шкалы
    # нормализации в StyleRose.plot_fuzzy_rose (см. src/visualizer.py).
    all_authors_ranges = {
        name: [(f.a, f.b, f.c) for f in profile.features]
        for name, profile in profiles.items()
    }

    for anonymous_file in anonim_files:
        # Анализируем файл
        best_author, results, anon_features, similarity_details = analyze_anonymous_file(
            profiles, anonymous_file
        )

        # Определяем базовое имя файла для сохранения графиков
        file_basename = os.path.splitext(os.path.basename(anonymous_file))[0]

        feature_names_short = config.FEATURE_LIST_SHORT
        # Единая палитра цветов авторов из config.py (раньше здесь был
        # локальный словарь только на 4 автора, из-за чего drugkov/saharnov/
        # kolas/maur/bryl всегда рисовались цветом по умолчанию)
        author_colors = config.AUTHOR_COLORS

        # ===== ВИЗУАЛИЗАЦИЯ ДЛЯ ЭТОГО ФАЙЛА =====
        print(f"\n📊 Строим визуализацию для {os.path.basename(anonymous_file)}...")

        # 1. Индивидуальные розы для каждого автора: закрашенная полоса —
        #    реальный диапазон [a, c] автора, линия — типичное значение (b),
        #    отдельная линия — сам анонимный текст.
        for author_name in profiles.keys():
            try:
                author_color = author_colors.get(author_name, author_colors['default'])

                fig_single = StyleRose.plot_fuzzy_rose(
                    all_authors_ranges,
                    anon_features,
                    feature_names_short,
                    authors_to_plot=[author_name],
                    author_colors={author_name: author_color},
                    title=f"{author_name} vs {file_basename}\n(уверенность {results[author_name]:.1%})",
                )

                filename = f'{config.OUTPUT_DIR}{file_basename}_{author_name}_vs_anon.png'
                plt.savefig(filename, dpi=150, bbox_inches='tight')
                print(f"  ✅ {filename}")
                plt.close(fig_single)

            except Exception as e:
                print(f"  ❌ Ошибка при построении розы для {author_name}: {e}")

        # 2. Общая роза (все авторы на одном графике, та же шкала осей)
        try:
            fig_all = StyleRose.plot_fuzzy_rose(
                all_authors_ranges,
                anon_features,
                feature_names_short,
                author_colors=author_colors,
                title=f"Все авторы vs {file_basename}",
            )
            filename = f'{config.OUTPUT_DIR}{file_basename}_all_authors.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"  ✅ {filename}")
            plt.close(fig_all)
        except Exception as e:
            print(f"  ❌ Ошибка при построении общей розы: {e}")

        # 3. Графики важности признаков для каждого автора
        for author_name, (sims, weights, contribs) in similarity_details.items():
            try:
                fig = StyleRose.plot_feature_importance(
                    author_name, sims, weights, contribs, feature_names_short,
                    title=f"{author_name}: {file_basename} (сходство {results[author_name]:.1%})"
                )
                filename = f'{config.OUTPUT_DIR}{file_basename}_importance_{author_name}.png'
                plt.savefig(filename, dpi=150, bbox_inches='tight')
                print(f"  ✅ {filename}")
                plt.close(fig)
            except Exception as e:
                print(f"  ❌ Ошибка при построении графика важности для {author_name}: {e}")

        # 4. Сравнительная диаграмма авторов
        try:
            fig2 = StyleRose.plot_authors_comparison(
                results,
                title=f"Сравнение уверенности: {file_basename}"
            )
            filename = f'{config.OUTPUT_DIR}{file_basename}_authors_comparison.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"  ✅ {filename}")
            plt.close(fig2)
        except Exception as e:
            print(f"  ❌ Ошибка при построении сравнительной диаграммы: {e}")

        # 5. Тепловая карта
        try:
            authors_raw_data = {}
            for name, profile in profiles.items():
                typical_values = []
                for feat_func in profile.features:
                    typical_values.append(feat_func.b)
                authors_raw_data[name] = typical_values

            authors_raw_data['Аноним'] = anon_features.tolist()

            n_features = len(feature_names_short)
            authors_data_norm = {name: [0] * n_features for name in authors_raw_data.keys()}

            for i in range(n_features):
                col_values = [authors_raw_data[name][i] for name in authors_raw_data.keys()]
                min_val = min(col_values)
                max_val = max(col_values)

                if max_val > min_val:
                    for name in authors_raw_data.keys():
                        val = authors_raw_data[name][i]
                        authors_data_norm[name][i] = (val - min_val) / (max_val - min_val)
                else:
                    for name in authors_raw_data.keys():
                        authors_data_norm[name][i] = 0.5

            fig3 = StyleRose.plot_feature_heatmap(
                authors_data_norm, feature_names_short,
                title=f"Тепловая карта: {file_basename}"
            )
            filename = f'{config.OUTPUT_DIR}{file_basename}_heatmap.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"  ✅ {filename}")
            plt.close(fig3)
        except Exception as e:
            print(f"  ❌ Ошибка при построении тепловой карты: {e}")

        print(f"\n✅ Завершён анализ файла: {os.path.basename(anonymous_file)}")

    print("\n" + "=" * 60)
    print("✅ ИССЛЕДОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 60)
    print(f"\n📁 Все графики сохранены в папку {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()