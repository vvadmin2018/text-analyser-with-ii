# main.py - Главный файл запуска исследования
# Запускается командой: python src/main.py
import numpy as np
import os
import pickle
from src.feature_extractor import FeatureExtractor
from src.profile_builder import AuthorProfile, TriangularMembership
from src.identifier import FuzzyDetective
from src.visualizer import StyleRose
import matplotlib.pyplot as plt
from src import config

# ============================================
# ЭТАП 1: ПОДГОТОВКА - извлекаем признаки из текстов
# ============================================

#def prepare_texts_folder():
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
    #extractor = FeatureExtractor()
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
            if hasattr(profile, 'features') and len(profile.features) > 0:
                valid_profiles[name] = profile
                print(f"  ✅ {name}: {len(profile.features)} функций")
            else:
                print(f"  ❌ {name}: портрет поврежден, будет перестроен")

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

def visualize_detailed_results(profiles, anon_features, results, similarity_details):
    """
    Детальная визуализация результатов с весами
    similarity_details: словарь с данными {author_name: (similarities, weights, contributions)}
    """
    from visualizer import StyleRose

    print("\n Строим детальную визуализацию...")

    feature_names_short = config.FEATURE_LIST_SHORT

    # 1. Для каждого автора строим график важности признаков
    for author_name, (sims, weights, contribs) in similarity_details.items():
        print(f"\n  Строим график для {author_name}...")
        fig = StyleRose.plot_feature_importance(
            author_name, sims, weights, contribs, feature_names_short,
            title=f"Анализ признаков: {author_name} (сходство {results[author_name]:.1%})"
        )
        plt.savefig(f'{config.OUTPUT_DIR}feature_importance_{author_name}.png', dpi=150, bbox_inches='tight')
        print(f"  ✅ feature_importance_{author_name}.png сохранён")
        plt.close(fig)

    # 2. Сравнительная диаграмма авторов
    print("\n  Строим сравнительную диаграмму...")
    fig2 = StyleRose.plot_authors_comparison(results, title="Сравнение уверенности идентификации")
    filename='{config.OUTPUT_DIR}authors_comparison.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print("  ✅ authors_comparison.png сохранён")
    plt.show()

    # 3. Тепловая карта всех признаков (нормализация по признакам)
    print("\n  Строим тепловую карту...")

    # Собираем данные
    authors_raw_data = {}
    for name, profile in profiles.items():
        typical_values = []
        for feat_func in profile.features:
            typical_values.append(feat_func.b)
        authors_raw_data[name] = typical_values

    authors_raw_data['Аноним'] = anon_features.tolist()

    # Нормализуем КАЖДЫЙ ПРИЗНАК отдельно (по столбцам)
    n_features = len(feature_names_short)
    authors_data = {name: [0] * n_features for name in authors_raw_data.keys()}

    for i in range(n_features):
        # Собираем значения i-го признака у всех авторов
        col_values = [authors_raw_data[name][i] for name in authors_raw_data.keys()]
        min_val = min(col_values)
        max_val = max(col_values)

        # Нормализуем i-й признак для всех авторов
        if max_val > min_val:
            for name in authors_raw_data.keys():
                val = authors_raw_data[name][i]
                authors_data[name][i] = (val - min_val) / (max_val - min_val)
        else:
            # Если все значения одинаковы, ставим 0.5
            for name in authors_raw_data.keys():
                authors_data[name][i] = 0.5

    if config.LEVEL_LOG == "DEBUG":
        for name, values in authors_data.items():
            print(f"  {name}: {[f'{v:.3f}' for v in values[:5]]}...")

    fig3 = StyleRose.plot_feature_heatmap(
        authors_data, feature_names_short,
        title="Сравнение профилей авторов (нормализация по признакам)"
    )
    filename='{config.OUTPUT_DIR}feature_heatmap.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print("  ✅ feature_heatmap.png сохранён")
    plt.close(fig3)
    print("\n  ✅ Все графики сохранены!")


def visualize_results(profiles, anon_features, results):
    """
    Строит графики с нормализацией по максимуму и размытыми секторами
    """
    print("\n📊 Строим улучшенную визуализацию с размытыми секторами...")

    # Подготавливаем данные
    authors_for_plot = {}
    authors_dispersion = {}

    for name, profile in profiles.items():
        typical_values = []
        dispersion_values = []

        for feat_func in profile.features:
            typical_values.append(feat_func.b)
            # Используем размах как меру неопределённости
            dispersion = (feat_func.c - feat_func.a) / 4  # треть размаха для лучшей виуализации
            dispersion_values.append(dispersion)

        authors_for_plot[name] = typical_values
        authors_dispersion[name] = dispersion_values

    # Названия признаков
    feature_names = [
        'Предл', 'Дисп', 'Абзац', '?', '!', '...', 'Прям. речь',
        'TTR', 'Сущ', 'Глаг', 'Прил', 'ДлСл',
        ',', '—', ':', 'Союз', 'Предлоги'
    ]

    # Дисперсия для анонимного текста (не используется, но оставляем для совместимости)
    anon_dispersion = []
    for i in range(len(anon_features)):
        avg_disp = np.mean([authors_dispersion.get(name, [0] * 14)[i]
                            for name in authors_dispersion.keys()])
        anon_dispersion.append(avg_disp)

    # Выводим исходные значения для наглядности
    if config.LEVEL_LOG == "DEBUG":
        print("\n📊 Исходные значения (до нормализации):")
        print("-" * 70)
        print(f"{'Признак':<12} {'Пушкин':<10} {'Лермонтов':<10} {'Толстой':<10} {'Аноним':<10} {'ДиспПуш':<10} {'ДиспЛер':<10} {'ДиспТолстой':<10} {'ДиспАноним':<10}")
        print("-" * 70)

    pushkin_values = authors_for_plot.get('pushkin', [])
    lermontov_values = authors_for_plot.get('lermontov', [])
    tolstoy_values = authors_for_plot.get('tolstoy', [])
    pushkin_disp = authors_dispersion.get('pushkin', [])
    lermontov_disp = authors_dispersion.get('lermontov', [])
    tolstoy_disp = authors_dispersion.get('tolstoy', [])

    for i, name in enumerate(feature_names):
        p_val = pushkin_values[i] if i < len(pushkin_values) else 0
        l_val = lermontov_values[i] if i < len(lermontov_values) else 0
        t_val = tolstoy_values[i] if i < len(tolstoy_values) else 0
        a_val = anon_features[i]
        p_disp = pushkin_disp[i] if i < len(pushkin_disp) else 0
        l_disp = lermontov_disp[i] if i < len(lermontov_disp) else 0
        t_disp = tolstoy_disp[i] if i < len(tolstoy_disp) else 0
        a_disp = anon_dispersion[i]

        if config.LEVEL_LOG == "DEBUG":
            print(f"{name:<12} {p_val:<10.3f} {l_val:<10.3f} {t_val:<10.3f} {a_val:<10.3f} {p_disp:<10.3f} {l_disp:<10.3f} {t_disp:<10.3f} {a_disp:<10.3f}")

    # ===== ЦВЕТА ДЛЯ КАЖДОГО АВТОРА =====
    author_colors = {
        'pushkin': '#FF4B4B',      # ярко-красный
        'lermontov': '#4B7BFF',    # ярко-синий
        'tolstoy': '#FFB44B',      # оранжевый
        'bulichev': '#FF4BFF',      # розовый
        'default': '#4B7BFF'       # синий по умолчанию
    }

    # ===== 1. ОБЩАЯ РОЗА ВЕТРОВ (все авторы + аноним) =====
    try:
        fig_all = StyleRose.plot_fuzzy_rose(
            authors_for_plot,
            anon_features,
            feature_names,
            profiles_dispersion=authors_dispersion,
            anonymous_dispersion=anon_dispersion,
            title=f"Нечёткая роза ветров: все авторы vs аноним",
            author_colors = author_colors  # передаем цвета
        )
        filename='{config.OUTPUT_DIR}fuzzy_rose_all_authors.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print("\n  ✅ fuzzy_rose_all_authors.png сохранён (все авторы)")
        plt.close(fig_all)
    except Exception as e:
        print(f"  ❌ Ошибка при построении общей розы: {e}")

    # ===== 2. ИНДИВИДУАЛЬНЫЕ РОЗЫ (каждый автор отдельно с анонимом) =====
    print("\n📊 Строим индивидуальные розы для каждого автора...")

    for author_name in authors_for_plot.keys():
        try:
            # Создаем словарь только с одним автором
            single_author_dict = {author_name: authors_for_plot[author_name]}
            single_dispersion_dict = {author_name: authors_dispersion[author_name]}

            # Получаем цвет для этого автора
            author_color = author_colors.get(author_name, author_colors['default'])


            # Строим розу для пары "автор vs аноним"
            fig_single = StyleRose.plot_fuzzy_rose(
                single_author_dict,
                anon_features,
                feature_names,
                profiles_dispersion=single_dispersion_dict,
                anonymous_dispersion=anon_dispersion,
                title=f"Сравнение стилей: {author_name} vs анонимный текст\n(уверенность {results[author_name]:.1%})",
                author_colors = {author_name: author_color}  # передаем цвет для этого автора
            )

            # Сохраняем с именем автора
            filename = f'{config.OUTPUT_DIR}fuzzy_rose_{author_name}_vs_anon.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"  ✅ {filename} сохранён (цвет: {author_color})")
            plt.close(fig_single)

        except Exception as e:
            print(f"  ❌ Ошибка при построении розы для {author_name}: {e}")

    # ===== 4. КЛАССИЧЕСКАЯ РОЗА (для сравнения) =====
    try:
        fig_classic = StyleRose.plot_max_normalized(
            authors_for_plot,
            anon_features,
            feature_names,
            title=f"Классическая роза ветров: все авторы vs аноним"
        )
        filename = f'{config.OUTPUT_DIR}style_rose_classic.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print("  ✅ style_rose_classic.png сохранён (классическая версия)")
        plt.close(fig_classic)
    except Exception as e:
        print(f"  ❌ Ошибка при построении классической розы: {e}")

def create_pdf_report(profiles, anon_features, results, similarity_details, anonymous_file, best_author, authors_for_plot, authors_dispersion, anon_dispersion, feature_names, feature_names_short):
    """
    Создает PDF-отчет со всеми графиками
    
    Args:
        profiles: словарь профилей авторов
        anon_features: признаки анонимного текста
        results: результаты сходства
        similarity_details: детали сходства
        anonymous_file: путь к файлу анонимного текста
        best_author: определенный автор
        authors_for_plot: данные для визуализации
        authors_dispersion: дисперсия профилей
        anon_dispersion: дисперсия анонима
        feature_names: полные названия признаков
        feature_names_short: короткие названия признаков
    """
    from matplotlib.backends.backend_pdf import PdfPages

    print("\n📑 Создаем PDF-отчет...")

    with PdfPages('style_analysis_report.pdf') as pdf:

        # 1. Титульная страница
        fig = plt.figure(figsize=(11, 8.5))
        plt.axis('off')
        plt.text(0.5, 0.7, 'Анализ авторства текста',
                 fontsize=24, ha='center', fontweight='bold')
        plt.text(0.5, 0.6, 'Нечёткая логика и стилевые профили',
                 fontsize=18, ha='center')
        plt.text(0.5, 0.4, f'Анонимный текст: {os.path.basename(anonymous_file)}',
                 fontsize=14, ha='center')
        plt.text(0.5, 0.3, f'Определён автор: {best_author} (уверенность {results[best_author]:.1%})',
                 fontsize=14, ha='center', color='green')
        pdf.savefig()
        plt.close(fig)

        # 2. Индивидуальные розы для каждого автора
        for author_name in authors_for_plot.keys():
            single_author_dict = {author_name: authors_for_plot[author_name]}
            single_dispersion_dict = {author_name: authors_dispersion[author_name]}

            fig = StyleRose.plot_fuzzy_rose(
                single_author_dict,
                anon_features,
                feature_names,
                profiles_dispersion=single_dispersion_dict,
                anonymous_dispersion=anon_dispersion,
                title=f"Сравнение: {author_name} vs аноним\n(уверенность {results[author_name]:.1%})"
            )
            pdf.savefig()
            plt.close(fig)

        # 3. Общая роза
        fig = StyleRose.plot_fuzzy_rose(
            authors_for_plot,
            anon_features,
            feature_names,
            profiles_dispersion=authors_dispersion,
            anonymous_dispersion=anon_dispersion,
            title="Нечёткая роза ветров: все авторы"
        )
        pdf.savefig()
        plt.close(fig)

        # 4. Графики важности признаков
        for author_name, (sims, weights, contribs) in similarity_details.items():
            fig = StyleRose.plot_feature_importance(
                author_name, sims, weights, contribs, feature_names_short,
                title=f"Важность признаков: {author_name}"
            )
            pdf.savefig()
            plt.close(fig)

        # 5. Сравнительная диаграмма
        fig = StyleRose.plot_authors_comparison(results)
        pdf.savefig()
        plt.close(fig)

        # 6. Тепловая карта
        authors_data = {}
        for name, profile in profiles.items():
            typical_values = [f.b for f in profile.features]
            max_val = max(typical_values)
            authors_data[name] = [v / max_val for v in typical_values] if max_val > 0 else typical_values

        anon_max = max(anon_features)
        authors_data['Аноним'] = [v / anon_max for v in anon_features] if anon_max > 0 else anon_features.tolist()

        fig = StyleRose.plot_feature_heatmap(
            authors_data, feature_names_short,
            title="Тепловая карта признаков"
        )
        pdf.savefig()
        plt.close(fig)

    print("  ✅ style_analysis_report.pdf сохранён")

# ============================================
# ГЛАВНАЯ ФУНКЦИЯ - ЗАПУСК ВСЕГО ПРОЦЕССА
# ============================================

def analyze_anonymous_file(profiles, anonymous_file):
    """
    Анализирует один анонимный файл и возвращает результаты
    """
    print(f"\n{'='*60}")
    print(f"🔍 АНАЛИЗ ФАЙЛА: {os.path.basename(anonymous_file)}")
    print(f"{'='*60}")
    
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
           'Предл', 'Дисп.', 'Абзац', '?', '!','...', 'Прям. речь', 'TTR',
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
    for anonymous_file in anonim_files:
        # Анализируем файл
        best_author, results, anon_features, similarity_details = analyze_anonymous_file(
            profiles, anonymous_file
        )
        
        # Определяем базовое имя файла для сохранения графиков
        file_basename = os.path.splitext(os.path.basename(anonymous_file))[0]
        
        # Подготовка данных для визуализации
        authors_for_plot = {}
        authors_dispersion = {}
        
        for name, profile in profiles.items():
            typical_values = []
            dispersion_values = []
            
            for feat_func in profile.features:
                typical_values.append(feat_func.b)
                dispersion = (feat_func.c - feat_func.a) / 4
                dispersion_values.append(dispersion)
            
            authors_for_plot[name] = typical_values
            authors_dispersion[name] = dispersion_values
        
        anon_dispersion = []
        for i in range(len(anon_features)):
            avg_disp = np.mean([authors_dispersion.get(name, [0] * 14)[i]
                                for name in authors_dispersion.keys()])
            anon_dispersion.append(avg_disp)
        
        feature_names = [
            'Предл', 'Дисп', 'Абзац', '?', '!', '...', 'Прям. речь',
            'TTR', 'Сущ', 'Глаг', 'Прил', 'ДлСл', ',', '—', ':', 'Союз', 'Предлоги'
        ]
        feature_names_short = config.FEATURE_LIST_SHORT
        
        # ===== ЦВЕТА ДЛЯ КАЖДОГО АВТОРА =====
        author_colors = {
            'pushkin': '#FF4B4B',      # ярко-красный
            'lermontov': '#4B7BFF',    # ярко-синий
            'tolstoy': '#FFB44B',      # оранжевый
            'bulichev': '#FF4BFF',     # розовый
            'default': '#4B7BFF'       # синий по умолчанию
        }
        
        # ===== ВИЗУАЛИЗАЦИЯ ДЛЯ ЭТОГО ФАЙЛА =====
        print(f"\n📊 Строим визуализацию для {os.path.basename(anonymous_file)}...")
        
        # 1. Индивидуальные розы для каждого автора
        for author_name in authors_for_plot.keys():
            try:
                single_author_dict = {author_name: authors_for_plot[author_name]}
                single_dispersion_dict = {author_name: authors_dispersion[author_name]}
                author_color = author_colors.get(author_name, author_colors['default'])
                
                fig_single = StyleRose.plot_fuzzy_rose(
                    single_author_dict,
                    anon_features,
                    feature_names,
                    profiles_dispersion=single_dispersion_dict,
                    anonymous_dispersion=anon_dispersion,
                    title=f"{author_name} vs {file_basename}\n(уверенность {results[author_name]:.1%})",
                    author_colors={author_name: author_color}
                )
                
                filename = f'{config.OUTPUT_DIR}{file_basename}_{author_name}_vs_anon.png'
                plt.savefig(filename, dpi=150, bbox_inches='tight')
                print(f"  ✅ {filename}")
                plt.close(fig_single)
                
            except Exception as e:
                print(f"  ❌ Ошибка при построении розы для {author_name}: {e}")
        
        # 2. Общая роза ветров (все авторы)
        try:
            fig_all = StyleRose.plot_fuzzy_rose(
                authors_for_plot,
                anon_features,
                feature_names,
                profiles_dispersion=authors_dispersion,
                anonymous_dispersion=anon_dispersion,
                title=f"Все авторы vs {file_basename}",
                author_colors=author_colors
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