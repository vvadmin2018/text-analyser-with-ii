# main.py - Обновлённая точка входа
"""
Text Analyser - Стилометрический анализ текстов
Использует нечёткую логику для определения авторства

Запуск: python main.py
"""

import sys
import os
import numpy as np

# Добавляем путь к новому коду
sys.path.insert(0, os.path.dirname(__file__))

from src.core.feature_extractor import FeatureExtractor
from src.core.profile_builder import ProfileBuilder
from src.core.identifier import FuzzyDetective
from src.core.models import AuthorProfile
from src.visualization.visualizer import StyleRose
from src.utils.text_utils import load_authors_texts
import config
import pickle
import matplotlib.pyplot as plt


def build_and_save_profiles():
    """Строит и сохраняет профили авторов"""
    print("=" * 60)
    print("📚 ЭТАП 1: ПОСТРОЕНИЕ ПРОФИЛЕЙ АВТОРОВ")
    print("=" * 60)
    
    # Загружаем тексты
    authors_data = load_authors_texts(config.BASE_PATH, config.AUTHORS_LIST)
    
    if not authors_data:
        print("❌ Нет текстов для обучения!")
        return None
    
    # Строим профили
    builder = ProfileBuilder(function_type='triangular')
    profiles = {}
    
    for author_name, texts in authors_data.items():
        profile = builder.build_from_texts(author_name, texts, use_quantiles=True)
        profiles[author_name] = profile
    
    # Сохраняем
    with open('authors_profiles.pkl', 'wb') as f:
        pickle.dump(profiles, f)
    print(f"\n✅ Профили сохранены в authors_profiles.pkl")
    
    return profiles


def load_profiles():
    """Загружает сохранённые профили"""
    if os.path.exists('authors_profiles.pkl'):
        with open('authors_profiles.pkl', 'rb') as f:
            profiles = pickle.load(f)
        print(f"✅ Загружено {len(profiles)} профилей")
        return profiles
    return None


def identify_author(profiles, anonymous_file):
    """Идентифицирует автора анонимного текста"""
    print("\n" + "=" * 60)
    print("🔍 ЭТАП 2: ИДЕНТИФИКАЦИЯ АВТОРА")
    print("=" * 60)
    
    if not os.path.exists(anonymous_file):
        print(f"❌ Файл не найден: {anonymous_file}")
        return None
    
    # Читаем текст
    with open(anonymous_file, 'r', encoding='utf-8') as f:
        anonymous_text = f.read()
    
    print(f"📄 Анализируем: {anonymous_file}")
    print(f"   Длина: {len(anonymous_text)} символов")
    
    # Создаём детектива
    detector = FuzzyDetective(profiles)
    
    # Идентифицируем с деталями
    result = detector.identify_with_details(anonymous_text)
    
    best_author = result['best_author']
    confidence = result['confidence']
    
    print(f"\n🎯 РЕЗУЛЬТАТ: {best_author}")
    print(f"   Уверенность: {confidence:.1%}")
    
    # Печатаем все результаты
    print("\n   Сходство с авторами:")
    for author, score in result['all_results'].items():
        marker = "→" if author == best_author else " "
        print(f"   {marker} {author}: {score:.3f}")
    
    if confidence < config.CONFIDENCE_THRESHOLD:
        print("\n⚠️  Уверенность ниже порога - возможно, автор не из списка")
    
    return result


def visualize_results(profiles, anon_features, results, similarity_details, anonymous_file):
    """Визуализирует результаты"""
    print("\n" + "=" * 60)
    print("📊 ЭТАП 3: ВИЗУАЛИЗАЦИЯ")
    print("=" * 60)
    
    feature_names = config.FEATURE_LIST_SHORT
    
    # Подготавливаем данные
    authors_for_plot = {}
    authors_dispersion = {}
    
    for name, profile in profiles.items():
        typical_values = [f.b for f in profile.features]
        dispersion = [(f.c - f.a) / 4 for f in profile.features]
        authors_for_plot[name] = typical_values
        authors_dispersion[name] = dispersion
    
    # Дисперсия анонима
    anon_dispersion = []
    for i in range(len(anon_features)):
        avg_disp = np.mean([authors_dispersion.get(name, [0]*17)[i] 
                           for name in authors_dispersion])
        anon_dispersion.append(avg_disp)
    
    # 1. Индивидуальные розы
    print("\n  Строим индивидуальные сравнения...")
    for author_name in authors_for_plot:
        single_dict = {author_name: authors_for_plot[author_name]}
        single_disp = {author_name: authors_dispersion[author_name]}
        
        fig = StyleRose.plot_fuzzy_rose(
            single_dict, anon_features, feature_names,
            profiles_dispersion=single_disp,
            anonymous_dispersion=anon_dispersion,
            title=f"Сравнение: {author_name} vs аноним\n(уверенность {results[author_name]:.1%})"
        )
        
        filename = f'./output/fuzzy_rose_{author_name}_vs_anon.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    ✅ {filename}")
    
    # 2. Общая роза
    print("\n  Строим общую розу ветров...")
    fig = StyleRose.plot_fuzzy_rose(
        authors_for_plot, anon_features, feature_names,
        profiles_dispersion=authors_dispersion,
        anonymous_dispersion=anon_dispersion,
        title="Нечёткая роза ветров: все авторы"
    )
    plt.savefig('./output/fuzzy_rose_all_authors.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("    ✅ fuzzy_rose_all_authors.png")
    
    # 3. Важность признаков
    print("\n  Строим графики важности признаков...")
    for author_name, details in similarity_details.items():
        sims, weights, contribs = details
        fig = StyleRose.plot_feature_importance(
            author_name, sims, weights, contribs, feature_names,
            title=f"Важность признаков: {author_name}"
        )
        filename = f'./output/feature_importance_{author_name}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    ✅ {filename}")
    
    # 4. Сравнение авторов
    print("\n  Строим сравнительную диаграмму...")
    fig = StyleRose.plot_authors_comparison(results)
    plt.savefig('./output/authors_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("    ✅ authors_comparison.png")
    
    # 5. Тепловая карта
    print("\n  Строим тепловую карту...")
    fig = StyleRose.plot_feature_heatmap(authors_for_plot, feature_names)
    plt.savefig('./output/feature_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("    ✅ feature_heatmap.png")
    
    # 6. PDF отчёт
    print("\n  Создаём PDF отчёт...")
    best_author = max(results, key=results.get)
    StyleRose.create_pdf_report(
        profiles, anon_features, results, similarity_details,
        anonymous_file, best_author,
        output_path='style_analysis_report.pdf'
    )
    
    print("\n✅ Все визуализации сохранены!")


def main():
    """Главная функция"""
    print("\n" + "=" * 60)
    print("🕵️  TEXT ANALYSER - Определение авторства текста")
    print("   Версия 2.0 с улучшенной архитектурой")
    print("=" * 60)
    
    # Шаг 1: Загрузка или построение профилей
    profiles = load_profiles()
    
    if profiles is None:
        print("\n🔄 Профили не найдены, строим новые...")
        profiles = build_and_save_profiles()
        
        if profiles is None:
            print("\n❌ Ошибка при построении профилей")
            return
    else:
        print(f"\n✅ Используем готовые профили: {list(profiles.keys())}")
    
    # Шаг 2: Идентификация
    result = identify_author(profiles, config.ANONIM_TEXT)
    
    if result is None:
        return
    
    # Шаг 3: Визуализация
    anon_features = np.array(result['features'])
    visualize_results(
        profiles, 
        anon_features,
        result['all_results'],
        result['details'],
        config.ANONIM_TEXT
    )
    
    # Финал
    print("\n" + "=" * 60)
    print("✅ АНАЛИЗ ЗАВЕРШЁН")
    print("=" * 60)
    print(f"\n📁 Результаты сохранены в папке output/")
    print(f"📄 PDF отчёт: style_analysis_report.pdf")
    print(f"\n🎯 Определённый автор: {result['best_author']}")
    print(f"   Уверенность: {result['confidence']:.1%}\n")


if __name__ == "__main__":
    main()
