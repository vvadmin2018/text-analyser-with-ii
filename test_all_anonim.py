# -*- coding: utf-8 -*-
"""
Тестирование всех анонимных текстов
"""
import os
import pickle
from feature_extractor import FeatureExtractor
from profile_builder import AuthorProfile

BASE_PATH = '/workspace/texts'
AUTHORS_LIST = ['pushkin', 'lermontov', 'tolstoy', 'bulichev']

def load_profiles(filename="authors_profiles.pkl"):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            profiles = pickle.load(f)
        return profiles
    return None

def test_anonymous_text(profiles, anonymous_filepath):
    with open(anonymous_filepath, 'r', encoding='utf-8') as f:
        anonymous_text = f.read()
    
    extractor = FeatureExtractor()
    anon_features = extractor.extract(anonymous_text)
    
    results = {}
    for author_name, profile in profiles.items():
        similarity, _ = profile.similarity_with_details(anon_features)
        results[author_name] = similarity
    
    best_author = max(results, key=results.get)
    best_score = results[best_author]
    
    return best_author, best_score, results

profiles = load_profiles()
if not profiles:
    print("❌ Портреты не загружены!")
    exit(1)

print("=" * 70)
print("ТЕСТИРОВАНИЕ ВСЕХ АНОНИМНЫХ ТЕКСТОВ")
print("=" * 70)

anonim_path = os.path.join(BASE_PATH, 'anonim')
for filename in sorted(os.listdir(anonim_path)):
    if filename.endswith('.txt'):
        filepath = os.path.join(anonim_path, filename)
        best_author, best_score, results = test_anonymous_text(profiles, filepath)
        
        print(f"\n📄 {filename}")
        print(f"   Размер: {os.path.getsize(filepath)} символов")
        print(f"   🎯 Результат: {best_author} ({best_score:.1%})")
        
        # Показываем все результаты
        for author, score in sorted(results.items(), key=lambda x: -x[1]):
            marker = " >>>" if author == best_author else ""
            print(f"      - {author}: {score:.1%}{marker}")
        
        # Проверка порога 70%
        if best_score >= 0.70:
            print(f"   ✅ Уверенность ВЫСОКАЯ (≥70%)")
        elif best_score >= 0.50:
            print(f"   ⚠️  Уверенность СРЕДНЯЯ (50-70%)")
        else:
            print(f"   ❌ Уверенность НИЗКАЯ (<50%) - авторство не определено")

print("\n" + "=" * 70)
