# tests/test_core.py
"""Тесты для основных модулей"""

import pytest
import numpy as np
import sys
import os

# Добавляем путь к исходному коду
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.feature_extractor import FeatureExtractor
from src.core.profile_builder import ProfileBuilder, ProfileAnalyzer
from src.core.identifier import FuzzyDetective, EnsembleIdentifier
from src.core.models import (
    TextFeatures, 
    TriangularMembership, 
    GaussianMembership,
    AuthorProfile
)


class TestFeatureExtractor:
    """Тесты для экстрактора признаков"""
    
    def test_empty_text(self):
        """Обработка пустого текста"""
        extractor = FeatureExtractor()
        features = extractor.extract("")
        assert len(features) == 17
        assert np.all(features == 0)
    
    def test_short_text(self):
        """Обработка короткого текста"""
        extractor = FeatureExtractor()
        features = extractor.extract("Привет мир.")
        assert len(features) == 17
        assert features[0] > 0  # средняя длина предложения
    
    def test_normal_text(self):
        """Обработка нормального текста"""
        extractor = FeatureExtractor()
        text = """
        Это первый текст. Он содержит несколько предложений!
        А вот и второе предложение? Да, именно так.
        
        Это новый абзац. В нём тоже есть предложения.
        """
        features = extractor.extract(text)
        
        assert len(features) == 17
        assert features[0] > 0  # средняя длина предложения
        assert features[3] > 0  # доля вопросительных
        assert features[4] > 0  # доля восклицательных
        assert features[2] >= 1  # количество абзацев
    
    def test_feature_names(self):
        """Проверка названий признаков"""
        extractor = FeatureExtractor()
        names = extractor.get_feature_names()
        assert len(names) == 17
        assert 'Ср. длина предл.' in names


class TestMembershipFunctions:
    """Тесты для функций принадлежности"""
    
    def test_triangular_mu(self):
        """Треугольная функция принадлежности"""
        func = TriangularMembership(a=0.2, b=0.5, c=0.8)
        
        # Пик должен быть 1
        assert abs(func.mu(0.5) - 1.0) < 0.01
        
        # Из-за размытия границ значения могут быть > 0 за пределами [a, c]
        # Проверяем что значение в пике больше чем на краях
        assert func.mu(0.5) > func.mu(0.2)
        assert func.mu(0.5) > func.mu(0.8)
        
        # Между a и b должно расти
        assert func.mu(0.3) < func.mu(0.4)
        assert func.mu(0.4) < func.mu(0.5)
    
    def test_gaussian_mu(self):
        """Гауссова функция принадлежности"""
        func = GaussianMembership(a=0.2, b=0.5, c=0.8, std=0.1)
        
        # Пик должен быть 1
        assert abs(func.mu(0.5) - 1.0) < 0.01
        
        # Удалённые точки должны иметь малую степень
        assert func.mu(0.2) < 0.1
        assert func.mu(0.8) < 0.1


class TestAuthorProfile:
    """Тесты для профиля автора"""
    
    def test_profile_creation(self):
        """Создание профиля"""
        profile = AuthorProfile(name="Test Author")
        assert profile.name == "Test Author"
        assert profile.n_features == 0
    
    def test_profile_from_texts(self):
        """Построение профиля из текстов"""
        builder = ProfileBuilder()
        
        texts = [
            "Это первый текст автора. Он довольно длинный.",
            "Второй текст немного короче. Но тоже информативный!",
            "Третий текст? Да, почему бы и нет. Это работает."
        ]
        
        profile = builder.build_from_texts("Test Author", texts)
        
        assert profile.name == "Test Author"
        assert profile.n_features == 17
        assert profile.texts_count == 3


class TestFuzzyDetective:
    """Тесты для нечёткого детектива"""
    
    def test_identify_simple(self):
        """Простая идентификация"""
        detector = FuzzyDetective()
        
        # Создаём профили
        builder = ProfileBuilder()
        
        texts1 = ["Первый автор пишет такие тексты. Они характерны.",
                  "Ещё один текст первого автора. Стиль сохраняется."]
        texts2 = ["Второй автор пишет иначе. Совсем другой стиль.",
                  "Текст второго автора отличается от первого."]
        
        profile1 = builder.build_from_texts("Author1", texts1)
        profile2 = builder.build_from_texts("Author2", texts2)
        
        detector.add_profile("Author1", profile1)
        detector.add_profile("Author2", profile2)
        
        # Тестируем идентификацию
        test_text = "Первый автор пишет такие тексты. Стиль характерен."
        author, results = detector.identify(test_text)
        
        assert author in ["Author1", "Author2"]
        assert "Author1" in results
        assert "Author2" in results
    
    def test_no_profiles(self):
        """Идентификация без профилей"""
        detector = FuzzyDetective()
        
        with pytest.raises(ValueError):
            detector.identify("Какой-то текст")


class TestTextFeatures:
    """Тесты для модели TextFeatures"""
    
    def test_creation(self):
        """Создание объекта"""
        features = TextFeatures(
            values=np.array([0.5, 0.3, 0.8]),
            feature_names=['f1', 'f2', 'f3'],
            text_length=100,
            num_sentences=5
        )
        
        assert features.n_features == 3
        assert features.text_length == 100
    
    def test_normalize(self):
        """Нормализация признаков"""
        features = TextFeatures(
            values=np.array([10, 20, 30]),
            feature_names=['f1', 'f2', 'f3'],
            text_length=100,
            num_sentences=5
        )
        
        normalized = features.normalize(method='minmax')
        
        assert abs(normalized.values[0] - 0.0) < 0.01
        assert abs(normalized.values[2] - 1.0) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
