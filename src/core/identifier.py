# src/core/identifier.py
"""Идентификация авторства текста"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter

from .models import AuthorProfile, TextFeatures
from .feature_extractor import FeatureExtractor


class FuzzyDetective:
    """
    Нечёткий детектив - определение авторства на основе нечёткой логики.
    
    Поддерживает:
    - Взвешенное сходство с профилями
    - Детальный анализ вклада признаков
    - Бутстрэп оценку уверенности
    """
    
    def __init__(self, profiles: Optional[Dict[str, AuthorProfile]] = None):
        """
        Инициализация детектива
        
        Args:
            profiles: словарь профилей авторов {имя: профиль}
        """
        self.profiles = profiles or {}
        self.extractor = FeatureExtractor()
    
    def add_profile(self, name: str, profile: AuthorProfile):
        """Добавляет профиль автора"""
        self.profiles[name] = profile
    
    def identify(self, text: str, 
                 use_weights: bool = True,
                 threshold: float = 0.5) -> Tuple[str, Dict[str, float]]:
        """
        Определяет автора текста
        
        Args:
            text: анонимный текст
            use_weights: использовать ли веса признаков
            threshold: порог уверенности
            
        Returns:
            (имя_автора, результаты_сходства)
        """
        features = self.extractor.extract(text)
        return self.identify_features(features, use_weights, threshold)
    
    def identify_features(self, features: np.ndarray,
                          use_weights: bool = True,
                          threshold: float = 0.5) -> Tuple[str, Dict[str, float]]:
        """
        Определяет автора по вектору признаков
        
        Args:
            features: вектор признаков
            use_weights: использовать ли веса
            threshold: порог уверенности
            
        Returns:
            (имя_автора, результаты_сходства)
        """
        if not self.profiles:
            raise ValueError("Нет профилей авторов для сравнения")
        
        results = {}
        
        for name, profile in self.profiles.items():
            similarity = profile.similarity(features) if hasattr(profile, 'similarity') else \
                        self._calculate_similarity(profile, features)
            results[name] = similarity
        
        # Находим лучшего
        best_author = max(results, key=results.get)
        best_score = results[best_author]
        
        if best_score < threshold:
            return "Автор не определён", results
        
        return best_author, results
    
    def _calculate_similarity(self, profile: AuthorProfile, 
                             features: np.ndarray) -> float:
        """Вычисляет сходство с профилем"""
        if not profile.features:
            return 0.0
        
        similarities = []
        for i, func in enumerate(profile.features):
            if i < len(features):
                mu = func.mu(features[i])
                similarities.append(mu)
            else:
                similarities.append(0.0)
        
        return np.mean(similarities)
    
    def identify_with_details(self, text: str) -> Dict:
        """
        Полная идентификация с деталями
        
        Returns:
            словарь с результатами и деталями
        """
        features = self.extractor.extract(text)
        results = {}
        details = {}
        
        for name, profile in self.profiles.items():
            if hasattr(profile, 'similarity_with_details'):
                sim, detail = profile.similarity_with_details(features)
            else:
                sim = self._calculate_similarity(profile, features)
                detail = {'similarities': [], 'weights': [], 'contributions': []}
            
            results[name] = sim
            details[name] = detail
        
        best_author = max(results, key=results.get)
        
        return {
            'best_author': best_author,
            'confidence': results[best_author],
            'all_results': results,
            'details': details,
            'features': features.tolist(),
        }
    
    def identify_with_confidence(self, text: str, 
                                 n_bootstrap: int = 1000) -> Dict:
        """
        Идентификация с оценкой уверенности через бутстрэп
        
        Args:
            text: анонимный текст
            n_bootstrap: количество бутстрэп итераций
            
        Returns:
            словарь с результатами и доверительными интервалами
        """
        features = self.extractor.extract(text)
        n_features = len(features)
        
        # Основные результаты
        base_results = {}
        bootstrap_scores = {name: [] for name in self.profiles}
        
        for name, profile in self.profiles.items():
            # Базовое сходство
            sim = self._calculate_similarity(profile, features)
            base_results[name] = sim
            
            # Бутстрэп
            for _ in range(n_bootstrap):
                indices = np.random.choice(n_features, n_features, replace=True)
                boot_features = features[indices]
                boot_sim = self._calculate_similarity(profile, boot_features)
                bootstrap_scores[name].append(boot_sim)
        
        # Вычисляем доверительные интервалы
        confidence_intervals = {}
        for name in self.profiles:
            scores = bootstrap_scores[name]
            ci_lower = np.percentile(scores, 2.5)
            ci_upper = np.percentile(scores, 97.5)
            confidence_intervals[name] = (ci_lower, ci_upper)
        
        best_author = max(base_results, key=base_results.get)
        
        return {
            'best_author': best_author,
            'confidence': base_results[best_author],
            'confidence_interval': confidence_intervals[best_author],
            'all_results': base_results,
            'all_intervals': confidence_intervals,
        }


class EnsembleIdentifier:
    """
    Ансамбль методов идентификации.
    
    Комбинирует:
    - Нечёткую логику
    - Машинное обучение (SVM, Random Forest)
    - Статистические методы
    """
    
    def __init__(self, profiles: Optional[Dict[str, AuthorProfile]] = None):
        """
        Инициализация ансамбля
        
        Args:
            profiles: словари профилей авторов
        """
        self.profiles = profiles or {}
        self.fuzzy_detector = FuzzyDetective(profiles)
        self.ml_models = {}
        self.is_trained = False
    
    def train_ml_models(self, texts: List[str], labels: List[str]):
        """
        Обучает ML модели
        
        Args:
            texts: обучающие тексты
            labels: метки авторов
        """
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        
        extractor = FeatureExtractor()
        
        # Извлекаем признаки
        X = np.array([extractor.extract(text) for text in texts])
        y = np.array(labels)
        
        # Нормализуем
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Обучаем модели
        self.ml_models['svm'] = SVC(kernel='rbf', probability=True)
        self.ml_models['svm'].fit(X_scaled, y)
        
        self.ml_models['rf'] = RandomForestClassifier(n_estimators=100)
        self.ml_models['rf'].fit(X_scaled, y)
        
        self.is_trained = True
        print(f"✅ ML модели обучены на {len(texts)} текстах")
    
    def predict(self, text: str, method: str = 'ensemble') -> str:
        """
        Предсказывает автора
        
        Args:
            text: анонимный текст
            method: метод ('fuzzy', 'svm', 'rf', 'ensemble')
            
        Returns:
            имя автора
        """
        if method == 'fuzzy':
            author, _ = self.fuzzy_detector.identify(text)
            return author
        
        if not self.is_trained:
            print("⚠️  ML модели не обучены, используем нечёткую логику")
            author, _ = self.fuzzy_detector.identify(text)
            return author
        
        extractor = FeatureExtractor()
        features = extractor.extract(text).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        
        if method == 'svm':
            return self.ml_models['svm'].predict(features_scaled)[0]
        
        if method == 'rf':
            return self.ml_models['rf'].predict(features_scaled)[0]
        
        if method == 'ensemble':
            # Взвешенное голосование
            predictions = {}
            
            # Нечёткая логика
            fuzzy_author, fuzzy_scores = self.fuzzy_detector.identify(text)
            for author, score in fuzzy_scores.items():
                predictions[author] = score * 0.4  # вес 40%
            
            # SVM вероятности
            features_scaled = self.scaler.transform(features)
            svm_probs = self.ml_models['svm'].predict_proba(features_scaled)[0]
            for author, prob in zip(self.ml_models['svm'].classes_, svm_probs):
                if author in predictions:
                    predictions[author] += prob * 0.3
                else:
                    predictions[author] = prob * 0.3
            
            # Random Forest вероятности
            rf_probs = self.ml_models['rf'].predict_proba(features_scaled)[0]
            for author, prob in zip(self.ml_models['rf'].classes_, rf_probs):
                if author in predictions:
                    predictions[author] += prob * 0.3
                else:
                    predictions[author] = prob * 0.3
            
            return max(predictions, key=predictions.get)
        
        raise ValueError(f"Неизвестный метод: {method}")
    
    def evaluate(self, texts: List[str], true_labels: List[str]) -> Dict:
        """
        Оценивает качество идентификации
        
        Returns:
            словарь с метриками
        """
        from sklearn.metrics import accuracy_score, classification_report
        
        predictions = [self.predict(text) for text in texts]
        
        accuracy = accuracy_score(true_labels, predictions)
        
        return {
            'accuracy': accuracy,
            'predictions': predictions,
            'true_labels': true_labels,
            'report': classification_report(true_labels, predictions),
        }
