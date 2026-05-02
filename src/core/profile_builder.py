# src/core/profile_builder.py
"""Построение нечётких профилей авторов"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from .models import (
    AuthorProfile, 
    TriangularMembership, 
    GaussianMembership,
    MembershipFunction,
    TextFeatures
)
from .feature_extractor import FeatureExtractor


class ProfileBuilder:
    """
    Строит нечёткие профили авторов на основе наборов текстов.
    
    Поддерживает:
    - Треугольные функции принадлежности
    - Гауссовы функции принадлежности
    - Адаптивное вычисление параметров из квантилей
    """
    
    def __init__(self, function_type: str = 'triangular'):
        """
        Инициализация билдера профилей
        
        Args:
            function_type: тип функций принадлежности ('triangular' или 'gaussian')
        """
        self.function_type = function_type
        self.extractor = FeatureExtractor()
    
    def build_from_texts(self, name: str, texts: List[str], 
                         use_quantiles: bool = True) -> AuthorProfile:
        """
        Строит профиль автора из списка текстов
        
        Args:
            name: имя автора
            texts: список текстов
            use_quantiles: использовать ли квантили вместо min/max
            
        Returns:
            AuthorProfile объект
        """
        print(f"\n🖌️  Строим портрет для {name}")
        
        # Собираем все значения признаков
        all_values: List[np.ndarray] = []
        
        for idx, text in enumerate(texts):
            if isinstance(text, bytes):
                text = text.decode('utf-8', errors='ignore')
            elif not isinstance(text, str):
                text = str(text)
            
            try:
                features = self.extractor.extract(text)
                all_values.append(features)
                print(f"  Текст {idx + 1}/{len(texts)}: {len(features)} признаков")
            except Exception as e:
                print(f"  ❌ Ошибка при обработке текста {idx + 1}: {e}")
        
        if not all_values:
            print(f"  ⚠️  Нет данных для построения профиля {name}")
            return AuthorProfile(name=name, feature_names=self.extractor.get_feature_names())
        
        # Преобразуем в массив
        X = np.array(all_values)
        n_features = X.shape[1]
        
        # Строим функции принадлежности
        features = []
        
        for i in range(n_features):
            values = X[:, i]
            
            if use_quantiles and len(values) >= 3:
                # Используем квантили для устойчивости к выбросам
                q1 = np.percentile(values, 25)
                q2 = np.percentile(values, 50)  # медиана
                q3 = np.percentile(values, 75)
                
                # Расширяем диапазон для покрытия 95% данных
                iqr = q3 - q1
                a = max(0, q1 - 0.5 * iqr)
                b = q2
                c = q3 + 0.5 * iqr
            else:
                # Классический подход min/max/mean
                a = np.min(values)
                c = np.max(values)
                b = np.mean(values)
                
                # Защита от вырожденных случаев
                if a == b == c:
                    a = max(0, a - 0.1)
                    c = c + 0.1
                    b = (a + c) / 2
            
            # Создаём функцию принадлежности
            if self.function_type == 'gaussian':
                std = (c - a) / 4
                func = GaussianMembership(a=a, b=b, c=c, std=std)
            else:
                func = TriangularMembership(a=a, b=b, c=c, softening=0.8)
            
            features.append(func)
        
        profile = AuthorProfile(
            name=name,
            features=features,
            texts_count=len(texts),
            feature_names=self.extractor.get_feature_names()
        )
        
        print(f"  ✅ Портрет для {name} построен! {len(features)} функций")
        return profile
    
    def build_from_features(self, name: str, 
                           features_list: List[np.ndarray]) -> AuthorProfile:
        """
        Строит профиль из готовых векторов признаков
        
        Args:
            name: имя автора
            features_list: список векторов признаков
            
        Returns:
            AuthorProfile объект
        """
        if not features_list:
            return AuthorProfile(name=name)
        
        X = np.array(features_list)
        n_features = X.shape[1]
        
        features = []
        
        for i in range(n_features):
            values = X[:, i]
            
            # Используем робастную статистику
            median = np.median(values)
            mad = np.median(np.abs(values - median))  # Median Absolute Deviation
            
            a = max(0, median - 3 * mad)
            b = median
            c = median + 3 * mad
            
            # Защита от вырождения
            if c - a < 1e-6:
                a = b - 0.1
                c = b + 0.1
            
            if self.function_type == 'gaussian':
                func = GaussianMembership(a=a, b=b, c=c, std=mad if mad > 0 else 0.1)
            else:
                func = TriangularMembership(a=a, b=b, c=c)
            
            features.append(func)
        
        return AuthorProfile(
            name=name,
            features=features,
            texts_count=len(features_list),
            feature_names=self.extractor.get_feature_names()
        )


class ProfileAnalyzer:
    """Анализ и сравнение профилей авторов"""
    
    @staticmethod
    def calculate_feature_importance(profiles: Dict[str, AuthorProfile], 
                                     labels: np.ndarray) -> np.ndarray:
        """
        Вычисляет важность признаков на основе дискриминативной способности
        
        Args:
            profiles: словарь профилей
            labels: метки авторов для каждого профиля
            
        Returns:
            массив важности признаков
        """
        from sklearn.feature_selection import f_classif
        
        # Собираем данные
        X = np.array([p.get_typical_values() for p in profiles.values()])
        
        if len(np.unique(labels)) < 2:
            # Если только один автор, все признаки одинаково важны
            return np.ones(X.shape[1])
        
        # Вычисляем F-статистики
        F_values, _ = f_classif(X, labels)
        
        # Нормализуем
        weights = F_values / (F_values.sum() + 1e-10)
        
        return weights
    
    @staticmethod
    def compare_profiles(profile1: AuthorProfile, 
                        profile2: AuthorProfile) -> Dict[str, float]:
        """
        Сравнивает два профиля по каждому признаку
        
        Returns:
            словарь с различиями по признакам
        """
        if profile1.n_features != profile2.n_features:
            raise ValueError("Профили имеют разное количество признаков")
        
        differences = {}
        
        for i, (f1, f2) in enumerate(zip(profile1.features, profile2.features)):
            diff = abs(f1.b - f2.b)
            overlap = ProfileAnalyzer.calculate_overlap(f1, f2)
            differences[f"feature_{i}"] = {
                'diff': diff,
                'overlap': overlap,
                'ratio': diff / (f1.support + f2.support + 1e-10)
            }
        
        return differences
    
    @staticmethod
    def calculate_overlap(f1: MembershipFunction, 
                         f2: MembershipFunction) -> float:
        """Вычисляет степень перекрытия двух функций принадлежности"""
        # Дискретизируем и вычисляем пересечение
        x_min = min(f1.a, f2.a)
        x_max = max(f1.c if hasattr(f1, 'c') else f1.b + 3*f1.std, 
                   f2.c if hasattr(f2, 'c') else f2.b + 3*f2.std)
        
        x_values = np.linspace(x_min, x_max, 100)
        mu1 = np.array([f1.mu(x) for x in x_values])
        mu2 = np.array([f2.mu(x) for x in x_values])
        
        intersection = np.sum(np.minimum(mu1, mu2))
        union = np.sum(np.maximum(mu1, mu2))
        
        return intersection / (union + 1e-10)
