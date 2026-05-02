# src/core/models.py
"""Модели данных для стилометрического анализа"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import numpy as np


@dataclass
class TextFeatures:
    """Вектор признаков текста"""
    values: np.ndarray
    feature_names: List[str]
    text_length: int
    num_sentences: int
    num_words: int = 0
    
    def __post_init__(self):
        if self.num_words == 0:
            self.num_words = self.text_length
    
    @property
    def n_features(self) -> int:
        return len(self.values)
    
    def normalize(self, method: str = 'robust') -> 'TextFeatures':
        """Нормализует признаки"""
        if method == 'robust':
            # Robust normalization (устойчив к выбросам)
            median = np.median(self.values)
            iqr = np.percentile(self.values, 75) - np.percentile(self.values, 25)
            if iqr > 0:
                normalized_values = (self.values - median) / iqr
            else:
                normalized_values = self.values - median
        elif method == 'minmax':
            min_val = np.min(self.values)
            max_val = np.max(self.values)
            if max_val > min_val:
                normalized_values = (self.values - min_val) / (max_val - min_val)
            else:
                normalized_values = self.values
        else:
            normalized_values = self.values
        
        return TextFeatures(
            values=normalized_values,
            feature_names=self.feature_names.copy(),
            text_length=self.text_length,
            num_sentences=self.num_sentences,
            num_words=self.num_words
        )


@dataclass
class MembershipFunction:
    """Базовый класс функции принадлежности"""
    a: float  # левая граница
    b: float  # пик (наиболее вероятное значение)
    c: float  # правая граница
    softening: float = 0.8
    
    def mu(self, x: float) -> float:
        """Вычисляет степень принадлежности"""
        raise NotImplementedError
    
    @property
    def support(self) -> float:
        """Ширина носителя функции"""
        return self.c - self.a
    
    @property
    def center(self) -> float:
        """Центр функции"""
        return (self.a + self.c) / 2


@dataclass
class TriangularMembership(MembershipFunction):
    """Треугольная функция принадлежности с размытыми границами"""
    
    def __post_init__(self):
        # Вычисляем параметр размытия
        if self.c != self.a:
            self.softening_value = self.softening * (self.c - self.a)
        else:
            self.softening_value = 0.1
    
    def mu(self, x: float) -> float:
        """
        Вычисляет степень принадлежности x к нечёткому множеству
        
        Args:
            x: значение
            
        Returns:
            степень принадлежности от 0 до 1
        """
        # Расширяем границы на величину softening
        a_soft = self.a - self.softening_value
        c_soft = self.c + self.softening_value
        
        if x <= a_soft or x >= c_soft:
            return 0.001
        elif abs(x - self.b) < 1e-10:  # x == self.b
            return 1.0
        elif x < self.b:
            if abs(self.b - self.a) < 1e-10:
                return 1.0
            # Плавный подъем с квадратичным сглаживанием
            ratio = (x - a_soft) / (self.b - a_soft)
            return ratio ** 0.5
        else:
            if abs(self.c - self.b) < 1e-10:
                return 1.0
            # Плавный спуск с квадратичным сглаживанием
            ratio = (c_soft - x) / (c_soft - self.b)
            return ratio ** 0.5
    
    def __repr__(self):
        return f"TriangularMembership(a={self.a:.3f}, b={self.b:.3f}, c={self.c:.3f})"


@dataclass
class GaussianMembership(MembershipFunction):
    """Гауссова функция принадлежности"""
    std: float = 0.1  # стандартное отклонение
    
    def __post_init__(self):
        # Если std не задан, вычисляем из диапазона
        if self.std <= 0:
            self.std = (self.c - self.a) / 4
    
    def mu(self, x: float) -> float:
        """
        Вычисляет степень принадлежности по гауссовой функции
        
        μ(x) = exp(-(x - b)² / (2σ²))
        """
        return np.exp(-((x - self.b) ** 2) / (2 * self.std ** 2))
    
    def __repr__(self):
        return f"GaussianMembership(mean={self.b:.3f}, std={self.std:.3f})"


@dataclass
class AuthorProfile:
    """Профиль автора - набор функций принадлежности для каждого признака"""
    name: str
    features: List[MembershipFunction] = field(default_factory=list)
    texts_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    feature_names: List[str] = field(default_factory=list)
    
    @property
    def n_features(self) -> int:
        return len(self.features)
    
    def get_typical_values(self) -> np.ndarray:
        """Возвращает типичные значения (пики функций)"""
        return np.array([f.b for f in self.features])
    
    def get_ranges(self) -> List[Tuple[float, float]]:
        """Возвращает диапазоны [a, c] для каждой функции"""
        return [(f.a, f.c) for f in self.features]
    
    def to_dict(self) -> Dict:
        """Сериализует профиль в словарь"""
        return {
            'name': self.name,
            'features': [
                {
                    'type': type(f).__name__,
                    'a': f.a,
                    'b': f.b,
                    'c': f.c if hasattr(f, 'c') else None,
                    'std': f.std if hasattr(f, 'std') else None,
                    'softening': f.softening if hasattr(f, 'softening') else None,
                }
                for f in self.features
            ],
            'texts_count': self.texts_count,
            'created_at': self.created_at,
            'feature_names': self.feature_names,
        }
