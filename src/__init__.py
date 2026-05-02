# Text Analyser - Стилометрический анализ текстов
"""
Пакет для стилометрического анализа и определения авторства текстов
с использованием нечёткой логики.
"""

__version__ = "2.0.0"
__author__ = "Text Analysis Team"

from .core.feature_extractor import FeatureExtractor
from .core.profile_builder import AuthorProfile, TriangularMembership, GaussianMembership
from .core.identifier import FuzzyDetective, EnsembleIdentifier
from .visualization.visualizer import StyleRose, InteractiveVisualizer
from .utils.text_utils import preprocess_text, load_texts_from_folder

__all__ = [
    'FeatureExtractor',
    'AuthorProfile',
    'TriangularMembership',
    'GaussianMembership',
    'FuzzyDetective',
    'EnsembleIdentifier',
    'StyleRose',
    'InteractiveVisualizer',
    'preprocess_text',
    'load_texts_from_folder',
]
