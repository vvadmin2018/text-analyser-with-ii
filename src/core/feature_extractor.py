# src/core/feature_extractor.py
"""Извлечение стилевых признаков из текста"""

import re
import string
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter

try:
    import pymorphy3 as pymorphy2
except ImportError:
    try:
        import pymorphy2
    except ImportError:
        pymorphy2 = None

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Добавляем новые признаки
from .models import TextFeatures


class FeatureExtractor:
    """
    Извлекает стилевые признаки из текста.
    
    Поддерживаемые признаки:
    - Синтаксические: длина предложений, дисперсия, абзацы, пунктуация
    - Лексические: TTR, части речи, средняя длина слова
    - Пунктуационные: запятые, тире, двоеточия
    - Служебные: союзы, предлоги
    - N-граммы символов (опционально)
    """
    
    def __init__(self, use_ngrams: bool = False, ngram_size: int = 3):
        """
        Инициализация экстрактора признаков
        
        Args:
            use_ngrams: использовать ли n-граммы символов
            ngram_size: размер n-грамм
        """
        self.use_ngrams = use_ngrams
        self.ngram_size = ngram_size
        
        # Загружаем стоп-слова
        try:
            self.stopwords = set(nltk.corpus.stopwords.words('russian'))
        except LookupError:
            nltk.download('stopwords')
            self.stopwords = set(nltk.corpus.stopwords.words('russian'))
        
        # Инициализируем морфологический анализатор
        if pymorphy2:
            self.morph = pymorphy2.MorphAnalyzer()
        else:
            self.morph = None
            print("⚠️  pymorphy2/pymorphy3 не установлен. Морфологический анализ будет ограничен.")
        
        # Словари служебных частей речи
        self.conjunctions = {
            'и', 'а', 'но', 'да', 'или', 'либо', 'что', 'чтобы', 'если',
            'когда', 'потому', 'так', 'как', 'чем', 'однако', 'зато', 'тоже',
            'также', 'причем', 'притом', 'поэтому', 'зачем', 'отчего', 'хотя'
        }
        
        self.prepositions = {
            'в', 'во', 'на', 'с', 'со', 'к', 'ко', 'у', 'о', 'об', 'от',
            'ото', 'из', 'изо', 'за', 'для', 'без', 'безо', 'до', 'при',
            'про', 'через', 'сквозь', 'между', 'среди', 'над', 'под', 'перед',
            'возле', 'около', 'вокруг', 'мимо', 'после', 'ради', 'вроде'
        }
        
        # Стандартные названия признаков
        self.feature_names = [
            'Ср. длина предл.', 'Дисп. предл', 'Ср. длина абзаца',
            'Доля ?', 'Доля !', 'Доля ...', 'Прямая речь', 'Лекс. богатство',
            'Доля сущ.', 'Доля глаголов', 'Доля прилаг.', 'Ср. длина слова',
            'Запятые', 'Тире', 'Двоеточия', 'Доля союзов', 'Доля предлогов'
        ]
    
    def extract(self, text: str) -> np.ndarray:
        """
        Извлекает все признаки из текста
        
        Args:
            text: исходный текст
            
        Returns:
            массив признаков形状 (n_features,)
        """
        # Защита от пустого текста
        if not text or len(text.strip()) < 10:
            return np.zeros(len(self.feature_names))
        
        # Токенизация
        sentences = sent_tokenize(text)
        words_raw = word_tokenize(text.lower())
        words = [w for w in words_raw if w not in string.punctuation and w.isalpha()]
        words_original = [w for w in word_tokenize(text) if w not in string.punctuation and w.isalpha()]
        
        if len(words) == 0 or len(sentences) == 0:
            return np.zeros(len(self.feature_names))
        
        features = []
        
        # ===== Группа А: Синтаксические признаки =====
        
        # A1: Средняя длина предложения (медиана)
        sent_lengths = []
        for sent in sentences:
            sent_words = [w for w in word_tokenize(sent) 
                         if w not in string.punctuation and w.isalpha()]
            sent_lengths.append(len(sent_words))
        
        features.append(np.median(sent_lengths) if sent_lengths else 0)
        
        # A2: Дисперсия длины предложений
        features.append(np.var(sent_lengths) if len(sent_lengths) > 1 else 0)
        
        # A3: Средняя длина абзаца (медиана)
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p for p in paragraphs if p.strip() and not p.startswith(('–', '—'))]
        
        if paragraphs:
            para_lengths = [len(sent_tokenize(p)) for p in paragraphs]
            features.append(np.median(para_lengths))
        else:
            features.append(1)
        
        # A4: Доля вопросительных предложений
        q_count = sum(1 for s in sentences if '?' in s)
        features.append(q_count / len(sentences))
        
        # A5: Доля восклицательных предложений
        ex_count = sum(1 for s in sentences if '!' in s)
        features.append(ex_count / len(sentences))
        
        # A6: Доля троеточий
        dot3_count = sum(1 for s in sentences if '...' in s or '!..' in s)
        features.append(dot3_count / len(sentences))
        
        # A7: Доля прямой речи
        direct_speech = [s for s in sentences if s.startswith(('–', '—'))]
        features.append(len(direct_speech) / len(sentences))
        
        # ===== Группа Б: Лексические признаки =====
        
        # Б1: Лексическое богатство (TTR)
        unique_words = set(words)
        features.append(len(unique_words) / len(words) if words else 0)
        
        # Морфологический анализ
        pos_counts = Counter()
        words_main = []
        
        if self.morph:
            for word in words_original:
                try:
                    parsed = self.morph.parse(word)[0]
                    pos = parsed.tag.POS
                    if pos:
                        pos_counts[pos] += 1
                    if pos in ['NOUN', 'VERB', 'INFN', 'GRND', 'PRTF', 'ADJF']:
                        words_main.append(word)
                except Exception:
                    continue
        
        total_words = len(words_original)
        
        # Б2: Доля существительных
        nouns = pos_counts.get('NOUN', 0) + pos_counts.get('NUMR', 0)
        features.append(nouns / total_words if total_words > 0 else 0)
        
        # Б3: Доля глаголов
        verbs = (pos_counts.get('VERB', 0) + pos_counts.get('INFN', 0) +
                 pos_counts.get('GRND', 0) + pos_counts.get('PRTS', 0) +
                 pos_counts.get('PRTF', 0))
        features.append(verbs / total_words if total_words > 0 else 0)
        
        # Б4: Доля прилагательных
        adjs = pos_counts.get('ADJF', 0) + pos_counts.get('ADJS', 0)
        features.append(adjs / total_words if total_words > 0 else 0)
        
        # Б5: Средняя длина слова
        if words_main:
            features.append(np.mean([len(w) for w in words_main]))
        else:
            features.append(0)
        
        # ===== Группа В: Пунктуационно-служебные признаки =====
        
        # В1: Частота запятых
        features.append(text.count(',') / len(sentences))
        
        # В2: Частота тире
        features.append((text.count('—') + text.count('–')) / len(sentences))
        
        # В3: Частота двоеточий
        features.append(text.count(':') / len(sentences))
        
        # В4: Доля союзов
        conj_morph = pos_counts.get('CONJ', 0)
        conj_dict = sum(1 for w in words if w in self.conjunctions)
        features.append(max(conj_morph, conj_dict) / total_words if total_words > 0 else 0)
        
        # В5: Доля предлогов
        prep_morph = pos_counts.get('PREP', 0)
        prep_dict = sum(1 for w in words if w in self.prepositions)
        features.append(max(prep_morph, prep_dict) / total_words if total_words > 0 else 0)
        
        result = np.array(features, dtype=np.float64)
        return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    
    def extract_with_details(self, text: str) -> TextFeatures:
        """
        Извлекает признаки с дополнительной информацией
        
        Args:
            text: исходный текст
            
        Returns:
            TextFeatures объект с метаданными
        """
        features = self.extract(text)
        
        sentences = sent_tokenize(text)
        words = [w for w in word_tokenize(text) if w.isalpha()]
        
        return TextFeatures(
            values=features,
            feature_names=self.feature_names.copy(),
            text_length=len(text),
            num_sentences=len(sentences),
            num_words=len(words)
        )
    
    def extract_char_ngrams(self, text: str, n: Optional[int] = None) -> Dict[str, float]:
        """
        Извлекает частоты n-грамм символов
        
        Args:
            text: исходный текст
            n: размер n-грамм (если None, используется self.ngram_size)
            
        Returns:
            словарь {n-грамма: частота}
        """
        if n is None:
            n = self.ngram_size
        
        # Очищаем текст
        clean_text = re.sub(r'\s+', ' ', text.lower())
        
        # Извлекаем n-граммы
        ngrams = [clean_text[i:i+n] for i in range(len(clean_text) - n + 1)]
        counter = Counter(ngrams)
        
        # Нормализуем
        total = sum(counter.values())
        return {k: v / total for k, v in counter.items()}
    
    def get_feature_names(self) -> List[str]:
        """Возвращает названия признаков"""
        return self.feature_names.copy()
