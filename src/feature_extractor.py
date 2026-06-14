# feature_extractor.py
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import string
import numpy as np
import re
from src import config

# Определяем доступные морфологические анализаторы
try:
    import pymorphy3 as pymorphy2

    PY_MORPHY_AVAILABLE = True
except ImportError:
    try:
        import pymorphy2

        PY_MORPHY_AVAILABLE = True
    except ImportError:
        PY_MORPHY_AVAILABLE = False
        print("⚠️ pymorphy2/pymorphy3 не установлен. Морфологический анализ будет ограничен.")

# Stanza для белорусского языка
try:
    import stanza

    STANZA_AVAILABLE = True
except ImportError:
    STANZA_AVAILABLE = False
    print("⚠️ stanza не установлен. Белорусский язык не поддерживается.")


class Language:
    """Коды поддерживаемых языков"""
    RUSSIAN = 'ru'
    BELARUSIAN = 'be'


class FeatureExtractor:
    """Извлекает стилевые признаки из текста с поддержкой русского и белорусского языков"""

    def __init__(self, language=Language.RUSSIAN):
        """
        Инициализация анализатора

        Параметры:
        - language: 'ru' для русского, 'be' для белорусского
        """
        self.language = language

        # Загружаем стоп-слова в зависимости от языка
        if language == Language.RUSSIAN:
            self.stopwords = set(nltk.corpus.stopwords.words('russian'))
        else:
            # Для белорусского используем базовый набор стоп-слов
            self.stopwords = self._get_belarusian_stopwords()

        # Инициализируем морфологический анализатор
        if PY_MORPHY_AVAILABLE:
            self.morph = pymorphy2.MorphAnalyzer()
        else:
            self.morph = None

        # Инициализируем Stanza для белорусского языка
        self.stanza_nlp = None
        if language == Language.BELARUSIAN and STANZA_AVAILABLE:
            try:
                # Загружаем модель для белорусского языка
                self.stanza_nlp = stanza.Pipeline(
                    'be',
                    processors='tokenize,pos,lemma',
                    use_gpu=False,
                    verbose=False
                )
                print("✅ Stanza для белорусского языка загружена")
            except Exception as e:
                print(f"⚠️ Не удалось загрузить Stanza для белорусского: {e}")

        # Словари для союзов и предлогов
        if language == Language.RUSSIAN:
            self._init_russian_dictionaries()
        else:
            self._init_belarusian_dictionaries()

    def _init_russian_dictionaries(self):
        """Инициализация словарей для русского языка"""
        self.conjunctions = set([
            'и', 'а', 'но', 'да', 'или', 'либо', 'что', 'чтобы', 'если',
            'когда', 'потому', 'так', 'как', 'чем', 'однако', 'зато', 'тоже',
            'также', 'причем', 'притом', 'потому', 'поэтому', 'зачем', 'отчего'
        ])

        self.prepositions = set([
            'в', 'во', 'на', 'с', 'со', 'к', 'ко', 'у', 'о', 'об', 'от',
            'ото', 'из', 'изо', 'за', 'для', 'без', 'безо', 'до', 'при',
            'про', 'через', 'сквозь', 'между', 'среди', 'над', 'под', 'перед',
            'возле', 'около', 'вокруг', 'мимо', 'после', 'ради', 'вроде'
        ])

    def _init_belarusian_dictionaries(self):
        """Инициализация словарей для белорусского языка"""
        self.conjunctions = set([
            'і', 'а', 'але', 'ды', 'ці', 'альбо', 'што', 'каб', 'калі',
            'таму', 'так', 'як', 'чым', 'аднак', 'затое', 'тожа', 'таксама'
        ])

        self.prepositions = set([
            'у', 'ў', 'на', 'з', 'са', 'да', 'к', 'а', 'аб', 'ад',
            'ада', 'ад', 'з', 'за', 'для', 'без', 'бяз', 'да', 'пры',
            'пра', 'праз', 'скрозь', 'паміж', 'сярод', 'над', 'пад', 'перад'
        ])

    def _get_belarusian_stopwords(self):
        """Возвращает базовый набор стоп-слов для белорусского языка"""
        return set([
            'і', 'а', 'не', 'што', 'на', 'ў', 'з', 'да', 'па', 'за',
            'як', 'так', 'каб', 'яго', 'яна', 'яно', 'яны', 'мы', 'вы',
            'яны', 'гэта', 'гэты', 'гэтая', 'гэтае', 'гэтыя'
        ])

    def _analyze_with_pymorphy(self, words_original):
        """Анализ текста с помощью pymorphy2 (для русского языка)"""
        pos_counts = {
            'NOUN': 0, 'ADJF': 0, 'ADJS': 0, 'VERB': 0, 'INFN': 0,
            'PRTS': 0, 'PRTF': 0, 'GRND': 0, 'NUMR': 0, 'ADVB': 0,
            'NPRO': 0, 'PRED': 0, 'PREP': 0, 'CONJ': 0, 'PRCL': 0, 'INTJ': 0,
        }

        words_main = []

        for word in words_original:
            try:
                parsed = self.morph.parse(word)[0]
                pos = parsed.tag.POS
                if pos and pos in pos_counts:
                    pos_counts[pos] += 1
                if pos in ['NOUN', 'VERB', 'INFN', 'GRND', 'PRTF', 'ADJF']:
                    words_main.append(word)
            except Exception:
                continue

        total = len(words_original)

        # Подсчёт частей речи
        nouns = pos_counts['NOUN'] + pos_counts['NUMR']
        verbs = (pos_counts['VERB'] + pos_counts['INFN'] + pos_counts['GRND'] +
                 pos_counts['PRTS'] + pos_counts['PRTF'])
        adjs = pos_counts['ADJF'] + pos_counts['ADJS']

        # Союзы и предлоги
        conj_total = pos_counts['CONJ']
        prep_total = pos_counts['PREP']

        # Дополнительно проверяем по словарям
        conj_by_dict = sum(1 for w in words_original if w.lower() in self.conjunctions)
        prep_by_dict = sum(1 for w in words_original if w.lower() in self.prepositions)

        conj_total = max(conj_total, conj_by_dict)
        prep_total = max(prep_total, prep_by_dict)

        return nouns, verbs, adjs, conj_total, prep_total, total, words_main

    def _analyze_with_stanza(self, text):
        """Анализ текста с помощью Stanza (для белорусского языка)"""
        if self.stanza_nlp is None:
            return 0, 0, 0, 0, 0, 0, []

        doc = self.stanza_nlp(text)

        nouns = 0
        verbs = 0
        adjs = 0
        preps = 0
        conjs = 0
        words_main = []

        for sent in doc.sentences:
            for word in sent.words:
                pos = word.upos
                if pos == 'NOUN':
                    nouns += 1
                    words_main.append(word.text)
                elif pos == 'VERB':
                    verbs += 1
                    words_main.append(word.text)
                elif pos == 'ADJ':
                    adjs += 1
                    words_main.append(word.text)
                elif pos == 'ADP':
                    preps += 1
                elif pos in ('CCONJ', 'SCONJ'):
                    conjs += 1

        total = len([w for s in doc.sentences for w in s.words])

        return nouns, verbs, adjs, conjs, preps, total, words_main

    def extract(self, text):
        """
        Возвращает массив признаков

        Признаки:
        A1  - средняя длина предложения (медиана)
        A2  - дисперсия длины предложений
        A3  - средняя длина абзаца (медиана)
        A4  - доля вопросительных предложений
        A5  - доля восклицательных предложений
        A6  - доля предложений с троеточиями
        A7  - доля предложений с прямой речью
        Б1  - лексическое богатство (RTTR)
        Б2  - доля существительных
        Б3  - доля глаголов
        Б4  - доля прилагательных
        Б5  - средняя длина слова
        В1  - частота запятых
        В2  - частота тире
        В3  - частота двоеточий
        В4  - доля союзов
        В5  - доля предлогов
        """
        num_props = config.N_FEATURES

        # Защита от пустого текста
        if not text or len(text.strip()) < 10:
            return np.zeros(num_props)

        # Предварительная обработка
        sentences = sent_tokenize(text)
        sentences_without_primaya_rech = [s for s in sentences if not s.startswith('–') and not s.startswith('—')]
        sentences_primaya_rech = [s for s in sentences if s.startswith('–') or s.startswith('—')]

        # Токенизация слов
        words = word_tokenize(text.lower())
        words = [w for w in words if w not in string.punctuation and w.isalpha()]

        words_original = word_tokenize(text)
        words_original = [w for w in words_original if w not in string.punctuation and w.isalpha()]

        if len(words) == 0 or len(sentences) == 0:
            return np.zeros(num_props)

        features = []

        # ===== Группа А: Синтаксические признаки =====

        # A1 и A2: Длины предложений
        sent_lengths = []
        for sent in sentences_without_primaya_rech:
            sent_words = word_tokenize(sent)
            sent_words = [w for w in sent_words if w not in string.punctuation and w.isalpha()]
            sent_lengths.append(len(sent_words))

        features.append(np.median(sent_lengths))  # A1
        features.append(np.var(sent_lengths) if len(sent_lengths) > 1 else 0)  # A2

        # A3: Средняя длина абзаца
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p for p in paragraphs if p.strip() and not p.startswith('–') and not p.startswith('—')]

        if paragraphs:
            para_lengths = [len(sent_tokenize(p)) for p in paragraphs]
            features.append(np.median(para_lengths))
        else:
            features.append(1)

        # A4-A7: Знаки препинания в концах предложений
        total_sentences = len(sentences)
        features.append(sum(1 for s in sentences if '?' in s) / total_sentences)  # A4
        features.append(sum(1 for s in sentences if '!' in s) / total_sentences)  # A5
        features.append(sum(1 for s in sentences if '...' in s or '!..' in s) / total_sentences)  # A6
        features.append(len(sentences_primaya_rech) / total_sentences)  # A7

        # ===== Группа Б: Лексические признаки =====

        # Б1: RTTR
        unique_words = set(words)
        rttr = len(unique_words) / (len(words) ** 0.5)
        features.append(rttr)

        # ===== Морфологический анализ (выбор метода по языку) =====
        if self.language == Language.BELARUSIAN and STANZA_AVAILABLE:
            nouns, verbs, adjs, conjs, preps, total_words, words_main = self._analyze_with_stanza(text)
        else:
            nouns, verbs, adjs, conjs, preps, total_words, words_main = self._analyze_with_pymorphy(words_original)

        # Б2-Б5: Части речи и длина слова
        features.append(nouns / total_words if total_words > 0 else 0)  # Б2
        features.append(verbs / total_words if total_words > 0 else 0)  # Б3
        features.append(adjs / total_words if total_words > 0 else 0)  # Б4

        word_lengths = [len(w) for w in words_main]
        features.append(np.mean(word_lengths) if word_lengths else 0)  # Б5

        # ===== Группа В: Пунктуация и служебные слова =====
        features.append(text.count(',') / total_sentences)  # В1
        features.append((text.count('—') + text.count('–')) / total_sentences)  # В2
        features.append(text.count(':') / total_sentences)  # В3
        features.append(conjs / total_words if total_words > 0 else 0)  # В4
        features.append(preps / total_words if total_words > 0 else 0)  # В5

        # Преобразуем в numpy array
        result = np.array(features)
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

        return result
