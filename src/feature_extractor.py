# feature_extractor.py
"""Извлечение стилевых признаков текста (17 признаков, группы А/Б/В)."""
import logging
import re
import string
from collections import Counter

import nltk
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize

from src import config

logger = logging.getLogger(__name__)

# Определяем доступные морфологические анализаторы
try:
    import pymorphy3 as pymorphy2

    PY_MORPHY_AVAILABLE = True
except ImportError:
    PY_MORPHY_AVAILABLE = False
    logger.warning("⚠️ pymorphy3 не установлен. Морфологический анализ будет ограничен.")

# Stanza для белорусского языка
try:
    import stanza

    STANZA_AVAILABLE = True
except ImportError:
    STANZA_AVAILABLE = False


class Language:
    """Коды поддерживаемых языков"""
    RUSSIAN = 'ru'
    BELARUSIAN = 'be'


# Буквы, которые есть только в одном из двух алфавитов. В белорусском нет
# «и», «щ» и «ъ»; в русском нет «ў» и «і». Этого хватает, чтобы различить
# языки на любом связном тексте: служебные слова с этими буквами частотны.
_BELARUSIAN_ONLY = 'ўі'
_RUSSIAN_ONLY = 'ищъ'


def detect_language(text):
    """Определяет язык текста по специфичным буквам алфавита.

    Возвращает Language.RUSSIAN, Language.BELARUSIAN или None, если
    отличительных букв в тексте нет (очень короткий отрывок, латиница,
    текст без этих литер) — тогда честнее не гадать.
    """
    lower = (text or '').lower()
    be = sum(lower.count(ch) for ch in _BELARUSIAN_ONLY)
    ru = sum(lower.count(ch) for ch in _RUSSIAN_ONLY)

    if be == 0 and ru == 0:
        return None
    return Language.BELARUSIAN if be > ru else Language.RUSSIAN


class FeatureExtractor:
    """Извлекает стилевые признаки из текста с поддержкой русского и белорусского языков"""

    def __init__(self, language=Language.RUSSIAN):
        """
        Инициализация анализатора

        Параметры:
        - language: 'ru' для русского, 'be' для белорусского
        """
        self.language = language
        self.stanza_nlp = None
        # Заполняется, если полноценный морфоанализ для выбранного языка
        # недоступен и признаки считаются огрублённым способом. Веб-приложение
        # показывает это пользователю — молча выдавать уверенный процент,
        # посчитанный по суффиксной эвристике, нельзя.
        self.degraded_reason = None

        logger.debug("FeatureExtractor: язык %s", self.language)

        if language == Language.RUSSIAN:
            self.stopwords = set(nltk.corpus.stopwords.words('russian'))
            self._init_russian_dictionaries()
        else:
            self.stopwords = self._get_belarusian_stopwords()
            self._init_belarusian_dictionaries()

        # Морфологический анализатор для русского
        self.morph = pymorphy2.MorphAnalyzer() if PY_MORPHY_AVAILABLE else None
        if language == Language.RUSSIAN and self.morph is None:
            self.degraded_reason = (
                "не установлен pymorphy3 — части речи определяются по окончаниям слов"
            )

        if language == Language.BELARUSIAN:
            self._init_stanza()

    def _init_stanza(self):
        """Поднимает Stanza-пайплайн для белорусского.

        Если модель недоступна, признаки считаются суффиксной эвристикой и
        поднимается degraded_reason. Раньше выбор анализатора шёл по флагу
        STANZA_AVAILABLE (импортируется ли модуль), а _analyze_with_stanza при
        незагруженном пайплайне возвращал сплошные нули — Б2-Б5 и В4-В5
        обнулялись, и пользователю показывался уверенный процент по мусору.
        """
        if not STANZA_AVAILABLE:
            self.degraded_reason = (
                "части речи белорусского текста определяются по окончаниям слов. Результат анализа может быть некорректным."
            )
            return

        try:
            self.stanza_nlp = stanza.Pipeline(
                'be',
                processors='tokenize,pos,lemma',
                use_gpu=False,
                verbose=False
            )
            logger.info("✅ Stanza для белорусского языка загружена")
        except Exception as e:
            logger.warning("⚠️ Не удалось загрузить словарь (Stanza) для белорусского: %s", e)
            self.degraded_reason = (
                f"Словарь (stanza) установлен, но модель не поднялась ({e})."
            )

    def _init_russian_dictionaries(self):
        """Инициализация словарей для русского языка"""
        self.conjunctions = {
            'и', 'а', 'но', 'да', 'или', 'либо', 'что', 'чтобы', 'если',
            'когда', 'потому', 'так', 'как', 'чем', 'однако', 'зато', 'тоже',
            'также', 'причем', 'притом', 'поэтому', 'зачем', 'отчего'
        }

        self.prepositions = {
            'в', 'во', 'на', 'с', 'со', 'к', 'ко', 'у', 'о', 'об', 'от',
            'ото', 'из', 'изо', 'за', 'для', 'без', 'безо', 'до', 'при',
            'про', 'через', 'сквозь', 'между', 'среди', 'над', 'под', 'перед',
            'возле', 'около', 'вокруг', 'мимо', 'после', 'ради', 'вроде'
        }

    def _init_belarusian_dictionaries(self):
        """Инициализация словарей для белорусского языка"""
        self.conjunctions = {
            'і', 'а', 'але', 'ды', 'ці', 'альбо', 'што', 'каб', 'калі',
            'таму', 'так', 'як', 'чым', 'аднак', 'затое', 'тожа', 'таксама'
        }

        self.prepositions = {
            'у', 'ў', 'на', 'з', 'са', 'да', 'к', 'а', 'аб', 'ад', 'ада',
            'за', 'для', 'без', 'бяз', 'пры', 'пра', 'праз', 'скрозь',
            'паміж', 'сярод', 'над', 'пад', 'перад'
        }

    def _get_belarusian_stopwords(self):
        """Возвращает базовый набор стоп-слов для белорусского языка"""
        return {
            'і', 'а', 'не', 'што', 'на', 'ў', 'з', 'да', 'па', 'за',
            'як', 'так', 'каб', 'яго', 'яна', 'яно', 'яны', 'мы', 'вы',
            'гэта', 'гэты', 'гэтая', 'гэтае', 'гэтыя'
        }

    def _mattr(self, words, window_size=None):
        """
        MATTR — Moving-Average Type-Token Ratio (Covington & McFall, 2010).

        Обычный TTR/RTTR сильно "плывёт" вместе с длиной текста: для короткого
        анонимного отрывка в полсотни слов и для главы автора в пару тысяч слов
        значения получаются совершенно разного порядка, даже если стиль
        идентичен.

        MATTR устойчив к длине: считаем TTR не по всему тексту сразу, а
        как среднее TTR по скользящим окнам ФИКСИРОВАННОГО размера — тогда
        50-словный отрывок и 2000-словная глава сравниваются на равных
        условиях, а не по случайно попавшей под руку длине куска.
        """
        n = len(words)
        if n == 0:
            return 0.0

        w = window_size or config.MATTR_WINDOW

        if n <= w:
            # Текст короче окна (бывает у совсем коротких анонимных
            # отрывков) — окно ужимать некуда, считаем обычный TTR по
            # всему, что есть.
            return len(set(words)) / n

        counts = Counter(words[:w])
        ttrs = [len(counts) / w]

        for i in range(w, n):
            old_word, new_word = words[i - w], words[i]
            counts[old_word] -= 1
            if counts[old_word] == 0:
                del counts[old_word]
            counts[new_word] += 1
            ttrs.append(len(counts) / w)

        return float(np.mean(ttrs))

    def _analyze_with_pymorphy(self, words_original):
        """Анализ текста с помощью pymorphy (для русского языка)"""
        if self.morph is None:
            return self._analyze_fallback(words_original)

        MAIN_POS = {'NOUN', 'VERB', 'INFN', 'GRND', 'PRTF', 'ADJF'}
        pos_counts = Counter()
        words_main = []

        for word in words_original:
            try:
                pos = self.morph.parse(word)[0].tag.POS
            except Exception:
                continue
            if pos:
                pos_counts[pos] += 1
            if pos in MAIN_POS:
                words_main.append(word)

        total = len(words_original)

        nouns = pos_counts['NOUN'] + pos_counts['NUMR']
        verbs = (pos_counts['VERB'] + pos_counts['INFN'] + pos_counts['GRND'] +
                 pos_counts['PRTS'] + pos_counts['PRTF'])
        adjs = pos_counts['ADJF'] + pos_counts['ADJS']

        # Морфологию дополняем словарями: pymorphy часто размечает частотные
        # служебные слова как частицы/наречия, поэтому берём большую из двух оценок.
        conj_total = max(pos_counts['CONJ'],
                         sum(1 for w in words_original if w.lower() in self.conjunctions))
        prep_total = max(pos_counts['PREP'],
                         sum(1 for w in words_original if w.lower() in self.prepositions))

        # Если pymorphy не распознал ни одного слова — используем fallback
        if total > 0 and nouns == 0 and verbs == 0 and adjs == 0 and not words_main:
            return self._analyze_fallback(words_original)

        return nouns, verbs, adjs, conj_total, prep_total, total, words_main

    # Суффиксные правила разбираются от длинного окончания к короткому:
    # проверка «сначала существительные, потом остальные» отправляла в
    # существительные почти всё, потому что односимвольные -а/-я/-о/-е/-ы/-і
    # перехватывали и «вялікая», и «добры» раньше, чем до них доходила
    # проверка прилагательных. На русском корпусе (эталон — pymorphy3) один
    # только порядок разбора поднимает точность с 55.8% до 71.0%.

    _RU_ENDINGS = (
        [(e, 'ADJ') for e in ('ый', 'ий', 'ой', 'ая', 'яя', 'ое', 'ее', 'ые',
                              'ие', 'ым', 'им', 'ых', 'их')] +
        [(e, 'VERB') for e in ('ться', 'тся', 'ть', 'ти', 'чь', 'ет', 'ют',
                               'ит', 'ат', 'ят', 'ла', 'ли', 'ло', 'л')] +
        [(e, 'NOUN') for e in ('ам', 'ах', 'ей', 'ям', 'ях', 'ой', 'а', 'я',
                               'о', 'е', 'ы', 'и', 'у', 'ю', 'ь', 'й')]
    )

    # Белорусский разбирался русскими окончаниями. Инфинитив на -ць (чытаць)
    # не совпадал ни с одним правилом и уходил в существительные по конечному
    # -ь; прошедшее время на -ў (глядзеў) не совпадало вообще ни с чем и
    # попадало в существительные веткой else. На корпусе texts/kolas +
    # texts/bryl + texts/maur (47600 слов) старые правила давали 96.9%
    # существительных, 1.0% глаголов и 2.1% прилагательных — то есть Б2 почти
    # константа, а Б3 и Б4 — шум около нуля. Новые дают 65.0 / 24.9 / 10.1,
    # что уже похоже на прозу.
    _BE_ENDINGS = (
        # -ога/-ому — ударные варианты -ага/-аму: аканье не действует под
        # ударением, поэтому «маладога», а не «маладага».
        [(e, 'ADJ') for e in ('ейшая', 'ейшыя', 'ейшы', 'ага', 'яга', 'ога',
                              'аму', 'яму', 'ому',
                              'ымі', 'імі', 'ыя', 'ія', 'ая', 'яя',
                              'ае', 'яе', 'ую', 'юю', 'ых', 'іх', 'ым', 'ім')] +
        [(e, 'VERB') for e in ('ацца', 'ыцца', 'іцца', 'ецца', 'аць', 'яць',
                               'ыць', 'іць', 'ець', 'уць', 'юць', 'сці',
                               'лася', 'ліся', 'ўся', 'ла', 'лі', 'ло',
                               'еш', 'аш', 'ыш', 'іш', 'еце', 'аце',
                               'аў', 'еў', 'ыў', 'іў', 'ў')] +
        # Исключения, которые иначе перехватывались бы глагольными правилами:
        # -оў/-яў здесь родительный падеж множественного числа (слоў, вучняў),
        # а -сьці — наречие (дзесьці), а не инфинитив.
        [(e, 'NOUN') for e in ('сьці', 'оў', 'яў', 'осць', 'асць',
                               'ам', 'ах', 'ям', 'ях', 'ой', 'ай', 'яй', 'ей',
                               'а', 'я', 'о', 'е', 'у', 'ю', 'ь', 'й', 'ы', 'і')]
    )

    @staticmethod
    def _sorted_rules(rules):
        """Длинные окончания раньше коротких — иначе -ы съедает -ыя."""
        return tuple(sorted(rules, key=lambda r: -len(r[0])))

    _RU_RULES = None
    _BE_RULES = None

    def _pos_rules(self):
        cls = type(self)
        if self.language == Language.BELARUSIAN:
            if cls._BE_RULES is None:
                cls._BE_RULES = self._sorted_rules(cls._BE_ENDINGS)
            return cls._BE_RULES
        if cls._RU_RULES is None:
            cls._RU_RULES = self._sorted_rules(cls._RU_ENDINGS)
        return cls._RU_RULES

    def _analyze_fallback(self, words_original):
        """Оценка частей речи по суффиксам и словарям.

        Приблизительная: без словаря омонимию не снять — «добры» (прил.) и
        «сталы» (сущ.) на конце одинаковы. Ручная проверка выборки из 50
        белорусских слов даёт порядка 85% верных разборов знаменательных
        частей речи. Для стилометрии это терпимо: все авторы и анонимный
        текст размечаются одним и тем же способом, поэтому систематическая
        ошибка частично сокращается при сравнении.
        """
        total = len(words_original)
        if total == 0:
            return 0, 0, 0, 0, 0, 0, []

        conj_total = sum(1 for w in words_original if w.lower() in self.conjunctions)
        prep_total = sum(1 for w in words_original if w.lower() in self.prepositions)

        rules = self._pos_rules()
        counts = Counter()
        words_main = []

        for w in words_original:
            wl = w.lower()
            if len(wl) <= 2 or wl in self.stopwords:
                continue

            for ending, tag in rules:
                if wl.endswith(ending):
                    counts[tag] += 1
                    break
            else:
                counts['NOUN'] += 1
            words_main.append(w)

        return (counts['NOUN'], counts['VERB'], counts['ADJ'],
                conj_total, prep_total, total, words_main)

    def _analyze_with_stanza(self, text):
        """Анализ текста с помощью Stanza (для белорусского языка)"""
        doc = self.stanza_nlp(text)

        nouns = verbs = adjs = preps = conjs = total = 0
        words_main = []

        for sent in doc.sentences:
            for word in sent.words:
                total += 1
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

        return nouns, verbs, adjs, conjs, preps, total, words_main

    def _analyze_morphology(self, text, words_original):
        """Выбирает морфологический анализатор под язык и доступные модели."""
        if self.stanza_nlp is not None:
            return self._analyze_with_stanza(text)
        if self.language == Language.BELARUSIAN:
            # pymorphy знает только русский, поэтому для белорусского без
            # Stanza идём в суффиксную эвристику (о чём сказано в degraded_reason).
            return self._analyze_fallback(words_original)
        return self._analyze_with_pymorphy(words_original)

    # Тире в начале строки — признак реплики диалога. Помимо длинного (—) и
    # среднего (–) тире учитывается обычный дефис: в части текстов корпуса
    # диалоги набраны именно им.
    DIRECT_SPEECH_MARKERS = ('–', '—', '-')

    @classmethod
    def _is_direct_speech(cls, sentence):
        return sentence.startswith(cls.DIRECT_SPEECH_MARKERS)

    @staticmethod
    def _content_words(tokens):
        """Оставляет только словесные токены (без пунктуации и чисел)."""
        return [t for t in tokens
                if t not in string.punctuation and any(c.isalpha() for c in t)]

    def describe(self, text):
        """Общая статистика текста для показа пользователю.

        Считается той же токенизацией, что и признаки в extract(), — иначе
        число слов на экране расходилось бы с тем, по которому на самом деле
        построен анализ. Слова здесь — только словесные токены: знаки
        препинания и чистые числа не в счёт.
        """
        text = text or ''
        if not text.strip():
            return {'chars': 0, 'chars_no_spaces': 0, 'paragraphs': 0,
                    'sentences': 0, 'words': 0, 'language': None}

        paragraphs = [p for p in re.split(r'\n\s*\n', text) if p.strip()]

        return {
            'chars': len(text),
            'chars_no_spaces': sum(1 for ch in text if not ch.isspace()),
            # Абзац без пустой строки-разделителя не выделяется, поэтому текст
            # одним куском — это один абзац, а не ноль.
            'paragraphs': len(paragraphs) or 1,
            'sentences': len(sent_tokenize(text)),
            'words': len(self._content_words(word_tokenize(text))),
            'language': detect_language(text),
        }

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
        Б1  - лексическое богатство (MATTR, устойчив к длине текста)
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

        sentences = sent_tokenize(text)

        # Одна токенизация на весь текст вместо двух проходов
        # (раньше отдельно токенизировался text и text.lower()).
        words_original = self._content_words(word_tokenize(text))
        words = [w.lower() for w in words_original]

        if not words or not sentences:
            return np.zeros(num_props)

        features = []

        # ===== Группа А: Синтаксические признаки =====

        # A1 и A2: Длины предложений (без прямой речи — она меряется отдельно
        # признаком A7 и сильно занижала бы медианную длину предложения).
        narrative = [s for s in sentences if not self._is_direct_speech(s)]
        direct_speech = [s for s in sentences if self._is_direct_speech(s)]
        # Сплошная прямая речь (диалог целиком) оставляла список пустым, а
        # np.median([]) — это RuntimeWarning и nan, который затем молча
        # превращался в 0. Считаем по всем предложениям, раз других нет.
        length_source = narrative or sentences
        sent_lengths = [len(self._content_words(word_tokenize(s))) for s in length_source]

        features.append(np.median(sent_lengths))  # A1
        features.append(np.var(sent_lengths) if len(sent_lengths) > 1 else 0)  # A2

        # A3: Средняя длина абзаца
        paragraphs = [p for p in re.split(r'\n\s*\n', text)
                      if p.strip() and not self._is_direct_speech(p)]

        if paragraphs:
            features.append(np.median([len(sent_tokenize(p)) for p in paragraphs]))
        else:
            features.append(1)

        # A4-A7: Знаки препинания в концах предложений
        total_sentences = len(sentences)
        features.append(sum(1 for s in sentences if '?' in s) / total_sentences)  # A4
        features.append(sum(1 for s in sentences if '!' in s) / total_sentences)  # A5
        features.append(sum(1 for s in sentences if '...' in s or '!..' in s) / total_sentences)  # A6
        features.append(len(direct_speech) / total_sentences)  # A7

        # ===== Группа Б: Лексические признаки =====

        features.append(self._mattr(words))  # Б1

        # ===== Морфологический анализ (выбор метода по языку) =====
        nouns, verbs, adjs, conjs, preps, total_words, words_main = \
            self._analyze_morphology(text, words_original)
        logger.debug("Морфология: сущ=%d глаг=%d прил=%d всего=%d",
                     nouns, verbs, adjs, total_words)

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

        result = np.nan_to_num(np.array(features), nan=0.0, posinf=0.0, neginf=0.0)
        logger.debug("Вектор признаков: %s", result)
        return result
