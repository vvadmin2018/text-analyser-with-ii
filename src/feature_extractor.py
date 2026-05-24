# feature_extractor.py
import nltk
# Скачиваем необходимые данные NLTK (можно закомментировать после первого запуска)
# nltk.download('stopwords')
# nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
import string
import numpy as np
try:
    import pymorphy3 as pymorphy2
except ImportError:
    import pymorphy2
import re
from src import config

class FeatureExtractor:
    """Извлекает стилевые признаки из текста с использованием pymorphy2 для морфологического анализа"""

    def __init__(self):
        # Загружаем стоп-слова для русского языка
        self.stopwords = set(nltk.corpus.stopwords.words('russian'))

        # Инициализируем морфологический анализатор pymorphy2
        self.morph = pymorphy2.MorphAnalyzer()

        # Отдельные множества для союзов
        self.conjunctions = set([
            'и', 'а', 'но', 'да', 'или', 'либо', 'что', 'чтобы', 'если',
            'когда', 'потому', 'так', 'как', 'чем', 'однако', 'зато', 'тоже',
            'также', 'причем', 'притом', 'потому', 'поэтому', 'зачем', 'отчего'
        ])

        # Отдельные множества для предлогов
        self.prepositions = set([
            'в', 'во', 'на', 'с', 'со', 'к', 'ко', 'у', 'о', 'об', 'от',
            'ото', 'из', 'изо', 'за', 'для', 'без', 'безо', 'до', 'при',
            'про', 'через', 'сквозь', 'между', 'среди', 'над', 'под', 'перед',
            'возле', 'около', 'вокруг', 'мимо', 'после', 'ради', 'вроде'
        ])

    def extract(self, text):
        """
        Возвращает массив признаков

        Признаки:
        A1  - средняя длина предложения
        A2  - дисперсия длины предложений
        A3  - средняя длина абзаца (в предложениях)
        A4  - доля вопросительных предложений
        A5  - доля восклицательных предложений
        A6  - доля предложений с троеточиями
        A7  - доля предложений с прямой речью
        Б1  - лексическое богатство (TTR)
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

        # Токенизация слов (для частотного анализа)
        words = word_tokenize(text.lower())
        words = [w for w in words if w not in string.punctuation and w.isalpha()]

        # Токенизация для морфологического анализа (сохраняем оригинальный регистр)
        words_original = word_tokenize(text)
        words_original = [w for w in words_original if w not in string.punctuation and w.isalpha()]

        # Если нет слов или предложений, возвращаем нули
        if len(words) == 0 or len(sentences) == 0:
            return np.zeros(num_props)

        features = []

        # ===== Группа А: Синтаксические признаки =====

        # A1: Средняя длина предложения (в словах)
        sent_lengths = []
        for sent in sentences_without_primaya_rech:
            sent_words = word_tokenize(sent)
            sent_words = [w for w in sent_words if w not in string.punctuation and w.isalpha()]
            sent_lengths.append(len(sent_words))

        # print(f"DEBUGPRINT      A1-USED: Средняя длина предложения (в словах): {np.mean(sent_lengths)}")
        if config.LEVEL_LOG == "DEBUG":
            print(f"DEBUGPRINT      A1-NEW: Средняя длина предложения (в словах): {np.median(sent_lengths)}")
        features.append(np.median(sent_lengths))  # A1

        # A2: Дисперсия длины предложения
        if len(sent_lengths) > 1:
            variance = np.var(sent_lengths)
        else:
            variance = 0
        features.append(variance)
        if config.LEVEL_LOG == "DEBUG":
            print(f"DEBUGPRINT      A2 (дисперсия): {variance:.2f}")

        # A3: Средняя длина абзаца (разделяем по двойному переносу строки)
        # Ищем блоки текста, разделенные одним или несколькими переносами
        paragraphs = re.split(r'\n\s*\n', text)  # один или несколько переносов с пробелами
        #Исключаем прямую речь из абзацев
        paragraphs = [p for p in paragraphs if (p.strip() and not p.startswith('–') and not p.startswith('—'))]

        if paragraphs:
            para_lengths = []
            for p in paragraphs:
                p_sents = sent_tokenize(p)
                para_lengths.append(len(p_sents))

            if config.LEVEL_LOG == "DEBUG":
                print(f"DEBUGPRINT    A3: Колво абзацев {len(paragraphs)}")
                #print(f"DEBUGPRINT    A2-USED-AVG: Средняя длина абзаца {np.mean(para_lengths)}")
                print(f"DEBUGPRINT    A3-MEDIANA: Средняя длина абзаца {np.median(para_lengths)}")
            features.append(np.median(para_lengths))
        else:
            print("Считаем весь текст один абзацем")
            features.append(1)  # если нет абзацев, считаем весь текст одним абзацем

        # A4: Доля вопросительных предложений
        q_marks = sum(1 for sent in sentences if '?' in sent)

        if config.LEVEL_LOG == "DEBUG":
            print(f"DEBUGPRINT    A4: Колво предложений {len(sentences)}")
            print(f"DEBUGPRINT    A4: Доля вопросительных предложений {q_marks / len(sentences)}")
        features.append(q_marks / len(sentences))

        # A5: Доля восклицательных предложений
        ex_marks = sum(1 for sent in sentences if '!' in sent)

        if config.LEVEL_LOG == "DEBUG":
            print(f"DEBUGPRINT    A5: Доля восклицательных предложений {ex_marks / len(sentences)}")
        features.append(ex_marks / len(sentences))

        # A6: Доля троеточий (новый)
        dot3_marks = sum(1 for sent in sentences if '...' in sent or '!..' in sent)

        if config.LEVEL_LOG == "DEBUG":
            print(f"DEBUGPRINT    A6: Колво предложений c троеточиями {dot3_marks}")
            print(f"DEBUGPRINT    A6: Доля предложений c троеточиями {dot3_marks / len(sentences)}")
        features.append(dot3_marks / len(sentences))

        # A7: Доля предложений с прямой речью (новый)

        if config.LEVEL_LOG == "DEBUG":
            print(f"DEBUGPRINT    A7: Колво предложений c прямой речью {len(sentences_primaya_rech)}")
            print(f"DEBUGPRINT    A7: Доля предложений c прямой речью {len(sentences_primaya_rech) / len(sentences)}")
        features.append(len(sentences_primaya_rech) / len(sentences))

        # ===== Группа Б: Лексические признаки =====

        # Б1: Лексическое богатство (TTR - type-token ratio)
        #unique_words = set(words)

        #if config.LEVEL_LOG == "DEBUG":
        #    print(f"DEBUGPRINT    Б1: Лексическое богатство {len(unique_words) / len(words)}")
        #features.append(len(unique_words) / len(words))  # Б1

        # Б1: Лексическое богатство (RTTR - root type-token ratio)
        unique_words = set(words)
        total_words = len(words)

        # RTTR = types / sqrt(tokens)
        rttr = len(unique_words) / (total_words ** 0.5)

        if config.LEVEL_LOG == "DEBUG":
            print(
                f"DEBUGPRINT    Б1: Лексическое богатство RTTR = {rttr:.4f} (types={len(unique_words)}, tokens={total_words})")
        features.append(rttr)  # Б1

        # ===== МОРФОЛОГИЧЕСКИЙ АНАЛИЗ С ПОМОЩЬЮ PYMORPHY2 =====
        # Счетчики для частей речи
        pos_counts = {
            'NOUN': 0,    # существительное
            'ADJF': 0,    # прилагательное (полное)
            'ADJS': 0,    # прилагательное (краткое)
            'VERB': 0,    # глагол
            'INFN': 0,    # инфинитив
            'PRTS': 0,    # причастие (краткое)
            'PRTF': 0,    # причастие (полное)
            'GRND': 0,    # деепричастие
            'NUMR': 0,    # числительное
            'ADVB': 0,    # наречие
            'NPRO': 0,    # местоимение
            'PRED': 0,    # предикатив
            'PREP': 0,    # предлог
            'CONJ': 0,    # союз
            'PRCL': 0,    # частица
            'INTJ': 0,    # междометие
        }

        # Анализируем каждое слово
        words_main = []
        for word in words_original:
            try:
                # Получаем разбор слова
                parsed = self.morph.parse(word)[0]
                pos = parsed.tag.POS
                if pos and pos in pos_counts:
                    pos_counts[pos] += 1
                if pos in ['NOUN', 'VERB', 'INFN', 'GRND', 'PRTF', 'ADJF']:
                    words_main.append(word)

            except Exception as e:
                # В случае ошибки пропускаем слово
                continue

        total_words = len(words_original)

        # Б2: Доля существительных (включая все формы)
        nouns = (pos_counts['NOUN'] + pos_counts['NUMR'])  # существительные и числительные
        features.append(nouns / total_words if total_words > 0 else 0)  # Б2
        if config.LEVEL_LOG == "DEBUG":
            print(f"DEBUGPRINT      Б2: Существительных: {nouns}")
            print(f"DEBUGPRINT      Б2: Общее колво слов: {total_words}")
            print(f"DEBUGPRINT      Б2: Доля существительных: {nouns / total_words if total_words > 0 else 0}")

        # Б3: Доля глаголов (включая все формы)
        verbs = (pos_counts['VERB'] + pos_counts['INFN'] + pos_counts['GRND'] +
                 pos_counts['PRTS'] + pos_counts['PRTF'])  # глаголы, инфинитивы, причастия, деепричастия
        features.append(verbs / total_words if total_words > 0 else 0)  # Б3

        if config.LEVEL_LOG == "DEBUG":
            print(f"DEBUGPRINT      Б3: Глаголов: {verbs}")
            print(f"DEBUGPRINT      Б3: Доля глаголов: {verbs / total_words if total_words > 0 else 0}")

        # Б4: Доля прилагательных
        adjs = pos_counts['ADJF'] + pos_counts['ADJS']
        features.append(adjs / total_words if total_words > 0 else 0)  # Б4

        if config.LEVEL_LOG == "DEBUG":
            print(f"DEBUGPRINT      Б4: Прилагательных: {adjs}")
            print(f"DEBUGPRINT      Б4: Доля прилагательных: {adjs / total_words if total_words > 0 else 0}")

        # Б5: Средняя длина слова
        word_lengths = [len(w) for w in words_main]

        if config.LEVEL_LOG == "DEBUG":
            print(f"DEBUGPRINT      Б5: Средняя длина слова: {np.mean(word_lengths)}")
        features.append(np.mean(word_lengths))  # Б5

        # ===== Группа В: Пунктуационно-служебные признаки =====

        # В1: Частота запятых (на предложение)
        features.append(text.count(',') / len(sentences))  # В1

        if config.LEVEL_LOG == "DEBUG":
            print(f"DEBUGPRINT      В1: Колво запятых: {text.count(',')}")
            print(f"DEBUGPRINT      В1: Колво предложений: {len(sentences)}")
            print(f"DEBUGPRINT      В1: Частота запятых (на предложение): {text.count(',') / len(sentences)}")

        # В2: Частота тире (на предложение)
        features.append((text.count('—') + text.count('–')) / len(sentences))  # В2
        if config.LEVEL_LOG == "DEBUG":
            print(f"DEBUGPRINT      В2: Колво тире: {(text.count('—') + text.count('–'))}")
            print(f"DEBUGPRINT      В2: Частота тире (на предложение): {(text.count('—') + text.count('–')) / len(sentences)}")

        # В3: Частота двоеточий (на предложение)
        features.append((text.count(':') + text.count(':')) / len(sentences))  # В3

        if config.LEVEL_LOG == "DEBUG":
            print(f"DEBUGPRINT      В3: Колво двоеточий: {(text.count(':') + text.count(':'))}")
            print(f"DEBUGPRINT      В3: Частота двоеточий: {(text.count(':') + text.count(':')) / len(sentences)}")

        # В4: Доля союзов (используем и pymorphy2, и словарь для надежности)
        conj_by_morph = pos_counts['CONJ']
        conj_by_dict = sum(1 for w in words if w in self.conjunctions)
        conj_total = max(conj_by_morph, conj_by_dict)  # берем максимум для надежности
        features.append(conj_total / total_words if total_words > 0 else 0)  # В4

        if config.LEVEL_LOG == "DEBUG":
            print(f"DEBUGPRINT      В4: Союзов (либа): {conj_by_morph}")
            print(f"DEBUGPRINT      В4: Союзов (словарь): {conj_by_dict}")
            print(f"DEBUGPRINT      В4: Доля союзов: {conj_total / total_words if total_words > 0 else 0}")

        # В5: Доля предлогов
        prep_by_morph = pos_counts['PREP']
        prep_by_dict = sum(1 for w in words if w in self.prepositions)
        prep_total = max(prep_by_morph, prep_by_dict)  # берем максимум для надежности
        features.append(prep_total / total_words if total_words > 0 else 0)  # В5

        if config.LEVEL_LOG == "DEBUG":
            print(f"DEBUGPRINT      В5: Предлогов (либа): {prep_by_morph}")
            print(f"DEBUGPRINT      В5: Предлогов (словарь): {prep_by_dict}")
            print(f"DEBUGPRINT      В5: Доля предлогов: {prep_total / total_words if total_words > 0 else 0}")

        # Преобразуем в numpy array
        result = np.array(features)

        # Защита от NaN и Inf
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

        return result