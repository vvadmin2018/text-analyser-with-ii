# config.py
# Глобальные настройки для всего проекта

import os
from datetime import datetime

LEVEL_LOG = "INFO"
#LEVEL_LOG = "DEBUG"
ANONIM_TEXT = "texts/anonim/anonim-8-bulichev-alisa.txt"
BASE_PATH = "texts/"
#AUTHORS_LIST = ["pushkin", "lermontov", "tolstoy", "bulichev"]
AUTHORS_LIST = ["bulichev", "drugkov", "saharnov"]
RUSSIAN_AUTHORS_LIST = ["bulichev", "drugkov", "saharnov"]
BELARUSIAN_AUTHORS_LIST = ["kolas", "maur", "bryl"]
AUTHORS_LIST = RUSSIAN_AUTHORS_LIST  # backward compat

AUTHOR_LABELS = {
    'pushkin': 'Пушкин',
    'tolstoy': 'Толстой',
    'bulichev': 'Кир Булычёв',
    'drugkov': 'Юрий Дружков (Постников)',
    'saharnov': 'Святослав Сахарнов',
    'kolas': 'Колас',
    'maur': 'Маўр',
    'bryl': 'Брыль',
}
OUTPUT_DIR_MAIN = "output/"

# Создаём подпапку с timestamp в формате год-месяц-день-час-минута
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
OUTPUT_DIR = os.path.join("output", timestamp, "")

# Параметры анализа
N_FEATURES = 17
FEATURE_LIST = [
    'Ср. длина предл.', 'Дисп. предл', 'Ср. длина абзаца',
    'Доля ?', 'Доля !', 'Доля ...', 'Прямая речь', 'Лекс. богатство', 'Доля сущ.',
    'Доля глаголов', 'Доля прилаг.', 'Ср. длина слова',
    'Запятые', 'Тире', 'Двоеточия', 'Доля союзов', 'Доля предлогов'
]
FEATURE_LIST_SHORT = [
    'Дл. Предл', 'Дисп.', 'Абз', '?', '!', '...', 'Прям. речь', 'TTR', 'Сущ', 'Глаг', 'Прил',
    'ДлСл', ',', '—', ':', 'Союз', 'Предлог'
]
SOFTENING = 0.8  # Увеличено для более плавных переходов между авторами и лучшей работы с короткими текстами

# Веса признаков по умолчанию - оптимизированы для лучшей дифференциации авторов
DEFAULT_WEIGHTS = [
    2.0,  # 0: Ср. длина предл. (важный признак)
    2.2,  # 1: Дисп. предл (сильно различается у авторов)
    1.8,  # 2: Ср. длина абзаца
    1.5,  # 3: Доля ?
    1.5,  # 4: Доля !
    1.3,  # 5: Доля ...
    1.6,  # 6: Прямая речь (важно для Булычёва)
    2.0,  # 7: Лекс. богатство
    1.8,  # 8: Доля сущ.
    1.8,  # 9: Доля глаголов
    1.6,  # 10: Доля прилаг.
    1.5,  # 11: Ср. длина слова
    1.6,  # 12: Запятые
    1.8,  # 13: Тире (важно для Лермонтова)
    1.4,  # 14: Двоеточия
    1.5,  # 15: Доля союзов (важно для Толстого)
    1.5,  # 16: Доля предлогов
]

# Цвета для авторов
AUTHOR_COLORS = {
    'pushkin': '#FF4B4B',      # ярко-красный
    'lermontov': '#4B7BFF',    # ярко-синий
    'tolstoy': '#FFB44B',      # оранжевый
    'bulichev': '#FF4BFF',     # розовый
    'kolas': '#2E8B57',        # зелёный
    'maur': '#8B4513',         # коричневый
    'bryl': '#4169E1',         # синий
    'drugkov': '#FF6347',      # томатный
    'saharnov': '#00CED1',     # бирюзовый
    'default': '#4B7BFF'       # синий по умолчанию
}

# Порог уверенности идентификации
CONFIDENCE_THRESHOLD = 0.5
