# Text Analyser 2.0

Стилометрический анализ текстов с использованием нечёткой логики для определения авторства.

## 🚀 Быстрый старт

```bash
# Установка зависимостей
pip install -r requirements.txt

# Запуск анализа
python main_new.py
```

## 📁 Структура проекта

```
text-analyser/
├── src/                          # Исходный код
│   ├── __init__.py
│   ├── core/                     # Основные компоненты
│   │   ├── models.py             # Модели данных
│   │   ├── feature_extractor.py  # Извлечение признаков
│   │   ├── profile_builder.py    # Построение профилей
│   │   └── identifier.py         # Идентификация
│   ├── visualization/            # Визуализация
│   │   └── visualizer.py
│   └── utils/                    # Утилиты
│       └── text_utils.py
├── tests/                        # Тесты
│   └── test_core.py
├── texts/                        # Тексты авторов
│   ├── pushkin/
│   ├── lermontov/
│   ├── tolstoy/
│   └── bulichev/
├── output/                       # Результаты
├── main_new.py                   # Точка входа
├── config.py                     # Конфигурация
└── requirements.txt              # Зависимости
```

## ✨ Новые возможности v2.0

### 1. Улучшенная архитектура
- **Модульность**: Код разделён на логические модули
- **Dataclasses**: Типизированные модели данных
- **Расширяемость**: Легко добавлять новые признаки и методы

### 2. Улучшенные алгоритмы
- **Квантили вместо min/max**: Устойчивость к выбросам
- **Гауссовы функции**: Альтернатива треугольным
- **Бутстрэп оценка**: Доверительные интервалы
- **Ансамбль методов**: Комбинация нечёткой логики и ML

### 3. Новые признаки (планируется)
- N-граммы символов
- Синтаксические зависимости
- Эмоциональная окраска

### 4. Визуализация
- **PDF отчёты**: Автоматическая генерация
- **Интерактивные графики**: Plotly (опционально)
- **Тепловые карты**: Сравнение профилей

### 5. Тестирование
- Покрытие ключевых компонентов
- pytest для запуска

## 📊 Использование

### Базовый анализ

```python
from src.core.feature_extractor import FeatureExtractor
from src.core.profile_builder import ProfileBuilder
from src.core.identifier import FuzzyDetective

# Извлечение признаков
extractor = FeatureExtractor()
features = extractor.extract("Текст для анализа...")

# Построение профиля
builder = ProfileBuilder()
profile = builder.build_from_texts("Автор", ["текст1", "текст2"])

# Идентификация
detector = FuzzyDetective({"Автор": profile})
author, results = detector.identify("Анонимный текст...")
```

### С оценкой уверенности

```python
result = detector.identify_with_confidence(text, n_bootstrap=1000)
print(f"Автор: {result['best_author']}")
print(f"Уверенность: {result['confidence']:.1%}")
print(f"Доверительный интервал: {result['confidence_interval']}")
```

### Ансамбль методов

```python
from src.core.identifier import EnsembleIdentifier

ensemble = EnsembleIdentifier(profiles)
ensemble.train_ml_models(texts, labels)

author = ensemble.predict(text, method='ensemble')
```

## 🧪 Тестирование

```bash
pytest tests/ -v
```

## 📈 Метрики качества

Для оценки точности используйте кросс-валидацию:

```python
from sklearn.model_selection import cross_val_score

ensemble = EnsembleIdentifier(profiles)
ensemble.train_ml_models(train_texts, train_labels)

metrics = ensemble.evaluate(test_texts, test_labels)
print(f"Accuracy: {metrics['accuracy']:.3f}")
print(metrics['report'])
```

## 🔧 Конфигурация

Измените `config.py` для настройки:

```python
AUTHORS_LIST = ["pushkin", "lermontov", "tolstoy"]
ANONIM_TEXT = "texts/anonim/unknown.txt"
N_FEATURES = 17
CONFI DENCE_THRESHOLD = 0.5
```

## 📝 Лицензия

MIT

## 👥 Авторы

Text Analysis Team
