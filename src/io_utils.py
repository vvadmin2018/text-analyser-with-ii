# io_utils.py
"""Чтение текстов с перебором кодировок.

Каскад `utf-8 → cp1251 → koi8-r → latin-1` раньше жил внутри
main.build_authors_profiles() и был недоступен веб-приложению, из-за чего
загруженный пользователем файл в cp1251 прочитать было нечем. Вынесен сюда,
чтобы обучение и веб-приложение читали тексты одинаково.
"""
import logging
import os

from src import config

logger = logging.getLogger(__name__)


def decode_text(raw_data, source_name="<буфер>"):
    """Декодирует байты, перебирая кодировки из config.TEXT_ENCODINGS.

    Возвращает строку или None, если ни одна кодировка не подошла.
    """
    for encoding in config.TEXT_ENCODINGS:
        try:
            text = raw_data.decode(encoding)
        except UnicodeDecodeError:
            continue
        logger.debug("Файл %s прочитан в кодировке %s", source_name, encoding)
        return text
    return None


def read_text_file(filepath):
    """Читает .txt файл с перебором кодировок. None — если не удалось."""
    with open(filepath, 'rb') as f:
        raw_data = f.read()
    return decode_text(raw_data, os.path.basename(filepath))


def list_txt_files(directory):
    """Отсортированный список путей ко всем .txt в каталоге (пусто, если нет)."""
    if not os.path.isdir(directory):
        return []
    return [
        os.path.join(directory, name)
        for name in sorted(os.listdir(directory))
        if name.endswith(".txt")
    ]


def load_author_texts(author, base_path=None):
    """Читает все непустые .txt автора. Возвращает список строк."""
    base_path = base_path or config.BASE_PATH
    author_path = os.path.join(base_path, author)

    if not os.path.isdir(author_path):
        logger.warning("Папка %s не найдена, пропускаем", author_path)
        return []

    texts = []
    for filepath in list_txt_files(author_path):
        filename = os.path.basename(filepath)
        try:
            text = read_text_file(filepath)
        except OSError as e:
            logger.warning("Ошибка при чтении %s: %s", filename, e)
            continue

        if text is None:
            logger.warning("Не удалось определить кодировку файла %s", filename)
            continue
        if not text.strip():
            logger.warning("Файл %s пустой, пропускаем", filename)
            continue

        logger.debug("  %s: %d символов", filename, len(text))
        texts.append(text)

    return texts
