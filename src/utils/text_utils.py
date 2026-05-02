# src/utils/text_utils.py
"""Утилиты для работы с текстами"""

import os
import re
from typing import List, Dict, Tuple
from pathlib import Path


def preprocess_text(text: str, 
                    remove_extra_spaces: bool = True,
                    normalize_quotes: bool = True) -> str:
    """
    Предварительная обработка текста
    
    Args:
        text: исходный текст
        remove_extra_spaces: удалять ли лишние пробелы
        normalize_quotes: нормализовать ли кавычки
        
    Returns:
        обработанный текст
    """
    if not text:
        return ""
    
    # Нормализуем переносы строк
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\r', '\n', text)
    
    # Удаляем лишние пробелы
    if remove_extra_spaces:
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n[ \t]*\n', '\n\n', text)
    
    # Нормализуем кавычки
    if normalize_quotes:
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
    
    # Удаляем BOM если есть
    if text.startswith('\ufeff'):
        text = text[1:]
    
    return text.strip()


def load_texts_from_folder(folder_path: str, 
                           encoding: str = 'utf-8',
                           recursive: bool = False) -> List[Tuple[str, str]]:
    """
    Загружает тексты из папки
    
    Args:
        folder_path: путь к папке
        encoding: кодировка файлов
        recursive: загружать ли рекурсивно из подпапок
        
    Returns:
        список кортежей (имя_файла, текст)
    """
    texts = []
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"⚠️  Папка не найдена: {folder_path}")
        return texts
    
    # Определяем режим обхода
    glob_pattern = '**/*.txt' if recursive else '*.txt'
    
    for filepath in folder.glob(glob_pattern):
        try:
            # Пробуем разные кодировки
            encodings_to_try = [encoding, 'utf-8', 'cp1251', 'koi8-r', 'latin-1']
            
            text = None
            for enc in encodings_to_try:
                try:
                    with open(filepath, 'r', encoding=enc) as f:
                        text = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if text is None:
                print(f"  ❌ Не удалось прочитать {filepath}")
                continue
            
            # Предобрабатываем
            text = preprocess_text(text)
            
            if len(text.strip()) > 0:
                relative_name = str(filepath.relative_to(folder))
                texts.append((relative_name, text))
                
        except Exception as e:
            print(f"  ❌ Ошибка при чтении {filepath}: {e}")
    
    print(f"✅ Загружено {len(texts)} текстов из {folder_path}")
    return texts


def load_authors_texts(base_path: str, 
                       authors_list: List[str]) -> Dict[str, List[str]]:
    """
    Загружает тексты авторов из структуры папок
    
    Структура:
        base_path/
            pushkin/
                text1.txt
                text2.txt
            lermontov/
                text1.txt
    
    Args:
        base_path: базовая папка
        authors_list: список имён авторов (имена папок)
        
    Returns:
        словарь {автор: [тексты]}
    """
    authors_data = {}
    
    for author in authors_list:
        author_path = os.path.join(base_path, author)
        
        if not os.path.exists(author_path):
            print(f"⚠️  Папка автора не найдена: {author_path}")
            continue
        
        texts_list = load_texts_from_folder(author_path)
        
        if texts_list:
            authors_data[author] = [text for _, text in texts_list]
            print(f"  ✅ {author}: {len(authors_data[author])} текстов")
        else:
            print(f"  ⚠️  Нет текстов для автора: {author}")
    
    return authors_data


def split_text_into_chunks(text: str, 
                           chunk_size: int = 1000,
                           overlap: int = 100) -> List[str]:
    """
    Разбивает текст на перекрывающиеся чанки
    
    Args:
        text: исходный текст
        chunk_size: размер чанка в символах
        overlap: перекрытие между чанками
        
    Returns:
        список чанков
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Если это не последний чанк, ищем границу предложения
        if end < len(text):
            # Ищем ближайшую точку
            last_period = text.rfind('.', start, end)
            last_excl = text.rfind('!', start, end)
            last_quest = text.rfind('?', start, end)
            
            boundary = max(last_period, last_excl, last_quest)
            
            if boundary > start + chunk_size // 2:
                end = boundary + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
    
    return chunks


def calculate_text_statistics(text: str) -> Dict:
    """
    Вычисляет базовую статистику текста
    
    Returns:
        словарь со статистикой
    """
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    words = re.findall(r'\b\w+\b', text.lower())
    
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    return {
        'characters': len(text),
        'characters_no_spaces': len(re.sub(r'\s', '', text)),
        'words': len(words),
        'sentences': len(sentences),
        'paragraphs': len(paragraphs),
        'avg_word_length': sum(len(w) for w in words) / len(words) if words else 0,
        'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
        'unique_words': len(set(words)),
        'lexical_diversity': len(set(words)) / len(words) if words else 0,
    }
