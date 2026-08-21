#!/usr/bin/env python3
"""Пакетная проверка: прогоняет все тексты из texts/anonim/ через обученные
профили и печатает сводку.

Заменяет прежний test_all_anonim.py, который не запускался: он импортировал
`from feature_extractor import ...` вместо `from src.feature_extractor`,
жёстко ссылался на /workspace/texts, содержал устаревший список авторов и
вызывал exit(1) прямо в теле модуля.

Использование:
    python scripts/batch_check.py
    python scripts/batch_check.py --profiles authors_profiles_ru.pkl --train
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, io_utils                              # noqa: E402
from src.feature_extractor import FeatureExtractor, Language  # noqa: E402
from src.identifier import identify                           # noqa: E402
from main import (build_authors_profiles, create_fuzzy_profiles,  # noqa: E402
                  load_profiles, save_profiles)

logger = logging.getLogger(__name__)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profiles", default="authors_profiles.pkl",
                        help="файл с сохранёнными портретами")
    parser.add_argument("--train", action="store_true",
                        help="обучить портреты, если их нет")
    parser.add_argument("--language", default=Language.RUSSIAN,
                        choices=[Language.RUSSIAN, Language.BELARUSIAN],
                        help="язык анализа")
    parser.add_argument("--anon-dir", default=None,
                        help="папка с анонимными текстами")
    parser.add_argument("--debug", action="store_true", help="подробный лог")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config.configure_logging("DEBUG" if args.debug else "INFO")

    profiles = load_profiles(args.profiles)
    if profiles is None:
        if not args.train:
            logger.error("❌ Портреты не загружены (%s). Запустите с --train "
                         "или выполните python main.py", args.profiles)
            return 1
        authors = (config.BELARUSIAN_AUTHORS_LIST
                   if args.language == Language.BELARUSIAN
                   else config.RUSSIAN_AUTHORS_LIST)
        authors_data = build_authors_profiles(authors=authors)
        if not authors_data:
            logger.error("❌ Не найдено текстов для обучения")
            return 1
        profiles = create_fuzzy_profiles(authors_data, language=args.language,
                                         save_report=False)
        save_profiles(profiles, args.profiles)

    anon_dir = args.anon_dir or os.path.join(config.BASE_PATH, config.ANON_DIR_NAME)
    anon_files = io_utils.list_txt_files(anon_dir)
    if not anon_files:
        logger.error("❌ В папке %s не найдено .txt файлов", anon_dir)
        return 1

    extractor = FeatureExtractor(language=args.language)

    print("=" * 70)
    print("ПАКЕТНАЯ ПРОВЕРКА АНОНИМНЫХ ТЕКСТОВ")
    print("=" * 70)

    undecided = 0
    for filepath in anon_files:
        text = io_utils.read_text_file(filepath)
        if text is None:
            print(f"\n📄 {os.path.basename(filepath)}\n   ❌ не удалось прочитать файл")
            continue

        best_author, results, _details = identify(profiles, extractor.extract(text))
        best_score = results[best_author]

        print(f"\n📄 {os.path.basename(filepath)}")
        print(f"   Размер: {len(text)} символов")
        print(f"   🎯 Результат: {config.AUTHOR_LABELS.get(best_author, best_author)} "
              f"({best_score:.1%})")

        for author, score in sorted(results.items(), key=lambda x: -x[1]):
            marker = " >>>" if author == best_author else ""
            print(f"      - {config.AUTHOR_LABELS.get(author, author)}: {score:.1%}{marker}")

        if best_score >= config.HIGH_CONFIDENCE_THRESHOLD:
            print(f"   ✅ Уверенность ВЫСОКАЯ (≥{config.HIGH_CONFIDENCE_THRESHOLD:.0%})")
        elif best_score >= config.CONFIDENCE_THRESHOLD:
            print(f"   ⚠️  Уверенность СРЕДНЯЯ "
                  f"({config.CONFIDENCE_THRESHOLD:.0%}-{config.HIGH_CONFIDENCE_THRESHOLD:.0%})")
        else:
            undecided += 1
            print(f"   ❌ Уверенность НИЗКАЯ (<{config.CONFIDENCE_THRESHOLD:.0%}) — "
                  f"авторство не определено")

    print("\n" + "=" * 70)
    print(f"Проанализировано файлов: {len(anon_files)}, не определено: {undecided}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
