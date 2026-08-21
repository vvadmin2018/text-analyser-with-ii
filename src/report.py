# report.py
"""Сводные таблицы признаков автора: DataFrame, HTML и файлы отчёта.

Раньше это жило прямо в AuthorProfile: класс, который должен строить нечёткий
портрет, заодно генерировал 130 строк HTML/CSS и писал файлы на диск. Здесь всё
то же самое, но отдельно от модели — и доступно веб-приложению для показа
таблицы прямо на странице (get_summary_html → build_table_html).
"""
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd

from src import config
from src.visualizer import (
    ANON_ACCENT, GRID_COLOR, INK_TEXT, INK_TEXT_SOFT, PAPER_BG,
    _author_color, _display_name,
)

logger = logging.getLogger(__name__)

STATS_ROW_LABELS = ('Среднее', 'Медиана')


def build_stats(feature_matrix):
    """Собирает статистику по матрице признаков (строки — тексты).

    Возвращает dict с ключами mean/median/text_count/dataframe — он же
    сохраняется в AuthorProfile.feature_stats и переживает pickle.
    """
    df = pd.DataFrame(feature_matrix, columns=config.FEATURE_LIST)
    df.insert(0, 'Текст №', range(1, len(df) + 1))
    for col in config.FEATURE_LIST:
        df[col] = df[col].round(3)

    columns = np.asarray(feature_matrix, dtype=float)
    return {
        'mean': [round(float(v), 3) for v in columns.mean(axis=0)],
        'median': [round(float(v), 3) for v in np.median(columns, axis=0)],
        'text_count': len(feature_matrix),
        'dataframe': df,
    }


def make_df_with_stats(stats):
    """Достраивает DataFrame строками 'Среднее' и 'Медиана'."""
    rows = [stats['dataframe']]
    for label, values in zip(STATS_ROW_LABELS, (stats['mean'], stats['median'])):
        row = {'Текст №': label}
        row.update(dict(zip(config.FEATURE_LIST, values)))
        rows.append(pd.DataFrame([row]))
    return pd.concat(rows, ignore_index=True)


def log_summary_table(name, stats):
    """Печатает сводную таблицу в лог (для CLI-прогона main.py)."""
    if not logger.isEnabledFor(logging.INFO):
        return
    ruler = "=" * (len(config.FEATURE_LIST) * 12 + 15)
    logger.info("\n  📊 Сводная таблица признаков для автора '%s':", _display_name(name))
    logger.info("  %s", ruler)
    logger.info("%s", stats['dataframe'].to_string(index=False))
    logger.info("  %s", "-" * len(ruler))
    logger.info("  Среднее:  %s", '  '.join(f'{v:>10}' for v in stats['mean']))
    logger.info("  Медиана:  %s", '  '.join(f'{v:>10}' for v in stats['median']))
    logger.info("  %s", ruler)


def build_table_html(name, stats):
    """Строит полный HTML-документ сводной таблицы в теме "бумага и чернила".

    Ничего не пишет на диск — этим занимается save_summary_table; тот же HTML
    веб-приложение показывает прямо на странице.
    """
    df_with_stats = make_df_with_stats(stats)
    author_display = _display_name(name)
    author_color = config.AUTHOR_COLORS.get(name) or _author_color(name)

    # ===== Компактность: таблица должна умещаться в одну страницу =====
    # При 17 признаках + колонка "Текст №" (18 колонок) полные описательные
    # названия ("Ср. длина предл." и т.п.) и много знаков после запятой
    # гарантированно выталкивают таблицу за пределы экрана вправо. Поэтому:
    #   1) заголовки колонок — короткие подписи (те же, что и на графиках);
    #   2) числа округляются при отображении;
    #   3) table-layout: fixed + width: 100% — таблица физически не может
    #      стать шире родительского контейнера, колонки просто сжимаются.
    short_names = dict(zip(config.FEATURE_LIST, config.FEATURE_LIST_SHORT))
    display_df = df_with_stats.rename(columns=short_names)
    feature_cols = [short_names.get(c, c) for c in config.FEATURE_LIST]

    # Строки "Среднее"/"Медиана" — как заметка золотистыми чернилами на
    # полях: мягкая заливка акцентным цветом и линия сверху, чтобы сразу
    # отличались от построчных значений отдельных текстов.
    def _highlight_stats_row(row):
        if row.iloc[0] in STATS_ROW_LABELS:
            return [f'background-color: {ANON_ACCENT}2E; font-weight: 700; '
                    f'border-top: 2px solid {ANON_ACCENT};'] * len(row)
        return [''] * len(row)

    styler = (
        display_df.style
        .format('{:.3f}', subset=feature_cols)
        .apply(_highlight_stats_row, axis=1)
        .set_table_attributes('class="data-table"')
        .hide(axis='index')
        .set_table_styles([
            {'selector': 'table', 'props': [
                ('table-layout', 'fixed'), ('width', '100%'),
            ]},
            {'selector': 'th', 'props': [
                ('background-color', author_color), ('color', PAPER_BG),
                ('padding', '5px 3px'), ('font-weight', '600'),
                ('font-size', '10.5px'), ('line-height', '1.2'),
                ('word-break', 'break-word'), ('white-space', 'normal'),
                ('border', f'1px solid {author_color}'),
            ]},
            {'selector': 'td', 'props': [
                ('padding', '4px 3px'), ('text-align', 'center'),
                ('font-size', '11px'),
                ('border', f'1px solid {GRID_COLOR}'),
            ]},
            {'selector': 'th:first-child, td:first-child', 'props': [
                ('width', '60px'),
            ]},
            {'selector': 'tbody tr:nth-child(odd)', 'props': [('background-color', PAPER_BG)]},
            {'selector': 'tbody tr:nth-child(even)', 'props': [('background-color', '#F3ECDD')]},
            {'selector': 'tbody tr:hover', 'props': [('background-color', '#EFE3C8')]},
        ], overwrite=False)
    )

    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>Сводная таблица признаков — {author_display}</title>
    <style>
        * {{ box-sizing: border-box; }}
        html, body {{ width: 100%; }}
        body {{
            font-family: Verdana, Geneva, sans-serif;
            background: {PAPER_BG};
            color: {INK_TEXT};
            margin: 0;
            padding: 18px 22px 28px;
        }}
        .sheet {{
            width: 100%;
            max-width: 100%;
            margin: 0 auto;
            background: #FFFFFF;
            border: 1px solid {GRID_COLOR};
            border-radius: 10px;
            padding: 18px 20px 22px;
            box-shadow: 0 6px 18px rgba(42, 35, 28, 0.10);
        }}
        h1 {{
            font-family: Georgia, "Times New Roman", serif;
            font-size: 19px;
            color: {author_color};
            border-bottom: 3px solid {author_color};
            padding-bottom: 8px;
            margin: 0 0 6px;
        }}
        .meta {{ color: {INK_TEXT_SOFT}; font-size: 12.5px; margin: 2px 0; }}
        table.data-table {{
            border-collapse: collapse;
            table-layout: fixed;
            width: 100%;
            margin-top: 16px;
            font-size: 11px;
        }}
        .footer-note {{
            margin-top: 14px;
            font-size: 11.5px;
            color: {INK_TEXT_SOFT};
            font-style: italic;
        }}
    </style>
</head>
<body>
    <div class="sheet">
        <h1>Сводная таблица признаков автора: {author_display}</h1>
        <p class="meta">Текстов в обучающей выборке: {stats['text_count']}</p>
        <p class="meta">Дата генерации: {datetime.now().strftime("%d.%m.%Y %H:%M")}</p>
        {styler.to_html()}
        <p class="footer-note">Значения округлены до 3 знаков после запятой.</p>
    </div>
</body>
</html>"""


def save_summary_table(name, stats, output_dir=None):
    """Сохраняет сводную таблицу признаков в output/ в форматах HTML и TXT.

    Возвращает (html_path, txt_path).
    """
    df_with_stats = make_df_with_stats(stats)
    author_display = _display_name(name)

    base_dir = output_dir or config.OUTPUT_DIR_MAIN
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    author_dir = os.path.join(base_dir, f"{timestamp}_{name}")
    os.makedirs(author_dir, exist_ok=True)

    html_file = os.path.join(author_dir, f"{name}_summary.html")
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(build_table_html(name, stats))

    ruler = "=" * (len(config.FEATURE_LIST) * 12 + 15)
    txt_file = os.path.join(author_dir, f"{name}_summary.txt")
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write(f"Сводная таблица признаков автора: {author_display}\n")
        f.write(f"Количество текстов: {stats['text_count']}\n")
        f.write(f"Дата генерации: {datetime.now().strftime('%d.%m.%Y %H:%M')}\n")
        f.write(ruler + "\n\n")
        f.write(df_with_stats.to_string(index=False))
        f.write("\n\n" + ruler + "\n")
        f.write("Примечание: все значения округлены до 3 знаков после запятой\n")

    logger.info("  💾 Таблица сохранена в:\n     HTML: %s\n     TXT:  %s", html_file, txt_file)
    return html_file, txt_file
