# app.py — веб-интерфейс THinkING (Streamlit)
"""Веб-приложение: выбор языка, ввод текста, анализ авторства и графики."""
import base64
import io
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import nltk                      # noqa: E402
import streamlit as st           # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import config, io_utils, visualizer                  # noqa: E402
from src.feature_extractor import (FeatureExtractor, Language,  # noqa: E402
                                   detect_language)
from src.identifier import identify                           # noqa: E402
from src.profile_builder import AuthorProfile                 # noqa: E402
from src.visualizer import StyleRose                          # noqa: E402
from main import build_authors_profiles, load_profiles, save_profiles  # noqa: E402

APP_DIR = os.path.dirname(os.path.abspath(__file__))

LANG_OPTIONS = {
    "Русский": {
        "lang": Language.RUSSIAN,
        "authors": config.RUSSIAN_AUTHORS_LIST,
        "pickle": "authors_profiles_ru.pkl",
    },
    "Белорусский": {
        "lang": Language.BELARUSIAN,
        "authors": config.BELARUSIAN_AUTHORS_LIST,
        "pickle": "authors_profiles_be.pkl",
    },
}

st.set_page_config(page_title="THinkING", layout="wide")
config.configure_logging()


# ============================================================
# Кэшируемые ресурсы
#
# Раньше всё это выполнялось на КАЖДЫЙ rerun Streamlit (то есть на каждое
# нажатие любой кнопки): три обращения nltk.download, чтение и base64-кодирование
# PNG, а главное — конструктор FeatureExtractor, который поднимает
# pymorphy3.MorphAnalyzer (~1-2 с), а для белорусского ещё и пайплайн Stanza.
# ============================================================

@st.cache_resource(show_spinner=False)
def ensure_nltk_data():
    for package in ('punkt', 'punkt_tab', 'stopwords'):
        nltk.download(package, quiet=True)
    return True


@st.cache_resource(show_spinner="Загружаем морфологический анализатор...")
def get_extractor(language):
    ensure_nltk_data()
    return FeatureExtractor(language=language)


@st.cache_data(show_spinner=False)
def get_ghost_b64(dark_mode):
    name = "ghost_transparent_light_ink.png" if dark_mode else "ghost_transparent_dark_ink.png"
    with open(os.path.join(APP_DIR, "resources", name), "rb") as f:
        return base64.b64encode(f.read()).decode()


def author_display(name):
    return config.AUTHOR_LABELS.get(name, name)


def fig_to_png(fig):
    """PNG-байты фигуры для st.download_button.

    Раньше приложение вместо этого создавало на сервере output/<timestamp>/ и
    писало туда PNG на каждый показ результата: пользователь этих файлов не
    видел, а каталог рос с каждым анализом.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    return buf.getvalue()


def show_chart(fig, download_name, key):
    """Рисует график, даёт кнопку скачивания и освобождает фигуру."""
    st.pyplot(fig)
    st.download_button("⬇ Скачать PNG", fig_to_png(fig), file_name=download_name,
                       mime="image/png", key=key)
    plt.close(fig)


def language_warning(text, language):
    """Грубая проверка, что текст написан на выбранном языке.

    Без неё можно было проанализировать русский текст белорусскими профилями
    и получить уверенный, но бессмысленный процент.

    Опирается на тот же detect_language, что и блок статистики: иначе
    предупреждение и показанный пользователю «предполагаемый язык» могли бы
    противоречить друг другу.
    """
    detected = detect_language(text)
    if detected is None or detected == language:
        return None

    if language == Language.BELARUSIAN:
        return ("Похоже, текст не на белорусском языке — специфичных букв «ў» и «і» "
                "в нём нет. Результат может быть бессмысленным.")
    return ("Похоже, текст на белорусском языке. Переключите язык анализа, "
            "иначе результат будет бессмысленным.")


LANGUAGE_NAMES = {
    Language.RUSSIAN: "русский",
    Language.BELARUSIAN: "белорусский",
}


def render_text_stats(stats, selected_language):
    """Общая информация о разобранном тексте.

    Числа берутся из FeatureExtractor.describe(), то есть считаны той же
    токенизацией, что и признаки. Показывать здесь свой, отдельный подсчёт
    слов было бы хуже, чем не показывать никакого: расхождение с анализом
    выглядит как ошибка, даже когда оба числа по-своему верны.
    """
    if not stats:
        return

    st.markdown("**О тексте:**")

    detected = stats.get("language")
    if detected is None:
        language_line = "не определён"
    else:
        language_line = LANGUAGE_NAMES.get(detected, detected)
        if detected != selected_language:
            language_line += " ⚠️"

    row1 = st.columns(3)
    row1[0].metric("Символов", f"{stats['chars']:,}".replace(",", " "))
    row1[1].metric("Слов", f"{stats['words']:,}".replace(",", " "))
    row1[2].metric("Предложений", f"{stats['sentences']:,}".replace(",", " "))

    row2 = st.columns(3)
    row2[0].metric("Абзацев", f"{stats['paragraphs']:,}".replace(",", " "))
    row2[1].metric("Без пробелов", f"{stats['chars_no_spaces']:,}".replace(",", " "))
    row2[2].metric("Язык", language_line)

    st.caption("Слова — без знаков препинания и чисел, как их считает анализатор. "
               "Язык определяется по буквам «ў», «і» против «и», «щ», «ъ».")


# ============================================================
# Состояние сессии
# ============================================================

DEFAULT_STATE = {
    "dark_mode": False,
    "profiles": None,
    "last_lang": None,
    "results": None,
    "input_text": "",
    "analyze_requested": False,
    "message": None,
    "last_upload": None,
}
for key, value in DEFAULT_STATE.items():
    st.session_state.setdefault(key, value)


def reset_analysis():
    st.session_state.results = None
    st.session_state.message = None


def clear_input():
    st.session_state.input_text = ""
    reset_analysis()


def request_analysis():
    st.session_state.analyze_requested = True


def forget_profiles(profile_path):
    st.session_state.profiles = None
    st.session_state.input_text = ""
    reset_analysis()
    if os.path.exists(profile_path):
        os.remove(profile_path)


# ============================================================
# Сайдбар: настройки и профили
# ============================================================

st.sidebar.header("Настройки")
st.session_state.dark_mode = st.sidebar.toggle("🌙 Тёмная тема",
                                               value=st.session_state.dark_mode)
lang_name = st.sidebar.selectbox("Язык анализа", list(LANG_OPTIONS.keys()), key="lang_name")
lang_cfg = LANG_OPTIONS[lang_name]
cur_lang = lang_cfg["lang"]
profile_path = lang_cfg["pickle"]

if st.session_state.last_lang != lang_name:
    st.session_state.profiles = None
    st.session_state.input_text = ""
    st.session_state.last_lang = lang_name
    reset_analysis()

# Графики следуют за темой приложения — иначе белое полотно matplotlib
# било по глазам на тёмном фоне страницы.
visualizer.use_theme(dark=st.session_state.dark_mode)


def train_profiles():
    """Обучает профили выбранного языка, показывая прогресс по авторам."""
    authors_data = build_authors_profiles(authors=lang_cfg["authors"])
    if not authors_data:
        st.sidebar.error(f"Нет текстов для обучения. Добавьте .txt в "
                         f"{config.BASE_PATH}<автор>/")
        return None

    extractor = get_extractor(cur_lang)
    progress = st.sidebar.progress(0.0, "Обучение...")
    new_profiles = {}

    for i, (author_name, texts) in enumerate(authors_data.items()):
        progress.progress(i / len(authors_data),
                          f"Обучаем: {author_display(author_name)} ({len(texts)} текстов)")
        profile = AuthorProfile(author_name)
        # save_report=False: таблица показывается на странице (вкладка
        # «Профили авторов»), плодить файлы на сервере незачем.
        profile.build_from_texts(texts, language=cur_lang, save_report=False,
                                 extractor=extractor)
        new_profiles[author_name] = profile

    progress.progress(1.0, "Готово")
    save_profiles(new_profiles, profile_path)
    return new_profiles


def render_profiles_sidebar(profiles):
    """Список обученных авторов и кнопка переобучения.

    Раньше этот блок был написан дважды — в ветке «профили только что
    загружены» и в ветке «профили уже в сессии», строка в строку.
    """
    st.sidebar.success("Загружены профили:")
    for name in profiles:
        st.sidebar.markdown(f"- {author_display(name)}")
    if st.sidebar.button("🔄 Переобучить", use_container_width=True):
        forget_profiles(profile_path)
        st.rerun()


st.sidebar.divider()
st.sidebar.subheader("Профили авторов")

if st.session_state.profiles is None:
    st.session_state.profiles = load_profiles(profile_path,
                                              authors=lang_cfg["authors"])

if st.session_state.profiles is not None:
    render_profiles_sidebar(st.session_state.profiles)
else:
    st.sidebar.warning("Профили не найдены")
    if st.sidebar.button("Обучить профили", use_container_width=True):
        st.session_state.profiles = train_profiles()
        if st.session_state.profiles:
            st.rerun()


# ============================================================
# Анализ (выполняется до отрисовки, чтобы разметка знала о результате)
# ============================================================

def run_analysis():
    text = st.session_state.input_text
    profiles = st.session_state.profiles

    if profiles is None:
        st.session_state.message = ("error", "Сначала обучите профили авторов.")
        return
    if not text.strip():
        st.session_state.message = ("warning", "Введите текст для анализа.")
        return
    if len(text.strip()) < config.MIN_TEXT_LENGTH:
        st.session_state.message = ("warning", (
            f"Текст слишком короткий (минимум {config.MIN_TEXT_LENGTH} символов, "
            f"сейчас {len(text.strip())})."))
        return

    with st.spinner("Анализ..."):
        extractor = get_extractor(cur_lang)
        try:
            anon_features = extractor.extract(text)
        except Exception as e:
            st.session_state.message = ("error", f"Не удалось разобрать текст: {e}")
            return

        best_author, results, similarity_details = identify(profiles, anon_features)

    st.session_state.message = None
    st.session_state.results = {
        "best_author": best_author,
        "best_score": results[best_author],
        "results": results,
        "anon_features": anon_features,
        "similarity_details": similarity_details,
        "degraded": extractor.degraded_reason,
        "lang_warning": language_warning(text, cur_lang),
        "stats": extractor.describe(text),
    }


if st.session_state.analyze_requested:
    st.session_state.analyze_requested = False
    run_analysis()


# ============================================================
# Оформление
#
# Тёмную тему приходится делать через CSS: Streamlit не умеет переключать
# [theme] из config.toml во время работы приложения. Но палитра теперь задана
# один раз через CSS-переменные, а правила опираются на стабильные
# data-testid-селекторы. Раньше здесь было ~120 строк !important-ов, включая
# хэш-классы вида .st-emotion-cache-10trblm — они генерируются сборкой
# Streamlit и меняются от версии к версии, так что тёмная тема ломалась бы на
# первом же обновлении пакета.
# ============================================================

LIGHT_VARS = """
    --paper: #F5F0E1;
    --paper-raised: #FAF6ED;
    --sidebar: #ECE6D3;
    --ink: #2A231C;
    --ink-soft: #6B6153;
    --border: #DCD2B8;
    --accent: #B8860B;
    --accent-ink: #FFFFFF;
    --accent-hover: #A0760A;
    --retrain: #C4A882;
    --retrain-ink: #3A2A1A;
"""

DARK_VARS = """
    --paper: #0E1117;
    --paper-raised: #1A1D24;
    --sidebar: #1E2028;
    --ink: #E8E3D8;
    --ink-soft: #A2998A;
    --border: #3A3F4B;
    --accent: #C79A2B;
    --accent-ink: #14100A;
    --accent-hover: #D8AB3C;
    --retrain: #3A3F4B;
    --retrain-ink: #E8E3D8;
"""

APP_CSS = f"""
<style>
.stApp {{ {DARK_VARS if st.session_state.dark_mode else LIGHT_VARS} }}

.stApp, [data-testid="stHeader"] {{ background-color: var(--paper); }}
.stApp {{ color: var(--ink); }}
[data-testid="stSidebar"] {{ background-color: var(--sidebar); }}
[data-testid="stDecoration"] {{ display: none; }}

.stApp h1, .stApp h2, .stApp h3, .stApp h4,
[data-testid="stMarkdownContainer"], .stApp label, .stApp p, .stApp li {{
    color: var(--ink);
}}
.subtitle, .stCaption, [data-testid="stCaptionContainer"] {{ color: var(--ink-soft); }}
h1 {{ margin-top: -24px; padding-top: 0; }}

.stTextArea textarea,
[data-baseweb="select"] > div,
[data-testid="stFileUploaderDropzone"] {{
    background-color: var(--paper-raised);
    color: var(--ink);
    border-color: var(--border);
}}
.stTextArea textarea::placeholder {{ color: var(--ink-soft); }}

.stButton button {{
    background-color: var(--paper-raised);
    color: var(--ink);
    border-color: var(--border);
}}
.stButton button:hover {{ border-color: var(--accent); color: var(--accent); }}

.stButton button[kind="primary"],
[data-testid="stBaseButton-primary"],
[data-testid="stDownloadButton"] button {{
    background-color: var(--accent);
    color: var(--accent-ink);
    border-color: var(--accent);
}}
.stButton button[kind="primary"]:hover,
[data-testid="stBaseButton-primary"]:hover,
[data-testid="stDownloadButton"] button:hover {{
    background-color: var(--accent-hover);
    border-color: var(--accent-hover);
    color: var(--accent-ink);
}}

/* Радужная рамка вокруг поля ввода в фокусе — своего аналога у Streamlit нет */
.stTextArea {{ position: relative; }}
.stTextArea:focus-within::before {{
    content: '';
    position: absolute;
    inset: -3px;
    border-radius: 4px;
    background: conic-gradient(#E91E63, #9C27B0, #2196F3, #00BCD4,
                               #4CAF50, #FFEB3B, #FF9800, #E91E63);
    z-index: -1;
    animation: spin 4s linear infinite;
    pointer-events: none;
}}
@keyframes spin {{ to {{ transform: rotate(360deg); }} }}
.stTextArea textarea:focus {{
    outline: none;
    border-color: transparent;
    box-shadow: none;
}}

.version-badge {{
    position: fixed; bottom: 8px; right: 12px;
    font-size: 11px; color: var(--ink-soft); opacity: 0.6;
}}
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)

st.markdown(f"""
<div style="display:flex; align-items:center; gap:8px;">
  <div>
    <h1 style="margin:0; padding:0;">THinkING</h1>
    <p class="subtitle" style="margin:0; font-size:14px;">Думающие чернила</p>
  </div>
  <img src="data:image/png;base64,{get_ghost_b64(st.session_state.dark_mode)}"
       alt="THinkING" style="height:200px; width:auto; margin-left:16px;">
</div>
""", unsafe_allow_html=True)

st.divider()


# ============================================================
# Основная разметка
# ============================================================

results_state = st.session_state.results
col_input, col_charts = st.columns([1, 3] if results_state else [3, 2], gap="medium")

with col_input:
    if results_state is None:
        uploaded = st.file_uploader(
            "Или загрузите .txt файл", type=["txt"],
            label_visibility="collapsed",
            help=f"Не больше {config.MAX_UPLOAD_MB} МБ")
        # Читаем каждый файл один раз, иначе он бы затирал правки пользователя
        # на каждом rerun.
        if uploaded is not None and uploaded.name != st.session_state.last_upload:
            # Streamlit отсекает файлы больше server.maxUploadSize сам, но
            # только если конфиг долетел до сервера: при запуске из другого
            # каталога .streamlit/config.toml не подхватывается. Дублируем
            # проверку здесь, чтобы лимит не зависел от способа запуска.
            if uploaded.size > config.MAX_UPLOAD_MB * 1024 * 1024:
                st.error(f"Файл {uploaded.name} больше {config.MAX_UPLOAD_MB} МБ "
                         f"({uploaded.size / 1024 / 1024:.1f} МБ)")
                st.session_state.last_upload = uploaded.name
            else:
                decoded = io_utils.decode_text(uploaded.getvalue(), uploaded.name)
                if decoded is None:
                    st.error(f"Не удалось определить кодировку файла {uploaded.name}")
                else:
                    st.session_state.last_upload = uploaded.name
                    st.session_state.input_text = decoded
                    st.rerun()

        st.text_area("Введите текст для анализа:", height=250, key="input_text",
                     placeholder="Вставьте текст на русском или белорусском языке...")

        text_len = len(st.session_state.input_text)
        st.caption(f"Длина текста: {text_len} символов "
                   f"(минимум {config.MIN_TEXT_LENGTH})")

        col_a, col_b = st.columns(2, gap="small")
        with col_a:
            st.button("Анализировать", on_click=request_analysis, type="primary",
                      use_container_width=True)
        with col_b:
            st.button("✕ Очистить", on_click=clear_input,
                      disabled=not st.session_state.input_text.strip(),
                      use_container_width=True)

        if st.session_state.message:
            level, text = st.session_state.message
            getattr(st, level)(text)
    else:
        st.button("🔄 Новый анализ", on_click=reset_analysis, use_container_width=True)

        r = results_state
        if r["degraded"]:
            st.warning(f"Морфология работает в упрощённом режиме: {r['degraded']}")
        if r["lang_warning"]:
            st.warning(r["lang_warning"])

        score = r["best_score"]
        if score >= config.CONFIDENCE_THRESHOLD:
            high = score >= config.HIGH_CONFIDENCE_THRESHOLD
            color = "green" if high else "orange"
            label = "Высокая уверенность" if high else "Средняя уверенность"
            st.markdown(f"<h3 style='color:{color};'>{author_display(r['best_author'])}</h3>",
                        unsafe_allow_html=True)
            st.markdown(f"**{score:.1%}** — {label}")
        else:
            st.markdown("<h3>Автор не определён</h3>", unsafe_allow_html=True)
            st.markdown(
                f"Ни один автор не достиг порога уверенности "
                f"({config.CONFIDENCE_THRESHOLD:.0%}). Лучший результат: "
                f"**{author_display(r['best_author'])}** — {score:.1%}")

        st.divider()
        render_text_stats(r.get("stats"), cur_lang)

        st.divider()
        st.markdown("**Все авторы:**")
        for author, author_score in sorted(r["results"].items(), key=lambda x: -x[1]):
            st.markdown(f"{author_display(author)}: {author_score:.1%}")

with col_charts:
    if results_state and st.session_state.profiles:
        r = results_state
        profiles = st.session_state.profiles
        anon_features = r["anon_features"]
        best_author = r["best_author"]
        feature_names = config.FEATURE_LIST_SHORT

        all_authors_ranges = {
            name: [(f.a, f.b, f.c) for f in profile.features]
            for name, profile in profiles.items()
        }

        tab_summary, tab_features, tab_profiles = st.tabs(
            ["Схожесть", "Вклад признаков", "Профили авторов"])

        with tab_summary:
            chart_col1, chart_col2 = st.columns(2, gap="medium")
            with chart_col1:
                show_chart(
                    StyleRose.plot_authors_comparison(
                        r["results"], title="Схожесть с авторами", figsize=(9, 6.5)),
                    "authors_comparison.png", "dl_comparison")
            with chart_col2:
                try:
                    show_chart(
                        StyleRose.plot_fuzzy_rose(
                            all_authors_ranges, anon_features, feature_names,
                            authors_to_plot=[best_author],
                            title=f"{author_display(best_author)} vs аноним "
                                  f"({r['best_score']:.1%})",
                            figsize=(7, 7)),
                        f"{best_author}_vs_anon.png", "dl_rose_best")
                except Exception as e:
                    st.warning(f"Не удалось построить итоговую розу: {e}")

        with tab_features:
            # similarity_details считались и раньше, но никуда не выводились —
            # а это самая содержательная часть анализа: видно, какие именно
            # признаки дали совпадение, а какие ему противоречат.
            st.caption("Какие признаки дали совпадение с автором, а какие — нет.")
            sims, weights, contribs = r["similarity_details"][best_author]
            try:
                show_chart(
                    StyleRose.plot_feature_importance(
                        best_author, sims, weights, contribs, feature_names,
                        title=f"{author_display(best_author)} "
                              f"(сходство {r['best_score']:.1%})",
                        figsize=(14, 5.5)),
                    f"{best_author}_importance.png", "dl_importance")
            except Exception as e:
                st.warning(f"Не удалось построить график важности признаков: {e}")

        with tab_profiles:
            st.caption("Значения признаков по каждому обучающему тексту автора.")
            for name, profile in profiles.items():
                with st.expander(author_display(name)):
                    html = profile.get_summary_html()
                    if html is None:
                        st.info("Таблица недоступна: профиль обучен старой версией "
                                "кода. Нажмите «Переобучить» в боковой панели.")
                    else:
                        st.iframe(html, height=420)

st.markdown(f"<div class='version-badge'>v{config.VERSION}</div>",
            unsafe_allow_html=True)
