# app.py — веб-интерфейс THinkING (Streamlit)
"""Веб-приложение: выбор языка, ввод текста, анализ авторства и графики."""
import base64
import io
import logging
import os
import sys
from html import escape as html_escape

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
logger = logging.getLogger(__name__)

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


def text_size_mb(text):
    """Размер текста в мегабайтах, по байтам UTF-8.

    Не по символам: кириллица в UTF-8 занимает два байта на букву, поэтому
    счёт по символам показал бы вдвое меньше, чем текст весит на самом деле,
    и «3 МБ» в интерфейсе означали бы 6 МБ.
    """
    return len((text or "").encode("utf-8")) / (1024 * 1024)


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
    return ("Похоже, текст на белорусском языке. Для корректного анализа необходимо переключить язык.")


def describe_text(extractor, text):
    """Статистика текста, которая не имеет права уронить анализ.

    Блок «О тексте» — необязательное дополнение к результату, поэтому любая
    ошибка здесь стоит показанной панели, а не всего разбора.

    Отдельно ловится AttributeError: get_extractor обёрнут в
    @st.cache_resource, и при обновлении кода без перезапуска процесса (так
    иногда деплоит Streamlit Cloud) в кэше остаётся объект, созданный прежней
    версией класса — у него describe ещё нет, хотя app.py уже новый. Ровно на
    этом приложение и падало: трассировка обрывалась на самом вызове, без
    единого кадра внутри feature_extractor.
    """
    describe = getattr(extractor, "describe", None)
    if describe is None:
        logger.warning("У анализатора нет метода describe — вероятно, в кэше "
                       "остался объект от прежней версии кода")
        return None
    try:
        return describe(text)
    except Exception as e:
        logger.warning("Не удалось собрать статистику текста: %s", e)
        return None


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

    Вёрстка своя, а не st.columns + st.metric. Причина в том, что Streamlit
    решает про перенос колонок сам: узкой колонке он ставит
    min-width: calc(100% - 24px), и шесть метрик на телефоне разъезжались на
    шесть строк вместо двух рядов по три. Заставить их встать в ряд можно, но
    тогда st.metric режет длинные значения многоточием — «58 939»
    превращалось в «8…». Своя сетка позволяет переносить значение на вторую
    строку вместо обрезки и не зависит от внутренних data-testid Streamlit.
    """
    if not stats:
        return

    detected = stats.get("language")
    if detected is None:
        language_line = "не определён"
    else:
        language_line = LANGUAGE_NAMES.get(detected, detected)
        if detected != selected_language:
            language_line += " ⚠️"

    def number(value):
        # Неразрывный пробел как разделитель разрядов: иначе «58 939» может
        # переломиться пополам в узкой ячейке.
        return f"{value:,}".replace(",", "\u00A0")

    cells = [
        ("Символов", number(stats["chars"])),
        ("Слов", number(stats["words"])),
        ("Предложений", number(stats["sentences"])),
        ("Абзацев", number(stats["paragraphs"])),
        ("Без пробелов", number(stats["chars_no_spaces"])),
        ("Язык", language_line),
    ]
    items = "".join(
        f'<div class="text-stats__cell">'
        f'<span class="text-stats__label">{html_escape(label)}</span>'
        f'<span class="text-stats__value">{html_escape(value)}</span>'
        f'</div>'
        for label, value in cells
    )

    st.markdown(
        f'<div class="text-stats"><p class="text-stats__title">О тексте:</p>'
        f'<div class="text-stats__grid">{items}</div></div>',
        unsafe_allow_html=True)

    st.caption("Слова — без знаков препинания и чисел. "
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
    # {название языка: {автор: путь}} — случайные тексты для ссылок «тестового
    # запуска». Выбираются один раз за сессию: если бы выбор шёл на каждый
    # rerun, ссылка меняла бы текст под рукой у пользователя.
    "sample_paths": {},
    # Какой тестовый текст сейчас лежит в форме (для подписи под полем).
    "sample_loaded": None,
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


def sample_paths_for(lang_name, authors):
    """Случайные тексты корпуса для выбранного языка: по одному на автора."""
    cache = st.session_state.sample_paths
    if lang_name not in cache:
        cache[lang_name] = io_utils.pick_sample_texts(
            authors, base_path=os.path.join(APP_DIR, config.BASE_PATH))
    return cache[lang_name]


def load_sample(path, author):
    """Кладёт тестовый текст в форму; дальше всё как при ручном вводе.

    Анализ здесь не запускается: пользователь видит текст в поле, может его
    поправить и сам нажимает «Анализировать».
    """
    name = os.path.basename(path)
    try:
        text = io_utils.read_text_file(path)
    except OSError as e:
        st.session_state.message = ("error", f"Не удалось открыть {name}: {e}")
        return
    if text is None:
        st.session_state.message = (
            "error", f"Не удалось определить кодировку файла {name}")
        return
    if not text.strip():
        st.session_state.message = ("error", f"Файл {name} пустой")
        return

    st.session_state.input_text = text
    st.session_state.message = None
    st.session_state.sample_loaded = {"author": author, "file": name, "text": text}


def same_text(a, b):
    """Совпадают ли тексты с точностью до перевода строки.

    Файлы корпуса бывают с CRLF, а текстовое поле в браузере склеивает их
    в LF — без этого подпись о тестовом тексте пропадала бы сама по себе.
    """
    return (a or "").replace("\r\n", "\n") == (b or "").replace("\r\n", "\n")


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


def render_sample_links():
    """Ссылки «тестового запуска»: по одной на автора выбранного языка.

    Streamlit не умеет ссылку, которая вызывает Python, поэтому это кнопки
    type="tertiary" (без рамки и фона), оформленные как ссылки в APP_CSS по
    ключу контейнера. Ключ, а не data-testid кнопки: он стабилен и
    не затрагивает остальные кнопки страницы.
    """
    paths = sample_paths_for(lang_name, lang_cfg["authors"])
    if not paths:
        return

    # Порядок ссылок — как в config.SAMPLE_LINK_LABELS; авторы без подписи
    # (добавленные в корпус позже) идут в конце.
    order = list(config.SAMPLE_LINK_LABELS)
    ordered = sorted(paths, key=lambda a: order.index(a) if a in order else len(order))

    st.caption("Тестовый запуск — случайный текст автора из корпуса:")
    with st.container(key="sample-links"):
        for author in ordered:
            path = paths[author]
            st.button(config.SAMPLE_LINK_LABELS.get(author, author_display(author)),
                      key=f"sample_{author}", type="tertiary",
                      on_click=load_sample, args=(path, author))


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
    size_mb = text_size_mb(text)
    if size_mb > config.MAX_TEXT_MB:
        st.session_state.message = ("error", (
            f"Текст слишком большой: {size_mb:.1f} МБ при пределе "
            f"{config.MAX_TEXT_MB} МБ. Разберите его частями."))
        return

    with st.spinner("Анализ..."):
        extractor = get_extractor(cur_lang)
        try:
            anon_features = extractor.extract(text)
        except Exception as e:
            st.session_state.message = ("error", f"Не удалось разобрать текст: {e}")
            return

        best_author, results, similarity_details = identify(profiles, anon_features)
        stats = describe_text(extractor, text)

    st.session_state.message = None
    st.session_state.results = {
        "best_author": best_author,
        "best_score": results[best_author],
        "results": results,
        "anon_features": anon_features,
        "similarity_details": similarity_details,
        "degraded": extractor.degraded_reason,
        "lang_warning": language_warning(text, cur_lang),
        "stats": stats,
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
    --link: #8A5A00;
    --link-hover: #5E3D00;
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
    --link: #E0B54A;
    --link-hover: #F0CB72;
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

/* ===== Ссылки «тестового запуска» =====
   Кнопки type="tertiary", похожие на ссылки. Селектор .stButton нужен, чтобы
   перебить общее правило .stButton button выше (фон и рамка карточки). */
.st-key-sample-links {{ gap: 0.15rem; margin-bottom: 0.75rem; }}
.st-key-sample-links .stButton button {{
    background-color: transparent;
    border: none;
    padding: 2px 0;
    min-height: 0;
    height: auto;
    justify-content: flex-start;
    text-align: left;
}}
.st-key-sample-links .stButton button p {{
    color: var(--link);
    text-decoration: underline;
    text-underline-offset: 3px;
    text-align: left;
}}
.st-key-sample-links .stButton button:hover p,
.st-key-sample-links .stButton button:focus-visible p {{
    color: var(--link-hover);
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

/* ===== Блок «О тексте» =====
   Своя сетка вместо st.columns: Streamlit складывает узкие колонки в столбик
   (min-width: calc(100% - 24px) при flex-wrap: wrap), из-за чего шесть метрик
   на телефоне занимали шесть строк. Здесь три колонки заданы жёстко на всех
   ширинах, а длинное значение переносится на вторую строку — st.metric в этом
   случае обрезал бы его многоточием. */
.text-stats {{ margin: 0 0 4px; }}
.text-stats__title {{
    font-weight: 700;
    margin: 0 0 10px;
    color: var(--ink);
}}
.text-stats__grid {{
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px 10px;
}}
.text-stats__cell {{ display: flex; flex-direction: column; min-width: 0; }}
.text-stats__label {{
    font-size: 0.78rem;
    line-height: 1.2;
    color: var(--ink-soft);
    margin-bottom: 2px;
    /* Две строки резервируются всегда: в узкой ячейке «Предложений» и «Без
       пробелов» переносятся, а «Слов» нет, и значения вставали на разной
       высоте — ряд переставал читаться как ряд. */
    min-height: 2.4em;
}}
.text-stats__value {{
    font-size: 1.15rem;
    line-height: 1.25;
    font-weight: 600;
    color: var(--ink);
    overflow-wrap: break-word;
}}
@media (max-width: 640px) {{
    .text-stats__label {{ font-size: 0.72rem; }}
    .text-stats__value {{ font-size: 1.05rem; }}
}}

.version-badge {{
    position: fixed; bottom: 8px; right: 12px;
    font-size: 11px; color: var(--ink-soft); opacity: 0.6;
}}

/* ===== Шапка =====
   Название — одно слово, но Streamlit задаёт overflow-wrap:break-word, из-за
   чего браузер вправе разорвать его посреди букв. Флекс-элемент с текстом
   сжимался под давлением картинки высотой 200px, и на телефоне «G»
   перепрыгивала на отдельную строку. Лечится тремя вещами сразу: запретом
   переноса внутри слова, запретом сжатия текстового блока ниже содержимого и
   уменьшением картинки на узком экране. */
.app-header {{
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
}}
.app-header__text {{
    /* 0 1 auto, а не 1 1 auto: с flex-grow блок растягивался на всю свободную
       ширину, и картинка отъезжала к правому краю вместо того, чтобы стоять
       вплотную к названию. До перехода на классы здесь был <div> без стилей,
       то есть ровно это поведение. */
    flex: 0 1 auto;
    min-width: 0;
}}
.app-header h1 {{
    margin: 0;
    padding: 0;
    white-space: nowrap;
    overflow-wrap: normal;
    word-break: keep-all;
    hyphens: none;
    font-size: clamp(1.9rem, 9vw, 2.75rem);
}}
.app-header__ghost {{
    height: 200px;
    width: auto;
    max-width: 40%;
    object-fit: contain;
    margin-left: 16px;
    flex: 0 1 auto;
}}
@media (max-width: 640px) {{
    .app-header__ghost {{
        height: 96px;
        margin-left: 8px;
    }}
}}
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)

st.markdown(f"""
<div class="app-header">
  <div class="app-header__text">
    <h1>THinkING</h1>
    <p class="subtitle" style="margin:0; font-size:14px;">Думающие чернила</p>
  </div>
  <img src="data:image/png;base64,{get_ghost_b64(st.session_state.dark_mode)}"
       alt="THinkING" class="app-header__ghost">
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

        render_sample_links()

        st.text_area("Введите текст для анализа:", height=250, key="input_text",
                     placeholder="Вставьте текст на русском или белорусском языке...")

        text_len = len(st.session_state.input_text)
        size_mb = text_size_mb(st.session_state.input_text)
        counter = (f"Длина текста: {text_len} символов "
                   f"(минимум {config.MIN_TEXT_LENGTH})")
        # Мегабайты показываются, только когда до предела уже недалеко: на
        # обычном тексте это лишний шум, а вот упереться в лимит на кнопке
        # «Анализировать», ничего о нём не зная, — неприятный сюрприз.
        if size_mb >= config.MAX_TEXT_MB / 10:
            counter += f" · {size_mb:.1f} из {config.MAX_TEXT_MB} МБ"
        if size_mb > config.MAX_TEXT_MB:
            st.caption(counter)
            st.error(f"Текст больше {config.MAX_TEXT_MB} МБ — анализ не запустится.")
        else:
            st.caption(counter)

        loaded = st.session_state.sample_loaded
        if loaded and same_text(loaded["text"], st.session_state.input_text):
            st.caption(
                f"Тестовый текст")

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
