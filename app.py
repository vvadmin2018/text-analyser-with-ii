import streamlit as st
import nltk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import config
from src.feature_extractor import FeatureExtractor, Language
from src.profile_builder import AuthorProfile
from src.visualizer import StyleRose
from main import build_authors_profiles, save_profiles, load_profiles

LANG_OPTIONS = {
    "Русский": {
        "lang": Language.RUSSIAN,
        "authors": config.RUSSIAN_AUTHORS_LIST,
        "pickle": "authors_profiles_ru.pkl",
    },
    "Беларуская": {
        "lang": Language.BELARUSIAN,
        "authors": config.BELARUSIAN_AUTHORS_LIST,
        "pickle": "authors_profiles_be.pkl",
    },
}

st.set_page_config(page_title="GhostQuill", layout="wide")

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "profiles" not in st.session_state:
    st.session_state.profiles = None
if "last_lang" not in st.session_state:
    st.session_state.last_lang = None
if "results" not in st.session_state:
    st.session_state.results = None

st.sidebar.header("Настройки")
st.session_state.dark_mode = st.sidebar.checkbox("🌙 Тёмная тема", value=st.session_state.dark_mode)
lang_name = st.sidebar.selectbox("Язык анализа", list(LANG_OPTIONS.keys()))
lang_cfg = LANG_OPTIONS[lang_name]
cur_lang = lang_cfg["lang"]

if st.session_state.last_lang != lang_name:
    st.session_state.profiles = None
    st.session_state.results = None
    st.session_state.last_lang = lang_name

profile_path = lang_cfg["pickle"]

DARK_EXTRA = ("""
<style>
.stApp { background-color: #0e1117; }
.stApp > header { background-color: #0e1117 !important; }
section[data-testid="stSidebar"] {
    background-color: #1e2028 !important;
    color: #fafafa !important;
}
.stTextArea textarea, .stSelectbox > div, .stSelectbox [data-baseweb="select"] span {
    background-color: #262730 !important; color: #fafafa !important;
}
.stTextArea label, .stSelectbox label, label, .stCheckbox label {
    color: #fafafa !important;
}
.stSelectbox [data-baseweb="select"] svg { fill: #fafafa !important; }
.stSelectbox [data-baseweb="popover"] { background-color: #1e2028 !important; }
.stSelectbox [data-baseweb="popover"] li { color: #fafafa !important; }
.stSelectbox [data-baseweb="popover"] li:hover { background-color: #363840 !important; }
.stButton button {
    background-color: #262730; color: #fafafa !important; border-color: #555;
}
.stButton button:hover { background-color: #363840; border-color: #777; }
.stAlert { background-color: #262730; color: #fafafa; border-color: #555; }
.stTextArea textarea::placeholder { color: #aaa !important; }
[data-testid="stMainMenu"], [data-testid="stMainMenu"] * { color: #fafafa !important; }
[data-testid="stMainMenu"] svg { fill: #fafafa !important; }
[data-testid="stToolbar"] button, [data-testid="stToolbar"] button * { color: #fafafa !important; }
[data-testid="stToolbar"] button svg { fill: #fafafa !important; }
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] .st-emotion-cache-10trblm,
section[data-testid="stSidebar"] .st-emotion-cache-1wmy9hl {
    color: #fafafa !important;
}
</style>
""" if st.session_state.dark_mode else "")

st.markdown(f"""
<style>
@keyframes fly {{
    0% {{ transform: translateX(0px) translateY(0px); }}
    25% {{ transform: translateX(50px) translateY(-12px); }}
    50% {{ transform: translateX(0px) translateY(0px); }}
    75% {{ transform: translateX(-50px) translateY(-12px); }}
    100% {{ transform: translateX(0px) translateY(0px); }}
}}
.ghost-fly {{
    animation: fly 5s ease-in-out infinite;
    font-size: 64px;
    text-align: center;
    display: block;
    line-height: 1;
    cursor: default;
    margin-bottom: -30px;
    margin-top: -20px;
}}
h1 {{
    margin-top: -24px !important;
    padding-top: 0 !important;
}}
</style>
{DARK_EXTRA}
<div class="ghost-fly">👻</div>
""", unsafe_allow_html=True)

st.title("🪶 GhostQuill")


def compute_plot_data(profiles, anon_features):
    authors_for_plot = {}
    authors_dispersion = {}
    for name, profile in profiles.items():
        typical = [f.b for f in profile.features]
        dispersion = [(f.c - f.a) / 4 for f in profile.features]
        authors_for_plot[name] = typical
        authors_dispersion[name] = dispersion
    anon_dispersion = []
    for i in range(len(anon_features)):
        vals = [authors_dispersion.get(name, [0] * len(anon_features))[i]
                for name in authors_dispersion]
        anon_dispersion.append(np.mean(vals) if vals else 0)
    return authors_for_plot, authors_dispersion, anon_dispersion


def author_display(name):
    return config.AUTHOR_LABELS.get(name, name)


st.sidebar.markdown("---")
st.sidebar.subheader("Профили авторов")

if st.session_state.profiles is None:
    profiles = load_profiles(profile_path)
    if profiles is not None:
        st.session_state.profiles = profiles
        st.sidebar.success("Загружены профили:")
        for name in profiles:
            st.sidebar.markdown(f"- {author_display(name)}")
    else:
        st.sidebar.warning("Профили не найдены")
        if st.sidebar.button("Обучить профили"):
            with st.spinner("Обучение..."):
                config.AUTHORS_LIST = lang_cfg["authors"]
                authors_data = build_authors_profiles()
                if not authors_data:
                    st.error("Нет текстов для обучения. Добавьте файлы .txt в texts/")
                else:
                    new_profiles = {}
                    for author_name, texts in authors_data.items():
                        profile = AuthorProfile(author_name)
                        profile.build_from_texts(texts, language=cur_lang)
                        new_profiles[author_name] = profile
                    save_profiles(new_profiles, profile_path)
                    st.session_state.profiles = new_profiles
                    st.rerun()
else:
    profiles = st.session_state.profiles
    st.sidebar.success("Загружены профили:")
    for name in profiles:
        st.sidebar.markdown(f"- {author_display(name)}")
    if st.sidebar.button("Переобучить"):
        st.session_state.profiles = None
        st.session_state.results = None
        if os.path.exists(profile_path):
            os.remove(profile_path)
        st.rerun()

st.markdown("---")
col_input, col_charts = st.columns([3, 2], gap="medium")

with col_input:
    user_text = st.text_area("Введите текст для анализа:", height=250,
                             placeholder="Вставьте текст на русском или белорусском языке...")

    if st.button("🔍 Анализировать", type="primary"):
        if st.session_state.profiles is None:
            st.error("Сначала обучите профили авторов.")
        elif not user_text.strip():
            st.warning("Введите текст для анализа.")
        elif len(user_text.strip()) < 10:
            st.warning("Текст слишком короткий (минимум 10 символов).")
        else:
            profiles = st.session_state.profiles
            with st.spinner("Анализ..."):
                extractor = FeatureExtractor(language=cur_lang)
                anon_features = extractor.extract(user_text)

                results = {}
                similarity_details = {}
                for author_name, profile in profiles.items():
                    sim, details = profile.similarity_with_details(anon_features)
                    results[author_name] = sim
                    similarity_details[author_name] = details

                best_author = max(results, key=results.get)
                best_score = results[best_author]

            st.session_state.results = {
                "best_author": best_author,
                "best_score": best_score,
                "results": results,
                "anon_features": anon_features,
                "similarity_details": similarity_details,
                "profiles": profiles,
            }
            st.rerun()

    if st.session_state.results:
        r = st.session_state.results
        st.markdown("---")
        score_pct = r["best_score"] * 100
        if r["best_score"] >= 0.7:
            color = "green"
            label = "Высокая уверенность"
        elif r["best_score"] >= 0.5:
            color = "orange"
            label = "Средняя уверенность"
        else:
            color = "red"
            label = "Низкая уверенность"
        st.markdown(
            f"<h3 style='color:{color};'>{author_display(r['best_author'])}</h3>",
            unsafe_allow_html=True
        )
        st.markdown(f"**{score_pct:.1f}%** — {label}")
        st.markdown("---")
        st.markdown("**Все авторы:**")
        for author, score in sorted(r["results"].items(), key=lambda x: -x[1]):
            marker = "✅" if author == r["best_author"] else ""
            st.markdown(f"{author_display(author)}: {score:.1%} {marker}")

with col_charts:
    if st.session_state.results and st.session_state.profiles:
        r = st.session_state.results
        profiles = r["profiles"]
        anon_features = r["anon_features"]
        results = r["results"]
        best_author = r["best_author"]
        best_score = r["best_score"]

        authors_for_plot, authors_dispersion, anon_dispersion = \
            compute_plot_data(profiles, anon_features)
        feature_names = config.FEATURE_LIST_SHORT
        author_color = config.AUTHOR_COLORS.get(best_author, config.AUTHOR_COLORS['default'])

        st.subheader("📊 Графики")

        fig1 = StyleRose.plot_authors_comparison(
            results, title=f"Сравнение уверенности — {lang_name}"
        )
        fig1.set_size_inches(5, 3)
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)

        single_dict = {best_author: authors_for_plot[best_author]}
        single_disp = {best_author: authors_dispersion[best_author]}
        fig2 = StyleRose.plot_fuzzy_rose(
            single_dict, anon_features, feature_names,
            profiles_dispersion=single_disp,
            anonymous_dispersion=anon_dispersion,
            title=f"{author_display(best_author)} vs аноним ({best_score:.1%})",
            author_colors={best_author: author_color}
        )
        fig2.set_size_inches(5, 3.5)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)
