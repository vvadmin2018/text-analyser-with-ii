import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import io

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
st.title("👻 GhostQuill 🪶 — ")

if "profiles" not in st.session_state:
    st.session_state.profiles = None
if "last_lang" not in st.session_state:
    st.session_state.last_lang = None
if "results" not in st.session_state:
    st.session_state.results = None

st.sidebar.header("Настройки")
lang_name = st.sidebar.selectbox("Язык анализа", list(LANG_OPTIONS.keys()))
lang_cfg = LANG_OPTIONS[lang_name]
cur_lang = lang_cfg["lang"]

if st.session_state.last_lang != lang_name:
    st.session_state.profiles = None
    st.session_state.results = None
    st.session_state.last_lang = lang_name

profile_path = lang_cfg["pickle"]


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
        st.caption("Нажмите ▲ чтобы развернуть")

        fig1 = StyleRose.plot_authors_comparison(
            results, title=f"Сравнение уверенности — {lang_name}"
        )
        fig1.set_size_inches(4, 3)
        st.pyplot(fig1, use_container_width=True)
        with st.expander("🔍 Увеличить"):
            fig1b = StyleRose.plot_authors_comparison(
                results, title=f"Сравнение уверенности — {lang_name}"
            )
            fig1b.set_size_inches(8, 5)
            st.pyplot(fig1b, use_container_width=True)
            plt.close(fig1b)
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
        fig2.set_size_inches(4, 3)
        st.pyplot(fig2, use_container_width=True)
        with st.expander("🔍 Увеличить"):
            fig2b = StyleRose.plot_fuzzy_rose(
                single_dict, anon_features, feature_names,
                profiles_dispersion=single_disp,
                anonymous_dispersion=anon_dispersion,
                title=f"{author_display(best_author)} vs аноним ({best_score:.1%})",
                author_colors={best_author: author_color}
            )
            fig2b.set_size_inches(8, 6)
            st.pyplot(fig2b, use_container_width=True)
            plt.close(fig2b)
        plt.close(fig2)
