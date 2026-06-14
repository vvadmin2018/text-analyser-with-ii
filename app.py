import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

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

st.set_page_config(page_title="Нечёткий детектив", layout="wide")
st.title("🕵️ Нечёткий детектив — Определение авторства текста")

if "profiles" not in st.session_state:
    st.session_state.profiles = None
if "last_lang" not in st.session_state:
    st.session_state.last_lang = None

st.sidebar.header("Настройки")
lang_name = st.sidebar.selectbox("Язык анализа", list(LANG_OPTIONS.keys()))
lang_cfg = LANG_OPTIONS[lang_name]
cur_lang = lang_cfg["lang"]

if st.session_state.last_lang != lang_name:
    st.session_state.profiles = None
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
        st.sidebar.success(f"Загружены профили ({len(profiles)} авторов)")
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
    st.sidebar.success(f"Профили готовы ({len(st.session_state.profiles)} авторов)")
    for name in st.session_state.profiles:
        st.sidebar.markdown(f"- {author_display(name)}")
    if st.sidebar.button("Переобучить"):
        st.session_state.profiles = None
        if os.path.exists(profile_path):
            os.remove(profile_path)
        st.rerun()

st.markdown("---")
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

        st.markdown("---")
        st.header("Результаты")
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Наиболее вероятный автор")
            score_pct = best_score * 100
            if best_score >= 0.7:
                color = "green"
                label = "Высокая уверенность"
            elif best_score >= 0.5:
                color = "orange"
                label = "Средняя уверенность"
            else:
                color = "red"
                label = "Низкая уверенность"

            st.markdown(
                f"<h2 style='color:{color};'>{author_display(best_author)}</h2>",
                unsafe_allow_html=True
            )
            st.markdown(f"**Уверенность:** {score_pct:.1f}%")
            st.markdown(f"**{label}**")

            st.markdown("---")
            st.subheader("Все авторы")
            for author, score in sorted(results.items(), key=lambda x: -x[1]):
                marker = "✅" if author == best_author else ""
                st.markdown(f"{author_display(author)}: {score:.1%} {marker}")

        with col2:
            with st.spinner("Построение графиков..."):
                authors_for_plot, authors_dispersion, anon_dispersion = \
                    compute_plot_data(profiles, anon_features)
                feature_names = config.FEATURE_LIST_SHORT

                fig1 = StyleRose.plot_authors_comparison(
                    results,
                    title=f"Сравнение уверенности — {lang_name}"
                )
                st.pyplot(fig1)
                plt.close(fig1)

                author_color = config.AUTHOR_COLORS.get(best_author, config.AUTHOR_COLORS['default'])
                single_dict = {best_author: authors_for_plot[best_author]}
                single_disp = {best_author: authors_dispersion[best_author]}

                fig2 = StyleRose.plot_fuzzy_rose(
                    single_dict,
                    anon_features,
                    feature_names,
                    profiles_dispersion=single_disp,
                    anonymous_dispersion=anon_dispersion,
                    title=f"Сравнение стилей: {author_display(best_author)} vs аноним\n(уверенность {best_score:.1%})",
                    author_colors={best_author: author_color}
                )
                st.pyplot(fig2)
                plt.close(fig2)
