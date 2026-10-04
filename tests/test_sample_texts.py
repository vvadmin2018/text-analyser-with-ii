# -*- coding: utf-8 -*-
"""Ссылки «тестового запуска»: случайный текст каждого автора в форму ввода."""
import os
import random

import pytest

from src import config, io_utils

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXTS = os.path.join(ROOT, "texts")


# ---------- io_utils.pick_sample_texts ----------

@pytest.fixture
def corpus(tmp_path):
    """Мини-корпус: два автора, у одного .txt вперемешку с мусором."""
    (tmp_path / "alpha").mkdir()
    for name in ("a1.txt", "a2.txt", "a3.txt", "notes.md", "a.pkl"):
        (tmp_path / "alpha" / name).write_text("текст", encoding="utf-8")
    (tmp_path / "beta").mkdir()
    (tmp_path / "beta" / "b1.txt").write_text("текст", encoding="utf-8")
    (tmp_path / "anonim").mkdir()
    (tmp_path / "anonim" / "x.txt").write_text("текст", encoding="utf-8")
    (tmp_path / "empty").mkdir()
    return str(tmp_path)


def test_one_text_per_author_from_own_folder(corpus):
    picked = io_utils.pick_sample_texts(["alpha", "beta"], base_path=corpus)

    assert list(picked) == ["alpha", "beta"]
    assert os.path.dirname(picked["alpha"]) == os.path.join(corpus, "alpha")
    assert os.path.dirname(picked["beta"]) == os.path.join(corpus, "beta")
    assert picked["beta"].endswith("b1.txt")


def test_only_txt_files_are_candidates(corpus):
    for seed in range(30):
        picked = io_utils.pick_sample_texts(
            ["alpha"], base_path=corpus, rng=random.Random(seed))
        assert picked["alpha"].endswith(".txt")


def test_choice_is_actually_random(corpus):
    """Не «всегда первый файл»: на разных зёрнах выпадают разные."""
    chosen = {
        os.path.basename(io_utils.pick_sample_texts(
            ["alpha"], base_path=corpus, rng=random.Random(seed))["alpha"])
        for seed in range(40)
    }
    assert chosen == {"a1.txt", "a2.txt", "a3.txt"}


def test_same_seed_gives_same_pick(corpus):
    first = io_utils.pick_sample_texts(["alpha"], base_path=corpus, rng=random.Random(7))
    second = io_utils.pick_sample_texts(["alpha"], base_path=corpus, rng=random.Random(7))
    assert first == second


def test_anonim_is_never_picked_unless_asked(corpus):
    """Слепой набор texts/anonim/ — не для тестовых запусков."""
    for seed in range(20):
        picked = io_utils.pick_sample_texts(
            ["alpha", "beta"], base_path=corpus, rng=random.Random(seed))
        # Путь считаем от корня корпуса: tmp_path и сам может содержать слово
        # из имени теста.
        assert all(os.path.relpath(path, corpus).split(os.sep)[0] != "anonim"
                   for path in picked.values())


def test_authors_without_texts_are_skipped(corpus):
    picked = io_utils.pick_sample_texts(
        ["alpha", "empty", "missing"], base_path=corpus)
    assert list(picked) == ["alpha"]


def test_no_authors_no_picks(corpus):
    assert io_utils.pick_sample_texts([], base_path=corpus) == {}


# ---------- конфиг ----------

@pytest.mark.parametrize("lang", sorted(config.AUTHORS_BY_LANGUAGE))
def test_every_active_author_has_a_link_label(lang):
    """Добавили автора в список — не забыли подпись, а не получили slug."""
    for author in config.AUTHORS_BY_LANGUAGE[lang]:
        assert config.SAMPLE_LINK_LABELS.get(author), author


def test_every_active_author_has_texts_on_disk():
    for authors in config.AUTHORS_BY_LANGUAGE.values():
        picked = io_utils.pick_sample_texts(authors, base_path=TEXTS)
        assert sorted(picked) == sorted(authors)


# ---------- веб-приложение ----------

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

RU_LABELS = {
    "saharnov": "«История корабля» С. Сахарнова",
    "drugkov": "«Приключения Карандаша и Самоделкина» Ю. Дружкова",
    "bulichev": "«Приключения Алисы» К. Булычёва",
}
BE_LABELS = {
    "baravikova": "Р. Баравікова",
    "maur": "Я. Маўр",
    "misko": "П. Місько",
}


@pytest.fixture
def app(monkeypatch):
    # AppTest гоняет скрипт в этом же процессе, а относительный texts/ из
    # config.BASE_PATH зависит от cwd — как и при обычном запуске.
    monkeypatch.chdir(ROOT)
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=60)
    at.run()
    assert not at.exception
    return at


def switch_language(at, name):
    at.sidebar.selectbox(key="lang_name").select(name).run()
    assert not at.exception


def link_buttons(at):
    return {b.key: b for b in at.button if b.key and b.key.startswith("sample_")}


def corpus_texts(author):
    return {
        io_utils.read_text_file(p).replace("\r\n", "\n")
        for p in io_utils.list_txt_files(os.path.join(TEXTS, author))
    }


def test_russian_links_use_requested_titles(app):
    links = link_buttons(app)
    assert {k: b.label for k, b in links.items()} == {
        f"sample_{a}": label for a, label in RU_LABELS.items()}


def test_belarusian_links_use_author_names(app):
    switch_language(app, "Белорусский")
    links = link_buttons(app)
    assert {k: b.label for k, b in links.items()} == {
        f"sample_{a}": label for a, label in BE_LABELS.items()}


def test_links_are_tertiary_buttons(app):
    """Именно ссылки, а не карточки-кнопки: стиль привязан к tertiary."""
    for button in link_buttons(app).values():
        assert button.proto.type == "tertiary"


def test_no_anonim_among_links(app):
    for lang in ("Русский", "Белорусский"):
        switch_language(app, lang)
        for key in link_buttons(app):
            assert "anonim" not in key


@pytest.mark.parametrize("lang,labels", [("Русский", RU_LABELS),
                                         ("Белорусский", BE_LABELS)])
def test_click_loads_author_text_into_form(app, lang, labels):
    """Клик кладёт в поле текст именно этого автора; анализ не стартует."""
    switch_language(app, lang)
    for author in labels:
        link_buttons(app)[f"sample_{author}"].click().run()
        assert not app.exception

        loaded = app.text_area(key="input_text").value.replace("\r\n", "\n")
        assert loaded in corpus_texts(author), author
        # Анализ запускает только кнопка «Анализировать».
        assert app.session_state.results is None
        assert not app.session_state.analyze_requested


def test_click_with_first_file_is_exact(app, monkeypatch):
    """Фиксируем выбор — и проверяем, что в форме байт в байт файл с диска."""
    monkeypatch.setattr(random, "choice", lambda seq: seq[0])
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=60)
    at.run()

    first = io_utils.list_txt_files(os.path.join(config.BASE_PATH, "saharnov"))[0]
    at.button(key="sample_saharnov").click().run()

    assert at.text_area(key="input_text").value == io_utils.read_text_file(first)


def test_pick_is_stable_across_reruns(app):
    """Ссылка не должна менять текст при каждом нажатии чего-то на странице."""
    first = dict(app.session_state.sample_paths["Русский"])
    app.run()
    app.run()
    assert app.session_state.sample_paths["Русский"] == first


def test_picks_are_kept_per_language(app):
    ru = dict(app.session_state.sample_paths["Русский"])
    switch_language(app, "Белорусский")
    switch_language(app, "Русский")
    assert app.session_state.sample_paths["Русский"] == ru
    assert set(app.session_state.sample_paths["Белорусский"]) == set(BE_LABELS)


def test_loaded_sample_is_announced_as_non_blind(app):
    link_buttons(app)["sample_bulichev"].click().run()
    captions = " ".join(c.value for c in app.caption)
    assert "Проверка не слепая" in captions
    assert ".txt" in captions


def test_announcement_disappears_after_edit(app):
    link_buttons(app)["sample_bulichev"].click().run()
    app.text_area(key="input_text").set_value("совсем другой текст").run()
    assert "Проверка не слепая" not in " ".join(c.value for c in app.caption)


def test_language_switch_clears_loaded_sample(app):
    link_buttons(app)["sample_saharnov"].click().run()
    switch_language(app, "Белорусский")
    assert app.text_area(key="input_text").value == ""
    assert "Проверка не слепая" not in " ".join(c.value for c in app.caption)


def test_links_sit_between_uploader_and_text_area():
    """Порядок на странице: загрузка файла → ссылки → поле ввода."""
    source = open(os.path.join(ROOT, "app.py"), encoding="utf-8").read()
    uploader = source.index("st.file_uploader(")
    links = source.index("        render_sample_links()")
    text_area = source.index("st.text_area(")
    assert uploader < links < text_area


def test_unreadable_sample_shows_error_not_crash(app, tmp_path):
    """Файл исчез между выбором и кликом — сообщение, а не трассировка."""
    gone = str(tmp_path / "gone.txt")
    app.session_state.sample_paths["Русский"]["saharnov"] = gone
    app.run()
    link_buttons(app)["sample_saharnov"].click().run()

    assert not app.exception
    assert any("gone.txt" in e.value for e in app.error)
    assert app.text_area(key="input_text").value == ""
