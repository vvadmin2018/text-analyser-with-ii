# text-analyser-with-ii

Authorship attribution for Russian and Belarusian literary texts using fuzzy
logic (triangular membership functions over 17 stylometric features).

## Quick start

```bash
pip install -r requirements-dev.txt
python -c "import nltk; [nltk.download(p) for p in ('punkt','punkt_tab','stopwords')]"

streamlit run app.py              # UI: language, text input/upload, charts
python main.py                    # full pipeline: train + analyze texts/anonim/ + PNGs
python scripts/batch_check.py     # batch summary over texts/anonim/
pytest                            # tests
```

## Architecture

- `src/` — core package:
  - `config.py` — authors, weights, thresholds, palette, `configure_logging()`
  - `feature_extractor.py` — the 17 features; pymorphy3 (ru) / stanza (be)
  - `profile_builder.py` — `TriangularMembership`, `AuthorProfile`
  - `identifier.py` — `identify(profiles, features)`, shared by CLI and UI
  - `visualizer.py` — charts; `use_theme(dark=…)` switches light/dark palette
  - `report.py` — per-author feature tables (HTML/TXT)
  - `io_utils.py` — encoding-cascade text loading
- `main.py` — offline pipeline (train → identify → visualize)
- `app.py` — Streamlit UI
- `scripts/batch_check.py` — batch check with argparse
- `tests/` — pytest suite
- `texts/{author}/` — training texts; `texts/anonim/` — texts to identify
- `output/<YYYY-MM-DD-HH-MM>/` — timestamped output from `main.py` (gitignored)

## Gotchas

- **Pickle cache.** Profiles go to `authors_profiles.pkl` (CLI) and
  `authors_profiles_ru.pkl` / `authors_profiles_be.pkl` (UI), all gitignored.
  `load_profiles` returns `None` if *any* profile has the wrong feature count —
  it is all-or-nothing on purpose, so analysis never silently runs against a
  partial author list. Delete the file or use "Переобучить" to retrain.
- **Belarusian needs `stanza`**, which is commented out in `requirements.txt`.
  Without it `FeatureExtractor` falls back to a suffix heuristic and sets
  `degraded_reason`; the UI shows that as a warning. Do not let this path go
  silent — it used to return all-zero morphology features.
- **`config.AUTHORS_LIST` is a default, not a channel.** Pass authors explicitly
  (`build_authors_profiles(authors=…)`); the UI must not mutate the module
  global, since one Streamlit process serves both languages.
- **Thresholds live in `config`** (`CONFIDENCE_THRESHOLD`,
  `HIGH_CONFIDENCE_THRESHOLD`, `MIN_TEXT_LENGTH`). Don't hardcode 0.5/0.6/0.7.
- **Logging, not `print`.** Modules use `logging.getLogger(__name__)`; entry
  points call `config.configure_logging()`. `feature_extractor.extract()` runs
  in a hot loop — keep diagnostics at `DEBUG`.
- **The web app writes nothing to disk.** Charts are streamed to the browser and
  offered via `st.download_button`; profile training in the UI passes
  `save_report=False`. Only `main.py` writes to `output/`.
- **Encoding fallback.** `utf-8 → cp1251 → koi8-r → latin-1`, via
  `src/io_utils.py` (used by both training and the UI's file uploader).
- **Comments and UI strings are in Russian.** Match the surrounding style.
- **Streamlit dark mode is CSS.** Streamlit cannot swap `[theme]` at runtime, so
  `app.py` defines both palettes as CSS variables. Target stable
  `data-testid`/`kind` selectors only — never `.st-emotion-cache-*` hashes,
  which change between Streamlit releases.

## Corpus

`texts/` holds `bulichev` (10), `drugkov` (7), `saharnov` (7) for Russian and
`kolas` (7), `maur` (14), `bryl` (7) for Belarusian, plus `anonim` (4) for
verification. `pushkin` (9), `lermontov` (12) and `tolstoy` (10) are present but
not in the active author lists.
