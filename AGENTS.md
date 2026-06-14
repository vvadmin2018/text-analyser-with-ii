# text-analyser-with-ii

Russian literary authorship attribution tool using fuzzy logic (triangular membership functions).

## Quick start

```bash
python main.py                    # full pipeline: train profiles + analyze all anon texts
streamlit run app.py              # UI: language selection, input, analysis + charts
python test_all_anonim.py         # batch test (has broken imports, see below)
```

## Architecture

- `src/` — core package: `config.py`, `feature_extractor.py`, `profile_builder.py`, `identifier.py`, `visualizer.py`
- `main.py` — entrypoint (train → identify → visualize)
- `app.py` — Streamlit UI: language selection, text input, analysis + charts
- `texts/{author}/` — training texts per author; `texts/anonim/` — texts to identify
- `output/<YYYY-MM-DD-HH-MM>/` — timestamped PNG output

## Gotchas

- **No dependency manifest.** Install: `pip install numpy nltk matplotlib pandas pymorphy3` (stanza optional for Belarusian). For UI: `pip install streamlit`.
- **Pickle cache.** Profiles saved to `authors_profiles.pkl` (gitignored). Delete it to force retraining. UI saves separately per language: `authors_profiles_ru.pkl` / `authors_profiles_be.pkl`.
- **Config.** `src/config.py` controls author list, feature weights, log level (`INFO`/`DEBUG`). `lermontov` is commented out of `AUTHORS_LIST` by default. `RUSSIAN_AUTHORS_LIST` / `BELARUSIAN_AUTHORS_LIST` used by UI.
- **`test_all_anonim.py` has broken imports.** Lines 7-8 use `from feature_extractor import ...` instead of `from src.feature_extractor import ...`. Patch them to run.
- **Encoding fallback.** Text loading tries `utf-8` → `cp1251` → `koi8-r` → `latin-1`.
- **No test framework, no linter, no CI.** `test_all_anonim.py` is the only test script.
- **Comments in Russian.** Codebase is in Russian for analyzing Russian literary texts.
- **`create_*.py` scripts** are deprecated — they use hardcoded `/workspace/` paths (container-only).
- **Belarusian author dirs** (`texts/kolas/`, `texts/maur/`, `texts/bryl/`) exist but are empty. Add `.txt` files before training in UI with Belarusian selected.
