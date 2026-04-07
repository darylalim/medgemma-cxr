# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Streamlit app using Google's MedGemma 1.5 4B for chest X-ray analysis: anatomy localization with bounding boxes and longitudinal comparison of two CXRs.

## Commands

Uses `uv` for package and project management. Use `uv run python -m pytest` (not bare `pytest`) so `streamlit_app` is importable from the project root.

```bash
uv sync                                                                       # install deps
uv run ruff check streamlit_app.py tests/                                     # lint
uv run ruff format streamlit_app.py tests/                                    # format
uv run ty check streamlit_app.py tests/                                       # type check
uv run python -m pytest tests/ -v                                             # test all
uv run python -m pytest tests/test_bbox.py::test_parse_bboxes_from_json_block -v  # test one
uv run streamlit run streamlit_app.py                                         # run app
```

## Architecture

Single-file app (`streamlit_app.py`) with two modes selected via sidebar radio:

**Localize Anatomy** (bounding box detection):
```
Upload → pad_image_to_square() → run_inference() → parse_bboxes() → draw_bboxes() → display
```

**Compare CXRs** (longitudinal comparison):
```
Upload 2 images → pad_image_to_square() each → run_comparison() → display text
```

- `main` — sidebar radio switches between `_localize_ui()` and `_compare_ui()`
- `pad_image_to_square` — converts to RGB via PIL, pads shorter side with zeros to make a square numpy array
- `run_inference` — formats `PROMPT_TEMPLATE` with the anatomy name, runs the HuggingFace pipeline
- `run_comparison` — sends two images with `COMPARISON_PROMPT` to the pipeline, returns free-text comparison
- `parse_bboxes` — tries four regex patterns in priority order to extract bbox JSON from model output
- `draw_bboxes` — converts normalized `[0, 1000]` coords to pixels, draws red rectangles and labels
- `load_model` — cached via `@st.cache_resource`, uses MPS on Apple Silicon with float32 (MPS limitation)

## Rules

- Do not pass generation parameters as both loose kwargs and a `GenerationConfig` object — pick one approach
- When using `GenerationConfig`, start from the model's existing config (`pipe.model.generation_config.to_dict()`) to preserve model-specific defaults
- `st.image` uses `width="stretch"`, not `use_container_width=True` (deprecated)
- `pad_image_to_square` handles RGB conversion — do not call `.convert("RGB")` before passing images to it
- Do not add module-level imports that are only used in a subset of test functions — import locally instead
- Always verify changes work end-to-end with the model before claiming a fix, not just that tests pass

## Key details

- Bounding box coords are `[y0, x0, y1, x1]` normalized to `[0, 1000]`
- Model responses may contain `<unused95>` thinking tokens that get stripped before parsing
- `load_dotenv()` runs before other imports (causes E402 ruff warnings — intentional)
- Requires `HF_TOKEN` environment variable (via `.env` or exported)
