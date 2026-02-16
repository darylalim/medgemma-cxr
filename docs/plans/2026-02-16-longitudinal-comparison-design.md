# Longitudinal CXR Comparison — Design

## Overview

Add a mode to the Streamlit app that lets users upload two chest X-rays (prior and current) and get a free-text longitudinal comparison from MedGemma.

## Approach

Inline in `streamlit_app.py` (single-file architecture preserved). Sidebar radio switches between "Localize Anatomy" and "Compare CXRs".

## New components

- `COMPARISON_PROMPT` — fixed prompt requesting longitudinal CXR comparison details
- `run_comparison(pipe, image1, image2)` — sends two images + comparison prompt to the pipeline, strips `<unused95>` tokens, returns free-text response

## UI flow (Compare mode)

1. Sidebar: two file uploaders ("Prior CXR", "Current CXR") + "Compare" button
2. Main area: both images side-by-side in two columns
3. On click: pad both to square, load model, run comparison
4. Display model response below images via `st.markdown`

## Reused components

- `load_model` — unchanged
- `pad_image_to_square` — unchanged
- Localize mode — unchanged, wrapped in mode conditional

## Testing

- `tests/test_comparison.py` — test `run_comparison` with mocked pipeline (returns string, strips thinking tokens)
