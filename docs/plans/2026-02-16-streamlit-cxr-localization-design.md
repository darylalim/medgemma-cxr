# Streamlit CXR Anatomy Localization App — Design

## Overview

Single-file Streamlit app (`streamlit_app.py`) that wraps the MedGemma 1.5 4B CXR anatomy localization notebook into an interactive web UI. Runs locally on Mac with MPS (Apple Silicon).

## Architecture

Single-file, Approach A. All logic in `streamlit_app.py` (~200 lines).

## UI Layout

- **Sidebar**: File uploader (PNG/JPG), text input for anatomy name, "Localize" button
- **Main area**: Two columns — left shows uploaded original, right shows annotated image with bounding boxes after inference. Spinner during model loading/inference.

## Model Loading & MPS Compatibility

- `@st.cache_resource` to load model once across reruns
- `float32` dtype (MPS doesn't fully support bfloat16)
- `device_map="auto"` routes to MPS on Apple Silicon
- HF token from `HF_TOKEN` environment variable

## Image Processing & Inference Flow

1. **Upload**: PNG/JPG via `st.file_uploader`, open with PIL
2. **Preprocess**: Pad to square (grayscale→RGB, RGBA→RGB, zero-pad shorter dimension) — same logic as notebook
3. **Prompt**: Structured prompt requesting `[y0, x0, y1, x1]` bounding boxes normalized to [0, 1000], chain-of-thought reasoning, JSON output
4. **Inference**: Pipeline with `do_sample=False`, `max_new_tokens=1000`. Strip thinking tokens (`<unused94>`/`<unused95>`)
5. **Parse**: Extract JSON from `` ```json...``` `` block, parse into list of `{"box_2d": [...], "label": "..."}`
6. **Draw**: Convert normalized [0, 1000] coords to pixel coords on square image, draw rectangles + labels with PIL `ImageDraw`
7. **Display**: Annotated image in right column

## Error Handling

- No `HF_TOKEN`: `st.error` with setup instructions
- Model loading fails: Catch exception, show error
- No JSON in response: Show raw text with warning
- Invalid image: Catch PIL exception, show `st.error`

No retry logic or fallbacks.

## Dependencies

Already in `requirements.txt`: accelerate, streamlit, torch, transformers. Add: scikit-image, Pillow (likely already transitive but make explicit).
