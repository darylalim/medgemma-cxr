# Longitudinal CXR Comparison Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a "Compare CXRs" mode to the Streamlit app so users can upload two chest X-rays and get a free-text longitudinal comparison from MedGemma.

**Architecture:** Add a sidebar radio to switch between "Localize Anatomy" (existing) and "Compare CXRs" (new). The comparison mode adds a `COMPARISON_PROMPT` constant and a `run_comparison()` function, plus a new UI branch in `main()`. Everything stays in `streamlit_app.py`.

**Tech Stack:** Streamlit, HuggingFace Transformers pipeline, PIL, NumPy (all existing)

---

### Task 1: Add `run_comparison` function with tests

**Files:**
- Modify: `streamlit_app.py:105` (add constant + function after existing inference section)
- Create: `tests/test_comparison.py`

**Step 1: Write the failing test**

Create `tests/test_comparison.py`:

```python
from unittest.mock import MagicMock

from PIL import Image
import numpy as np


def test_run_comparison_returns_string():
    """run_comparison should return the model's text response."""
    from streamlit_app import run_comparison

    pipe = MagicMock()
    pipe.tokenizer.eos_token_id = 0
    pipe.return_value = [
        {"generated_text": [{"role": "assistant", "content": "Comparison result text."}]}
    ]

    img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    result = run_comparison(pipe, img, img)

    assert isinstance(result, str)
    assert "Comparison result text." in result


def test_run_comparison_strips_thinking_tokens():
    """run_comparison should strip <unused95> thinking tokens."""
    from streamlit_app import run_comparison

    pipe = MagicMock()
    pipe.tokenizer.eos_token_id = 0
    pipe.return_value = [
        {"generated_text": [{"role": "assistant", "content": "<unused95>Actual comparison."}]}
    ]

    img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    result = run_comparison(pipe, img, img)

    assert "<unused95>" not in result
    assert "Actual comparison." in result


def test_run_comparison_sends_two_images():
    """run_comparison should include both images in the message to the pipeline."""
    from streamlit_app import run_comparison

    pipe = MagicMock()
    pipe.tokenizer.eos_token_id = 0
    pipe.return_value = [
        {"generated_text": [{"role": "assistant", "content": "Result."}]}
    ]

    img1 = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    img2 = Image.fromarray(np.ones((100, 100, 3), dtype=np.uint8) * 255)
    run_comparison(pipe, img1, img2)

    call_kwargs = pipe.call_args
    messages = call_kwargs.kwargs.get("text") or call_kwargs[1].get("text") or call_kwargs[0][0]
    user_content = messages[0]["content"]
    image_items = [item for item in user_content if item["type"] == "image"]
    assert len(image_items) == 2
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_comparison.py -v`
Expected: FAIL with `ImportError` — `run_comparison` does not exist yet.

**Step 3: Write minimal implementation**

Add to `streamlit_app.py` after `run_inference` (around line 133), before the `# --- Streamlit UI ---` section:

```python
COMPARISON_PROMPT = (
    "Provide a comparison of these two images and include details "
    "which students should take note of when reading longitudinal CXR"
)


def run_comparison(pipe, image1: Image.Image, image2: Image.Image) -> str:
    """Run MedGemma longitudinal comparison on two CXR images."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image1},
                {"type": "image", "image": image2},
                {"type": "text", "text": COMPARISON_PROMPT},
            ],
        }
    ]
    output = pipe(
        text=messages,
        max_new_tokens=1000,
        max_length=None,
        do_sample=False,
        pad_token_id=pipe.tokenizer.eos_token_id,
    )
    response = output[0]["generated_text"][-1]["content"]

    if "<unused95>" in response:
        response = response.split("<unused95>", 1)[1].lstrip()

    return response
```

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_comparison.py -v`
Expected: 3 PASS

**Step 5: Run all existing tests to verify no regressions**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add streamlit_app.py tests/test_comparison.py
git commit -m "feat: add run_comparison function for longitudinal CXR comparison"
```

---

### Task 2: Add sidebar mode switcher and comparison UI

**Files:**
- Modify: `streamlit_app.py:138-204` (the `main()` function)

**Step 1: Modify `main()` to add mode switcher and comparison UI**

Replace the `main()` function with:

```python
def main():
    st.set_page_config(page_title="MedGemma CXR", layout="wide")

    if not os.environ.get("HF_TOKEN"):
        st.error(
            "HF_TOKEN environment variable not set. "
            "Get a token at https://huggingface.co/settings/tokens "
            "and run: export HF_TOKEN=your_token"
        )
        st.stop()

    with st.sidebar:
        mode = st.radio("Mode", ["Localize Anatomy", "Compare CXRs"])
        st.header("Settings")

    if mode == "Localize Anatomy":
        st.title("CXR Anatomy Localization")
        _localize_ui()
    else:
        st.title("CXR Longitudinal Comparison")
        _compare_ui()


def _localize_ui():
    with st.sidebar:
        uploaded_file = st.file_uploader(
            "Upload a chest X-ray", type=["png", "jpg", "jpeg"]
        )
        object_name = st.text_input("Anatomy to localize", value="right clavicle")
        run_button = st.button("Localize", type="primary")

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
        except Exception as e:
            st.error(f"Could not open image: {e}")
            st.stop()

        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Original")
            st.image(image, width="stretch")

        if run_button:
            if not object_name.strip():
                st.warning("Please enter an anatomy name.")
                st.stop()

            image_array = pad_image_to_square(image)
            square_image = Image.fromarray(image_array)

            with st.spinner("Loading model..."):
                try:
                    pipe = load_model()
                except Exception as e:
                    st.error(f"Failed to load model: {e}")
                    st.stop()

            with st.spinner("Running inference..."):
                response = run_inference(pipe, square_image, object_name.strip())

            bboxes = parse_bboxes(response)
            if not bboxes:
                st.warning("Could not parse bounding boxes from model response.")
            else:
                annotated = draw_bboxes(square_image, bboxes)
                with col_right:
                    st.subheader("Localization")
                    st.image(annotated, width="stretch")

            with st.expander("Model response"):
                st.text(response)
    else:
        st.info("Upload a chest X-ray image in the sidebar to get started.")


def _compare_ui():
    with st.sidebar:
        prior_file = st.file_uploader(
            "Upload prior CXR", type=["png", "jpg", "jpeg"], key="prior"
        )
        current_file = st.file_uploader(
            "Upload current CXR", type=["png", "jpg", "jpeg"], key="current"
        )
        compare_button = st.button("Compare", type="primary")

    if prior_file is not None and current_file is not None:
        try:
            prior_image = Image.open(prior_file)
            current_image = Image.open(current_file)
        except Exception as e:
            st.error(f"Could not open image: {e}")
            st.stop()

        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Prior")
            st.image(prior_image, width="stretch")
        with col_right:
            st.subheader("Current")
            st.image(current_image, width="stretch")

        if compare_button:
            prior_array = pad_image_to_square(prior_image)
            current_array = pad_image_to_square(current_image)
            prior_square = Image.fromarray(prior_array)
            current_square = Image.fromarray(current_array)

            with st.spinner("Loading model..."):
                try:
                    pipe = load_model()
                except Exception as e:
                    st.error(f"Failed to load model: {e}")
                    st.stop()

            with st.spinner("Comparing images..."):
                response = run_comparison(pipe, prior_square, current_square)

            st.subheader("Comparison")
            st.markdown(response)

            with st.expander("Raw model response"):
                st.text(response)
    elif prior_file is not None or current_file is not None:
        st.info("Upload both a prior and current CXR to compare.")
    else:
        st.info("Upload two chest X-ray images in the sidebar to get started.")
```

**Step 2: Run all tests**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: All PASS

**Step 3: Lint and format**

Run: `.venv/bin/ruff check streamlit_app.py && .venv/bin/ruff format streamlit_app.py`
Expected: No errors

**Step 4: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: add sidebar mode switcher and Compare CXRs UI"
```
