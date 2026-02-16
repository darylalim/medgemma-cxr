# Streamlit CXR Anatomy Localization — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a single-file Streamlit app that loads MedGemma 1.5 4B, accepts a chest X-ray upload, localizes a user-specified anatomical structure, and draws bounding boxes on the image.

**Architecture:** Single-file `streamlit_app.py` with cached model loading, image preprocessing (pad to square), structured prompt construction, inference via HF pipeline, JSON parsing of bounding boxes, and PIL-based visualization. Runs on Mac with MPS.

**Tech Stack:** Streamlit, transformers, torch (MPS), PIL/Pillow, scikit-image, numpy

---

### Task 1: Update dependencies

**Files:**
- Modify: `requirements.txt`

**Step 1: Add missing dependencies**

Add `scikit-image`, `Pillow`, and `numpy` to `requirements.txt`. The final file:

```
accelerate
numpy
Pillow
pytest
python-dotenv
ruff
scikit-image
streamlit
torch
transformers
ty
```

**Step 2: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: All packages install successfully

**Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: add scikit-image, Pillow, numpy to dependencies"
```

---

### Task 2: Write tests for image preprocessing utilities

**Files:**
- Create: `tests/test_image_utils.py`

**Step 1: Write failing tests for pad_image_to_square**

```python
import numpy as np
from PIL import Image


def test_pad_square_image_unchanged():
    """A square image should not be padded."""
    from streamlit_app import pad_image_to_square

    img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    result = pad_image_to_square(img)
    assert result.shape[0] == result.shape[1]
    assert result.shape[0] == 100


def test_pad_wide_image():
    """A wide image should be padded vertically to become square."""
    from streamlit_app import pad_image_to_square

    img = Image.fromarray(np.zeros((50, 100, 3), dtype=np.uint8))
    result = pad_image_to_square(img)
    assert result.shape[0] == result.shape[1]
    assert result.shape[0] == 100


def test_pad_tall_image():
    """A tall image should be padded horizontally to become square."""
    from streamlit_app import pad_image_to_square

    img = Image.fromarray(np.zeros((100, 50, 3), dtype=np.uint8))
    result = pad_image_to_square(img)
    assert result.shape[0] == result.shape[1]
    assert result.shape[0] == 100


def test_pad_grayscale_converts_to_rgb():
    """A grayscale image should be converted to 3-channel RGB."""
    from streamlit_app import pad_image_to_square

    img = Image.fromarray(np.zeros((100, 100), dtype=np.uint8))
    result = pad_image_to_square(img)
    assert result.shape == (100, 100, 3)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_image_utils.py -v`
Expected: FAIL — `pad_image_to_square` not defined in `streamlit_app`

---

### Task 3: Write tests for bbox parsing and coordinate conversion

**Files:**
- Create: `tests/test_bbox.py`

**Step 1: Write failing tests for parse_bboxes and convert coordinates**

```python
import numpy as np
from PIL import Image


def test_parse_bboxes_from_json_block():
    """Should extract bbox list from a ```json...``` block in model response."""
    from streamlit_app import parse_bboxes

    response = (
        'Some reasoning here.\n'
        'Final Answer: ```json[{"box_2d": [100, 200, 300, 400], "label": "right clavicle"}]```'
    )
    result = parse_bboxes(response)
    assert len(result) == 1
    assert result[0]["box_2d"] == [100, 200, 300, 400]
    assert result[0]["label"] == "right clavicle"


def test_parse_bboxes_multiple():
    """Should parse multiple bounding boxes."""
    from streamlit_app import parse_bboxes

    response = '```json[{"box_2d": [10, 20, 30, 40], "label": "a"}, {"box_2d": [50, 60, 70, 80], "label": "b"}]```'
    result = parse_bboxes(response)
    assert len(result) == 2


def test_parse_bboxes_no_json():
    """Should return empty list when no JSON block found."""
    from streamlit_app import parse_bboxes

    result = parse_bboxes("No JSON here at all.")
    assert result == []


def test_draw_bboxes_returns_image():
    """draw_bboxes should return a PIL Image with boxes drawn."""
    from streamlit_app import draw_bboxes

    img = Image.fromarray(np.ones((200, 200, 3), dtype=np.uint8) * 128)
    bboxes = [{"box_2d": [100, 200, 300, 400], "label": "test"}]
    result = draw_bboxes(img, bboxes)
    assert isinstance(result, Image.Image)
    assert result.size == (200, 200)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_bbox.py -v`
Expected: FAIL — `parse_bboxes` and `draw_bboxes` not defined

---

### Task 4: Implement image preprocessing

**Files:**
- Modify: `streamlit_app.py`

**Step 1: Implement pad_image_to_square**

Write at the top of `streamlit_app.py`:

```python
import numpy as np
import skimage.color
import skimage.util
from PIL import Image


def pad_image_to_square(image: Image.Image) -> np.ndarray:
    """Pad a PIL Image to a square numpy array (RGB, uint8)."""
    image_array = np.array(image)
    image_array = skimage.util.img_as_ubyte(image_array)
    if len(image_array.shape) < 3:
        image_array = skimage.color.gray2rgb(image_array)
    if image_array.shape[2] == 4:
        image_array = skimage.color.rgba2rgb(image_array)
        image_array = skimage.util.img_as_ubyte(image_array)

    h, w = image_array.shape[:2]
    max_dim = max(h, w)
    if h < w:
        dh = w - h
        image_array = np.pad(
            image_array, ((dh // 2, dh - dh // 2), (0, 0), (0, 0))
        )
    elif w < h:
        dw = h - w
        image_array = np.pad(
            image_array, ((0, 0), (dw // 2, dw - dw // 2), (0, 0))
        )
    return image_array
```

**Step 2: Run tests to verify they pass**

Run: `pytest tests/test_image_utils.py -v`
Expected: All 4 tests PASS

**Step 3: Commit**

```bash
git add streamlit_app.py tests/test_image_utils.py
git commit -m "feat: add pad_image_to_square preprocessing"
```

---

### Task 5: Implement bbox parsing and drawing

**Files:**
- Modify: `streamlit_app.py`

**Step 1: Implement parse_bboxes**

```python
import json
import re
from PIL import ImageDraw, ImageFont


def parse_bboxes(response: str) -> list[dict]:
    """Extract bounding boxes from model response containing ```json...``` block."""
    match = re.search(r"```json\s*(\[.*?\])\s*```", response, re.DOTALL)
    if not match:
        return []
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return []
```

**Step 2: Implement draw_bboxes**

```python
BBOX_COLOR = (255, 0, 0)
BBOX_WIDTH = 3


def draw_bboxes(image: Image.Image, bboxes: list[dict]) -> Image.Image:
    """Draw bounding boxes on an image. Coords are normalized to [0, 1000]."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size

    for bbox in bboxes:
        coords = bbox.get("box_2d", [])
        label = bbox.get("label", "")
        if len(coords) != 4:
            continue
        y0, x0, y1, x1 = coords
        # Convert from [0, 1000] normalized to pixel coordinates
        x0_px = x0 / 1000 * w
        y0_px = y0 / 1000 * h
        x1_px = x1 / 1000 * w
        y1_px = y1 / 1000 * h
        draw.rectangle([x0_px, y0_px, x1_px, y1_px], outline=BBOX_COLOR, width=BBOX_WIDTH)
        if label:
            draw.text((x0_px, max(0, y0_px - 15)), label, fill=BBOX_COLOR)

    return img
```

**Step 3: Run tests to verify they pass**

Run: `pytest tests/test_bbox.py -v`
Expected: All 4 tests PASS

**Step 4: Commit**

```bash
git add streamlit_app.py tests/test_bbox.py
git commit -m "feat: add bbox parsing and drawing utilities"
```

---

### Task 6: Implement model loading and prompt construction

**Files:**
- Modify: `streamlit_app.py`

**Step 1: Add model loading with @st.cache_resource**

```python
import os
import streamlit as st
import torch
from transformers import pipeline


MODEL_ID = "google/medgemma-1.5-4b-it"


@st.cache_resource
def load_model():
    """Load MedGemma pipeline once, cached across Streamlit reruns."""
    return pipeline(
        "image-text-to-text",
        model=MODEL_ID,
        model_kwargs=dict(dtype=torch.float32, device_map="auto"),
    )
```

**Step 2: Add prompt builder**

```python
def build_prompt(object_name: str) -> str:
    """Build the structured localization prompt for a given anatomy name."""
    return f"""Instructions:
The following user query will require outputting bounding boxes. The format of bounding boxes coordinates is [y0, x0, y1, x1] where (y0, x0) must be top-left corner and (y1, x1) the bottom-right corner. This implies that x0 < x1 and y0 < y1. Always normalize the x and y coordinates the range [0, 1000], meaning that a bounding box starting at 15% of the image width would be associated with an x coordinate of 150. You MUST output a single parseable json list of objects enclosed into ```json...``` brackets, for instance ```json[{{"box_2d": [800, 3, 840, 471], "label": "car"}}, {{"box_2d": [400, 22, 600, 73], "label": "dog"}}]``` is a valid output. Now answer to the user query.

Remember "left" refers to the patient's left side where the heart is and sometimes underneath an L in the upper right corner of the image.

Query:
Where is the {object_name}? Don't give a final answer without reasoning. Output the final answer in the format "Final Answer: X" where X is a JSON list of objects. The object needs a "box_2d" and "label" key. Answer:"""
```

**Step 3: Add inference runner**

```python
def run_inference(pipe, image: Image.Image, object_name: str) -> str:
    """Run MedGemma inference and return the cleaned response text."""
    prompt = build_prompt(object_name)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    output = pipe(text=messages, max_new_tokens=1000, do_sample=False)
    response = output[0]["generated_text"][-1]["content"]

    # Strip thinking tokens
    if "<unused95>" in response:
        response = response.split("<unused95>", 1)[1].lstrip()

    return response
```

**Step 4: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: add model loading, prompt builder, and inference runner"
```

---

### Task 7: Implement main Streamlit UI

**Files:**
- Modify: `streamlit_app.py`

**Step 1: Add the main app function**

Append to `streamlit_app.py`:

```python
def main():
    st.set_page_config(page_title="CXR Anatomy Localization", layout="wide")
    st.title("CXR Anatomy Localization")

    # Check for HF token
    if not os.environ.get("HF_TOKEN"):
        st.error(
            "HF_TOKEN environment variable not set. "
            "Get a token at https://huggingface.co/settings/tokens "
            "and run: export HF_TOKEN=your_token"
        )
        st.stop()

    # Sidebar controls
    with st.sidebar:
        st.header("Settings")
        uploaded_file = st.file_uploader(
            "Upload a chest X-ray", type=["png", "jpg", "jpeg"]
        )
        object_name = st.text_input("Anatomy to localize", value="right clavicle")
        run_button = st.button("Localize", type="primary")

    # Main area
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"Could not open image: {e}")
            st.stop()

        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Original")
            st.image(image, use_container_width=True)

        if run_button:
            if not object_name.strip():
                st.warning("Please enter an anatomy name.")
                st.stop()

            # Preprocess
            image_array = pad_image_to_square(image)
            square_image = Image.fromarray(image_array)

            # Load model and run inference
            with st.spinner("Loading model..."):
                try:
                    pipe = load_model()
                except Exception as e:
                    st.error(f"Failed to load model: {e}")
                    st.stop()

            with st.spinner("Running inference..."):
                response = run_inference(pipe, square_image, object_name.strip())

            # Parse and draw
            bboxes = parse_bboxes(response)
            if not bboxes:
                st.warning("Could not parse bounding boxes from model response.")
                st.text(response)
            else:
                annotated = draw_bboxes(square_image, bboxes)
                with col_right:
                    st.subheader("Localization")
                    st.image(annotated, use_container_width=True)
    else:
        st.info("Upload a chest X-ray image in the sidebar to get started.")


if __name__ == "__main__":
    main()
```

**Step 2: Manual smoke test**

Run: `HF_TOKEN=your_token streamlit run streamlit_app.py`
Expected: App loads, sidebar shows uploader + text input + button. Upload an image, click Localize, see annotated result.

**Step 3: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: add Streamlit UI for CXR anatomy localization"
```

---

### Task 8: Final integration test

**Step 1: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests pass

**Step 2: Run linter**

Run: `ruff check streamlit_app.py tests/`
Expected: No errors (fix any that appear)

**Step 3: Final commit**

```bash
git add -A
git commit -m "chore: final cleanup and lint fixes"
```
