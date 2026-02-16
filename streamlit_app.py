import json
import os
import re

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageDraw
from transformers import pipeline

MODEL_ID = "google/medgemma-1.5-4b-it"
BBOX_COLOR = (255, 0, 0)
BBOX_WIDTH = 3


# --- Image preprocessing ---


def pad_image_to_square(image: Image.Image) -> np.ndarray:
    """Pad a PIL Image to a square numpy array (RGB, uint8)."""
    image_array = np.array(image.convert("RGB"))

    h, w = image_array.shape[:2]
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


# --- Bbox parsing and drawing ---


def parse_bboxes(response: str) -> list[dict]:
    """Extract bounding boxes from model response.

    Tries, in order: ```json [...]```, ``` [...]```, Final Answer: [...], bare [...].
    """
    patterns = [
        r"```json\s*(\[.*?\])\s*```",
        r"```\s*(\[.*?\])\s*```",
        r"Final Answer:\s*(\[.*?\])",
        r"(\[\s*\{.*?\}\s*\])",
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
    return []


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
        x0_px = x0 / 1000 * w
        y0_px = y0 / 1000 * h
        x1_px = x1 / 1000 * w
        y1_px = y1 / 1000 * h
        draw.rectangle(
            [x0_px, y0_px, x1_px, y1_px], outline=BBOX_COLOR, width=BBOX_WIDTH
        )
        if label:
            draw.text((x0_px, max(0, y0_px - 15)), label, fill=BBOX_COLOR)

    return img


# --- Model loading and inference ---


@st.cache_resource
def load_model():
    """Load MedGemma pipeline once, cached across Streamlit reruns."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return pipeline(
        "image-text-to-text",
        model=MODEL_ID,
        model_kwargs=dict(dtype=torch.float32),
        device=device,
    )


PROMPT_TEMPLATE = """Instructions:
The following user query will require outputting bounding boxes. The format of bounding boxes coordinates is [y0, x0, y1, x1] where (y0, x0) must be top-left corner and (y1, x1) the bottom-right corner. This implies that x0 < x1 and y0 < y1. Always normalize the x and y coordinates the range [0, 1000], meaning that a bounding box starting at 15% of the image width would be associated with an x coordinate of 150. You MUST output a single parseable json list of objects enclosed into ```json...``` brackets, for instance ```json[{{"box_2d": [800, 3, 840, 471], "label": "car"}}, {{"box_2d": [400, 22, 600, 73], "label": "dog"}}]``` is a valid output. Now answer to the user query.

Remember "left" refers to the patient's left side where the heart is and sometimes underneath an L in the upper right corner of the image.

Query:
Where is the {object_name}? Don't give a final answer without reasoning. Output the final answer in the format "Final Answer: X" where X is a JSON list of objects. The object needs a "box_2d" and "label" key. Answer:"""


def run_inference(pipe, image: Image.Image, object_name: str) -> str:
    """Run MedGemma inference and return the cleaned response text."""
    prompt = PROMPT_TEMPLATE.format(object_name=object_name)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    output = pipe(text=messages, max_new_tokens=1000, max_length=None, do_sample=False, pad_token_id=pipe.tokenizer.eos_token_id)
    response = output[0]["generated_text"][-1]["content"]

    if "<unused95>" in response:
        response = response.split("<unused95>", 1)[1].lstrip()

    return response


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


# --- Streamlit UI ---


def main():
    st.set_page_config(page_title="CXR Anatomy Localization", layout="wide")
    st.title("CXR Anatomy Localization")

    if not os.environ.get("HF_TOKEN"):
        st.error(
            "HF_TOKEN environment variable not set. "
            "Get a token at https://huggingface.co/settings/tokens "
            "and run: export HF_TOKEN=your_token"
        )
        st.stop()

    with st.sidebar:
        st.header("Settings")
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


if __name__ == "__main__":
    main()
