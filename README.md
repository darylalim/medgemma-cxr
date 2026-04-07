# MedGemma Chest X-ray (CXR)

Streamlit app using Google's MedGemma 1.5 4B for chest X-ray analysis: anatomy localization with bounding boxes and longitudinal comparison of two CXRs.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Create a `.env` file with your [Hugging Face token](https://huggingface.co/settings/tokens):

```
HF_TOKEN=your_token_here
```

## Usage

```bash
uv run streamlit run streamlit_app.py
```

The app has two modes, selectable via sidebar radio:

- **Localize Anatomy** — Upload a chest X-ray, enter an anatomy to localize (e.g. "right clavicle"), and click **Localize**. The app pads the image to a square, runs MedGemma inference, and draws bounding boxes on the result.
- **Compare CXRs** — Upload a prior and current chest X-ray and click **Compare**. The app runs MedGemma longitudinal comparison and displays a free-text analysis of changes between the two images.

## Development

```bash
uv run ruff check streamlit_app.py tests/    # lint
uv run ruff format streamlit_app.py tests/   # format
uv run ty check streamlit_app.py tests/      # type check
uv run python -m pytest tests/ -v            # test
```
