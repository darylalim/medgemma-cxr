# MedGemma Chest X-ray (CXR)

Streamlit app using Google's MedGemma 1.5 4B to localize anatomical structures in chest X-rays with bounding boxes.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with your [Hugging Face token](https://huggingface.co/settings/tokens):

```
HF_TOKEN=your_token_here
```

## Usage

```bash
streamlit run streamlit_app.py
```

Upload a chest X-ray image, enter an anatomy to localize (e.g. "right clavicle"), and click **Localize**. The app pads the image to a square, runs MedGemma inference, and draws bounding boxes on the result.

## Development

```bash
ruff check streamlit_app.py tests/    # lint
ruff format streamlit_app.py tests/   # format
ty check streamlit_app.py tests/      # type check
python -m pytest tests/ -v            # test
```
