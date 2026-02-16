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
