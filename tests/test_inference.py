from unittest.mock import MagicMock, patch

from PIL import Image
import numpy as np

from tests.conftest import make_generation_result


@patch("streamlit_app.generate")
@patch("streamlit_app.apply_chat_template", return_value="formatted_prompt")
def test_run_inference_returns_string(mock_template, mock_generate):
    """run_inference should return the model's text response."""
    from streamlit_app import run_inference

    mock_generate.return_value = make_generation_result("Some bbox response.")
    model = MagicMock()
    processor = MagicMock()

    img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    result = run_inference(model, processor, img, "right clavicle")

    assert isinstance(result, str)
    assert "Some bbox response." in result


@patch("streamlit_app.generate")
@patch("streamlit_app.apply_chat_template", return_value="formatted_prompt")
def test_run_inference_strips_thinking_tokens(mock_template, mock_generate):
    """run_inference should strip <unused95> thinking tokens."""
    from streamlit_app import run_inference

    mock_generate.return_value = make_generation_result("<unused95>Actual response.")
    model = MagicMock()
    processor = MagicMock()

    img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    result = run_inference(model, processor, img, "trachea")

    assert "<unused95>" not in result
    assert "Actual response." in result


@patch("streamlit_app.generate")
@patch("streamlit_app.apply_chat_template", return_value="formatted_prompt")
def test_run_inference_sends_single_image(mock_template, mock_generate):
    """run_inference should pass the image as a single-element list to generate()."""
    from streamlit_app import run_inference

    mock_generate.return_value = make_generation_result("Result.")
    model = MagicMock()
    processor = MagicMock()

    img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    run_inference(model, processor, img, "right clavicle")

    call_args = mock_generate.call_args
    image_list = call_args[0][3]  # 4th positional arg: [image]
    assert len(image_list) == 1
    assert image_list[0] is img


@patch("streamlit_app.generate")
@patch("streamlit_app.apply_chat_template", return_value="formatted_prompt")
def test_run_inference_formats_prompt_with_object_name(mock_template, mock_generate):
    """run_inference should format PROMPT_TEMPLATE with the given anatomy name."""
    from streamlit_app import run_inference

    mock_generate.return_value = make_generation_result("Result.")
    model = MagicMock()
    processor = MagicMock()

    img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    run_inference(model, processor, img, "aortic arch")

    template_args = mock_template.call_args
    prompt_arg = template_args[0][2]  # 3rd positional arg: the formatted prompt string
    assert "aortic arch" in prompt_arg
