from unittest.mock import MagicMock, patch

from PIL import Image
import numpy as np

from tests.conftest import make_generation_result


@patch("streamlit_app.generate")
@patch("streamlit_app.apply_chat_template", return_value="formatted_prompt")
def test_run_comparison_returns_string(mock_template, mock_generate):
    """run_comparison should return the model's text response."""
    from streamlit_app import run_comparison

    mock_generate.return_value = make_generation_result("Comparison result text.")
    model = MagicMock()
    processor = MagicMock()

    img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    result = run_comparison(model, processor, img, img)

    assert isinstance(result, str)
    assert "Comparison result text." in result


@patch("streamlit_app.generate")
@patch("streamlit_app.apply_chat_template", return_value="formatted_prompt")
def test_run_comparison_strips_thinking_tokens(mock_template, mock_generate):
    """run_comparison should strip <unused95> thinking tokens."""
    from streamlit_app import run_comparison

    mock_generate.return_value = make_generation_result("<unused95>Actual comparison.")
    model = MagicMock()
    processor = MagicMock()

    img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    result = run_comparison(model, processor, img, img)

    assert "<unused95>" not in result
    assert "Actual comparison." in result


@patch("streamlit_app.generate")
@patch("streamlit_app.apply_chat_template", return_value="formatted_prompt")
def test_run_comparison_sends_two_images(mock_template, mock_generate):
    """run_comparison should pass both images to generate()."""
    from streamlit_app import run_comparison

    mock_generate.return_value = make_generation_result("Result.")
    model = MagicMock()
    processor = MagicMock()

    img1 = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    img2 = Image.fromarray(np.ones((100, 100, 3), dtype=np.uint8) * 255)
    run_comparison(model, processor, img1, img2)

    call_args = mock_generate.call_args
    image_list = call_args[0][3]  # 4th positional arg: [image1, image2]
    assert len(image_list) == 2
    assert image_list[0] is img1
    assert image_list[1] is img2
