import numpy as np
from PIL import Image


def test_pad_square_image_unchanged():
    """A square image should not be padded."""
    from streamlit_app import pad_image_to_square

    img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    result = pad_image_to_square(img)
    assert result.shape == (100, 100, 3)


def test_pad_wide_image():
    """A wide image should be padded vertically to become square."""
    from streamlit_app import pad_image_to_square

    img = Image.fromarray(np.zeros((50, 100, 3), dtype=np.uint8))
    result = pad_image_to_square(img)
    assert result.shape[0] == result.shape[1] == 100


def test_pad_tall_image():
    """A tall image should be padded horizontally to become square."""
    from streamlit_app import pad_image_to_square

    img = Image.fromarray(np.zeros((100, 50, 3), dtype=np.uint8))
    result = pad_image_to_square(img)
    assert result.shape[0] == result.shape[1] == 100


def test_pad_grayscale_converts_to_rgb():
    """A grayscale image should be converted to 3-channel RGB."""
    from streamlit_app import pad_image_to_square

    img = Image.fromarray(np.zeros((100, 100), dtype=np.uint8))
    result = pad_image_to_square(img)
    assert result.shape == (100, 100, 3)


def test_pad_rgba_converts_to_rgb():
    """An RGBA image should be converted to 3-channel RGB."""
    from streamlit_app import pad_image_to_square

    img = Image.fromarray(np.zeros((100, 100, 4), dtype=np.uint8))
    result = pad_image_to_square(img)
    assert result.shape == (100, 100, 3)


def test_pad_returns_uint8():
    """Output array should always be uint8."""
    from streamlit_app import pad_image_to_square

    img = Image.fromarray(np.zeros((50, 100, 3), dtype=np.uint8))
    result = pad_image_to_square(img)
    assert result.dtype == np.uint8
