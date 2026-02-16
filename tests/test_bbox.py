from PIL import Image


def test_parse_bboxes_from_json_block():
    """Should extract bbox list from a ```json...``` block."""
    from streamlit_app import parse_bboxes

    response = (
        'Some reasoning here.\n'
        'Final Answer: ```json[{"box_2d": [100, 200, 300, 400], "label": "right clavicle"}]```'
    )
    result = parse_bboxes(response)
    assert len(result) == 1
    assert result[0]["box_2d"] == [100, 200, 300, 400]
    assert result[0]["label"] == "right clavicle"


def test_parse_bboxes_from_bare_fenced_block():
    """Should extract bbox list from a ``` ... ``` block without json tag."""
    from streamlit_app import parse_bboxes

    response = 'Reasoning.\n```[{"box_2d": [10, 20, 30, 40], "label": "aorta"}]```'
    result = parse_bboxes(response)
    assert len(result) == 1
    assert result[0]["label"] == "aorta"


def test_parse_bboxes_from_final_answer():
    """Should extract bbox list after 'Final Answer:' without fenced block."""
    from streamlit_app import parse_bboxes

    response = 'Reasoning here.\nFinal Answer: [{"box_2d": [50, 60, 70, 80], "label": "trachea"}]'
    result = parse_bboxes(response)
    assert len(result) == 1
    assert result[0]["label"] == "trachea"


def test_parse_bboxes_from_bare_json():
    """Should extract bbox list from a bare JSON array in the response."""
    from streamlit_app import parse_bboxes

    response = 'The structure is here [{"box_2d": [1, 2, 3, 4], "label": "spine"}] end.'
    result = parse_bboxes(response)
    assert len(result) == 1
    assert result[0]["label"] == "spine"


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


def test_parse_bboxes_invalid_json():
    """Should return empty list when JSON is malformed."""
    from streamlit_app import parse_bboxes

    result = parse_bboxes('```json[{"box_2d": [1, 2, 3, 4], "label": broken}]```')
    assert result == []


def test_draw_bboxes_returns_image():
    """draw_bboxes should return a PIL Image with boxes drawn."""
    import numpy as np

    from streamlit_app import draw_bboxes

    img = Image.fromarray(np.ones((200, 200, 3), dtype=np.uint8) * 128)
    bboxes = [{"box_2d": [100, 200, 300, 400], "label": "test"}]
    result = draw_bboxes(img, bboxes)
    assert isinstance(result, Image.Image)
    assert result.size == (200, 200)


def test_draw_bboxes_skips_invalid_coords():
    """draw_bboxes should skip entries with missing or wrong-length coords."""
    import numpy as np

    from streamlit_app import draw_bboxes

    img = Image.fromarray(np.ones((200, 200, 3), dtype=np.uint8) * 128)
    bboxes = [
        {"box_2d": [100, 200], "label": "too few"},
        {"label": "no coords"},
        {"box_2d": [100, 200, 300, 400], "label": "valid"},
    ]
    result = draw_bboxes(img, bboxes)
    assert isinstance(result, Image.Image)


def test_draw_bboxes_no_label():
    """draw_bboxes should handle entries without a label."""
    import numpy as np

    from streamlit_app import draw_bboxes

    img = Image.fromarray(np.ones((200, 200, 3), dtype=np.uint8) * 128)
    bboxes = [{"box_2d": [100, 200, 300, 400]}]
    result = draw_bboxes(img, bboxes)
    assert isinstance(result, Image.Image)
