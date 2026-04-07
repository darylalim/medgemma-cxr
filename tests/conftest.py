from unittest.mock import MagicMock


def make_generation_result(text: str):
    """Create a mock GenerationResult with a .text attribute."""
    result = MagicMock()
    result.text = text
    return result
