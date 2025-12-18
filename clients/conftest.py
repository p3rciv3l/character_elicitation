"""
Pytest configuration.

Note: OpenRouterClient now uses requests directly instead of the OpenAI SDK,
so no patching is needed. This file is kept for potential future fixtures.
"""
from unittest.mock import MagicMock
import pytest

# Mock fixture for backward compatibility with unit tests
_mock_openai_instance = MagicMock()
_mock_openai_class = MagicMock(return_value=_mock_openai_instance)

@pytest.fixture
def mock_openai():
    """Fixture providing a mocked OpenAI class for backward compatibility."""
    return _mock_openai_class
