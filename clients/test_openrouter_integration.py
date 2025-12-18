"""
Minimal integration tests for OpenRouterClient using real API calls.

These tests make actual API calls to OpenRouter using free models.
They are skipped if OPENROUTER_API_KEY is not set.

Run with: pytest test_openrouter_integration.py -v -s
"""

import os
import pytest
from pathlib import Path
from clients.openrouter_client import OpenRouterClient

# Try to load .env file explicitly for tests
try:
    from dotenv import load_dotenv
    
    # Load .env file from prod_env directory
    env_file = Path(__file__).parent.parent / "prod_env" / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"✓ Loaded .env from: {env_file}")
    else:
        print(f"✗ Warning: .env file not found at {env_file}")
except ImportError:
    print("✗ Warning: python-dotenv not installed, skipping .env loading")
except Exception as e:
    print(f"✗ Warning: Failed to load .env: {e}")

# Debug: Check if API key was loaded
_api_key = os.getenv("OPENROUTER_API_KEY")
if _api_key:
    print(f"✓ API key loaded successfully (length: {len(_api_key)} characters)")
else:
    print("✗ API key NOT found in environment")
    print(f"  Looked in: {Path(__file__).parent.parent / 'prod_env' / '.env'}")

# Skip all tests if API key is not available
pytestmark = pytest.mark.skipif(
    not _api_key,
    reason="OPENROUTER_API_KEY not set - skipping integration tests"
)

FREE_MODEL = "arcee-ai/trinity-mini:free"
TIMEOUT = 60.0  # 60 second timeout for API calls

class TestIntegration:
    """Minimal integration tests with real API calls."""

    def test_basic_generation(self):
        """Test basic text generation."""
        client = OpenRouterClient(
            model=FREE_MODEL,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            max_tokens=50,
            timeout=TIMEOUT,
        )
        
        response = client.generate([
            {"role": "user", "content": "Say hello in one sentence."}
        ])
        
        assert "choices" in response
        assert len(response["choices"]) > 0
        message = response["choices"][0]["message"]
        
        # Free models sometimes use "reasoning" field instead of "content"
        content = message.get("content") or message.get("reasoning", "")
        assert content is not None
        assert len(content) > 0, f"Got empty response. Full message: {message}"
        print(f"Response: {content}")

    def test_with_parameter_overrides(self):
        """Test that parameter overrides work."""
        client = OpenRouterClient(
            model=FREE_MODEL,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            temperature=0.1, 
            max_tokens=30,
            timeout=TIMEOUT,
        )
        
        response = client.generate([
            {"role": "user", "content": "Count to 3."}
        ])
        
        content = response["choices"][0]["message"]["content"]
        assert content is not None
        print(f"Response: {content}")

    def test_method_level_override(self):
        """Test that method-level parameter overrides work."""
        client = OpenRouterClient(
            model=FREE_MODEL,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            temperature=0.9,  # Client default
            max_tokens=30,
            timeout=TIMEOUT,
        )
        
        # Override in method call
        response = client.generate(
            [{"role": "user", "content": "Say 'test'"}],
            temperature=0.1,  # Method override
        )
        
        content = response["choices"][0]["message"]["content"]
        assert content is not None
        print(f"Response: {content}")
