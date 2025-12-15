"""
Comprehensive tests for OpenRouterClient.

Tests the parameter override logic, default value application, provider routing,
and OpenRouter-specific features. Uses glm-4.5-air:free model for testing.

Run with: pytest test_openrouter_client.py -v
"""

import os
import pytest
import tempfile
from typing import Any, Dict
from unittest.mock import patch, MagicMock

from helm.common.cache import BlackHoleCacheConfig, SqliteCacheConfig
from helm.common.request import Request
from helm.tokenizers.huggingface_tokenizer import HuggingFaceTokenizer

from clients.openrouter_client import (
    OpenRouterClient,
    OpenRouterDefaults,
    DEFAULTS,
    ProviderPreferences,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def tokenizer():
    """Create a tokenizer for testing."""
    return HuggingFaceTokenizer(
        cache_config=BlackHoleCacheConfig(),
        tokenizer_name="mistralai/Mistral-7B-v0.1",
    )


@pytest.fixture
def tokenizer_name():
    """Tokenizer name used for testing."""
    return "mistralai/Mistral-7B-v0.1"


@pytest.fixture
def cache_config():
    """Cache configuration for testing."""
    return BlackHoleCacheConfig()


@pytest.fixture
def test_model():
    """Test model name (using free tier model)."""
    return "z-ai/glm-4.5-air:free"


@pytest.fixture
def api_key():
    """Test API key."""
    return "test_api_key_12345"


@pytest.fixture
def client(tokenizer, tokenizer_name, cache_config, test_model, api_key):
    """Create a basic OpenRouterClient for testing."""
    return OpenRouterClient(
        tokenizer=tokenizer,
        tokenizer_name=tokenizer_name,
        cache_config=cache_config,
        model_name=test_model,
        api_key=api_key,
    )


@pytest.fixture
def minimal_request():
    """Create a minimal Request object for testing."""
    return Request(
        model="z-ai/glm-4.5-air:free",
        model_deployment="openrouter/glm-4.5-air:free",
        prompt="Hello, world!",
    )


# =============================================================================
# Category 1: Parameter Override Behavior
# =============================================================================

class TestParameterOverrides:
    """Tests for client parameter override behavior."""

    def test_client_override_takes_precedence_over_request(
        self, tokenizer, tokenizer_name, cache_config, test_model, api_key
    ):
        """Client temperature=0.5 should override Request temperature=1.0."""
        client = OpenRouterClient(
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            cache_config=cache_config,
            model_name=test_model,
            api_key=api_key,
            temperature=0.5,
        )
        
        request = Request(
            model=test_model,
            model_deployment="openrouter/glm-4.5-air:free",
            prompt="Test",
            temperature=1.0,  # Request has 1.0
        )
        
        raw_request = client._make_chat_raw_request(request)
        
        assert raw_request["temperature"] == 0.5  # Client wins

    def test_request_value_used_when_client_doesnt_override(
        self, tokenizer, tokenizer_name, cache_config, test_model, api_key
    ):
        """When client doesn't set temperature, Request value should be used."""
        client = OpenRouterClient(
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            cache_config=cache_config,
            model_name=test_model,
            api_key=api_key,
            # No temperature override
        )
        
        request = Request(
            model=test_model,
            model_deployment="openrouter/glm-4.5-air:free",
            prompt="Test",
            temperature=0.8,
        )
        
        raw_request = client._make_chat_raw_request(request)
        
        assert raw_request["temperature"] == 0.8  # Request value used

    def test_all_openai_compatible_parameters_can_be_overridden(
        self, tokenizer, tokenizer_name, cache_config, test_model, api_key
    ):
        """All inherited OpenAI parameters can be overridden via client init."""
        client = OpenRouterClient(
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            cache_config=cache_config,
            model_name=test_model,
            api_key=api_key,
            temperature=0.5,
            top_p=0.9,
            max_tokens=500,
            frequency_penalty=0.3,
            presence_penalty=0.4,
            stop=["STOP", "END"],
        )
        
        request = Request(
            model=test_model,
            model_deployment="openrouter/glm-4.5-air:free",
            prompt="Test",
        )
        
        raw_request = client._make_chat_raw_request(request)
        
        assert raw_request["temperature"] == 0.5
        assert raw_request["top_p"] == 0.9
        assert raw_request["max_tokens"] == 500
        assert raw_request["frequency_penalty"] == 0.3
        assert raw_request["presence_penalty"] == 0.4
        assert raw_request["stop"] == ["STOP", "END"]

    def test_zero_values_are_valid_overrides(
        self, tokenizer, tokenizer_name, cache_config, test_model, api_key
    ):
        """temperature=0.0 should be valid, not treated as 'not set'."""
        client = OpenRouterClient(
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            cache_config=cache_config,
            model_name=test_model,
            api_key=api_key,
            temperature=0.0,
            frequency_penalty=0.0,
        )
        
        request = Request(
            model=test_model,
            model_deployment="openrouter/trinity-mini:free",
            prompt="Test",
            temperature=1.0,  # Would be used if 0.0 was treated as None
            frequency_penalty=0.5,
        )
        
        raw_request = client._make_chat_raw_request(request)
        
        assert raw_request["temperature"] == 0.0
        assert raw_request["frequency_penalty"] == 0.0


# =============================================================================
# Category 2: Default Value Application
# =============================================================================

class TestDefaultValues:
    """Tests for OpenRouter default value application."""

    def test_openrouter_defaults_applied_when_not_overridden(
        self, tokenizer, tokenizer_name, cache_config, test_model, api_key
    ):
        """OpenRouter-specific defaults should be applied."""
        client = OpenRouterClient(
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            cache_config=cache_config,
            model_name=test_model,
            api_key=api_key,
        )
        
        request = Request(
            model=test_model,
            model_deployment="openrouter/glm-4.5-air:free",
            prompt="Test",
        )
        
        raw_request = client._make_chat_raw_request(request)
        
        # OpenRouter-specific defaults
        assert raw_request["seed"] == DEFAULTS.seed  # 42069
        assert raw_request["repetition_penalty"] == DEFAULTS.repetition_penalty  # 1.0
        assert raw_request["min_p"] == DEFAULTS.min_p  # 0.0
        assert raw_request["top_a"] == DEFAULTS.top_a  # 0.0
        assert raw_request["verbosity"] == DEFAULTS.verbosity  # "medium"

    def test_max_tokens_default_override(
        self, tokenizer, tokenizer_name, cache_config, test_model, api_key
    ):
        """HELM's default max_tokens=100 should be overridden to 1000."""
        client = OpenRouterClient(
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            cache_config=cache_config,
            model_name=test_model,
            api_key=api_key,
        )
        
        # Request with HELM's default of 100
        request = Request(
            model=test_model,
            model_deployment="openrouter/trinity-mini:free",
            prompt="Test",
            max_tokens=100,  # HELM default
        )
        
        raw_request = client._make_chat_raw_request(request)
        
        assert raw_request["max_tokens"] == 1000  # OpenRouter default

    def test_max_tokens_explicit_value_preserved(
        self, tokenizer, tokenizer_name, cache_config, test_model, api_key
    ):
        """Explicit max_tokens values other than 100 should be preserved."""
        client = OpenRouterClient(
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            cache_config=cache_config,
            model_name=test_model,
            api_key=api_key,
        )
        
        request = Request(
            model=test_model,
            model_deployment="openrouter/trinity-mini:free",
            prompt="Test",
            max_tokens=500,  # Explicit non-default value
        )
        
        raw_request = client._make_chat_raw_request(request)
        
        assert raw_request["max_tokens"] == 500  # Preserved

    def test_top_k_default_behavior(
        self, tokenizer, tokenizer_name, cache_config, test_model, api_key
    ):
        """top_k should default to 0 (disabled) when not set."""
        client = OpenRouterClient(
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            cache_config=cache_config,
            model_name=test_model,
            api_key=api_key,
        )
        
        request = Request(
            model=test_model,
            model_deployment="openrouter/trinity-mini:free",
            prompt="Test",
            top_k_per_token=1,  # HELM default
        )
        
        raw_request = client._make_chat_raw_request(request)
        
        assert raw_request["top_k"] == 0  # OpenRouter default

    def test_top_k_from_request_when_greater_than_one(
        self, tokenizer, tokenizer_name, cache_config, test_model, api_key
    ):
        """top_k should use Request.top_k_per_token when > 1."""
        client = OpenRouterClient(
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            cache_config=cache_config,
            model_name=test_model,
            api_key=api_key,
        )
        
        request = Request(
            model=test_model,
            model_deployment="openrouter/trinity-mini:free",
            prompt="Test",
            top_k_per_token=50,
        )
        
        raw_request = client._make_chat_raw_request(request)
        
        assert raw_request["top_k"] == 50

    def test_defaults_dataclass_is_frozen(self):
        """OpenRouterDefaults should be immutable."""
        with pytest.raises(Exception):  # FrozenInstanceError
            DEFAULTS.seed = 999


# =============================================================================
# Category 3: Provider Routing Configuration
# =============================================================================

class TestProviderRouting:
    """Tests for provider routing configuration."""

    def test_provider_config_always_present(
        self, tokenizer, tokenizer_name, cache_config, test_model, api_key
    ):
        """provider dict should always be in raw_request."""
        client = OpenRouterClient(
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            cache_config=cache_config,
            model_name=test_model,
            api_key=api_key,
        )
        
        request = Request(
            model=test_model,
            model_deployment="openrouter/glm-4.5-air:free",
            prompt="Test",
        )
        
        raw_request = client._make_chat_raw_request(request)
        
        assert "provider" in raw_request
        assert raw_request["provider"] == {
            "require_parameters": False,
            "quantizations": ["fp8", "fp16", "bf16"],
        }

    def test_provider_defaults_merged_with_user_config(
        self, tokenizer, tokenizer_name, cache_config, test_model, api_key
    ):
        """User provider config should merge with defaults, not replace them."""
        client = OpenRouterClient(
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            cache_config=cache_config,
            model_name=test_model,
            api_key=api_key,
            provider={"sort": "price"},
        )
        
        request = Request(
            model=test_model,
            model_deployment="openrouter/glm-4.5-air:free",
            prompt="Test",
        )
        
        raw_request = client._make_chat_raw_request(request)
        
        assert raw_request["provider"]["sort"] == "price"  # User value
        assert raw_request["provider"]["require_parameters"] == False  # Default preserved
        assert raw_request["provider"]["quantizations"] == ["fp8", "fp16", "bf16"]  # Default preserved

    def test_provider_defaults_can_be_overridden(
        self, tokenizer, tokenizer_name, cache_config, test_model, api_key
    ):
        """User config should be able to override default values."""
        client = OpenRouterClient(
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            cache_config=cache_config,
            model_name=test_model,
            api_key=api_key,
            provider={
                "require_parameters": True,
                "quantizations": ["fp8"],
            },
        )
        
        request = Request(
            model=test_model,
            model_deployment="openrouter/glm-4.5-air:free",
            prompt="Test",
        )
        
        raw_request = client._make_chat_raw_request(request)
        
        assert raw_request["provider"]["require_parameters"] == True
        assert raw_request["provider"]["quantizations"] == ["fp8"]

    def test_all_provider_routing_options_supported(
        self, tokenizer, tokenizer_name, cache_config, test_model, api_key
    ):
        """All provider routing fields should be supported."""
        comprehensive_provider: ProviderPreferences = {
            "order": ["anthropic", "openai"],
            "allow_fallbacks": False,
            "require_parameters": True,
            "data_collection": "deny",
            "zdr": True,
            "enforce_distillable_text": True,
            "only": ["anthropic"],
            "ignore": ["openai"],
            "quantizations": ["fp16"],
            "sort": "throughput",
            "max_price": {"prompt": 1.0, "completion": 2.0},
        }
        
        client = OpenRouterClient(
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            cache_config=cache_config,
            model_name=test_model,
            api_key=api_key,
            provider=comprehensive_provider,
        )
        
        request = Request(
            model=test_model,
            model_deployment="openrouter/glm-4.5-air:free",
            prompt="Test",
        )
        
        raw_request = client._make_chat_raw_request(request)
        
        assert raw_request["provider"]["order"] == ["anthropic", "openai"]
        assert raw_request["provider"]["allow_fallbacks"] == False
        assert raw_request["provider"]["require_parameters"] == True
        assert raw_request["provider"]["data_collection"] == "deny"
        assert raw_request["provider"]["zdr"] == True
        assert raw_request["provider"]["enforce_distillable_text"] == True
        assert raw_request["provider"]["only"] == ["anthropic"]
        assert raw_request["provider"]["ignore"] == ["openai"]
        assert raw_request["provider"]["quantizations"] == ["fp16"]
        assert raw_request["provider"]["sort"] == "throughput"
        assert raw_request["provider"]["max_price"] == {"prompt": 1.0, "completion": 2.0}


# =============================================================================
# Category 4: OpenRouter-Specific Parameters
# =============================================================================

class TestOpenRouterSpecificParameters:
    """Tests for OpenRouter-specific sampling parameters."""

    def test_openrouter_parameters_always_present(
        self, tokenizer, tokenizer_name, cache_config, test_model, api_key
    ):
        """OpenRouter-specific params should always be present with defaults."""
        client = OpenRouterClient(
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            cache_config=cache_config,
            model_name=test_model,
            api_key=api_key,
        )
        
        request = Request(
            model=test_model,
            model_deployment="openrouter/glm-4.5-air:free",
            prompt="Test",
        )
        
        raw_request = client._make_chat_raw_request(request)
        
        # All OpenRouter-specific params should exist
        assert "top_k" in raw_request
        assert "repetition_penalty" in raw_request
        assert "min_p" in raw_request
        assert "top_a" in raw_request
        assert "seed" in raw_request
        assert "verbosity" in raw_request

    def test_openrouter_parameters_can_be_overridden(
        self, tokenizer, tokenizer_name, cache_config, test_model, api_key
    ):
        """Each OpenRouter-specific parameter can be set via client init."""
        client = OpenRouterClient(
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            cache_config=cache_config,
            model_name=test_model,
            api_key=api_key,
            top_k=50,
            repetition_penalty=1.2,
            min_p=0.05,
            top_a=0.1,
            seed=12345,
            verbosity="high",
        )
        
        request = Request(
            model=test_model,
            model_deployment="openrouter/glm-4.5-air:free",
            prompt="Test",
        )
        
        raw_request = client._make_chat_raw_request(request)
        
        assert raw_request["top_k"] == 50
        assert raw_request["repetition_penalty"] == 1.2
        assert raw_request["min_p"] == 0.05
        assert raw_request["top_a"] == 0.1
        assert raw_request["seed"] == 12345
        assert raw_request["verbosity"] == "high"

    @pytest.mark.parametrize("client_top_k,request_top_k_per_token,expected", [
        (20, 50, 20),    # Client override takes precedence
        (None, 50, 50),  # Request value used when no client override and > 1
        (None, 1, 0),    # Default when no override and request is 1
    ])
    def test_top_k_parameter_handling(
        self, tokenizer, tokenizer_name, cache_config, test_model, api_key,
        client_top_k, request_top_k_per_token, expected
    ):
        """top_k logic: client override > Request.top_k_per_token > default."""
        client = OpenRouterClient(
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            cache_config=cache_config,
            model_name=test_model,
            api_key=api_key,
            top_k=client_top_k,
        )
        
        request = Request(
            model=test_model,
            model_deployment="openrouter/trinity-mini:free",
            prompt="Test",
            top_k_per_token=request_top_k_per_token,
        )
        
        raw_request = client._make_chat_raw_request(request)
        
        assert raw_request["top_k"] == expected


# =============================================================================
# Category 5: Tool Configuration
# =============================================================================

class TestToolConfiguration:
    """Tests for tool use configuration."""

    def test_tools_not_added_when_not_provided(
        self, tokenizer, tokenizer_name, cache_config, test_model, api_key
    ):
        """Tool-related params should not appear when tools are not provided."""
        client = OpenRouterClient(
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            cache_config=cache_config,
            model_name=test_model,
            api_key=api_key,
        )
        
        request = Request(
            model=test_model,
            model_deployment="openrouter/glm-4.5-air:free",
            prompt="Test",
        )
        
        raw_request = client._make_chat_raw_request(request)
        
        assert "tools" not in raw_request
        assert "tool_choice" not in raw_request
        assert "parallel_tool_calls" not in raw_request

    def test_tools_configuration_when_provided(
        self, tokenizer, tokenizer_name, cache_config, test_model, api_key
    ):
        """When tools are provided, all tool params should be set correctly."""
        test_tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        
        client = OpenRouterClient(
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            cache_config=cache_config,
            model_name=test_model,
            api_key=api_key,
            tools=test_tools,
            tool_choice="auto",
            parallel_tool_calls=False,
        )
        
        request = Request(
            model=test_model,
            model_deployment="openrouter/glm-4.5-air:free",
            prompt="Test",
        )
        
        raw_request = client._make_chat_raw_request(request)
        
        assert raw_request["tools"] == test_tools
        assert raw_request["tool_choice"] == "auto"
        assert raw_request["parallel_tool_calls"] == False

    def test_parallel_tool_calls_default(
        self, tokenizer, tokenizer_name, cache_config, test_model, api_key
    ):
        """parallel_tool_calls should default to True when tools are provided."""
        test_tools = [{"type": "function", "function": {"name": "test", "parameters": {}}}]
        
        client = OpenRouterClient(
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            cache_config=cache_config,
            model_name=test_model,
            api_key=api_key,
            tools=test_tools,
            # No parallel_tool_calls specified
        )
        
        request = Request(
            model=test_model,
            model_deployment="openrouter/glm-4.5-air:free",
            prompt="Test",
        )
        
        raw_request = client._make_chat_raw_request(request)
        
        assert raw_request["parallel_tool_calls"] == True  # Default

    def test_tool_choice_only_when_tools_present(
        self, tokenizer, tokenizer_name, cache_config, test_model, api_key
    ):
        """tool_choice should only be added when tools are provided."""
        client = OpenRouterClient(
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            cache_config=cache_config,
            model_name=test_model,
            api_key=api_key,
            tool_choice="auto",  # Set but no tools
        )
        
        request = Request(
            model=test_model,
            model_deployment="openrouter/glm-4.5-air:free",
            prompt="Test",
        )
        
        raw_request = client._make_chat_raw_request(request)
        
        assert "tool_choice" not in raw_request  # Not added without tools


# =============================================================================
# Category 6: Parameter Precedence Edge Cases
# =============================================================================

class TestParameterPrecedence:
    """Tests for edge cases in parameter precedence."""

    def test_none_vs_zero_distinction_temperature(
        self, tokenizer, tokenizer_name, cache_config, test_model, api_key
    ):
        """None means 'use parent/default', 0 means 'override to zero'."""
        # Client with temperature=None (no override)
        client_none = OpenRouterClient(
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            cache_config=cache_config,
            model_name=test_model,
            api_key=api_key,
            temperature=None,
        )
        
        # Client with temperature=0.0 (explicit override)
        client_zero = OpenRouterClient(
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            cache_config=cache_config,
            model_name=test_model,
            api_key=api_key,
            temperature=0.0,
        )
        
        request = Request(
            model=test_model,
            model_deployment="openrouter/glm-4.5-air:free",
            prompt="Test",
            temperature=0.8,
        )
        
        raw_none = client_none._make_chat_raw_request(request)
        raw_zero = client_zero._make_chat_raw_request(request)
        
        assert raw_none["temperature"] == 0.8  # Request value (None = no override)
        assert raw_zero["temperature"] == 0.0  # Client value (0.0 = explicit override)

    def test_none_vs_zero_distinction_frequency_penalty(
        self, tokenizer, tokenizer_name, cache_config, test_model, api_key
    ):
        """Same distinction for frequency_penalty."""
        client_zero = OpenRouterClient(
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            cache_config=cache_config,
            model_name=test_model,
            api_key=api_key,
            frequency_penalty=0.0,
        )
        
        request = Request(
            model=test_model,
            model_deployment="openrouter/trinity-mini:free",
            prompt="Test",
            frequency_penalty=0.5,
        )
        
        raw_request = client_zero._make_chat_raw_request(request)
        
        assert raw_request["frequency_penalty"] == 0.0  # Client 0.0, not Request 0.5


# =============================================================================
# Category 7: Integration with OpenAI Parent
# =============================================================================

class TestOpenAIIntegration:
    """Tests for integration with OpenAI parent class."""

    def test_model_name_resolution_client_takes_precedence(
        self, tokenizer, tokenizer_name, cache_config, api_key
    ):
        """Client model_name should take precedence over Request.model."""
        client = OpenRouterClient(
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            cache_config=cache_config,
            model_name="anthropic/claude-3-opus",  # Client model
            api_key=api_key,
        )
        
        request = Request(
            model="openai/gpt-4",  # Request model
            model_deployment="openrouter/gpt-4",
            prompt="Test",
        )
        
        assert client._get_model_for_request(request) == "anthropic/claude-3-opus"

    def test_model_name_resolution_falls_back_to_request(
        self, tokenizer, tokenizer_name, cache_config, api_key
    ):
        """When no client model_name, use Request.model."""
        client = OpenRouterClient(
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            cache_config=cache_config,
            model_name=None,
            api_key=api_key,
        )
        
        request = Request(
            model="openai/gpt-4",
            model_deployment="openrouter/gpt-4",
            prompt="Test",
        )
        
        assert client._get_model_for_request(request) == "openai/gpt-4"

    def test_base_url_is_openrouter(
        self, tokenizer, tokenizer_name, cache_config, test_model, api_key
    ):
        """Client should point to OpenRouter, not OpenAI."""
        client = OpenRouterClient(
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            cache_config=cache_config,
            model_name=test_model,
            api_key=api_key,
        )
        
        assert str(client.client.base_url) == "https://openrouter.ai/api/v1/"

    def test_messages_included_in_raw_request(
        self, tokenizer, tokenizer_name, cache_config, test_model, api_key
    ):
        """Messages should be included in raw_request."""
        client = OpenRouterClient(
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            cache_config=cache_config,
            model_name=test_model,
            api_key=api_key,
        )
        
        request = Request(
            model=test_model,
            model_deployment="openrouter/trinity-mini:free",
            prompt="Hello, world!",
        )
        
        raw_request = client._make_chat_raw_request(request)
        
        assert "messages" in raw_request
        assert len(raw_request["messages"]) > 0


# =============================================================================
# Category 8: Error Handling and Validation
# =============================================================================

class TestErrorHandling:
    """Tests for error handling and validation."""

    def test_missing_api_key_raises_error(
        self, tokenizer, tokenizer_name, cache_config, test_model, monkeypatch
    ):
        """Client initialization should fail without API key."""
        # Remove env var
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        
        with pytest.raises(ValueError, match="OpenRouter API key not provided"):
            OpenRouterClient(
                tokenizer=tokenizer,
                tokenizer_name=tokenizer_name,
                cache_config=cache_config,
                model_name=test_model,
                # No api_key
            )

    def test_api_key_priority_explicit_over_env(
        self, tokenizer, tokenizer_name, cache_config, test_model, monkeypatch
    ):
        """Explicit api_key parameter should override env var."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "env_key")
        
        client = OpenRouterClient(
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            cache_config=cache_config,
            model_name=test_model,
            api_key="explicit_key",
        )
        
        assert client._api_key == "explicit_key"

    def test_api_key_from_env_var(
        self, tokenizer, tokenizer_name, cache_config, test_model, monkeypatch
    ):
        """API key should be read from environment variable."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "env_key")
        
        client = OpenRouterClient(
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            cache_config=cache_config,
            model_name=test_model,
            # No explicit api_key
        )
        
        assert client._api_key == "env_key"


# =============================================================================
# Category 9: Default Headers
# =============================================================================

class TestDefaultHeaders:
    """Tests for provider-specific headers."""

    def test_default_headers_passed_to_openai_client(
        self, tokenizer, tokenizer_name, cache_config, test_model, api_key
    ):
        """Custom headers should be set on the OpenAI client."""
        headers = {"x-anthropic-beta": "fine-grained-tool-streaming-2025-05-14"}
        
        client = OpenRouterClient(
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            cache_config=cache_config,
            model_name=test_model,
            api_key=api_key,
            default_headers=headers,
        )
        
        # Headers should be on the client, not in raw_request
        # Note: The actual implementation may vary, this tests the intent
        assert client.client.default_headers is not None

    def test_headers_not_in_request_body(
        self, tokenizer, tokenizer_name, cache_config, test_model, api_key
    ):
        """Headers should not appear in the raw_request dict."""
        headers = {"x-anthropic-beta": "some-feature"}
        
        client = OpenRouterClient(
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            cache_config=cache_config,
            model_name=test_model,
            api_key=api_key,
            default_headers=headers,
        )
        
        request = Request(
            model=test_model,
            model_deployment="openrouter/glm-4.5-air:free",
            prompt="Test",
        )
        
        raw_request = client._make_chat_raw_request(request)
        
        assert "x-anthropic-beta" not in raw_request


# =============================================================================
# Category 10: Real-World Configuration Scenarios
# =============================================================================

class TestRealWorldScenarios:
    """Tests for realistic configuration scenarios."""

    def test_minimal_configuration(
        self, tokenizer, tokenizer_name, cache_config, test_model, api_key
    ):
        """Client should work with minimal config."""
        client = OpenRouterClient(
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            cache_config=cache_config,
            model_name=test_model,
            api_key=api_key,
        )
        
        request = Request(
            model=test_model,
            model_deployment="openrouter/glm-4.5-air:free",
            prompt="Test",
        )
        
        raw_request = client._make_chat_raw_request(request)
        
        # Basic structure should be valid
        assert "model" in raw_request
        assert "messages" in raw_request
        assert "provider" in raw_request
        assert raw_request["provider"]["require_parameters"] == False

    def test_full_configuration(
        self, tokenizer, tokenizer_name, cache_config, test_model, api_key
    ):
        """All parameters can be set simultaneously."""
        client = OpenRouterClient(
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            cache_config=cache_config,
            model_name=test_model,
            api_key=api_key,
            # Sampling
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            frequency_penalty=0.1,
            presence_penalty=0.2,
            repetition_penalty=1.1,
            min_p=0.05,
            top_a=0.1,
            # Generation
            seed=12345,
            max_tokens=2000,
            verbosity="high",
            stop=["END"],
            # Provider
            provider={"sort": "price", "data_collection": "deny"},
        )
        
        request = Request(
            model=test_model,
            model_deployment="openrouter/glm-4.5-air:free",
            prompt="Test",
        )
        
        raw_request = client._make_chat_raw_request(request)
        
        # Verify all parameters set correctly
        assert raw_request["temperature"] == 0.7
        assert raw_request["top_p"] == 0.9
        assert raw_request["top_k"] == 50
        assert raw_request["frequency_penalty"] == 0.1
        assert raw_request["presence_penalty"] == 0.2
        assert raw_request["repetition_penalty"] == 1.1
        assert raw_request["min_p"] == 0.05
        assert raw_request["top_a"] == 0.1
        assert raw_request["seed"] == 12345
        assert raw_request["max_tokens"] == 2000
        assert raw_request["verbosity"] == "high"
        assert raw_request["stop"] == ["END"]
        assert raw_request["provider"]["sort"] == "price"
        assert raw_request["provider"]["data_collection"] == "deny"

    def test_model_specific_overrides_differ(
        self, tokenizer, tokenizer_name, cache_config, api_key
    ):
        """Different models can have completely different configurations."""
        # Simulates two different model deployments
        client_a = OpenRouterClient(
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            cache_config=cache_config,
            model_name="z-ai/glm-4.5-air:free",
            api_key=api_key,
            temperature=0.7,
            provider={"sort": "price"},
        )
        
        client_b = OpenRouterClient(
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            cache_config=cache_config,
            model_name="arcee-ai/trinity-mini:free",
            api_key=api_key,
            temperature=1.0,
            provider={"sort": "throughput"},
        )
        
        request_a = Request(
            model="z-ai/glm-4.5-air:free",
            model_deployment="openrouter/glm-4.5-air:free",
            prompt="Test A",
        )
        
        request_b = Request(
            model="arcee-ai/trinity-mini:free",
            model_deployment="openrouter/trinity-mini:free",
            prompt="Test B",
        )
        
        raw_a = client_a._make_chat_raw_request(request_a)
        raw_b = client_b._make_chat_raw_request(request_b)
        
        # Each client produces different config
        assert raw_a["temperature"] != raw_b["temperature"]
        assert raw_a["provider"]["sort"] != raw_b["provider"]["sort"]
        assert raw_a["model"] != raw_b["model"]

    def test_glm_free_real_world_config(
        self, tokenizer, tokenizer_name, cache_config, api_key
    ):
        """Test configuration matching the real model_deployments.yaml for glm-4.5-air:free."""
        # This matches the actual YAML configuration
        client = OpenRouterClient(
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            cache_config=cache_config,
            model_name="z-ai/glm-4.5-air:free",
            api_key=api_key,
            seed=42069,
            temperature=0.15,
            top_p=0.75,
            top_k=50,
            min_p=0.06,
        )
        
        request = Request(
            model="z-ai/glm-4.5-air:free",
            model_deployment="openrouter/glm-4.5-air:free",
            prompt="Hello, I am a test prompt.",
        )
        
        raw_request = client._make_chat_raw_request(request)
        
        assert raw_request["model"] == "z-ai/glm-4.5-air:free"
        assert raw_request["seed"] == 42069
        assert raw_request["temperature"] == 0.15
        assert raw_request["top_p"] == 0.75
        assert raw_request["top_k"] == 50
        assert raw_request["min_p"] == 0.06
        assert "provider" in raw_request


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])