"""
Tests for standalone OpenRouterClient.

Tests the parameter override logic, default value application, provider routing,
and YAML configuration loading. Uses mocks for OpenAI API calls.

Run with: pytest test_openrouter_client.py -v
"""

import os
import pytest
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch, Mock

# OpenAI patching is handled in conftest.py which pytest loads first
from clients.openrouter_client import (
    OpenRouterClient,
    OpenRouterDefaults,
    DEFAULTS,
    ProviderPreferences,
    load_model_deployments,
    load_model_config,
    get_model_client,
)

# =============================================================================
# Test Fixtures
# =============================================================================

# OpenAI is already patched at module level above

@pytest.fixture
def api_key():
    """Test API key."""
    return "test_api_key_12345"


@pytest.fixture
def test_model():
    """Test model name."""
    return "z-ai/glm-4.5-air:free"


@pytest.fixture
def client(test_model, api_key):
    """Create a basic OpenRouterClient for testing."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": api_key}):
        return OpenRouterClient(
            model=test_model,
            api_key=api_key,
        )


@pytest.fixture
def sample_messages():
    """Sample messages for testing."""
    return [
        {"role": "user", "content": "Hello, world!"}
    ]


@pytest.fixture
def mock_yaml_content():
    """Sample YAML content for testing."""
    return """
models:
  - name: test-model
    model: openai/gpt-4
    temperature: 0.7
    top_p: 0.95
    seed: 42069
  - name: test-model-2
    model: anthropic/claude-3-opus
    temperature: 1.0
    provider:
      sort: price
"""


# =============================================================================
# Category 1: Client Initialization
# =============================================================================

class TestClientInitialization:
    """Tests for client initialization."""

    def test_basic_initialization(self, test_model, api_key):
        """Client should initialize with model and API key."""
        client = OpenRouterClient(
            model=test_model,
            api_key=api_key,
        )

        assert client.model == test_model
        assert client._api_key == api_key
        assert hasattr(client, '_defaults')  # Client is properly initialized

    def test_api_key_from_env_var(self, test_model, monkeypatch):
        """API key should be read from environment variable."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "env_key")
        
        client = OpenRouterClient(model=test_model)
        
        assert client._api_key == "env_key"

    def test_missing_api_key_raises_error(self, test_model, monkeypatch):
        """Client initialization should fail without API key."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        
        with pytest.raises(ValueError, match="OpenRouter API key not provided"):
            OpenRouterClient(model=test_model)

    def test_api_key_priority_explicit_over_env(self, test_model, monkeypatch):
        """Explicit api_key parameter should override env var."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "env_key")
        
        client = OpenRouterClient(
            model=test_model,
            api_key="explicit_key",
        )
        
        assert client._api_key == "explicit_key"

    def test_default_headers_passed_to_client(self, test_model, api_key):
        """Custom headers should be stored on the client."""
        headers = {"x-anthropic-beta": "fine-grained-tool-streaming-2025-05-14"}

        client = OpenRouterClient(
            model=test_model,
            api_key=api_key,
            default_headers=headers,
        )

        # Verify headers are stored correctly
        assert client.default_headers == headers


# =============================================================================
# Category 2: Default Value Application
# =============================================================================

class TestDefaultValues:
    """Tests for default value application."""

    def test_defaults_applied_when_not_overridden(self, client):
        """OpenRouter-specific defaults should be applied."""
        params = client._build_request_params()
        
        assert params["seed"] == DEFAULTS.seed  # 42069
        assert params["repetition_penalty"] == DEFAULTS.repetition_penalty  # 1.0
        assert params["min_p"] == DEFAULTS.min_p  # 0.0
        assert params["top_a"] == DEFAULTS.top_a  # 0.0
        assert params["verbosity"] == DEFAULTS.verbosity  # "medium"
        assert params["temperature"] == DEFAULTS.temperature  # 1.0
        assert params["top_p"] == DEFAULTS.top_p  # 1.0
        assert params["max_tokens"] == DEFAULTS.max_tokens  # 1000

    def test_provider_defaults_applied(self, client):
        """Provider defaults should be applied."""
        params = client._build_request_params()
        
        assert "provider" in params
        assert params["provider"]["require_parameters"] == False
        assert params["provider"]["quantizations"] == ["fp8", "fp16", "bf16"]

    def test_defaults_dataclass_is_frozen(self):
        """OpenRouterDefaults should be immutable."""
        with pytest.raises(Exception):  # FrozenInstanceError
            DEFAULTS.seed = 999


# =============================================================================
# Category 3: Parameter Override Behavior
# =============================================================================

class TestParameterOverrides:
    """Tests for parameter override behavior."""

    def test_client_override_takes_precedence_over_defaults(self, test_model, api_key):
        """Client temperature=0.5 should override default temperature=1.0."""
        client = OpenRouterClient(
            model=test_model,
            api_key=api_key,
            temperature=0.5,
        )
        
        params = client._build_request_params()
        
        assert params["temperature"] == 0.5  # Client override wins

    def test_method_override_takes_precedence_over_client_defaults(
        self, client
    ):
        """Method call override should take highest precedence."""
        params = client._build_request_params(temperature=0.8)
        
        assert params["temperature"] == 0.8

    def test_all_parameters_can_be_overridden(self, test_model, api_key):
        """All parameters can be overridden via client init."""
        client = OpenRouterClient(
            model=test_model,
            api_key=api_key,
            temperature=0.5,
            top_p=0.9,
            max_tokens=500,
            frequency_penalty=0.3,
            presence_penalty=0.4,
            stop=["STOP", "END"],
        )
        
        params = client._build_request_params()
        
        assert params["temperature"] == 0.5
        assert params["top_p"] == 0.9
        assert params["max_tokens"] == 500
        assert params["frequency_penalty"] == 0.3
        assert params["presence_penalty"] == 0.4
        assert params["stop"] == ["STOP", "END"]

    def test_zero_values_are_valid_overrides(self, test_model, api_key):
        """temperature=0.0 should be valid, not treated as 'not set'."""
        client = OpenRouterClient(
            model=test_model,
            api_key=api_key,
            temperature=0.0,
            frequency_penalty=0.0,
        )
        
        params = client._build_request_params()
        
        assert params["temperature"] == 0.0
        assert params["frequency_penalty"] == 0.0

    def test_none_values_not_included_in_params(self, test_model, api_key):
        """None values should not appear in final params."""
        client = OpenRouterClient(
            model=test_model,
            api_key=api_key,
            temperature=0.7,
            # top_p not set (will be None)
        )
        
        params = client._build_request_params()
        
        # top_p should use default, not be None
        assert params["top_p"] == DEFAULTS.top_p
        assert "top_p" in params


# =============================================================================
# Category 4: Provider Routing Configuration
# =============================================================================

class TestProviderRouting:
    """Tests for provider routing configuration."""

    def test_provider_config_always_present(self, client):
        """provider dict should always be in params."""
        params = client._build_request_params()
        
        assert "provider" in params
        assert params["provider"] == {
            "require_parameters": False,
            "quantizations": ["fp8", "fp16", "bf16"],
        }

    def test_provider_defaults_merged_with_user_config(self, test_model, api_key):
        """User provider config should merge with defaults, not replace them."""
        client = OpenRouterClient(
            model=test_model,
            api_key=api_key,
            provider={"sort": "price"},
        )
        
        params = client._build_request_params()
        
        assert params["provider"]["sort"] == "price"  # User value
        assert params["provider"]["require_parameters"] == False  # Default preserved
        assert params["provider"]["quantizations"] == ["fp8", "fp16", "bf16"]  # Default preserved

    def test_provider_defaults_can_be_overridden(self, test_model, api_key):
        """User config should be able to override default values."""
        client = OpenRouterClient(
            model=test_model,
            api_key=api_key,
            provider={
                "require_parameters": True,
                "quantizations": ["fp8"],
            },
        )
        
        params = client._build_request_params()
        
        assert params["provider"]["require_parameters"] == True
        assert params["provider"]["quantizations"] == ["fp8"]

    def test_all_provider_routing_options_supported(self, test_model, api_key):
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
            model=test_model,
            api_key=api_key,
            provider=comprehensive_provider,
        )
        
        params = client._build_request_params()
        
        assert params["provider"]["order"] == ["anthropic", "openai"]
        assert params["provider"]["allow_fallbacks"] == False
        assert params["provider"]["require_parameters"] == True
        assert params["provider"]["data_collection"] == "deny"
        assert params["provider"]["zdr"] == True
        assert params["provider"]["enforce_distillable_text"] == True
        assert params["provider"]["only"] == ["anthropic"]
        assert params["provider"]["ignore"] == ["openai"]
        assert params["provider"]["quantizations"] == ["fp16"]
        assert params["provider"]["sort"] == "throughput"
        assert params["provider"]["max_price"] == {"prompt": 1.0, "completion": 2.0}

    def test_provider_override_in_method_call(self, client):
        """Provider config can be overridden in method call."""
        params = client._build_request_params(
            provider={"sort": "latency"}
        )
        
        # Should merge with defaults
        assert params["provider"]["sort"] == "latency"
        assert params["provider"]["require_parameters"] == False  # Default preserved


# =============================================================================
# Category 5: OpenRouter-Specific Parameters
# =============================================================================

class TestOpenRouterSpecificParameters:
    """Tests for OpenRouter-specific sampling parameters."""

    def test_openrouter_parameters_always_present(self, client):
        """OpenRouter-specific params should always be present with defaults."""
        params = client._build_request_params()
        
        # All OpenRouter-specific params should exist
        assert "top_k" in params
        assert "repetition_penalty" in params
        assert "min_p" in params
        assert "top_a" in params
        assert "seed" in params
        assert "verbosity" in params

    def test_openrouter_parameters_can_be_overridden(self, test_model, api_key):
        """Each OpenRouter-specific parameter can be set via client init."""
        client = OpenRouterClient(
            model=test_model,
            api_key=api_key,
            top_k=50,
            repetition_penalty=1.2,
            min_p=0.05,
            top_a=0.1,
            seed=12345,
            verbosity="high",
        )
        
        params = client._build_request_params()
        
        assert params["top_k"] == 50
        assert params["repetition_penalty"] == 1.2
        assert params["min_p"] == 0.05
        assert params["top_a"] == 0.1
        assert params["seed"] == 12345
        assert params["verbosity"] == "high"


# =============================================================================
# Category 6: Tool Configuration
# =============================================================================

class TestToolConfiguration:
    """Tests for tool use configuration."""

    def test_tools_not_added_when_not_provided(self, client):
        """Tool-related params should not appear when tools are not provided."""
        params = client._build_request_params()
        
        assert "tools" not in params
        assert "tool_choice" not in params
        assert "parallel_tool_calls" not in params

    def test_tools_configuration_when_provided(self, test_model, api_key):
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
            model=test_model,
            api_key=api_key,
            tools=test_tools,
            tool_choice="auto",
            parallel_tool_calls=False,
        )
        
        params = client._build_request_params()
        
        assert params["tools"] == test_tools
        assert params["tool_choice"] == "auto"
        assert params["parallel_tool_calls"] == False

    def test_parallel_tool_calls_default(self, test_model, api_key):
        """parallel_tool_calls should default to True when tools are provided."""
        test_tools = [{"type": "function", "function": {"name": "test", "parameters": {}}}]
        
        client = OpenRouterClient(
            model=test_model,
            api_key=api_key,
            tools=test_tools,
            # No parallel_tool_calls specified
        )
        
        params = client._build_request_params()
        
        assert params["parallel_tool_calls"] == True  # Default


# =============================================================================
# Category 7: Generate Method
# =============================================================================

class TestGenerateMethod:
    """Tests for the generate() method."""

    @patch('clients.openrouter_client._get_requests')
    def test_generate_basic_call(self, mock_get_requests, sample_messages):
        """generate() should call OpenRouter API with correct parameters."""
        # Setup mock
        mock_requests = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello!"}}]
        }
        mock_requests.post.return_value = mock_response
        mock_get_requests.return_value = mock_requests

        # Create client and generate response
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test"}):
            client = OpenRouterClient(model="test-model", api_key="test")
            response = client.generate(sample_messages)

        # Verify HTTP request was made
        mock_requests.post.assert_called_once()
        call_args = mock_requests.post.call_args

        # Verify URL
        assert call_args.args[0] == "https://openrouter.ai/api/v1/chat/completions"

        # Verify headers
        headers = call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer test"
        assert headers["Content-Type"] == "application/json"

        # Verify JSON payload
        payload = call_args.kwargs["json"]
        assert payload["model"] == "test-model"
        assert payload["messages"] == sample_messages
        assert payload["stream"] == False
        assert "provider" in payload

    @patch('clients.openrouter_client._get_requests')
    def test_generate_with_overrides(self, mock_get_requests, sample_messages):
        """generate() should apply parameter overrides."""
        mock_requests = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "Hello!"}}]}
        mock_requests.post.return_value = mock_response
        mock_get_requests.return_value = mock_requests

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test"}):
            client = OpenRouterClient(
                model="test-model",
                api_key="test",
                temperature=0.7,
            )
            client.generate(sample_messages, temperature=0.9)

        call_args = mock_requests.post.call_args
        payload = call_args.kwargs["json"]
        assert payload["temperature"] == 0.9  # Override wins

    @patch('clients.openrouter_client._get_requests')
    def test_generate_streaming(self, mock_get_requests, sample_messages):
        """generate() should support streaming."""
        mock_requests = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "Hello!"}}]}
        mock_requests.post.return_value = mock_response
        mock_get_requests.return_value = mock_requests

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test"}):
            client = OpenRouterClient(model="test-model", api_key="test")
            client.generate(sample_messages, stream=True)

        call_args = mock_requests.post.call_args
        payload = call_args.kwargs["json"]
        assert payload["stream"] == True


# =============================================================================
# Category 8: YAML Configuration Loading
# =============================================================================

class TestYamlLoading:
    """Tests for YAML configuration loading."""

    def test_load_model_deployments(self, mock_yaml_content, tmp_path):
        """load_model_deployments() should load YAML file correctly."""
        yaml_file = tmp_path / "model_deployments.yaml"
        yaml_file.write_text(mock_yaml_content)
        
        configs = load_model_deployments(str(yaml_file))
        
        assert "test-model" in configs
        assert "test-model-2" in configs
        assert configs["test-model"]["model"] == "openai/gpt-4"
        assert configs["test-model"]["temperature"] == 0.7

    def test_load_model_deployments_default_path(self, mock_yaml_content, tmp_path, monkeypatch):
        """load_model_deployments() should use default path if not specified."""
        # Create prod_env directory structure
        prod_env = tmp_path / "prod_env"
        prod_env.mkdir()
        yaml_file = prod_env / "model_deployments.yaml"
        yaml_file.write_text(mock_yaml_content)
        
        # Mock __file__ to point to a file in tmp_path
        import clients.openrouter_client as or_module
        original_file = or_module.__file__
        try:
            # Create a fake __file__ path that will resolve to tmp_path
            fake_file = str(tmp_path / "clients" / "openrouter_client.py")
            or_module.__file__ = fake_file
            
            configs = load_model_deployments()
            
            assert "test-model" in configs
        finally:
            or_module.__file__ = original_file

    def test_load_model_config(self, mock_yaml_content, tmp_path):
        """load_model_config() should load specific model config."""
        yaml_file = tmp_path / "model_deployments.yaml"
        yaml_file.write_text(mock_yaml_content)
        
        config = load_model_config("test-model", str(yaml_file))
        
        assert config["model"] == "openai/gpt-4"
        assert config["temperature"] == 0.7
        assert config["seed"] == 42069

    def test_load_model_config_missing_model(self, mock_yaml_content, tmp_path):
        """load_model_config() should raise error for missing model."""
        yaml_file = tmp_path / "model_deployments.yaml"
        yaml_file.write_text(mock_yaml_content)
        
        with pytest.raises(ValueError, match="Model 'missing-model' not found"):
            load_model_config("missing-model", str(yaml_file))

    @patch('clients.openrouter_client.OpenRouterClient')
    def test_get_model_client(self, mock_client_class, mock_yaml_content, tmp_path):
        """get_model_client() should create configured client."""
        yaml_file = tmp_path / "model_deployments.yaml"
        yaml_file.write_text(mock_yaml_content)
        
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance
        
        client = get_model_client("test-model", "test-key", str(yaml_file))
        
        # Verify client was created with correct parameters
        mock_client_class.assert_called_once()
        call_kwargs = mock_client_class.call_args.kwargs
        
        assert call_kwargs["model"] == "openai/gpt-4"
        assert call_kwargs["api_key"] == "test-key"
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["seed"] == 42069

    @patch('clients.openrouter_client.OpenRouterClient')
    def test_get_model_client_with_provider(self, mock_client_class, mock_yaml_content, tmp_path):
        """get_model_client() should handle provider config."""
        yaml_file = tmp_path / "model_deployments.yaml"
        yaml_file.write_text(mock_yaml_content)
        
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance
        
        client = get_model_client("test-model-2", "test-key", str(yaml_file))
        
        call_kwargs = mock_client_class.call_args.kwargs
        assert call_kwargs["provider"]["sort"] == "price"

    @patch('clients.openrouter_client.OpenRouterClient')
    def test_get_model_client_with_overrides(self, mock_client_class, mock_yaml_content, tmp_path):
        """get_model_client() should allow parameter overrides."""
        yaml_file = tmp_path / "model_deployments.yaml"
        yaml_file.write_text(mock_yaml_content)
        
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance
        
        client = get_model_client(
            "test-model",
            "test-key",
            str(yaml_file),
            temperature=0.9,  # Override YAML value
        )
        
        call_kwargs = mock_client_class.call_args.kwargs
        assert call_kwargs["temperature"] == 0.9  # Override wins


# =============================================================================
# Category 9: Real-World Configuration Scenarios
# =============================================================================

class TestRealWorldScenarios:
    """Tests for realistic configuration scenarios."""

    def test_minimal_configuration(self, test_model, api_key):
        """Client should work with minimal config."""
        client = OpenRouterClient(
            model=test_model,
            api_key=api_key,
        )
        
        params = client._build_request_params()
        
        # Basic structure should be valid
        assert "model" not in params  # Model is added in generate()
        assert "provider" in params
        assert params["provider"]["require_parameters"] == False

    def test_full_configuration(self, test_model, api_key):
        """All parameters can be set simultaneously."""
        client = OpenRouterClient(
            model=test_model,
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
        
        params = client._build_request_params()
        
        # Verify all parameters set correctly
        assert params["temperature"] == 0.7
        assert params["top_p"] == 0.9
        assert params["top_k"] == 50
        assert params["frequency_penalty"] == 0.1
        assert params["presence_penalty"] == 0.2
        assert params["repetition_penalty"] == 1.1
        assert params["min_p"] == 0.05
        assert params["top_a"] == 0.1
        assert params["seed"] == 12345
        assert params["max_tokens"] == 2000
        assert params["verbosity"] == "high"
        assert params["stop"] == ["END"]
        assert params["provider"]["sort"] == "price"
        assert params["provider"]["data_collection"] == "deny"


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
