"""
Standalone OpenRouter Client

A simple client for OpenRouter's unified AI API that provides intelligent
provider routing across multiple model providers with built-in defaults and
easy parameter overrides.

References:
    - Parameters: https://openrouter.ai/docs/api/reference/parameters
    - Provider Routing: https://openrouter.ai/docs/guides/routing/provider-selection
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, TypedDict

# Note: We use requests directly instead of the OpenAI SDK to avoid import hang issues
# The OpenAI SDK can hang on import in some environments due to httpx/HTTP2 issues

# Lazy imports to avoid any module-level execution
_yaml = None
_requests = None

def _get_yaml():
    """Lazy import of yaml."""
    global _yaml
    if _yaml is None:
        try:
            import yaml
            _yaml = yaml
        except ImportError:
            raise ImportError(
                "pyyaml package is required. Install with: pip install pyyaml"
            )
    return _yaml

def _get_requests():
    """Lazy import of requests."""
    global _requests
    if _requests is None:
        try:
            import requests
            _requests = requests
        except ImportError:
            raise ImportError(
                "requests package is required. Install with: pip install requests"
            )
    return _requests


# =============================================================================
# Type Definitions
# =============================================================================

class ProviderPreferences(TypedDict, total=False):
    """
    OpenRouter provider routing configuration.
    
    Controls how requests are routed across providers for optimal cost,
    performance, and reliability.
    
    Reference: https://openrouter.ai/docs/guides/routing/provider-selection
    """
    order: List[str]
    """Provider slugs to try in order (e.g., ["anthropic", "openai"])."""
    
    allow_fallbacks: bool
    """Whether to allow backup providers when primary is unavailable. Default: True."""
    
    require_parameters: bool
    """Only use providers supporting all request parameters. Default: False."""
    
    data_collection: Literal["allow", "deny"]
    """Control whether to use providers that may store data. Default: "allow"."""
    
    zdr: bool
    """Restrict to Zero Data Retention endpoints only."""
    
    enforce_distillable_text: bool
    """Restrict to models allowing text distillation."""
    
    only: List[str]
    """Provider slugs to exclusively allow."""
    
    ignore: List[str]
    """Provider slugs to skip."""
    
    quantizations: List[str]
    """Quantization levels to filter by (e.g., ["fp8", "fp16", "bf16"])."""
    
    sort: Literal["price", "throughput", "latency"]
    """Sort providers by attribute. Disables load balancing when set."""
    
    max_price: Dict[str, float]
    """Maximum pricing (e.g., {"prompt": 1.0, "completion": 2.0} for $/M tokens)."""


# =============================================================================
# Default Configuration
# =============================================================================

@dataclass(frozen=True)
class OpenRouterDefaults:
    """
    Default parameter values for OpenRouter requests.
    
    These defaults are applied at the client level and can be overridden
    per-model via model_deployments.yaml configuration or method calls.
    """
    # Sampling parameters
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    min_p: float = 0.0
    top_a: float = 0.0
    
    # Generation control
    seed: int = 42069
    max_tokens: int = 1000
    verbosity: str = "medium"
    
    # Tool configuration
    parallel_tool_calls: bool = True
    
    # Provider routing
    provider: Dict[str, Any] = field(default_factory=lambda: {
        "require_parameters": False,
        "quantizations": ["fp8", "fp16", "bf16"],
    })


# Singleton instance for default values
DEFAULTS = OpenRouterDefaults()


# =============================================================================
# OpenRouter Client
# =============================================================================

class OpenRouterClient:
    """
    Standalone client for OpenRouter's unified AI API.
    
    Provides intelligent routing across multiple AI providers (OpenAI, Anthropic,
    Google, etc.) with support for cost optimization, latency preferences, and
    privacy controls.
    
    Example:
        >>> client = OpenRouterClient(
        ...     model="openai/gpt-4",
        ...     api_key="sk-...",
        ...     temperature=0.7,
        ...     provider={"sort": "price"},
        ... )
        >>> response = client.generate([
        ...     {"role": "user", "content": "Hello!"}
        ... ])
        >>> print(response.choices[0].message.content)
    """
    
    BASE_URL = "https://openrouter.ai/api/v1"
    
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = 30.0,
        # Sampling parameters
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        min_p: Optional[float] = None,
        top_a: Optional[float] = None,
        # Generation control
        seed: Optional[int] = None,
        max_tokens: Optional[int] = None,
        logit_bias: Optional[Dict[str, float]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        stop: Optional[List[str]] = None,
        verbosity: Optional[Literal["low", "medium", "high"]] = None,
        # Tool configuration
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        parallel_tool_calls: Optional[bool] = None,
        # Provider routing
        provider: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the OpenRouter client.
        
        Args:
            model: OpenRouter model identifier (e.g., "openai/gpt-4")
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            default_headers: HTTP headers for all requests (e.g., {"x-anthropic-beta": "..."})
            timeout: Request timeout in seconds (default: 30.0, None for no timeout)
            
            # Sampling (OpenAI-compatible)
            temperature: Controls randomness (0.0-2.0). Default: 1.0
            top_p: Nucleus sampling threshold (0.0-1.0). Default: 1.0
            top_k: Top-k sampling (0=disabled). Default: 0
            frequency_penalty: Penalize frequent tokens (-2.0-2.0). Default: 0.0
            presence_penalty: Penalize repeated tokens (-2.0-2.0). Default: 0.0
            
            # Sampling (OpenRouter-specific)
            repetition_penalty: Alternative repetition control (0.0-2.0). Default: 1.0
            min_p: Minimum probability threshold (0.0-1.0). Default: 0.0
            top_a: Dynamic top-p based on max probability (0.0-1.0). Default: 0.0
            
            # Generation
            seed: Random seed for deterministic outputs. Default: 42069
            max_tokens: Maximum tokens to generate. Default: 1000
            logit_bias: Token bias mapping
            logprobs: Whether to return log probabilities
            top_logprobs: Number of top logprobs to return (0-20)
            response_format: Output format specification
            stop: Stop sequences
            verbosity: Response verbosity ("low", "medium", "high"). Default: "medium"
            
            # Tools
            tools: Tool definitions for function calling
            tool_choice: Tool selection mode ("none", "auto", "required", or specific)
            parallel_tool_calls: Allow parallel function calls. Default: True
            
            # Provider Routing
            provider: Provider routing preferences (see ProviderPreferences)
        """
        # Resolve API key
        self._api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self._api_key:
            raise ValueError(
                "OpenRouter API key not provided. Set OPENROUTER_API_KEY environment "
                "variable or pass api_key parameter."
            )
        
        # Store model name and HTTP settings
        self.model = model
        self.timeout = timeout
        self.default_headers = default_headers or {}
        
        # Note: We use requests directly instead of OpenAI SDK to avoid import hangs
        # This maintains all parameter management logic while ensuring reliability
        
        # Store default parameters (None means use default, value means override)
        self._defaults = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "repetition_penalty": repetition_penalty,
            "min_p": min_p,
            "top_a": top_a,
            "seed": seed,
            "max_tokens": max_tokens,
            "logit_bias": logit_bias,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs,
            "response_format": response_format,
            "stop": stop,
            "verbosity": verbosity,
            "tools": tools,
            "tool_choice": tool_choice,
            "parallel_tool_calls": parallel_tool_calls,
        }
        
        # Build provider configuration with defaults
        self._default_provider = self._build_provider_config(provider)
    
    def _build_provider_config(
        self, 
        user_config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Build provider configuration by merging user settings with defaults.
        
        Args:
            user_config: User-provided provider preferences
            
        Returns:
            Complete provider configuration dictionary
        """
        # Start with defaults
        config = dict(DEFAULTS.provider)
        
        # Merge user config (user values override defaults)
        if user_config:
            config.update(user_config)
        
        return config
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        **override_params
    ):
        """
        Generate a completion using the OpenRouter API.
        
        Args:
            messages: List of message dicts with "role" and "content" keys
            stream: Whether to stream the response
            **override_params: Parameters to override client defaults and YAML config
            
        Returns:
            Response dict from OpenRouter API (OpenAI-compatible format)
            
        Example:
            >>> response = client.generate(
            ...     [{"role": "user", "content": "Hello!"}],
            ...     temperature=0.8,  # Override default
            ... )
            >>> print(response["choices"][0]["message"]["content"])
        """
        # Build request parameters with precedence:
        # override_params > client defaults > DEFAULTS
        params = self._build_request_params(**override_params)
        
        # Add messages and model
        params["model"] = self.model
        params["messages"] = messages
        params["stream"] = stream
        
        # Make API call using requests (avoiding OpenAI SDK import issues)
        requests = _get_requests()  # Lazy import
        
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            **self.default_headers,
        }
        
        response = requests.post(
            f"{self.BASE_URL}/chat/completions",
            headers=headers,
            json=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        
        return response.json()
    
    def _build_request_params(self, **override_params) -> Dict[str, Any]:
        """
        Build request parameters applying precedence:
        override_params > client defaults > DEFAULTS
        
        Args:
            **override_params: Parameters from method call
            
        Returns:
            Complete request parameters dictionary
        """
        params = {}
        
        # Apply defaults first
        self._apply_defaults(params)
        
        # Apply client-level overrides (only if not None)
        for key, value in self._defaults.items():
            if value is not None and key not in override_params:
                params[key] = value
        
        # Apply method-level overrides (highest precedence)
        params.update(override_params)
        
        # Tool configuration: set parallel_tool_calls default if tools are present
        # Check after all params are applied (override_params, client defaults, or DEFAULTS)
        if "tools" in params and params["tools"] is not None and "parallel_tool_calls" not in params:
            params["parallel_tool_calls"] = DEFAULTS.parallel_tool_calls
        
        # Always include provider config (merge with any override)
        provider_config = dict(self._default_provider)
        if "provider" in override_params:
            provider_config.update(override_params["provider"])
        params["provider"] = provider_config
        
        # Remove None values (except for explicit overrides)
        params = {k: v for k, v in params.items() if v is not None}
        
        return params
    
    def _apply_defaults(self, params: Dict[str, Any]) -> None:
        """
        Apply default values to parameters dict.
        
        Args:
            params: Parameters dict to modify in place
        """
        # Sampling parameters
        if "temperature" not in params:
            params["temperature"] = DEFAULTS.temperature
        if "top_p" not in params:
            params["top_p"] = DEFAULTS.top_p
        if "top_k" not in params:
            params["top_k"] = DEFAULTS.top_k
        if "frequency_penalty" not in params:
            params["frequency_penalty"] = DEFAULTS.frequency_penalty
        if "presence_penalty" not in params:
            params["presence_penalty"] = DEFAULTS.presence_penalty
        if "repetition_penalty" not in params:
            params["repetition_penalty"] = DEFAULTS.repetition_penalty
        if "min_p" not in params:
            params["min_p"] = DEFAULTS.min_p
        if "top_a" not in params:
            params["top_a"] = DEFAULTS.top_a
        
        # Generation control
        if "seed" not in params:
            params["seed"] = DEFAULTS.seed
        if "max_tokens" not in params:
            params["max_tokens"] = DEFAULTS.max_tokens
        if "verbosity" not in params:
            params["verbosity"] = DEFAULTS.verbosity


# =============================================================================
# YAML Configuration Loading
# =============================================================================

def load_model_deployments(
    yaml_path: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Load model deployments from YAML file.
    
    Args:
        yaml_path: Path to model_deployments.yaml. If None, looks for
                  prod_env/model_deployments.yaml relative to this file.
    
    Returns:
        Dictionary mapping model names to their configurations
    
    Example:
        >>> configs = load_model_deployments()
        >>> print(configs["gpt-5.1"])
    """
    if yaml_path is None:
        # Default to prod_env/model_deployments.yaml relative to this file
        current_dir = Path(__file__).parent.parent
        yaml_path = current_dir / "prod_env" / "model_deployments.yaml"
    
    yaml = _get_yaml()  # Lazy import
    
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Model deployments file not found: {yaml_path}"
        )
    
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    
    if data is None:
        raise ValueError(
            f"Invalid YAML file: {yaml_path} is empty or contains no data"
        )
    
    if "models" not in data:
        raise ValueError(
            f"Invalid YAML structure: expected 'models' key in {yaml_path}"
        )
    
    # Build dictionary mapping name -> config
    configs = {}
    for model_config in data["models"]:
        if "name" not in model_config:
            raise ValueError(
                f"Model config missing 'name' field: {model_config}"
            )
        if "model" not in model_config:
            raise ValueError(
                f"Model config missing 'model' field: {model_config}"
            )
        
        name = model_config["name"]
        configs[name] = model_config
    
    return configs


def load_model_config(
    name: str,
    yaml_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load configuration for a specific model by name.
    
    Args:
        name: Model name from YAML file
        yaml_path: Path to model_deployments.yaml (optional)
    
    Returns:
        Model configuration dictionary
    
    Example:
        >>> config = load_model_config("gpt-5.1")
        >>> print(config["model"])  # "openai/gpt-5.1:floor"
    """
    configs = load_model_deployments(yaml_path)
    
    if name not in configs:
        available = ", ".join(sorted(configs.keys()))
        raise ValueError(
            f"Model '{name}' not found. Available models: {available}"
        )
    
    return configs[name]


def get_model_client(
    name: str,
    api_key: Optional[str] = None,
    yaml_path: Optional[str] = None,
    **override_params
) -> OpenRouterClient:
    """
    Get a configured OpenRouterClient instance from YAML configuration.
    
    Args:
        name: Model name from YAML file
        api_key: OpenRouter API key (optional, uses env var if not provided)
        yaml_path: Path to model_deployments.yaml (optional)
        **override_params: Additional parameters to override YAML config
    
    Returns:
        Configured OpenRouterClient instance
    
    Example:
        >>> client = get_model_client("gpt-5.1")
        >>> response = client.generate([{"role": "user", "content": "Hello!"}])
    """
    config = load_model_config(name, yaml_path)
    
    # Extract model name (required)
    model = config.pop("model")
    
    # Extract provider config if present
    provider = config.pop("provider", None)
    
    # Extract name (not needed for client)
    config.pop("name", None)
    
    # Merge YAML config with override params (override_params take precedence)
    client_params = {**config, **override_params}
    
    # Add provider back if it exists
    if provider is not None:
        client_params["provider"] = provider
    
    # Create client
    return OpenRouterClient(
        model=model,
        api_key=api_key,
        **client_params
    )
