"""
OpenRouter Client for HELM

A comprehensive client for OpenRouter's unified AI API that provides intelligent
provider routing across multiple model providers. This client extends HELM's
OpenAI client with full support for OpenRouter-specific parameters and features.

References:
    - Parameters: https://openrouter.ai/docs/api/reference/parameters
    - Provider Routing: https://openrouter.ai/docs/guides/routing/provider-selection
"""

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, TypedDict
from typing_extensions import NotRequired

from helm.clients.client import CachingClient
from helm.clients.openai_client import OpenAIClient
from helm.common.cache import CacheConfig
from helm.common.object_spec import get_class_by_name
from helm.common.request import Request
from helm.tokenizers.tokenizer import Tokenizer

try:
    from openai import OpenAI
except ModuleNotFoundError as e:
    from helm.common.optional_dependencies import handle_module_not_found_error
    handle_module_not_found_error(e, ["openai"])


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


class OpenRouterRawRequest(TypedDict, total=False):
    """
    Complete OpenRouter API request structure.
    
    Combines OpenAI-compatible parameters with OpenRouter-specific extensions.
    """
    # Core request
    model: str
    messages: List[Dict[str, Any]]
    
    # Sampling parameters (OpenAI-compatible)
    temperature: float
    top_p: float
    max_tokens: int
    frequency_penalty: float
    presence_penalty: float
    stop: List[str]
    n: int
    
    # OpenRouter-specific sampling
    top_k: int
    repetition_penalty: float
    min_p: float
    top_a: float
    seed: int
    
    # Output control
    logit_bias: Dict[str, float]
    logprobs: bool
    top_logprobs: int
    response_format: Dict[str, Any]
    verbosity: Literal["low", "medium", "high"]
    
    # Tool use
    tools: List[Dict[str, Any]]
    tool_choice: Any
    parallel_tool_calls: bool
    
    # Provider routing
    provider: ProviderPreferences


# =============================================================================
# Default Configuration
# =============================================================================

@dataclass(frozen=True)
class OpenRouterDefaults:
    """
    Default parameter values for OpenRouter requests.
    
    These defaults are applied at the client level and can be overridden
    per-model via model_deployments.yaml configuration.
    
    Values align with OpenRouter's API defaults where possible, with sensible
    customizations for research/evaluation workloads.
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

class OpenRouterClient(OpenAIClient):
    """
    HELM client for OpenRouter's unified AI API.
    
    Provides intelligent routing across multiple AI providers (OpenAI, Anthropic,
    Google, etc.) with support for cost optimization, latency preferences, and
    privacy controls.
    
    All parameters can be configured via model_deployments.yaml:
    
        client_spec:
          class_name: "clients.openrouter_client.OpenRouterClient"
          args:
            model_name: "anthropic/claude-3-opus"
            temperature: 0.7
            provider:
              sort: "throughput"
              data_collection: "deny"
    
    Attributes:
        model_name: OpenRouter model identifier (e.g., "openai/gpt-4")
        
    Example:
        >>> client = OpenRouterClient(
        ...     tokenizer=tokenizer,
        ...     tokenizer_name="openai/gpt-4",
        ...     cache_config=cache_config,
        ...     model_name="openai/gpt-4",
        ...     temperature=0.7,
        ...     provider={"sort": "price"},
        ... )
    """
    
    BASE_URL = "https://openrouter.ai/api/v1/"
    
    def __init__(
        self,
        tokenizer_name: str,
        tokenizer: Tokenizer,
        cache_config: CacheConfig,
        # Authentication
        api_key: Optional[str] = None,
        # Model identification
        model_name: Optional[str] = None,
        output_processor: Optional[str] = None,
        # HTTP headers (for provider-specific features like Anthropic beta)
        default_headers: Optional[Dict[str, str]] = None,
        # =================================================================
        # Sampling Parameters
        # These override values from Request when specified
        # =================================================================
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        min_p: Optional[float] = None,
        top_a: Optional[float] = None,
        # =================================================================
        # Generation Control
        # =================================================================
        seed: Optional[int] = None,
        max_tokens: Optional[int] = None,
        logit_bias: Optional[Dict[str, float]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        stop: Optional[List[str]] = None,
        verbosity: Optional[Literal["low", "medium", "high"]] = None,
        # =================================================================
        # Tool Configuration
        # =================================================================
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        parallel_tool_calls: Optional[bool] = None,
        # =================================================================
        # Provider Routing
        # =================================================================
        provider: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the OpenRouter client.
        
        Args:
            tokenizer_name: Name of the tokenizer to use
            tokenizer: Tokenizer instance for token counting
            cache_config: Cache configuration for request caching
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            model_name: Model identifier on OpenRouter (e.g., "anthropic/claude-3-opus")
            output_processor: Optional function name for post-processing outputs
            default_headers: HTTP headers for all requests (e.g., {"x-anthropic-beta": "..."})
            
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
        
        # Initialize base caching client (skip OpenAI's __init__ to customize)
        CachingClient.__init__(self, cache_config=cache_config)
        
        # Initialize tokenizer (required by OpenAIClient)
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name
        
        # OpenAI client attributes
        self.openai_model_name = model_name
        self.reasoning_effort = None
        self.output_processor: Optional[Callable[[str], str]] = (
            get_class_by_name(output_processor) if output_processor else None
        )
        
        # Create OpenAI-compatible client pointed at OpenRouter
        self.client = OpenAI(
            api_key=self._api_key,
            base_url=self.BASE_URL,
            default_headers=default_headers,
        )
        
        # Store model name
        self.model_name = model_name
        
        # Store parameter overrides
        # Using None to indicate "use default", actual value to override
        self._params = _ParameterStore(
            # Sampling
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            top_a=top_a,
            # Generation
            seed=seed,
            max_tokens=max_tokens,
            logit_bias=logit_bias,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            response_format=response_format,
            stop=stop,
            verbosity=verbosity,
            # Tools
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
        )
        
        # Build provider configuration with defaults
        self._provider = self._build_provider_config(provider)
    
    def _build_provider_config(
        self, 
        user_config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Build provider configuration by merging user settings with defaults.
        
        Provider routing is a core feature of OpenRouter, so we always include
        sensible defaults while allowing full customization.
        
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
    
    def _make_chat_raw_request(self, request: Request) -> Dict[str, Any]:
        """
        Build the raw request for OpenRouter's chat completion API.
        
        This method:
        1. Gets base request from OpenAI parent (using Request values)
        2. Applies client-level parameter overrides
        3. Adds OpenRouter-specific parameters
        4. Adds provider routing configuration
        
        The precedence order is:
            Client override > Request value > HELM Request default
        
        Args:
            request: HELM Request object
            
        Returns:
            Complete request dictionary for OpenRouter API
        """
        # Get base request from OpenAI parent
        raw_request = super()._make_chat_raw_request(request)
        
        # Apply parameter overrides and add OpenRouter-specific params
        self._apply_parameter_overrides(raw_request, request)
        
        # Add provider routing (always present)
        raw_request["provider"] = self._provider
        
        return raw_request
    
    def _apply_parameter_overrides(
        self, 
        raw_request: Dict[str, Any], 
        request: Request
    ) -> None:
        """
        Apply client-level parameter overrides to the raw request.
        
        For each parameter:
        - If client has an override (not None): use the override
        - If client override is None: keep the value from parent (Request)
        - For OpenRouter-specific params: use client value or default
        
        Args:
            raw_request: Request dict to modify (mutated in place)
            request: Original HELM Request for reference
        """
        p = self._params
        
        # =====================================================================
        # OpenAI-Compatible Parameters (override if client specified)
        # =====================================================================
        
        if p.temperature is not None:
            raw_request["temperature"] = p.temperature
        
        if p.top_p is not None:
            raw_request["top_p"] = p.top_p
        
        if p.max_tokens is not None:
            raw_request["max_tokens"] = p.max_tokens
        elif raw_request.get("max_tokens") == 100:
            # Override HELM's low default of 100 with our default of 1000
            raw_request["max_tokens"] = DEFAULTS.max_tokens
        
        if p.frequency_penalty is not None:
            raw_request["frequency_penalty"] = p.frequency_penalty
        
        if p.presence_penalty is not None:
            raw_request["presence_penalty"] = p.presence_penalty
        
        if p.stop is not None:
            raw_request["stop"] = p.stop
        
        if p.response_format is not None:
            raw_request["response_format"] = p.response_format
        
        if p.logprobs is not None:
            raw_request["logprobs"] = p.logprobs
        
        if p.top_logprobs is not None:
            raw_request["top_logprobs"] = p.top_logprobs
        
        if p.logit_bias is not None:
            raw_request["logit_bias"] = p.logit_bias
        
        # =====================================================================
        # OpenRouter-Specific Sampling Parameters
        # =====================================================================
        
        # top_k: Use client value, fall back to HELM's top_k_per_token, then default
        if p.top_k is not None:
            raw_request["top_k"] = p.top_k
        elif hasattr(request, "top_k_per_token") and request.top_k_per_token > 1:
            raw_request["top_k"] = request.top_k_per_token
        else:
            raw_request["top_k"] = DEFAULTS.top_k
        
        # repetition_penalty: Client value or default
        raw_request["repetition_penalty"] = (
            p.repetition_penalty if p.repetition_penalty is not None 
            else DEFAULTS.repetition_penalty
        )
        
        # min_p: Client value or default
        raw_request["min_p"] = (
            p.min_p if p.min_p is not None 
            else DEFAULTS.min_p
        )
        
        # top_a: Client value or default
        raw_request["top_a"] = (
            p.top_a if p.top_a is not None 
            else DEFAULTS.top_a
        )
        
        # =====================================================================
        # Generation Control
        # =====================================================================
        
        # seed: Client value or default
        raw_request["seed"] = (
            p.seed if p.seed is not None 
            else DEFAULTS.seed
        )
        
        # verbosity: Client value or default
        raw_request["verbosity"] = (
            p.verbosity if p.verbosity is not None 
            else DEFAULTS.verbosity
        )
        
        # =====================================================================
        # Tool Configuration
        # =====================================================================
        
        # Only add tool-related params if tools are actually provided
        if p.tools is not None:
            raw_request["tools"] = p.tools
            
            # tool_choice only makes sense when tools are present
            if p.tool_choice is not None:
                raw_request["tool_choice"] = p.tool_choice
            
            # parallel_tool_calls: Client value or default
            raw_request["parallel_tool_calls"] = (
                p.parallel_tool_calls if p.parallel_tool_calls is not None
                else DEFAULTS.parallel_tool_calls
            )
    
    def _get_model_for_request(self, request: Request) -> str:
        """
        Get the model identifier for the API request.
        
        Args:
            request: HELM Request object
            
        Returns:
            OpenRouter model identifier (e.g., "anthropic/claude-3-opus")
        """
        return self.model_name or request.model


# =============================================================================
# Internal Helpers
# =============================================================================

@dataclass
class _ParameterStore:
    """
    Internal storage for client parameter overrides.
    
    All fields are Optional - None means "use default or parent value".
    This allows clean distinction between "not set" and "set to a value".
    """
    # Sampling
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    repetition_penalty: Optional[float] = None
    min_p: Optional[float] = None
    top_a: Optional[float] = None
    
    # Generation
    seed: Optional[int] = None
    max_tokens: Optional[int] = None
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    response_format: Optional[Dict[str, Any]] = None
    stop: Optional[List[str]] = None
    verbosity: Optional[Literal["low", "medium", "high"]] = None
    
    # Tools
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    parallel_tool_calls: Optional[bool] = None