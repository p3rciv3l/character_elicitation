"""
Standalone OpenRouter client for character elicitation research.

This module provides a simple client for OpenRouter's unified AI API with
built-in defaults and easy parameter overrides.
"""

from clients.openrouter_client import (
    OpenRouterClient,
    OpenRouterDefaults,
    ProviderPreferences,
    DEFAULTS,
    load_model_deployments,
    load_model_config,
    get_model_client,
)

__all__ = [
    "OpenRouterClient",
    "OpenRouterDefaults",
    "ProviderPreferences",
    "DEFAULTS",
    "load_model_deployments",
    "load_model_config",
    "get_model_client",
]
