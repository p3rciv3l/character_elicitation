"""
Custom HELM clients for character elicitation research.

This module provides extended clients for various AI providers with
additional features and parameter controls.
"""

from clients.openrouter_client import (
    OpenRouterClient,
    OpenRouterDefaults,
    ProviderPreferences,
    DEFAULTS,
)

__all__ = [
    "OpenRouterClient",
    "OpenRouterDefaults",
    "ProviderPreferences",
    "DEFAULTS",
]
