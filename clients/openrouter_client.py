import os
from typing import Any, Dict, Optional

from helm.clients.openai_client import OpenAIClient
from helm.common.cache import CacheConfig
from helm.common.request import Request
from helm.tokenizers.tokenizer import Tokenizer


class OpenRouterClient(OpenAIClient):
    """extends HELM's OpenAI client to support OpenRouter-specific parameters
    configured per-deployment in model_deployments.yaml."""
    
    DEFAULT_PROVIDER_CONFIG: Dict[str, Any] = {
        "require_parameters": True,
        "quantizations": ["fp8", "fp16", "bf16"],
    }

    def __init__(
        self,
        tokenizer_name: str,
        tokenizer: Tokenizer,
        cache_config: CacheConfig,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        output_processor: Optional[str] = None,
        provider: Optional[Dict[str, Any]] = None,
        enable_tools: bool = False,
        **sampling_defaults,
    ):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1/"
        super().__init__(
            tokenizer,
            tokenizer_name,
            cache_config=cache_config,
            output_processor=output_processor,
            base_url=self.base_url,
            api_key=self.api_key,
        )
        self.model_name = model_name
        self.provider = provider
        self.enable_tools = enable_tools
        self.sampling_defaults = sampling_defaults

    def _get_model_for_request(self, request: Request) -> str:
        return self.model_name or request.model

    def _make_chat_raw_request(self, request: Request) -> Dict[str, Any]:
        raw_request = super()._make_chat_raw_request(request)
        raw_request["model"] = self._get_model_for_request(request)
        
        for key in ("temperature", "top_p", "max_tokens", "presence_penalty", "frequency_penalty", "seed"):
            if key in self.sampling_defaults:
                raw_request[key] = self.sampling_defaults[key]
        
        if getattr(request, "logprobs", False):
            raw_request["logprobs"] = True
        if getattr(request, "top_logprobs", None) is not None:
            raw_request["top_logprobs"] = request.top_logprobs
        if getattr(request, "logit_bias", None):
            raw_request["logit_bias"] = request.logit_bias

        if self.enable_tools and getattr(request, "tools", None):
            raw_request["tools"] = request.tools
            if getattr(request, "tool_choice", None) is not None:
                raw_request["tool_choice"] = request.tool_choice
            raw_request["parallel_tool_calls"] = getattr(request, "parallel_tool_calls", False)

        extra: Dict[str, Any] = {}
        
        for key, skip_value in [("repetition_penalty", 1.0), ("min_p", 0.0), ("top_a", 0.0), ("top_k", 0)]:
            if key in self.sampling_defaults and self.sampling_defaults[key] != skip_value:
                extra[key] = self.sampling_defaults[key]

        if getattr(request, "verbosity", "medium") != "medium":
            extra["verbosity"] = request.verbosity
        if getattr(request, "structured_outputs", False):
            extra["structured_outputs"] = True

        # provider routing
        extra["provider"] = {**self.DEFAULT_PROVIDER_CONFIG, **(self.provider or {})}

        raw_request["extra_body"] = {**raw_request.get("extra_body", {}), **extra}
        return raw_request
