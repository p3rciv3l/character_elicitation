import os
from typing import Any, Dict, Optional

from helm.clients.openai_client import OpenAIClient
from helm.common.request import Request
from helm.tokenizers.tokenizer import Tokenizer


class OpenRouterClient(OpenAIClient):
    DEFAULT_PROVIDER_CONFIG: Dict[str, Any] = {
        # only route to providers that support all params in the request
        "require_parameters": True,
        "quantizations": ["fp8", "fp16", "bf16"],
    }

    def __init__(
        self,
        tokenizer_name: str,
        tokenizer: Tokenizer,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        output_processor: Optional[str] = None,
        provider: Optional[Dict[str, Any]] = None,
        enable_tools: bool = False,
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1/"
        super().__init__(
            tokenizer,
            tokenizer_name,
            output_processor=output_processor,
            base_url=self.base_url,
            api_key=self.api_key,
        )
        self.model_name = model_name
        self.provider = provider
        self.enable_tools = enable_tools

    def _get_model_for_request(self, request: Request) -> str:
        return self.model_name or request.model

    def _make_chat_raw_request(self, request: Request) -> Dict[str, Any]:
        """Build chat request payload with OpenRouter-specific parameters.

        Extends the base OpenAI-compatible payload with parameters supported by OpenRouter:
        https://openrouter.ai/docs/api/reference/parameters

        OpenRouter-specific parameters (not in OpenAI SDK) are passed via `extra_body`
        to avoid TypeError from the OpenAI Python SDK on unknown kwargs.
        See: https://github.com/stanford-crfm/helm/issues/2995
        """
        # base OpenAI-compatible payload from parent
        raw_request = super()._make_chat_raw_request(request)

        raw_request["model"] = self._get_model_for_request(request)

        if request.seed is not None:
            raw_request["seed"] = request.seed

        if request.logprobs:
            raw_request["logprobs"] = True

        if request.top_logprobs is not None:
            raw_request["top_logprobs"] = request.top_logprobs

        if request.logit_bias:
            raw_request["logit_bias"] = request.logit_bias

        # tool calling (disabled by default)
        if self.enable_tools:
            if request.tools:
                raw_request["tools"] = request.tools

            if request.tool_choice is not None:
                raw_request["tool_choice"] = request.tool_choice

            raw_request["parallel_tool_calls"] = request.parallel_tool_calls

        extra: Dict[str, Any] = {}

        if request.repetition_penalty != 1.0:
            extra["repetition_penalty"] = request.repetition_penalty

        if request.min_p > 0.0:
            extra["min_p"] = request.min_p

        if request.top_a > 0.0:
            extra["top_a"] = request.top_a

        if request.top_k_per_token > 0:
            extra["top_k"] = request.top_k_per_token

        if request.verbosity != "medium":
            extra["verbosity"] = request.verbosity

        # structured outputs works with response_format json_schema
        # see: https://openrouter.ai/docs/guides/features/structured-outputs
        if request.structured_outputs:
            extra["structured_outputs"] = True

        # provider routing config
        # see: https://openrouter.ai/docs/guides/routing/provider-selection
        provider_config = {**self.DEFAULT_PROVIDER_CONFIG}
        if self.provider:
            provider_config.update(self.provider)
        extra["provider"] = provider_config

        # merge into extra_body if any OpenRouter-specific params
        if extra:
            raw_request["extra_body"] = {**raw_request.get("extra_body", {}), **extra}

        return raw_request
