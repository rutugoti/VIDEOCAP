import pytest
import os
from typing import List

from src.providers.base import (
    ProviderConfig,
    ProviderError,
    AuthenticationError,
    RateLimitError,
    TimeoutError,
    ProviderUnavailableError,
    global_cost_tracker,
)
from src.providers.registry import ProviderRegistry
from src.providers.factory import ProviderFactory
from src.providers.mock_provider import MockProvider
from src.providers.openai_provider import OpenAIProvider
from src.providers.groq_provider import GroqProvider
from src.providers.ollama_provider import OllamaProvider


def test_provider_registry():
    # Verify standard registrations
    providers = ProviderRegistry.list_providers()
    assert "mock" in providers
    assert "fireworks" in providers
    assert "groq" in providers
    assert "openai" in providers
    assert "ollama" in providers

    # Test dynamic manual registration
    class TemporaryProvider(MockProvider):
        PROVIDER_NAME = "temp_prov"
        def get_name(self) -> str:
            return "temp_prov"

    ProviderRegistry.register("temp_prov", TemporaryProvider)
    assert "temp_prov" in ProviderRegistry.list_providers()
    assert ProviderRegistry.get("temp_prov") == TemporaryProvider

    # Unregister
    ProviderRegistry.unregister("temp_prov")
    assert "temp_prov" not in ProviderRegistry.list_providers()


def test_provider_capabilities():
    mock_prov = ProviderRegistry.get("mock")()
    caps = mock_prov.get_capabilities()
    assert caps.supports_vision is True
    assert caps.supports_audio is True
    assert caps.supports_json is True

    ollama_prov = ProviderRegistry.get("ollama")()
    ollama_caps = ollama_prov.get_capabilities()
    assert ollama_caps.supports_vision is False
    assert ollama_caps.supports_audio is False
    assert ollama_caps.supports_json is True


def test_provider_factory_fallback():
    # Define a custom failing provider
    class FailingProvider(MockProvider):
        PROVIDER_NAME = "failing"
        def __init__(self, api_key=None, base_url=None, **kwargs):
            super().__init__(api_key, base_url, **kwargs)
        def get_name(self) -> str:
            return "failing"
        def generate(self, prompt, system_prompt, config):
            raise RateLimitError("Rate limit hit on failing provider")

    # Register failing provider
    ProviderRegistry.register("failing", FailingProvider)

    try:
        # Create a LLM provider with fallback: failing -> mock
        llm = ProviderFactory.get_llm(["failing", "mock"])
        
        # Test generate: should fall back to mock and return the mock string successfully
        cfg = ProviderConfig(provider_name="test", model_name="test-model")
        res = llm.generate("test prompt", "system prompt", cfg)
        assert res.text is not None
        assert "Generic mock" in res.text or len(res.text) > 0
    finally:
        ProviderRegistry.unregister("failing")


def test_error_normalization():
    # Verify we catch and normalize errors correctly inside OpenAI/Fireworks adapters
    import openai
    import httpx
    from src.providers.openai_provider import _normalize_openai_error
    
    # 1. RateLimitError
    req = httpx.Request("POST", "https://api.openai.com")
    resp = httpx.Response(429, request=req)
    raw_rate_limit = openai.RateLimitError(
        message="Rate limit exceeded",
        response=resp,
        body=None
    )
    normalized = _normalize_openai_error(raw_rate_limit)
    assert isinstance(normalized, RateLimitError)

    # 2. AuthenticationError
    resp_auth = httpx.Response(401, request=req)
    raw_auth = openai.AuthenticationError(
        message="Auth failed",
        response=resp_auth,
        body=None
    )
    normalized_auth = _normalize_openai_error(raw_auth)
    assert isinstance(normalized_auth, AuthenticationError)


def test_cost_tracking():
    # Reset tracking
    global_cost_tracker.calls.clear()
    
    # Track a call
    global_cost_tracker.track_call(
        provider="openai",
        model="gpt-4o",
        prompt_tokens=100,
        completion_tokens=50,
        latency=1.5,
        estimated_cost=0.003
    )

    summary = global_cost_tracker.get_summary()
    assert summary["total_calls"] == 1
    assert summary["total_prompt_tokens"] == 100
    assert summary["total_completion_tokens"] == 50
    assert summary["total_latency"] == 1.5
    assert summary["total_cost"] == 0.003
