import json
import logging
import time
from typing import Any, Dict, List, Optional
import urllib.request
import urllib.error

from src.providers.base import (
    BaseProvider,
    LLMProvider,
    ProviderCapabilities,
    ProviderConfig,
    ProviderError,
    ProviderUnavailableError,
    LLMResponse,
    global_cost_tracker,
)
from src.providers.registry import ProviderRegistry

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """Local Ollama Adapter using standard REST API requests."""

    PROVIDER_NAME = "ollama"

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")

    def get_name(self) -> str:
        return "ollama"

    def get_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_vision=False,
            supports_audio=False,
            supports_json=True,
            supports_streaming=False,
        )

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        config: ProviderConfig
    ) -> LLMResponse:
        url = f"{self.base_url}/api/chat"
        model_name = config.model_name or "gemma:2b"
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": config.temperature or 0.7,
                "num_predict": config.max_tokens or 512
            }
        }

        # Handle json output format option if requested
        if config.extra_params.get("json_mode", False):
            payload["format"] = "json"

        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        timeout = config.timeout_seconds or 30

        try:
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=timeout) as response:
                res_body = response.read().decode("utf-8")
                latency = time.time() - t0
                
                res_data = json.loads(res_body)
                text = res_data["message"]["content"].strip()
                
                # Retrieve tokens if reported by Ollama
                prompt_tokens = res_data.get("prompt_eval_count", 0)
                completion_tokens = res_data.get("eval_count", 0)

                global_cost_tracker.track_call(
                    provider="ollama",
                    model=model_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency=latency
                )

                return LLMResponse(
                    text=text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency=latency
                )
        except urllib.error.URLError as e:
            logger.error(f"Ollama connection failed: {e}")
            raise ProviderUnavailableError(f"Local Ollama service unreachable at {self.base_url}: {e}") from e
        except Exception as e:
            logger.error(f"Ollama generate failed: {e}")
            raise ProviderError(f"Ollama generation call failed: {e}") from e


# Auto-register Ollama provider
ProviderRegistry.register("ollama", OllamaProvider)
