import base64
import json
import logging
import time
import random
from typing import Any, Dict, List, Optional
import urllib.request
import urllib.error

from src.providers.base import (
    BaseProvider,
    VisionProvider,
    SpeechProvider,
    OCRProvider,
    LLMProvider,
    ProviderCapabilities,
    ProviderConfig,
    ProviderError,
    AuthenticationError,
    RateLimitError,
    TimeoutError,
    ProviderUnavailableError,
    InvalidResponseError,
    VisionObservation,
    SpeechObservation,
    OCRObservation,
    LLMResponse,
    global_cost_tracker,
)
from src.providers.registry import ProviderRegistry

logger = logging.getLogger(__name__)


def _call_gemini_rest(
    endpoint: str,
    api_key: str,
    payload: Dict[str, Any],
    timeout: int = 30
) -> Dict[str, Any]:
    """Execute REST API request to Gemini API endpoint with automatic retries for rate limits."""
    url = f"{endpoint}?key={api_key}"
    headers = {"Content-Type": "application/json"}
    data = json.dumps(payload).encode("utf-8")
    
    max_retries = 5
    base_backoff = 2.0
    
    for attempt in range(max_retries + 1):
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                res_body = response.read().decode("utf-8")
                return json.loads(res_body)
        except urllib.error.HTTPError as e:
            status = e.code
            body_bytes = e.read()
            body = body_bytes.decode("utf-8")
            logger.error(f"Gemini REST Error {status} (attempt {attempt+1}/{max_retries+1}): {body}")
            
            if status in (401, 403):
                raise AuthenticationError(f"Gemini auth failed: {body}") from e
            elif status == 429:
                is_daily_limit = "requestsperday" in body.lower()
                sleep_time = None
                try:
                    err_data = json.loads(body)
                    details = err_data.get("error", {}).get("details", [])
                    for detail in details:
                        if "RetryInfo" in detail.get("@type", ""):
                            delay_str = detail.get("retryDelay", "")
                            if delay_str.endswith("s"):
                                sleep_time = float(delay_str[:-1]) + 0.5
                                break
                except Exception:
                    pass
                
                if is_daily_limit or (sleep_time is not None and sleep_time > 10.0):
                    logger.warning(f"Daily quota limit or long delay ({sleep_time}s) hit. Bypassing retries for this model.")
                    raise RateLimitError(f"Gemini daily quota limit exceeded: {body}") from e
                
                if attempt < max_retries:
                    if sleep_time is None:
                        sleep_time = base_backoff * (2 ** attempt) + random.uniform(0, 1)
                        
                    logger.warning(f"Rate limit hit. Retrying in {sleep_time:.2f}s...")
                    time.sleep(sleep_time)
                    continue
                else:
                    raise RateLimitError(f"Gemini rate limit exceeded after {max_retries} retries: {body}") from e
            elif status >= 500:
                if attempt < max_retries:
                    sleep_time = base_backoff * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Server error {status}. Retrying in {sleep_time:.2f}s...")
                    time.sleep(sleep_time)
                    continue
                else:
                    raise ProviderUnavailableError(f"Gemini server error after {max_retries} retries: {body}") from e
            raise ProviderError(f"Gemini API returned status {status}: {body}") from e
        except urllib.error.URLError as e:
            if attempt < max_retries:
                sleep_time = base_backoff * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"Connection failed: {e}. Retrying in {sleep_time:.2f}s...")
                time.sleep(sleep_time)
                continue
            raise ProviderUnavailableError(f"Gemini connection failed: {e}") from e
        except Exception as e:
            raise ProviderError(f"Gemini request failed: {e}") from e



class GeminiProvider(VisionProvider, OCRProvider, LLMProvider):
    """Google Gemini Adapter using direct REST API calls."""

    PROVIDER_NAME = "gemini"

    def __init__(self, api_key: str = "mock_key_for_testing", base_url: Optional[str] = None):
        self.api_key = api_key
        self.base_url = base_url or "https://generativelanguage.googleapis.com/v1beta/models"
        # Per-run circuit breaker: once the free-tier *daily* quota is exhausted, every
        # subsequent Gemini call would 429 too. Trip this flag on the first daily-quota
        # error so later calls fail fast (0 HTTP) and the fallback proxy moves on
        # immediately, instead of re-hitting the quota across every model candidate.
        self._daily_quota_hit = False

    def get_name(self) -> str:
        return "gemini"

    def get_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_vision=True,
            supports_audio=False,
            supports_json=True,
            supports_multimodal=True,
        )

    def _execute_with_model_fallback(
        self,
        base_model_name: str,
        payload: Dict[str, Any],
        timeout: int = 30
    ) -> tuple[Dict[str, Any], str]:
        """
        Execute the REST call, falling back to other Gemini models if quota/429 limits are hit.
        Returns a tuple of (response_json, model_used).
        """
        candidates = [
            base_model_name,
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-3.5-flash",
            "gemini-flash-latest"
        ]
        # De-duplicate candidates while preserving order
        seen = set()
        model_list = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                model_list.append(c)
                
        last_err = None
        for model in model_list:
            endpoint = f"{self.base_url}/{model}:generateContent"
            try:
                res = _call_gemini_rest(endpoint, self.api_key, payload, timeout=timeout)
                return res, model
            except RateLimitError as e:
                logger.warning(f"Model '{model}' hit rate limit/quota. Trying next fallback candidate...")
                last_err = e
            except Exception as e:
                if "auth failed" in str(e).lower() or isinstance(e, AuthenticationError):
                    raise
                logger.warning(f"Model '{model}' failed with error: {e}. Trying next candidate...")
                last_err = e
                
        raise ProviderError(f"All Gemini fallback models exhausted. Last error: {last_err}")


    # 1. Vision Adapter
    def analyze_frames(
        self,
        frames: List[Any],
        prompt: str,
        config: ProviderConfig
    ) -> List[VisionObservation]:
        if not frames:
            return []

        if self.api_key == "mock_key_for_testing":
            from src.providers.mock_provider import MockProvider
            mock_inst = MockProvider()
            return mock_inst.analyze_frames(frames, prompt, config)

        observations: List[VisionObservation] = []
        model_name = config.model_name or "gemini-2.5-flash"
        endpoint = f"{self.base_url}/{model_name}:generateContent"

        for frame in frames:
            if not getattr(frame, "frame_data", None):
                continue

            base64_image = base64.b64encode(frame.frame_data).decode("utf-8")
            
            system_instruction = (
                "You are an expert video analysis assistant. Analyze the image and extract observations. "
                "Return a JSON list of objects matching this schema: "
                '[{"content": "description", "confidence": 0.9, "observation_type": "action"}]'
            )

            payload = {
                "contents": [{
                    "parts": [
                        {"text": f"{system_instruction}\n\nPrompt: {prompt}"},
                        {
                            "inlineData": {
                                "mimeType": "image/jpeg",
                                "data": base64_image
                            }
                        }
                    ]
                }],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "temperature": config.temperature or 0.2,
                    "maxOutputTokens": config.max_tokens or 1024
                }
            }

            try:
                t0 = time.time()
                res, actual_model = self._execute_with_model_fallback(model_name, payload, timeout=config.timeout_seconds or 30)
                latency = time.time() - t0
                
                # Cost tracking
                global_cost_tracker.track_call(
                    provider="gemini",
                    model=actual_model,
                    prompt_tokens=0, # REST API doesn't return detailed tokens easily in all responses
                    completion_tokens=0,
                    latency=latency
                )

                text = res["candidates"][0]["content"]["parts"][0]["text"].strip()
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    parsed = [parsed]

                if isinstance(parsed, list):
                    for item in parsed:
                        observations.append(
                            VisionObservation(
                                content=item.get("content", "unspecified visual observation"),
                                timestamp=frame.timestamp,
                                confidence=item.get("confidence", 0.5),
                                observation_type=item.get("observation_type", "action")
                            )
                        )
            except Exception as e:
                logger.error(f"Gemini Vision call failed at {frame.timestamp}s: {e}")
                raise ProviderError(f"Gemini Vision call failed: {e}") from e

        return sorted(observations, key=lambda x: x.timestamp)

    # 2. OCR Adapter
    def extract_text(
        self,
        frames: List[Any],
        config: ProviderConfig
    ) -> List[OCRObservation]:
        if not frames:
            return []

        if self.api_key == "mock_key_for_testing":
            from src.providers.mock_provider import MockProvider
            mock_inst = MockProvider()
            return mock_inst.extract_text(frames, config)

        observations: List[OCRObservation] = []
        model_name = config.model_name or "gemini-2.5-flash"
        endpoint = f"{self.base_url}/{model_name}:generateContent"

        for frame in frames:
            if not getattr(frame, "frame_data", None):
                continue

            base64_image = base64.b64encode(frame.frame_data).decode("utf-8")
            
            system_instruction = (
                "You are an expert OCR assistant. Extract all visible text in the image. "
                "Return a JSON list of objects matching this schema: "
                '[{"text": "detected text content", "confidence": 0.9}]'
            )

            payload = {
                "contents": [{
                    "parts": [
                        {"text": system_instruction},
                        {
                            "inlineData": {
                                "mimeType": "image/jpeg",
                                "data": base64_image
                            }
                        }
                    ]
                }],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "temperature": 0.1,
                    "maxOutputTokens": config.max_tokens or 1024
                }
            }

            try:
                t0 = time.time()
                res, actual_model = self._execute_with_model_fallback(model_name, payload, timeout=config.timeout_seconds or 30)
                latency = time.time() - t0
                
                global_cost_tracker.track_call(
                    provider="gemini",
                    model=actual_model,
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency=latency
                )

                text = res["candidates"][0]["content"]["parts"][0]["text"].strip()
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    parsed = [parsed]

                if isinstance(parsed, list):
                    for item in parsed:
                        val = item.get("text", "").strip()
                        if val:
                            observations.append(
                                OCRObservation(
                                    content=val,
                                    timestamp=frame.timestamp,
                                    confidence=item.get("confidence", 0.5)
                                )
                            )
            except Exception as e:
                logger.error(f"Gemini OCR call failed at {frame.timestamp}s: {e}")
                raise ProviderError(f"Gemini OCR call failed: {e}") from e

        return sorted(observations, key=lambda x: x.timestamp)

    # 3. LLM Adapter
    def generate(
        self,
        prompt: str,
        system_prompt: str,
        config: ProviderConfig
    ) -> LLMResponse:
        if self.api_key == "mock_key_for_testing":
            from src.providers.mock_provider import MockProvider
            mock_inst = MockProvider()
            return mock_inst.generate(prompt, system_prompt, config)

        model_name = config.model_name or "gemini-2.5-flash"
        endpoint = f"{self.base_url}/{model_name}:generateContent"

        full_prompt = f"{system_prompt}\n\nUser Prompt:\n{prompt}"
        payload = {
            "contents": [{
                "parts": [{"text": full_prompt}]
            }],
            "generationConfig": {
                "temperature": config.temperature or 0.7,
                "maxOutputTokens": config.max_tokens or 512
            }
        }
        
        # Enable JSON mode if requested
        if config.extra_params.get("json_mode", False):
            payload["generationConfig"]["responseMimeType"] = "application/json"

        try:
            t0 = time.time()
            res, actual_model = self._execute_with_model_fallback(model_name, payload, timeout=config.timeout_seconds or 30)
            latency = time.time() - t0
            
            global_cost_tracker.track_call(
                provider="gemini",
                model=actual_model,
                prompt_tokens=0,
                completion_tokens=0,
                latency=latency
            )

            text = res["candidates"][0]["content"]["parts"][0]["text"].strip()
            return LLMResponse(
                text=text,
                latency=latency
            )
        except Exception as e:
            logger.error(f"Gemini LLM generate failed: {e}")
            raise ProviderError(f"Gemini LLM generate failed: {e}") from e


# Auto-register Gemini provider
ProviderRegistry.register("gemini", GeminiProvider)
