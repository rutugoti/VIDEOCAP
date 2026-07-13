import base64
import io
import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI
import openai

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
    VisionObservation,
    SpeechObservation,
    OCRObservation,
    LLMResponse,
    global_cost_tracker,
)
from src.providers.registry import ProviderRegistry

logger = logging.getLogger(__name__)


class _TokenRateLimiter:
    """
    Process-wide tokens-per-minute limiter for Groq calls.

    Groq's free/on-demand tier caps tokens-per-minute (TPM ~30k). Vision + OCR
    over many frames easily exceeds that in a single burst, which previously
    exhausted every model and failed the whole video. This paces requests so the
    minute budget is respected: a call that would overflow the window blocks until
    the window rolls over. The default leaves headroom under the real 30k limit and
    is overridable via the GROQ_TPM env var.
    """

    def __init__(self, tpm: int):
        self.tpm = max(1000, tpm)
        self._lock = threading.Lock()
        self._window_start = time.time()
        self._used = 0

    def acquire(self, tokens: int) -> None:
        tokens = max(0, int(tokens))
        with self._lock:
            now = time.time()
            if now - self._window_start >= 60.0:
                self._window_start = now
                self._used = 0
            if self._used + tokens > self.tpm and self._used > 0:
                sleep_for = 60.0 - (now - self._window_start)
                if sleep_for > 0:
                    logger.info(f"Groq TPM budget reached ({self._used}/{self.tpm}); pausing {sleep_for:.1f}s.")
                    time.sleep(sleep_for)
                self._window_start = time.time()
                self._used = 0
            self._used += tokens


# Shared across all GroqProvider instances so vision, OCR and LLM share one budget.
_GROQ_LIMITER = _TokenRateLimiter(int(os.environ.get("GROQ_TPM", "1000000")))

# Rough token cost of one downscaled (<=600px) image for llama-4 vision models.
_IMAGE_TOKEN_ESTIMATE = 2900


def _normalize_groq_error(e: Exception) -> Exception:
    """Map raw OpenAI SDK exceptions from Groq to normalized provider errors."""
    if isinstance(e, openai.AuthenticationError):
        return AuthenticationError(f"Groq Authentication failed: {e}")
    elif isinstance(e, openai.RateLimitError):
        return RateLimitError(f"Groq Rate limit exceeded: {e}")
    elif isinstance(e, openai.APITimeoutError):
        return TimeoutError(f"Groq Request timed out: {e}")
    elif isinstance(e, (openai.APIConnectionError, openai.InternalServerError)):
        return ProviderUnavailableError(f"Groq service unavailable: {e}")
    elif isinstance(e, openai.APIStatusError):
        status_code = getattr(e, "status_code", None)
        if status_code == 429:
            return RateLimitError(f"Groq Rate limit exceeded: {e}")
        elif status_code in (401, 403):
            return AuthenticationError(f"Groq Authentication failed: {e}")
        elif status_code >= 500:
            return ProviderUnavailableError(f"Groq Server error: {e}")
    return ProviderError(f"Groq Provider call failed: {e}")


def _call_groq_with_retry(api_call_fn, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 8.0):
    """Executes Groq client API call with exponential backoff."""
    attempt = 0
    while True:
        try:
            return api_call_fn()
        except (openai.RateLimitError, openai.InternalServerError, openai.APIConnectionError, openai.APITimeoutError) as e:
            attempt += 1
            if attempt > max_retries:
                raise _normalize_groq_error(e) from e
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            logger.warning(f"Groq transient error ({e}). Retrying in {delay:.2f}s...")
            time.sleep(delay)
        except Exception as e:
            raise _normalize_groq_error(e) from e


class GroqProvider(VisionProvider, SpeechProvider, OCRProvider, LLMProvider):
    """Groq Adapter using OpenAI-compatible SDK endpoint."""

    PROVIDER_NAME = "groq"

    # Current Groq model IDs (the older *-vision-preview models were decommissioned).
    DEFAULT_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
    DEFAULT_LLM_MODEL = "llama-3.3-70b-versatile"
    DEFAULT_SPEECH_MODEL = "whisper-large-v3"

    def __init__(self, api_key: str, base_url: Optional[str] = None):
        self.api_key = api_key
        self.base_url = base_url or "https://api.groq.com/openai/v1"
        self.client = OpenAI(api_key=api_key, base_url=self.base_url, timeout=30.0)

    def get_name(self) -> str:
        return "groq"

    def get_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_vision=True,
            supports_audio=True,
            supports_json=True,
            supports_multimodal=True,
        )

    def _execute_with_model_fallback(
        self,
        base_model_name: str,
        candidates: List[str],
        api_call_fn_generator,
        max_retries: int = 3
    ) -> tuple[Any, str]:
        """
        Execute API call, falling back to other Groq models if rate limits or errors are hit.
        Returns a tuple of (response, model_used).
        """
        # De-duplicate candidates while preserving order
        seen = set()
        model_list = []
        for m in [base_model_name] + candidates:
            if m and m not in seen:
                seen.add(m)
                model_list.append(m)

        last_err = None
        for model in model_list:
            try:
                fn = api_call_fn_generator(model)
                res = _call_groq_with_retry(fn, max_retries=max_retries)
                return res, model
            except (RateLimitError, ProviderUnavailableError) as e:
                logger.warning(f"Groq model '{model}' failed (rate limit/unavailable). Trying fallback...")
                last_err = e
            except Exception as e:
                if isinstance(e, AuthenticationError):
                    raise
                logger.warning(f"Groq model '{model}' error: {e}. Trying fallback...")
                last_err = e

        raise ProviderError(f"All Groq models exhausted. Last error: {last_err}")

    # 1. Vision Adapter
    def analyze_frames(
        self,
        frames: List[Any],
        prompt: str,
        config: ProviderConfig
    ) -> List[VisionObservation]:
        if not frames:
            return []

        observations: List[VisionObservation] = []
        model_name = config.model_name or self.DEFAULT_VISION_MODEL
        fallbacks = [self.DEFAULT_VISION_MODEL]

        system_prompt = (
            "You are an expert video analysis assistant. Analyze the sequence of keyframes and extract observations.\n"
            "Return a JSON object matching this schema:\n"
            "{\n"
            "  \"observations\": [\n"
            "    {\n"
            "      \"content\": \"description of what is seen in this frame\",\n"
            "      \"timestamp\": float (the exact timestamp of the frame this was observed in, e.g. 1.25),\n"
            "      \"confidence\": float (0.0 to 1.0),\n"
            "      \"observation_type\": \"object\" | \"action\" | \"scene\" | \"emotion\"\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "Return ONLY the valid JSON object. Do not wrap in markdown or add notes."
        )

        user_content = [
            {
                "type": "text",
                "text": "You are analyzing a sequence of keyframes from a video. Extract detailed visual observations. Focus on: people, actions, objects, emotions, sequence of events. For each observation, map it to the corresponding timestamp of the frame it was observed in."
            }
        ]

        valid_timestamps = []
        skipped_count = 0
        for idx, frame in enumerate(frames):
            if not getattr(frame, "frame_data", None):
                skipped_count += 1
                continue
            base64_image = base64.b64encode(frame.frame_data).decode("utf-8")
            image_url = f"data:image/jpeg;base64,{base64_image}"
            valid_timestamps.append(frame.timestamp)
            user_content.append({"type": "text", "text": f"\n[Frame {idx} at timestamp {frame.timestamp:.2f}s]:"})
            user_content.append({
                "type": "image_url",
                "image_url": {"url": image_url}
            })

        if skipped_count > 0:
            logger.warning(
                f"Groq Vision: skipped {skipped_count}/{len(frames)} frames with empty frame_data. "
                f"Only {len(valid_timestamps)} frames will be analyzed."
            )

        if not valid_timestamps:
            logger.error("Groq Vision: all frames had empty frame_data — returning empty observations.")
            return []

        user_content.append({"type": "text", "text": f"\nPrompt: {prompt}"})

        def _get_api_call(model):
            def _api_call():
                total_image_tokens = len(valid_timestamps) * _IMAGE_TOKEN_ESTIMATE
                _GROQ_LIMITER.acquire(total_image_tokens + (config.max_tokens or 2048))
                t0 = time.time()
                res = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    max_tokens=config.max_tokens or 2048,
                    temperature=config.temperature or 0.2,
                    response_format={"type": "json_object"}
                )
                latency = time.time() - t0
                global_cost_tracker.track_call(
                    provider="groq",
                    model=model,
                    prompt_tokens=res.usage.prompt_tokens if res.usage else 0,
                    completion_tokens=res.usage.completion_tokens if res.usage else 0,
                    latency=latency
                )
                return res
            return _api_call

        try:
            response, actual_model = self._execute_with_model_fallback(
                model_name,
                fallbacks,
                _get_api_call,
                max_retries=config.max_retries or 3
            )
            text = response.choices[0].message.content.strip()
            from src.shared.fireworks_providers import _parse_and_repair_json
            parsed = _parse_and_repair_json(text)
            if isinstance(parsed, dict):
                for val in parsed.values():
                    if isinstance(val, list) and all(isinstance(x, dict) for x in val):
                        parsed = val
                        break
                else:
                    parsed = [parsed]

            if isinstance(parsed, list):
                for item in parsed:
                    val = item.get("content", "").strip()
                    if not val:
                        continue

                    # Align the reported timestamp back to closest valid frame timestamp
                    t_val = item.get("timestamp")
                    if t_val is not None:
                        try:
                            t_val = float(t_val)
                            if valid_timestamps:
                                t_val = min(valid_timestamps, key=lambda x: abs(x - t_val))
                        except (ValueError, TypeError):
                            t_val = frames[0].timestamp
                    else:
                        t_val = frames[0].timestamp

                    raw_conf = item.get("confidence", 0.5)
                    observations.append(
                        VisionObservation(
                            content=val,
                            timestamp=t_val,
                            confidence=raw_conf,
                            observation_type=item.get("observation_type", "action")
                        )
                    )
            return sorted(observations, key=lambda x: x.timestamp)
        except Exception as e:
            logger.error(f"Groq Vision provider analysis failed: {e}")
            raise ProviderError(f"Groq Vision provider analysis failed: {e}") from e

    # 2. OCR Adapter
    def extract_text(
        self,
        frames: List[Any],
        config: ProviderConfig
    ) -> List[OCRObservation]:
        observations: List[OCRObservation] = []
        model_name = config.model_name or self.DEFAULT_VISION_MODEL
        fallbacks = [self.DEFAULT_VISION_MODEL]

        def _process_single(frame: Any) -> List[OCRObservation]:
            if not getattr(frame, "frame_data", None):
                return []

            base64_image = base64.b64encode(frame.frame_data).decode("utf-8")
            image_url = f"data:image/jpeg;base64,{base64_image}"

            system_prompt = (
                "You are an expert OCR and text detection assistant. Analyze the image and extract any visible text.\n"
                "Return a JSON object matching this schema:\n"
                "{\n"
                "  \"text_detections\": [\n"
                "    {\n"
                "      \"text\": \"detected text content\",\n"
                "      \"confidence\": float (0.0 to 1.0)\n"
                "    }\n"
                "  ]\n"
                "}"
            )

            def _get_api_call(model):
                def _api_call():
                    _GROQ_LIMITER.acquire(_IMAGE_TOKEN_ESTIMATE + (config.max_tokens or 1024))
                    t0 = time.time()
                    res = self.client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "Extract all readable text from this image."},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": image_url}
                                    }
                                ]
                            }
                        ],
                        max_tokens=config.max_tokens or 1024,
                        temperature=config.temperature or 0.1,
                        response_format={"type": "json_object"}
                    )
                    latency = time.time() - t0
                    global_cost_tracker.track_call(
                        provider="groq",
                        model=model,
                        prompt_tokens=res.usage.prompt_tokens if res.usage else 0,
                        completion_tokens=res.usage.completion_tokens if res.usage else 0,
                        latency=latency
                    )
                    return res
                return _api_call

            try:
                response, actual_model = self._execute_with_model_fallback(
                    model_name,
                    fallbacks,
                    _get_api_call,
                    max_retries=config.max_retries or 3
                )
                text = response.choices[0].message.content.strip()
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    for val in parsed.values():
                        if isinstance(val, list) and all(isinstance(x, dict) for x in val):
                            parsed = val
                            break
                    else:
                        parsed = [parsed]
                
                res_obs = []
                if isinstance(parsed, list):
                    for item in parsed:
                        val = item.get("text", "").strip()
                        if val:
                            res_obs.append(
                                OCRObservation(
                                    content=val,
                                    timestamp=frame.timestamp,
                                    confidence=item.get("confidence", 0.5)
                                )
                            )
                return res_obs
            except Exception as e:
                logger.error(f"Groq OCR failed at {frame.timestamp}s: {e}")
                raise _normalize_groq_error(e) from e

        try:
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(_process_single, f) for f in frames]
                for fut in futures:
                    observations.extend(fut.result())
            return sorted(observations, key=lambda x: x.timestamp)
        except Exception as e:
            raise ProviderError(f"Groq OCR extraction failed: {e}") from e

    # 3. Speech Adapter (Groq Whisper via OpenAI-compatible endpoint)
    def transcribe(
        self,
        frames: List[Any],
        config: ProviderConfig
    ) -> List[SpeechObservation]:
        if not frames:
            return []

        observations: List[SpeechObservation] = []
        model_name = config.model_name or self.DEFAULT_SPEECH_MODEL
        # whisper-v3 is not a valid Groq id; normalize legacy config values.
        if model_name in ("whisper-v3", "whisper-1", "whisper"):
            model_name = self.DEFAULT_SPEECH_MODEL

        def _transcribe_single(frame: Any) -> Optional[SpeechObservation]:
            if not getattr(frame, "audio_segment", None):
                return None

            audio_file = ("segment.wav", io.BytesIO(frame.audio_segment), "audio/wav")

            def _api_call():
                t0 = time.time()
                res = self.client.audio.transcriptions.create(
                    model=model_name,
                    file=audio_file,
                    language=config.extra_params.get("language", "en")
                )
                latency = time.time() - t0
                global_cost_tracker.track_call(
                    provider="groq",
                    model=model_name,
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency=latency
                )
                return res

            response = _call_groq_with_retry(_api_call, max_retries=config.max_retries or 3)
            text = (response.text or "").strip()
            if text:
                return SpeechObservation(content=text, timestamp=frame.timestamp, confidence=0.9)
            return None

        try:
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(_transcribe_single, f)
                    for f in frames if getattr(f, "audio_segment", None)
                ]
                for fut in futures:
                    res = fut.result()
                    if res:
                        observations.append(res)
            return sorted(observations, key=lambda x: x.timestamp)
        except Exception as e:
            raise ProviderError(f"Groq Speech transcription failed: {e}") from e

    # 4. LLM Adapter
    def generate(
        self,
        prompt: str,
        system_prompt: str,
        config: ProviderConfig
    ) -> LLMResponse:
        model_name = config.model_name or self.DEFAULT_LLM_MODEL
        fallbacks = ["llama-3.1-8b-instant"]
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        def _get_api_call(model):
            def _api_call():
                est = len(prompt + system_prompt) // 4 + (config.max_tokens or 512)
                _GROQ_LIMITER.acquire(est)
                t0 = time.time()
                fmt = {"type": "json_object"} if config.extra_params.get("json_mode", False) else None
                res = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=config.max_tokens or 512,
                    temperature=config.temperature or 0.7,
                    response_format=fmt
                )
                latency = time.time() - t0
                global_cost_tracker.track_call(
                    provider="groq",
                    model=model,
                    prompt_tokens=res.usage.prompt_tokens if res.usage else 0,
                    completion_tokens=res.usage.completion_tokens if res.usage else 0,
                    latency=latency
                )
                return res, latency
            return _api_call

        try:
            (response, latency), actual_model = self._execute_with_model_fallback(
                model_name,
                fallbacks,
                _get_api_call,
                max_retries=config.max_retries or 3
            )
            text = response.choices[0].message.content.strip()
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = response.usage.completion_tokens if response.usage else 0
            return LLMResponse(
                text=text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency=latency
            )
        except Exception as e:
            raise _normalize_groq_error(e) from e


# Auto-register Groq provider
ProviderRegistry.register("groq", GroqProvider)
