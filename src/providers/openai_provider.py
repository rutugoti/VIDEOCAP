import base64
import io
import json
import logging
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
    EmbeddingProvider,
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
    EmbeddingResponse,
    global_cost_tracker,
)
from src.providers.registry import ProviderRegistry

logger = logging.getLogger(__name__)


def _normalize_openai_error(e: Exception) -> Exception:
    """Map raw OpenAI SDK exceptions to normalized provider errors."""
    if isinstance(e, openai.AuthenticationError):
        return AuthenticationError(f"OpenAI Authentication failed: {e}")
    elif isinstance(e, openai.RateLimitError):
        return RateLimitError(f"OpenAI Rate limit exceeded: {e}")
    elif isinstance(e, openai.APITimeoutError):
        return TimeoutError(f"OpenAI Request timed out: {e}")
    elif isinstance(e, (openai.APIConnectionError, openai.InternalServerError)):
        return ProviderUnavailableError(f"OpenAI service unavailable: {e}")
    elif isinstance(e, openai.APIStatusError):
        status_code = getattr(e, "status_code", None)
        if status_code == 429:
            return RateLimitError(f"OpenAI Rate limit exceeded: {e}")
        elif status_code in (401, 403):
            return AuthenticationError(f"OpenAI Authentication failed: {e}")
        elif status_code >= 500:
            return ProviderUnavailableError(f"OpenAI Server error: {e}")
    return ProviderError(f"OpenAI Provider call failed: {e}")


def _call_openai_with_retry(api_call_fn, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 8.0):
    """Executes OpenAI client API call with exponential backoff."""
    attempt = 0
    while True:
        try:
            return api_call_fn()
        except (openai.RateLimitError, openai.InternalServerError, openai.APIConnectionError, openai.APITimeoutError) as e:
            attempt += 1
            if attempt > max_retries:
                raise _normalize_openai_error(e) from e
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            logger.warning(f"OpenAI transient error ({e}). Retrying in {delay:.2f}s...")
            time.sleep(delay)
        except Exception as e:
            raise _normalize_openai_error(e) from e


class OpenAIProvider(VisionProvider, SpeechProvider, OCRProvider, LLMProvider, EmbeddingProvider):
    """OpenAI Adapter implementing multi-modality and embedding interfaces."""

    PROVIDER_NAME = "openai"

    def __init__(self, api_key: str, base_url: Optional[str] = None):
        self.api_key = api_key
        self.base_url = base_url
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=30.0) if base_url else OpenAI(api_key=api_key, timeout=30.0)

    def get_name(self) -> str:
        return "openai"

    def get_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_vision=True,
            supports_audio=True,
            supports_json=True,
            supports_embeddings=True,
            supports_multimodal=True,
        )

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
        model_name = config.model_name or "gpt-4o"

        def _process_single(frame: Any) -> List[VisionObservation]:
            if not getattr(frame, "frame_data", None):
                return []

            base64_image = base64.b64encode(frame.frame_data).decode("utf-8")
            image_url = f"data:image/jpeg;base64,{base64_image}"

            system_prompt = (
                "You are an expert video analysis assistant. Analyze the image and extract observations.\n"
                "Return a JSON object matching this schema:\n"
                "{\n"
                "  \"observations\": [\n"
                "    {\n"
                "      \"content\": \"description\",\n"
                "      \"confidence\": float (0.0 to 1.0),\n"
                "      \"observation_type\": \"object\", \"action\", \"scene\", or \"emotion\"\n"
                "    }\n"
                "  ]\n"
                "}"
            )

            def _api_call():
                t0 = time.time()
                res = self.client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": image_url}
                                }
                            ]
                        }
                    ],
                    max_tokens=config.max_tokens or 1024,
                    temperature=config.temperature or 0.2,
                    response_format={"type": "json_object"}
                )
                latency = time.time() - t0
                global_cost_tracker.track_call(
                    provider="openai",
                    model=model_name,
                    prompt_tokens=res.usage.prompt_tokens if res.usage else 0,
                    completion_tokens=res.usage.completion_tokens if res.usage else 0,
                    latency=latency
                )
                return res

            try:
                response = _call_openai_with_retry(_api_call, max_retries=config.max_retries or 3)
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
                        res_obs.append(
                            VisionObservation(
                                content=item.get("content", "unspecified visual observation"),
                                timestamp=frame.timestamp,
                                confidence=item.get("confidence", 0.5),
                                observation_type=item.get("observation_type", "action")
                            )
                        )
                return res_obs
            except Exception as e:
                logger.error(f"OpenAI Vision failed at {frame.timestamp}s: {e}")
                raise _normalize_openai_error(e) from e

        try:
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(_process_single, f) for f in frames]
                for fut in futures:
                    observations.extend(fut.result())
            return sorted(observations, key=lambda x: x.timestamp)
        except Exception as e:
            raise ProviderError(f"OpenAI Vision failed: {e}") from e

    # 2. Speech Adapter
    def transcribe(
        self,
        frames: List[Any],
        config: ProviderConfig
    ) -> List[SpeechObservation]:
        if not frames:
            return []

        observations: List[SpeechObservation] = []
        model_name = config.model_name or "whisper-1"

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
                    provider="openai",
                    model=model_name,
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency=latency
                )
                return res

            try:
                response = _call_openai_with_retry(_api_call, max_retries=config.max_retries or 3)
                text = response.text.strip()
                if text:
                    return SpeechObservation(
                        content=text,
                        timestamp=frame.timestamp,
                        confidence=0.9
                    )
            except Exception as e:
                logger.error(f"OpenAI Whisper failed at {frame.timestamp}s: {e}")
                raise _normalize_openai_error(e) from e
            return None

        try:
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(_transcribe_single, f) for f in frames if getattr(f, "audio_segment", None)]
                for fut in futures:
                    res = fut.result()
                    if res:
                        observations.append(res)
            return sorted(observations, key=lambda x: x.timestamp)
        except Exception as e:
            raise ProviderError(f"OpenAI Speech transcription failed: {e}") from e

    # 3. OCR Adapter
    def extract_text(
        self,
        frames: List[Any],
        config: ProviderConfig
    ) -> List[OCRObservation]:
        # Reuse Vision model or vision endpoint
        observations: List[OCRObservation] = []
        model_name = config.model_name or "gpt-4o-mini"

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

            def _api_call():
                t0 = time.time()
                res = self.client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Extract all visible text in this image."},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": image_url}
                                }
                            ]
                        }
                    ],
                    max_tokens=config.max_tokens or 1024,
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                latency = time.time() - t0
                global_cost_tracker.track_call(
                    provider="openai",
                    model=model_name,
                    prompt_tokens=res.usage.prompt_tokens if res.usage else 0,
                    completion_tokens=res.usage.completion_tokens if res.usage else 0,
                    latency=latency
                )
                return res

            try:
                response = _call_openai_with_retry(_api_call, max_retries=config.max_retries or 3)
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
                logger.error(f"OpenAI OCR failed at {frame.timestamp}s: {e}")
                raise _normalize_openai_error(e) from e

        try:
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(_process_single, f) for f in frames]
                for fut in futures:
                    observations.extend(fut.result())
            return sorted(observations, key=lambda x: x.timestamp)
        except Exception as e:
            raise ProviderError(f"OpenAI OCR extraction failed: {e}") from e

    # 4. LLM Adapter
    def generate(
        self,
        prompt: str,
        system_prompt: str,
        config: ProviderConfig
    ) -> LLMResponse:
        model_name = config.model_name or "gpt-4o-mini"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        def _api_call():
            t0 = time.time()
            fmt = {"type": "json_object"} if config.extra_params.get("json_mode", False) else None
            res = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=config.max_tokens or 512,
                temperature=config.temperature or 0.7,
                response_format=fmt
            )
            latency = time.time() - t0
            global_cost_tracker.track_call(
                provider="openai",
                model=model_name,
                prompt_tokens=res.usage.prompt_tokens if res.usage else 0,
                completion_tokens=res.usage.completion_tokens if res.usage else 0,
                latency=latency
            )
            return res, latency

        try:
            response, latency = _call_openai_with_retry(_api_call, max_retries=config.max_retries or 3)
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
            raise _normalize_openai_error(e) from e

    # 5. Embedding Adapter
    def embed_query(
        self,
        text: str,
        config: ProviderConfig
    ) -> EmbeddingResponse:
        model_name = config.model_name or "text-embedding-3-small"

        def _api_call():
            t0 = time.time()
            res = self.client.embeddings.create(
                model=model_name,
                input=text
            )
            latency = time.time() - t0
            global_cost_tracker.track_call(
                provider="openai",
                model=model_name,
                prompt_tokens=res.usage.prompt_tokens if res.usage else 0,
                completion_tokens=0,
                latency=latency
            )
            return res, latency

        try:
            response, latency = _call_openai_with_retry(_api_call, max_retries=config.max_retries or 3)
            embedding = response.data[0].embedding
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            return EmbeddingResponse(
                embedding=embedding,
                prompt_tokens=prompt_tokens,
                latency=latency
            )
        except Exception as e:
            raise _normalize_openai_error(e) from e


# Auto-register OpenAI provider
ProviderRegistry.register("openai", OpenAIProvider)
