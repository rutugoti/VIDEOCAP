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


def _parse_and_repair_json(text: str) -> dict | list:
    """Robust parser that strips markdown/formatting wrappers and performs automatic repair attempts on malformed JSON."""
    text = text.strip()
    if not text:
        return {}

    if text.startswith("```"):
        newline_idx = text.find("\n")
        if newline_idx != -1:
            text = text[newline_idx:].strip()
        if text.endswith("```"):
            text = text[:-3].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start_bracket = text.find("[")
    end_bracket = text.rfind("]")
    start_brace = text.find("{")
    end_brace = text.rfind("}")

    if start_bracket != -1 and end_bracket != -1 and end_bracket > start_bracket:
        candidate = text[start_bracket:end_bracket + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
        candidate = text[start_brace:end_brace + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    cleaned = text.replace("None", "null").replace("True", "true").replace("False", "false")
    for boundary in ["{", "["]:
        idx = cleaned.find(boundary)
        if idx != -1:
            r_boundary = "}" if boundary == "{" else "]"
            ridx = cleaned.rfind(r_boundary)
            if ridx != -1:
                try:
                    return json.loads(cleaned[idx:ridx + 1])
                except json.JSONDecodeError:
                    pass

    return {}


def _normalize_openai_error(e: Exception) -> Exception:
    """Map raw OpenAI SDK exceptions to normalized provider errors."""
    if isinstance(e, openai.AuthenticationError):
        return AuthenticationError(f"Authentication failed: {e}")
    elif isinstance(e, openai.RateLimitError):
        return RateLimitError(f"Rate limit exceeded: {e}")
    elif isinstance(e, openai.APITimeoutError):
        return TimeoutError(f"Request timed out: {e}")
    elif isinstance(e, (openai.APIConnectionError, openai.InternalServerError)):
        return ProviderUnavailableError(f"Service unavailable or connection failed: {e}")
    elif isinstance(e, openai.APIStatusError):
        status_code = getattr(e, "status_code", None)
        if status_code == 429:
            return RateLimitError(f"Rate limit exceeded: {e}")
        elif status_code in (401, 403):
            return AuthenticationError(f"Authentication failed: {e}")
        elif status_code >= 500:
            return ProviderUnavailableError(f"Server error: {e}")
    return ProviderError(f"Provider call failed: {e}")


def _call_openai_with_retry(api_call_fn, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 8.0):
    """Executes OpenAI client API call with exponential backoff, retrying only transient failures."""
    attempt = 0
    while True:
        try:
            return api_call_fn()
        except (openai.RateLimitError, openai.InternalServerError, openai.APIConnectionError, openai.APITimeoutError) as e:
            attempt += 1
            if attempt > max_retries:
                logger.error(f"Transient error persisted after {max_retries} retries: {e}")
                raise _normalize_openai_error(e) from e
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            logger.warning(f"Transient API error ({e}). Retrying in {delay:.2f}s (Attempt {attempt}/{max_retries})...")
            time.sleep(delay)
        except openai.APIStatusError as e:
            status_code = getattr(e, "status_code", None)
            if status_code in (429, 500, 502, 503, 504):
                attempt += 1
                if attempt > max_retries:
                    raise _normalize_openai_error(e) from e
                delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
                logger.warning(f"Transient API status error {status_code}. Retrying in {delay:.2f}s (Attempt {attempt}/{max_retries})...")
                time.sleep(delay)
            else:
                raise _normalize_openai_error(e) from e
        except Exception as e:
            raise _normalize_openai_error(e) from e


class FireworksProvider(VisionProvider, SpeechProvider, OCRProvider, LLMProvider):
    """Fireworks Adapter implementing multi-modality interfaces (OpenAI-compatible)."""

    PROVIDER_NAME = "fireworks"

    def __init__(self, api_key: str = "mock_key_for_testing", base_url: Optional[str] = None):
        self.api_key = api_key
        self.base_url = base_url or "https://api.fireworks.ai/inference/v1"
        self.client = OpenAI(api_key=api_key, base_url=self.base_url)

    def get_name(self) -> str:
        return "fireworks"

    def get_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_vision=True,
            supports_audio=True,
            supports_json=True,
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

        # If key is mock, bypass and return mock data for offline tests
        if self.api_key == "mock_key_for_testing":
            logger.info("Using mock vision provider response for testing key.")
            # Delegate to standard mock
            from src.providers.mock_provider import MockProvider
            mock_inst = MockProvider()
            return mock_inst.analyze_frames(frames, prompt, config)

        observations: List[VisionObservation] = []

        def _process_single(frame: Any) -> List[VisionObservation]:
            if not getattr(frame, "frame_data", None):
                return []

            base64_image = base64.b64encode(frame.frame_data).decode("utf-8")
            image_url = f"data:image/jpeg;base64,{base64_image}"

            system_prompt = (
                "You are an expert video analysis assistant. Analyze the image and extract observations.\n"
                "Return a JSON list of objects matching this schema:\n"
                "[\n"
                "  {\n"
                "    \"content\": \"description of what is seen\",\n"
                "    \"confidence\": float (0.0 to 1.0),\n"
                "    \"observation_type\": \"object\", \"action\", \"scene\", or \"emotion\"\n"
                "  }\n"
                "]\n"
                "Return ONLY the valid JSON list. Do not wrap in markdown or add notes."
            )

            def _api_call():
                t0 = time.time()
                try:
                    res = self.client.chat.completions.create(
                        model=config.model_name,
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
                except openai.BadRequestError:
                    res = self.client.chat.completions.create(
                        model=config.model_name,
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
                        temperature=config.temperature or 0.2
                    )
                latency = time.time() - t0
                # Cost tracking
                prompt_tokens = res.usage.prompt_tokens if res.usage else 0
                completion_tokens = res.usage.completion_tokens if res.usage else 0
                global_cost_tracker.track_call(
                    provider="fireworks",
                    model=config.model_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency=latency
                )
                return res

            try:
                response = _call_openai_with_retry(_api_call, max_retries=config.max_retries or 3)
                text = response.choices[0].message.content.strip()
                parsed = _parse_and_repair_json(text)
                if isinstance(parsed, dict):
                    parsed = [parsed]
                
                res_obs = []
                if isinstance(parsed, list):
                    for item in parsed:
                        raw = item.get("confidence", 0.5)
                        res_obs.append(
                            VisionObservation(
                                content=item.get("content", "unspecified visual observation"),
                                timestamp=frame.timestamp,
                                confidence=raw,
                                observation_type=item.get("observation_type", "action")
                            )
                        )
                return res_obs
            except Exception as e:
                logger.error(f"Fireworks Vision API failed at {frame.timestamp}s: {e}")
                raise _normalize_openai_error(e) from e

        try:
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(_process_single, f) for f in frames]
                for fut in futures:
                    observations.extend(fut.result())
            return sorted(observations, key=lambda x: x.timestamp)
        except Exception as e:
            logger.error(f"Vision provider analysis failed: {e}")
            raise ProviderError(f"Vision provider analysis failed: {e}") from e

    # 2. Speech Adapter
    def transcribe(
        self,
        frames: List[Any],
        config: ProviderConfig
    ) -> List[SpeechObservation]:
        if not frames:
            return []

        if self.api_key == "mock_key_for_testing":
            logger.info("Using mock audio provider response for testing key.")
            from src.providers.mock_provider import MockProvider
            mock_inst = MockProvider()
            return mock_inst.transcribe(frames, config)

        observations: List[SpeechObservation] = []

        def _transcribe_single(frame: Any) -> Optional[SpeechObservation]:
            if not getattr(frame, "audio_segment", None):
                return None

            audio_file = ("segment.wav", io.BytesIO(frame.audio_segment), "audio/wav")
            
            def _api_call():
                t0 = time.time()
                res = self.client.audio.transcriptions.create(
                    model=config.model_name,
                    file=audio_file,
                    language=config.extra_params.get("language", "en")
                )
                latency = time.time() - t0
                global_cost_tracker.track_call(
                    provider="fireworks",
                    model=config.model_name,
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
                logger.error(f"Fireworks Whisper API failed at {frame.timestamp}s: {e}")
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
            logger.error(f"Audio provider transcription failed: {e}")
            raise ProviderError(f"Audio provider transcription failed: {e}") from e

    # 3. OCR Adapter
    def extract_text(
        self,
        frames: List[Any],
        config: ProviderConfig
    ) -> List[OCRObservation]:
        if not frames:
            return []

        if self.api_key == "mock_key_for_testing":
            logger.info("Using mock OCR provider response for testing key.")
            from src.providers.mock_provider import MockProvider
            mock_inst = MockProvider()
            return mock_inst.extract_text(frames, config)

        observations: List[OCRObservation] = []

        def _process_single(frame: Any) -> List[OCRObservation]:
            if not getattr(frame, "frame_data", None):
                return []

            base64_image = base64.b64encode(frame.frame_data).decode("utf-8")
            image_url = f"data:image/jpeg;base64,{base64_image}"

            system_prompt = (
                "You are an expert OCR and text detection assistant. Analyze the image and extract any visible text.\n"
                "This includes signs, subtitles, logos, labels, or brand names.\n"
                "Return a JSON list of objects matching this schema:\n"
                "[\n"
                "  {\n"
                "    \"text\": \"detected text content\",\n"
                "    \"confidence\": float (0.0 to 1.0)\n"
                "  }\n"
                "]\n"
                "Return ONLY the valid JSON list. If no text is visible, return an empty list."
            )

            def _api_call():
                t0 = time.time()
                try:
                    res = self.client.chat.completions.create(
                        model=config.model_name,
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
                except openai.BadRequestError:
                    res = self.client.chat.completions.create(
                        model=config.model_name,
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
                        temperature=0.1
                    )
                latency = time.time() - t0
                prompt_tokens = res.usage.prompt_tokens if res.usage else 0
                completion_tokens = res.usage.completion_tokens if res.usage else 0
                global_cost_tracker.track_call(
                    provider="fireworks",
                    model=config.model_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency=latency
                )
                return res

            try:
                response = _call_openai_with_retry(_api_call, max_retries=config.max_retries or 3)
                text = response.choices[0].message.content.strip()
                parsed = _parse_and_repair_json(text)
                if isinstance(parsed, dict):
                    parsed = [parsed]
                
                res_obs = []
                if isinstance(parsed, list):
                    for item in parsed:
                        val = item.get("text", "").strip()
                        if val:
                            raw = item.get("confidence", 0.5)
                            res_obs.append(
                                OCRObservation(
                                    content=val,
                                    timestamp=frame.timestamp,
                                    confidence=raw
                                )
                            )
                return res_obs
            except Exception as e:
                logger.error(f"Fireworks OCR API failed at {frame.timestamp}s: {e}")
                raise _normalize_openai_error(e) from e

        try:
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(_process_single, f) for f in frames]
                for fut in futures:
                    observations.extend(fut.result())
            return sorted(observations, key=lambda x: x.timestamp)
        except Exception as e:
            logger.error(f"OCR provider extraction failed: {e}")
            raise ProviderError(f"OCR provider extraction failed: {e}") from e

    # 4. LLM Adapter
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

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        def _api_call():
            t0 = time.time()
            try:
                res = self.client.chat.completions.create(
                    model=config.model_name,
                    messages=messages,
                    max_tokens=config.max_tokens or 512,
                    temperature=config.temperature or 0.7,
                    response_format={"type": "json_object"}
                )
            except openai.BadRequestError:
                res = self.client.chat.completions.create(
                    model=config.model_name,
                    messages=messages,
                    max_tokens=config.max_tokens or 512,
                    temperature=config.temperature or 0.7
                )
            latency = time.time() - t0
            prompt_tokens = res.usage.prompt_tokens if res.usage else 0
            completion_tokens = res.usage.completion_tokens if res.usage else 0
            global_cost_tracker.track_call(
                provider="fireworks",
                model=config.model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
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
            logger.error(f"Fireworks LLM generation failed: {e}")
            raise _normalize_openai_error(e) from e


# Auto-register Fireworks provider
ProviderRegistry.register("fireworks", FireworksProvider)
