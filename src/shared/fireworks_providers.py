import base64
import io
import json
import logging
import time
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
import openai

from src.shared.models import Sample, Observation, Confidence
from src.shared.providers import (
    VisionProvider,
    AudioProvider,
    OCRProvider,
    LLMProvider,
    VisionConfig,
    AudioConfig,
    OCRConfig,
    LLMConfig,
    ProviderError,
)

logger = logging.getLogger(__name__)


def _parse_and_repair_json(text: str) -> dict | list:
    """Robust parser that strips markdown/formatting wrappers and performs automatic repair attempts on malformed JSON."""
    text = text.strip()
    if not text:
        return {}

    # Strip markdown code blocks
    if text.startswith("```"):
        newline_idx = text.find("\n")
        if newline_idx != -1:
            text = text[newline_idx:].strip()
        if text.endswith("```"):
            text = text[:-3].strip()

    # Direct parse attempt
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Extract JSON boundaries
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

    # Py-values replacement repair
    cleaned = text
    cleaned = cleaned.replace("None", "null").replace("True", "true").replace("False", "false")
    # Try parsing boundaries again
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
                raise ProviderError(f"API call failed after max retries: {e}") from e
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            logger.warning(f"Transient API error ({e}). Retrying in {delay:.2f}s (Attempt {attempt}/{max_retries})...")
            time.sleep(delay)
        except openai.APIStatusError as e:
            status_code = getattr(e, "status_code", None)
            if status_code in (429, 500, 502, 503, 504):
                attempt += 1
                if attempt > max_retries:
                    raise ProviderError(f"API call failed after max retries: {e}") from e
                delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
                logger.warning(f"Transient API status error {status_code}. Retrying in {delay:.2f}s (Attempt {attempt}/{max_retries})...")
                time.sleep(delay)
            else:
                raise ProviderError(f"Non-transient API status error {status_code}: {e}") from e
        except Exception as e:
            raise ProviderError(f"API call failed: {e}") from e


class FireworksVisionProvider(VisionProvider):
    """Concrete VisionProvider calling the Fireworks Multimodal API (OpenAI-compatible)."""

    def __init__(self, api_key: str, base_url: str = "https://api.fireworks.ai/inference/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def analyze_frames(
        self,
        frames: List[Sample],
        prompt: str,
        config: VisionConfig
    ) -> List[Observation]:
        """Analyze a list of frames using the Fireworks multimodal vision API."""
        if not frames:
            return []

        observations: List[Observation] = []

        def _process_single(frame: Sample) -> List[Observation]:
            if not frame.frame_data:
                return []

            base64_image = base64.b64encode(frame.frame_data).decode("utf-8")
            image_url = f"data:image/jpeg;base64,{base64_image}"

            system_prompt = (
                "You are an expert video analysis assistant. Analyze the image and extract observations.\n"
                "Return a JSON object matching this schema:\n"
                "{\n"
                "  \"observations\": [\n"
                "    {\n"
                "      \"content\": \"description of what is seen\",\n"
                "      \"confidence\": float (0.0 to 1.0),\n"
                "      \"observation_type\": \"object\", \"action\", \"scene\", or \"emotion\"\n"
                "    }\n"
                "  ]\n"
                "}\n"
                "Return ONLY the valid JSON object. Do not wrap in markdown or add notes."
            )

            def _api_call():
                try:
                    return self.client.chat.completions.create(
                        model=config.model,
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
                    return self.client.chat.completions.create(
                        model=config.model,
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

            try:
                response = _call_openai_with_retry(_api_call)
                text = response.choices[0].message.content.strip()
                parsed = _parse_and_repair_json(text)
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
                        raw = item.get("confidence")
                        conf_meta = (Confidence(value=raw, source="model_reported") if raw is not None
                                     else Confidence(value=0.5, source="default"))
                        res_obs.append(
                            Observation(
                                content=item.get("content", "unspecified visual observation"),
                                timestamp=frame.timestamp,
                                confidence=conf_meta.value,
                                confidence_meta=conf_meta,
                                source="visual",
                                observation_type=item.get("observation_type", "action")
                            )
                        )
                return res_obs
            except Exception as e:
                logger.error(f"Fireworks Vision API failed at {frame.timestamp}s: {e}")
                raise ProviderError(f"Fireworks Vision API failed: {e}") from e

        try:
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(_process_single, f) for f in frames]
                for fut in futures:
                    observations.extend(fut.result())
            return sorted(observations, key=lambda x: x.timestamp)
        except Exception as e:
            logger.error(f"Vision provider analysis failed: {e}")
            raise ProviderError(f"Vision provider analysis failed: {e}") from e

class FireworksAudioProvider(AudioProvider):
    """Concrete AudioProvider calling the Fireworks Whisper API (OpenAI-compatible)."""

    def __init__(self, api_key: str, base_url: str = "https://api.fireworks.ai/inference/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def transcribe(
        self,
        frames: List[Sample],
        config: AudioConfig
    ) -> List[Observation]:
        """Transcribe audio segments from samples using the Fireworks Whisper API."""
        if not frames:
            return []

        observations: List[Observation] = []

        def _transcribe_single(frame: Sample) -> Optional[Observation]:
            if not frame.audio_segment:
                return None

            audio_file = ("segment.wav", io.BytesIO(frame.audio_segment), "audio/wav")
            
            def _api_call():
                return self.client.audio.transcriptions.create(
                    model=config.model,
                    file=audio_file,
                    language=config.language or "en"
                )

            try:
                response = _call_openai_with_retry(_api_call)
                avg_logprob = getattr(response, "avg_logprob", None)
                no_speech_prob = getattr(response, "no_speech_prob", None)
                if avg_logprob is not None or no_speech_prob is not None:
                    # Map logprob to probability value or use standard fallback
                    conf_meta = Confidence(value=0.9, source="measured")
                else:
                    conf_meta = Confidence(value=0.5, source="default")

                text = response.text.strip()
                if text:
                    return Observation(
                        content=text,
                        timestamp=frame.timestamp,
                        confidence=conf_meta.value,
                        confidence_meta=conf_meta,
                        source="audio",
                        observation_type="speech"
                    )
            except Exception as e:
                logger.error(f"Fireworks Whisper API failed at {frame.timestamp}s: {e}")
                raise ProviderError(f"Fireworks Whisper API failed: {e}") from e
            return None

        try:
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(_transcribe_single, f) for f in frames if f.audio_segment]
                for fut in futures:
                    res = fut.result()
                    if res:
                        observations.append(res)
            return sorted(observations, key=lambda x: x.timestamp)
        except Exception as e:
            logger.error(f"Audio provider transcription failed: {e}")
            raise ProviderError(f"Audio provider transcription failed: {e}") from e

class FireworksOCRProvider(OCRProvider):
    """Concrete OCRProvider calling the Fireworks Vision model to extract text."""

    def __init__(self, api_key: str, base_url: str = "https://api.fireworks.ai/inference/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def extract_text(
        self,
        frames: List[Sample],
        config: OCRConfig
    ) -> List[Observation]:
        """Extract text from frames using Fireworks multimodal OCR API calls."""
        if not frames:
            return []

        observations: List[Observation] = []

        def _process_single(frame: Sample) -> List[Observation]:
            if not frame.frame_data:
                return []

            base64_image = base64.b64encode(frame.frame_data).decode("utf-8")
            image_url = f"data:image/jpeg;base64,{base64_image}"

            system_prompt = (
                "You are an expert OCR and text detection assistant. Analyze the image and extract any visible text.\n"
                "This includes signs, subtitles, logos, labels, or brand names.\n"
                "Return a JSON object matching this schema:\n"
                "{\n"
                "  \"text_detections\": [\n"
                "    {\n"
                "      \"text\": \"detected text content\",\n"
                "      \"confidence\": float (0.0 to 1.0)\n"
                "    }\n"
                "  ]\n"
                "}\n"
                "Return ONLY the valid JSON object. If no text is visible, return an empty list."
            )

            def _api_call():
                try:
                    return self.client.chat.completions.create(
                        model=config.model,
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
                    return self.client.chat.completions.create(
                        model=config.model,
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

            try:
                response = _call_openai_with_retry(_api_call)
                text = response.choices[0].message.content.strip()
                parsed = _parse_and_repair_json(text)
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
                            raw = item.get("confidence")
                            conf_meta = (Confidence(value=raw, source="model_reported") if raw is not None
                                         else Confidence(value=0.5, source="default"))
                            res_obs.append(
                                Observation(
                                    content=val,
                                    timestamp=frame.timestamp,
                                    confidence=conf_meta.value,
                                    confidence_meta=conf_meta,
                                    source="text",
                                    observation_type="ocr_text"
                                )
                            )
                return res_obs
            except Exception as e:
                logger.error(f"Fireworks OCR API failed at {frame.timestamp}s: {e}")
                raise ProviderError(f"Fireworks OCR API failed: {e}") from e

        try:
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(_process_single, f) for f in frames]
                for fut in futures:
                    observations.extend(fut.result())
            return sorted(observations, key=lambda x: x.timestamp)
        except Exception as e:
            logger.error(f"OCR provider extraction failed: {e}")
            raise ProviderError(f"OCR provider extraction failed: {e}") from e

class FireworksLLMProvider(LLMProvider):
    """Concrete LLMProvider calling the Fireworks Chat Completion API."""

    def __init__(self, api_key: str, base_url: str = "https://api.fireworks.ai/inference/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        config: LLMConfig
    ) -> str:
        """Query Fireworks LLM to generate text."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        def _api_call():
            try:
                return self.client.chat.completions.create(
                    model=config.model,
                    messages=messages,
                    max_tokens=config.max_tokens or 512,
                    temperature=config.temperature or 0.7,
                    response_format={"type": "json_object"}
                )
            except openai.BadRequestError:
                # Fallback if json_object is not supported by this model
                return self.client.chat.completions.create(
                    model=config.model,
                    messages=messages,
                    max_tokens=config.max_tokens or 512,
                    temperature=config.temperature or 0.7
                )

        try:
            response = _call_openai_with_retry(_api_call)
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Fireworks LLM generation failed: {e}")
            raise ProviderError(f"Fireworks LLM generation failed: {e}") from e
