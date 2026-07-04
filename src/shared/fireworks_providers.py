import base64
import io
import json
import logging
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

from src.shared.models import Sample, Observation
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

        # If key is mock, bypass and return mock data for offline tests
        if self.api_key == "mock_key_for_testing":
            logger.info("Using mock vision provider response for testing key.")
            return self._get_mock_observations(frames)

        observations: List[Observation] = []

        def _process_single(frame: Sample) -> List[Observation]:
            if not frame.frame_data:
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

            try:
                response = self.client.chat.completions.create(
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
                text = response.choices[0].message.content.strip()
                parsed = self._parse_json_list(text)
                
                res_obs = []
                for item in parsed:
                    res_obs.append(
                        Observation(
                            content=item.get("content", "unspecified visual observation"),
                            timestamp=frame.timestamp,
                            confidence=item.get("confidence", 0.9),
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

    def _parse_json_list(self, text: str) -> List[dict]:
        """Robust parser to extract a JSON list from a chat response."""
        text = text.strip()
        if not text:
            return []

        # Strip markdown wrappers if any
        if text.startswith("```"):
            newline_idx = text.find("\n")
            if newline_idx != -1:
                text = text[newline_idx:].strip()
            if text.endswith("```"):
                text = text[:-3].strip()

        start = text.find("[")
        end = text.rfind("]")

        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(text[start:end + 1])
                if isinstance(data, list):
                    return data
            except Exception:
                pass

        # Try parsing as a single dictionary
        start_dict = text.find("{")
        end_dict = text.rfind("}")
        if start_dict != -1 and end_dict != -1 and end_dict > start_dict:
            try:
                data = json.loads(text[start_dict:end_dict + 1])
                if isinstance(data, dict):
                    return [data]
            except Exception:
                pass

        # Parse line by line fallback
        return [{"content": text[:200], "confidence": 0.8, "observation_type": "scene"}]

    def _get_mock_observations(self, frames: List[Sample]) -> List[Observation]:
        """Offline fallback to prevent API calls in tests."""
        obs = []
        for f in frames:
            obs.append(
                Observation(
                    content=f"action observed at timestamp {f.timestamp}s",
                    timestamp=f.timestamp,
                    confidence=0.90,
                    source="visual",
                    observation_type="action"
                )
            )
            if f.sample_index == 0:
                obs.append(
                    Observation(
                        content="kitchen scene with modern appliances",
                        timestamp=f.timestamp,
                        confidence=0.95,
                        source="visual",
                        observation_type="scene"
                    )
                )
        return obs


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

        if self.api_key == "mock_key_for_testing":
            logger.info("Using mock audio provider response for testing key.")
            return self._get_mock_transcriptions(frames)

        observations: List[Observation] = []

        def _transcribe_single(frame: Sample) -> Optional[Observation]:
            if not frame.audio_segment:
                return None

            audio_file = ("segment.wav", io.BytesIO(frame.audio_segment), "audio/wav")
            try:
                response = self.client.audio.transcriptions.create(
                    model=config.model,
                    file=audio_file,
                    language=config.language or "en"
                )
                text = response.text.strip()
                if text:
                    return Observation(
                        content=text,
                        timestamp=frame.timestamp,
                        confidence=0.9,
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

    def _get_mock_transcriptions(self, frames: List[Sample]) -> List[Observation]:
        obs = []
        for f in frames:
            if f.audio_segment is not None:
                obs.append(
                    Observation(
                        content=f"spoken words recorded at {f.timestamp}s",
                        timestamp=f.timestamp,
                        confidence=0.88,
                        source="audio",
                        observation_type="speech"
                    )
                )
        return obs


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

        if self.api_key == "mock_key_for_testing":
            logger.info("Using mock OCR provider response for testing key.")
            return self._get_mock_ocr(frames)

        observations: List[Observation] = []

        def _process_single(frame: Sample) -> List[Observation]:
            if not frame.frame_data:
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

            try:
                response = self.client.chat.completions.create(
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
                text = response.choices[0].message.content.strip()
                parsed = self._parse_json_list(text)
                
                res_obs = []
                for item in parsed:
                    val = item.get("text", "").strip()
                    if val:
                        res_obs.append(
                            Observation(
                                content=val,
                                timestamp=frame.timestamp,
                                confidence=item.get("confidence", 0.9),
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

    def _parse_json_list(self, text: str) -> List[dict]:
        text = text.strip()
        if not text:
            return []

        if text.startswith("```"):
            newline_idx = text.find("\n")
            if newline_idx != -1:
                text = text[newline_idx:].strip()
            if text.endswith("```"):
                text = text[:-3].strip()

        start = text.find("[")
        end = text.rfind("]")

        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(text[start:end + 1])
                if isinstance(data, list):
                    return data
            except Exception:
                pass

        start_dict = text.find("{")
        end_dict = text.rfind("}")
        if start_dict != -1 and end_dict != -1 and end_dict > start_dict:
            try:
                data = json.loads(text[start_dict:end_dict + 1])
                if isinstance(data, dict):
                    return [data]
            except Exception:
                pass

        return []

    def _get_mock_ocr(self, frames: List[Sample]) -> List[Observation]:
        obs = []
        for f in frames:
            if f.sample_index == len(frames) // 2:
                obs.append(
                    Observation(
                        content="mock brand logo text",
                        timestamp=f.timestamp,
                        confidence=0.92,
                        source="text",
                        observation_type="ocr_text"
                    )
                )
        return obs


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
        if self.api_key == "mock_key_for_testing":
            # Delegate to standard mock behavior logic
            from src.shared.providers import MockLLMProvider
            return MockLLMProvider().generate(prompt, system_prompt, config)

        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model=config.model,
                messages=messages,
                max_tokens=config.max_tokens or 512,
                temperature=config.temperature or 0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Fireworks LLM generation failed: {e}")
            raise ProviderError(f"Fireworks LLM generation failed: {e}") from e
