from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel

# Import core schemas
from src.shared.models import (
    Observation,
    Sample,
    Caption,
    Narrative,
    ValidationReport,
    CaptionValidation,
    Confidence,
)


# =====================================================================
# Common Exceptions
# =====================================================================

class ProviderError(Exception):
    """Generic API/model failure exception."""
    pass


class RateLimitError(ProviderError):
    """Exception raised when API rate limits are exceeded."""
    pass


class TokenLimitError(ProviderError):
    """Exception raised when input size exceeds the model's context window."""
    pass


class NoAudioError(Exception):
    """Exception raised when operations fail due to a missing audio track."""
    pass


class InvalidVideoError(Exception):
    """Exception raised when a video file is corrupted or unsupported."""
    pass


class ValidationError(Exception):
    """Exception raised when a caption fails semantic validation checks."""
    pass


# =====================================================================
# Config Schemas
# =====================================================================

from typing import Union

class VisionConfig(BaseModel):
    provider: Union[str, List[str]]
    model: str
    max_tokens: int = 4096
    temperature: float = 0.2


class AudioConfig(BaseModel):
    provider: Union[str, List[str]]
    model: str
    language: str = "en"


class OCRConfig(BaseModel):
    provider: Union[str, List[str]]
    model: str
    max_tokens: int = 1024


class LLMConfig(BaseModel):
    provider: Union[str, List[str]]
    model: str
    max_tokens: int = 512
    temperature: float = 0.7


class ValidatorConfig(BaseModel):
    provider: Union[str, List[str]]
    model: str
    max_tokens: int = 1024
    temperature: float = 0.1


# =====================================================================
# Provider Abstract Interfaces
# =====================================================================

class VisionProvider(ABC):
    """Interface for visual understanding models."""

    @abstractmethod
    def analyze_frames(
        self,
        frames: List[Sample],
        prompt: str,
        config: VisionConfig
    ) -> List[Observation]:
        """
        Analyze video frames and return visual observations.
        """
        pass


class AudioProvider(ABC):
    """Interface for speech-to-text models."""

    @abstractmethod
    def transcribe(
        self,
        frames: List[Sample],
        config: AudioConfig
    ) -> List[Observation]:
        """
        Transcribe audio and return speech observations.
        Note: Interface matches the input list of Samples containing audio segments.
        """
        pass


class OCRProvider(ABC):
    """Interface for on-screen text detection."""

    @abstractmethod
    def extract_text(
        self,
        frames: List[Sample],
        config: OCRConfig
    ) -> List[Observation]:
        """
        Detect and read text visible in video frames.
        """
        pass


class LLMProvider(ABC):
    """Interface for language model interactions."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str,
        config: LLMConfig
    ) -> str:
        """
        Generate text from a prompt.
        """
        pass


class Validator(ABC):
    """Interface for semantic validation."""

    @abstractmethod
    def validate(
        self,
        captions: Dict[str, Caption],
        narrative: Narrative,
        config: ValidatorConfig
    ) -> ValidationReport:
        """
        Validate captions against source narrative.
        """
        pass


# =====================================================================
# Mock Provider Implementations
# =====================================================================

class MockVisionProvider(VisionProvider):
    """Mock implementation of VisionProvider for offline testing."""

    def analyze_frames(
        self,
        frames: List[Sample],
        prompt: str,
        config: VisionConfig
    ) -> List[Observation]:
        observations = []
        # Create standard visual observations for active frames
        for f in frames:
            observations.append(
                Observation(
                    content=f"action observed at timestamp {f.timestamp}s",
                    timestamp=f.timestamp,
                    confidence=0.90,
                    confidence_meta=Confidence(value=0.90, source="default"),
                    source="visual",
                    observation_type="action"
                )
            )
            # Add general scene observation for the first frame
            if f.sample_index == 0:
                observations.append(
                    Observation(
                        content="kitchen scene with modern appliances",
                        timestamp=f.timestamp,
                        confidence=0.95,
                        confidence_meta=Confidence(value=0.95, source="default"),
                        source="visual",
                        observation_type="scene"
                    )
                )
        return observations


class MockAudioProvider(AudioProvider):
    """Mock implementation of AudioProvider for offline testing."""

    def transcribe(
        self,
        frames: List[Sample],
        config: AudioConfig
    ) -> List[Observation]:
        observations = []
        # Add mock speech observations for samples that have audio_segment
        for f in frames:
            if f.audio_segment is not None:
                observations.append(
                    Observation(
                        content=f"spoken words recorded at {f.timestamp}s",
                        timestamp=f.timestamp,
                        confidence=0.88,
                        confidence_meta=Confidence(value=0.88, source="default"),
                        source="audio",
                        observation_type="speech"
                    )
                )
        return observations


class MockOCRProvider(OCRProvider):
    """Mock implementation of OCRProvider for offline testing."""

    def extract_text(
        self,
        frames: List[Sample],
        config: OCRConfig
    ) -> List[Observation]:
        observations = []
        # Only add OCR text at middle frames for realistic testing
        for f in frames:
            if f.sample_index == len(frames) // 2:
                observations.append(
                    Observation(
                        content="mock brand logo text",
                        timestamp=f.timestamp,
                        confidence=0.92,
                        confidence_meta=Confidence(value=0.92, source="default"),
                        source="text",
                        observation_type="ocr_text"
                    )
                )
        return observations


class MockLLMProvider(LLMProvider):
    """Mock implementation of LLMProvider for offline testing."""

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        config: LLMConfig
    ) -> str:
        # Pre-canned mock responses depending on keyword detection
        prompt_lower = prompt.lower()
        sys_lower = system_prompt.lower()

        # P4.1: Unsupported-claim entailment adjudicator mock.
        # Classifies candidate terms; treats affect/manner words as unsupported.
        if "entailment adjudicator" in sys_lower:
            affect_words = ["happily", "angrily", "sadly", "finally", "furiously", "excitedly", "reluctantly"]
            unsupported = [w for w in affect_words if w in prompt_lower]
            import json as _json
            return _json.dumps({"supported": [], "entailed": [], "unsupported": unsupported})

        # P3.1: Causal relationship classifier mock
        if "causal relationship classifier" in sys_lower:
            # Use simple heuristics to decide mock verdict
            if "cause" in prompt_lower or "trigger" in prompt_lower or "start" in prompt_lower:
                return '{"relationship": "causal", "confidence": 0.85, "rationale": "Direct cause-effect from description keywords."}'
            elif "enable" in prompt_lower or "condition" in prompt_lower or "allow" in prompt_lower:
                return '{"relationship": "enables", "confidence": 0.75, "rationale": "Enabling condition detected."}'
            else:
                return '{"relationship": "temporal", "confidence": 0.60, "rationale": "Only temporal sequence observed."}'


        is_single_pass = (
            ("formal" in prompt_lower and "sarcastic" in prompt_lower and "tech_humor" in prompt_lower and "non_tech_humor" in prompt_lower) or
            ("formal" in sys_lower and "sarcastic" in sys_lower and "tech_humor" in sys_lower and "non_tech_humor" in sys_lower)
        )
        if is_single_pass:
            if "test_missing_key" in prompt_lower:
                return """{
                    "sarcastic": "Oh fantastic, another video where events happen. Absolutely groundbreaking.",
                    "tech_humor": "EventLoop returned status code 200 after resolving kitchen tasks successfully.",
                    "non_tech_humor": "Well, the kitchen is clean but the coffee is gone."
                }"""
            elif "test_over_budget" in prompt_lower:
                return """{
                    "formal": "This is a extremely long and verbose response designed specifically to exceed the maximum word count limit of thirty five words on the very first attempt to trigger the retry validation block and it contains extra words to exceed the threshold easily.",
                    "sarcastic": "Oh fantastic, another video where events happen. Absolutely groundbreaking.",
                    "tech_humor": "EventLoop returned status code 200 after resolving kitchen tasks successfully.",
                    "non_tech_humor": "Well, the kitchen is clean but the coffee is gone."
                }"""
            elif "test_low_separation" in prompt_lower:
                return """{
                    "formal": "A person enters the kitchen, starts the coffee maker, and exits.",
                    "sarcastic": "Oh fantastic, another video where events happen.",
                    "tech_humor": "EventLoop returned status code 200 after resolving kitchen tasks successfully.",
                    "non_tech_humor": "EventLoop returned status code 200 after resolving kitchen tasks successfully."
                }"""
            else:
                return """{
                    "formal": "A formal rewrite of the video narrative detailing events sequentially, ensuring all facts are preserved.",
                    "sarcastic": "Oh fantastic, another video where events happen. Absolutely groundbreaking, and I am completely thrilled by this development.",
                    "tech_humor": "EventLoop returned status code 200 after resolving kitchen tasks successfully and flushing the local cache.",
                    "non_tech_humor": "Well, the kitchen is clean but the coffee is gone. Classic morning tragedy that happens to the best of us."
                }"""

        # Check system prompt first (specific role indicators to avoid positive/negative rule collisions)
        if "fact-checker" in sys_lower or "validator" in sys_lower:
            caption_part = ""
            if "caption:" in prompt_lower:
                parts = prompt_lower.split("caption:")
                if len(parts) > 1:
                    after_caption = parts[1]
                    if "report your findings" in after_caption:
                        caption_part = after_caption.split("report your findings")[0].strip()
                    else:
                        caption_part = after_caption.strip()
            else:
                caption_part = prompt_lower

            if "hallucinate" in caption_part:
                return """{
                    "hallucinations": ["caption mentions a red car not in the source"],
                    "missing_facts": [],
                    "fact_drift": [],
                    "style_match": true,
                    "overall_pass": false
                }"""
            elif "drift" in caption_part:
                return """{
                    "hallucinations": [],
                    "missing_facts": [],
                    "fact_drift": ["coffee maker changed to toaster"],
                    "style_match": true,
                    "overall_pass": false
                }"""
            elif "leakage" in caption_part or "leak" in caption_part:
                return """{
                    "hallucinations": [],
                    "missing_facts": [],
                    "fact_drift": [],
                    "style_match": false,
                    "overall_pass": false
                }"""
            else:
                return """{
                    "hallucinations": [],
                    "missing_facts": [],
                    "fact_drift": [],
                    "style_match": true,
                    "overall_pass": true
                }"""
        elif "software engineer" in sys_lower or "tech_humor" in sys_lower:
            return "EventLoop returned status code 200 after resolving kitchen tasks successfully and flushing the local cache."
        elif "funny social media" in sys_lower or "non_tech_humor" in sys_lower or "non-tech" in sys_lower:
            return "Well, the kitchen is clean but the coffee is gone. Classic morning tragedy that happens to the best of us."
        elif "sarcastic" in sys_lower:
            return "Oh fantastic, another video where events happen. Absolutely groundbreaking, and I am completely thrilled by this development."
        elif "technical writer" in sys_lower or "formal" in sys_lower:
            return "A formal rewrite of the video narrative detailing events sequentially, ensuring all facts are preserved."

        # Fallback to prompt-based checks
        if "narrative" in prompt_lower or "compress" in prompt_lower or "summary" in prompt_lower:
            return """{
                "text": "A person enters the kitchen, starts the coffee maker, and exits.",
                "key_events": [
                    "enters kitchen",
                    "starts coffee maker",
                    "exits kitchen"
                ],
                "salience_scores": {
                    "enters kitchen": 0.50,
                    "starts coffee maker": 0.95,
                    "exits kitchen": 0.40
                },
                "evidence_mapping": {
                    "enters kitchen": ["visual"],
                    "starts coffee maker": ["visual", "audio"],
                    "exits kitchen": ["visual"]
                }
            }"""
        elif "fuse" in prompt_lower or "fusion" in prompt_lower:
            return """[
                {
                    "description": "A person enters the kitchen and inspects the modern appliances.",
                    "actors": ["person"],
                    "actions": ["enters", "inspects"],
                    "objects": ["appliances"],
                    "timestamp_start": 0.0,
                    "timestamp_end": 2.0,
                    "confidence": 0.95,
                    "evidence_sources": ["visual"],
                    "salience": 0.8
                },
                {
                    "description": "A person speaks and transcribes audio context.",
                    "actors": ["person"],
                    "actions": ["speaks"],
                    "objects": [],
                    "timestamp_start": 2.0,
                    "timestamp_end": 5.0,
                    "confidence": 0.90,
                    "evidence_sources": ["audio"],
                    "salience": 0.7
                }
            ]"""
        elif "formal" in prompt_lower:
            return "A formal rewrite of the video narrative detailing events sequentially, ensuring all facts are preserved."
        elif "sarcastic" in prompt_lower:
            return "Oh fantastic, another video where events happen. Absolutely groundbreaking, and I am completely thrilled by this development."
        elif "tech_humor" in prompt_lower or "software engineering" in prompt_lower or "programming" in prompt_lower or "tech metaphor" in prompt_lower:
            return "EventLoop returned status code 200 after resolving kitchen tasks successfully and flushing the local cache."
        elif "humor" in prompt_lower or "funny" in prompt_lower:
            return "Well, the kitchen is clean but the coffee is gone. Classic morning tragedy that happens to the best of us."
        return "Generic mock LLM text response."


class MockValidator(Validator):
    """Mock implementation of Validator for offline testing."""

    def validate(
        self,
        captions: Dict[str, Caption],
        narrative: Narrative,
        config: ValidatorConfig
    ) -> ValidationReport:
        per_caption_validation = {}
        for style, cap in captions.items():
            per_caption_validation[style] = CaptionValidation(
                style=style,
                passed=True,
                hallucinations=[],
                missing_facts=[],
                style_adherence=0.95
            )
        return ValidationReport(
            overall_pass=True,
            per_caption=per_caption_validation,
            hallucination_count=0,
            consistency_score=1.0
        )


# =====================================================================
# Legacy Compatibility Adapters
# =====================================================================
from src.providers.base import ProviderConfig

class LegacyVisionProviderAdapter(VisionProvider):
    def __init__(self, agnostic_provider):
        self.agnostic_provider = agnostic_provider

    def analyze_frames(self, frames: List[Sample], prompt: str, config: VisionConfig) -> List[Observation]:
        p_cfg = ProviderConfig(
            provider_name=config.provider,
            model_name=config.model,
            max_tokens=config.max_tokens,
            temperature=config.temperature
        )
        res_obs = self.agnostic_provider.analyze_frames(frames, prompt, p_cfg)
        out = []
        for o in res_obs:
            out.append(
                Observation(
                    content=o.content,
                    timestamp=o.timestamp,
                    confidence=o.confidence,
                    confidence_meta=Confidence(value=o.confidence, source="model_reported"),
                    source="visual",
                    observation_type=o.observation_type
                )
            )
        return out

class LegacyAudioProviderAdapter(AudioProvider):
    def __init__(self, agnostic_provider):
        self.agnostic_provider = agnostic_provider

    def transcribe(self, frames: List[Sample], config: AudioConfig) -> List[Observation]:
        p_cfg = ProviderConfig(
            provider_name=config.provider,
            model_name=config.model,
            extra_params={"language": config.language}
        )
        res_obs = self.agnostic_provider.transcribe(frames, p_cfg)
        out = []
        for o in res_obs:
            out.append(
                Observation(
                    content=o.content,
                    timestamp=o.timestamp,
                    confidence=o.confidence,
                    confidence_meta=Confidence(value=o.confidence, source="measured"),
                    source="audio",
                    observation_type="speech"
                )
            )
        return out

class LegacyOCRProviderAdapter(OCRProvider):
    def __init__(self, agnostic_provider):
        self.agnostic_provider = agnostic_provider

    def extract_text(self, frames: List[Sample], config: OCRConfig) -> List[Observation]:
        p_cfg = ProviderConfig(
            provider_name=config.provider,
            model_name=config.model,
            max_tokens=config.max_tokens
        )
        res_obs = self.agnostic_provider.extract_text(frames, p_cfg)
        out = []
        for o in res_obs:
            out.append(
                Observation(
                    content=o.content,
                    timestamp=o.timestamp,
                    confidence=o.confidence,
                    confidence_meta=Confidence(value=o.confidence, source="model_reported"),
                    source="text",
                    observation_type="ocr_text"
                )
            )
        return out

class LegacyLLMProviderAdapter(LLMProvider):
    def __init__(self, agnostic_provider):
        self.agnostic_provider = agnostic_provider

    def generate(self, prompt: str, system_prompt: str, config: LLMConfig) -> str:
        p_cfg = ProviderConfig(
            provider_name=config.provider,
            model_name=config.model,
            max_tokens=config.max_tokens,
            temperature=config.temperature
        )
        res = self.agnostic_provider.generate(prompt, system_prompt, p_cfg)
        return res.text

class LegacyValidatorAdapter(Validator):
    def __init__(self, agnostic_provider):
        self.agnostic_provider = agnostic_provider

    def validate(self, captions: Dict[str, Caption], narrative: Narrative, config: ValidatorConfig) -> ValidationReport:
        p_cfg = ProviderConfig(
            provider_name=config.provider,
            model_name=config.model,
            max_tokens=config.max_tokens,
            temperature=config.temperature
        )
        res = self.agnostic_provider.validate(captions, narrative, p_cfg)
        
        per_caption_validation = {}
        for style, cap in captions.items():
            per_caption_validation[style] = CaptionValidation(
                style=style,
                passed=res.passed,
                hallucinations=res.hallucinations,
                missing_facts=res.missing_facts,
                style_adherence=res.style_adherence
            )
        return ValidationReport(
            overall_pass=res.passed,
            per_caption=per_caption_validation,
            hallucination_count=len(res.hallucinations),
            consistency_score=res.overall_confidence
        )

