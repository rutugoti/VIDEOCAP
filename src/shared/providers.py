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

class VisionConfig(BaseModel):
    provider: str
    model: str
    max_tokens: int = 4096
    temperature: float = 0.2


class AudioConfig(BaseModel):
    provider: str
    model: str
    language: str = "en"


class OCRConfig(BaseModel):
    provider: str
    model: str
    max_tokens: int = 1024


class LLMConfig(BaseModel):
    provider: str
    model: str
    max_tokens: int = 512
    temperature: float = 0.7


class ValidatorConfig(BaseModel):
    provider: str
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
            return "A formal rewrite of the video narrative detailing events sequentially."
        elif "sarcastic" in prompt_lower:
            return "Oh fantastic, another video where events happen. Absolutely groundbreaking."
        elif "tech" in prompt_lower and "humor" in prompt_lower:
            return "EventLoop returned status code 200 after resolving kitchen tasks successfully."
        elif "humor" in prompt_lower:
            return "Well, the kitchen is clean but the coffee is gone. Classic morning tragedy."
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
