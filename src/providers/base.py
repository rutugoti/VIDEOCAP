import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

# =====================================================================
# Normalized Exceptions
# =====================================================================

class ProviderError(Exception):
    """Base exception for all provider errors."""
    pass

class AuthenticationError(ProviderError):
    """Raised when authentication credentials fail."""
    pass

class RateLimitError(ProviderError):
    """Raised when API rate limits are hit."""
    pass

class TimeoutError(ProviderError):
    """Raised when request times out."""
    pass

class QuotaExceededError(ProviderError):
    """Raised when account quota/billing limits are exceeded."""
    pass

class ProviderUnavailableError(ProviderError):
    """Raised when the remote service is down or unreachable."""
    pass

class InvalidResponseError(ProviderError):
    """Raised when response is unparsable or fails validation checks."""
    pass

# =====================================================================
# Capabilities Metadata
# =====================================================================

class ProviderCapabilities(BaseModel):
    supports_vision: bool = False
    supports_audio: bool = False
    supports_json: bool = False
    supports_function_calling: bool = False
    supports_streaming: bool = False
    supports_embeddings: bool = False
    supports_multimodal: bool = False
    supports_batch: bool = False

# =====================================================================
# Cost Tracking
# =====================================================================

class CostTracker:
    """Tracks token usage, latency, retries, and estimated cost across calls."""
    
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    def track_call(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency: float,
        retries: int = 0,
        estimated_cost: float = 0.0
    ):
        self.calls.append({
            "provider": provider,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency": latency,
            "retries": retries,
            "estimated_cost": estimated_cost
        })

    def get_summary(self) -> Dict[str, Any]:
        total_calls = len(self.calls)
        total_prompt_tokens = sum(c.get("prompt_tokens", 0) for c in self.calls)
        total_completion_tokens = sum(c.get("completion_tokens", 0) for c in self.calls)
        total_latency = sum(c.get("latency", 0.0) for c in self.calls)
        total_cost = sum(c.get("estimated_cost", 0.0) for c in self.calls)
        return {
            "total_calls": total_calls,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_latency": total_latency,
            "total_cost": total_cost,
        }

# Global cost tracker instance
global_cost_tracker = CostTracker()

# =====================================================================
# Standard Response Objects
# =====================================================================

class VisionObservation(BaseModel):
    content: str
    timestamp: float
    confidence: float
    observation_type: str  # object, action, scene, emotion, etc.

class SpeechObservation(BaseModel):
    content: str
    timestamp: float
    confidence: float

class OCRObservation(BaseModel):
    content: str
    timestamp: float
    confidence: float

class LLMResponse(BaseModel):
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency: float = 0.0
    estimated_cost: float = 0.0

class EmbeddingResponse(BaseModel):
    embedding: List[float]
    prompt_tokens: int = 0
    latency: float = 0.0
    estimated_cost: float = 0.0

class ValidationResult(BaseModel):
    passed: bool
    status: str  # PASS, FAIL, UNKNOWN
    hallucinations: List[str] = Field(default_factory=list)
    missing_facts: List[str] = Field(default_factory=list)
    style_adherence: float = 1.0
    overall_confidence: float = 1.0
    latency: float = 0.0
    estimated_cost: float = 0.0

# =====================================================================
# Provider Config Models
# =====================================================================

class ProviderConfig(BaseModel):
    provider_name: Union[str, List[str]]
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model_name: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    timeout_seconds: Optional[int] = None
    max_retries: Optional[int] = None
    extra_params: Dict[str, Any] = Field(default_factory=dict)

# =====================================================================
# Provider Abstract Interfaces
# =====================================================================

class BaseProvider(ABC):
    """Common base for all providers to expose name and capabilities."""
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the name identifier of this provider."""
        pass

    @abstractmethod
    def get_capabilities(self) -> ProviderCapabilities:
        """Return the capabilities of this provider."""
        pass


class VisionProvider(BaseProvider):
    """Interface for visual understanding models."""

    @abstractmethod
    def analyze_frames(
        self,
        frames: List[Any],  # List[Sample]
        prompt: str,
        config: ProviderConfig
    ) -> List[VisionObservation]:
        """Analyze video frames and return standard visual observations."""
        pass


class SpeechProvider(BaseProvider):
    """Interface for speech-to-text models."""

    @abstractmethod
    def transcribe(
        self,
        frames: List[Any],  # List[Sample]
        config: ProviderConfig
    ) -> List[SpeechObservation]:
        """Transcribe audio segments from frames and return standard speech observations."""
        pass


class OCRProvider(BaseProvider):
    """Interface for on-screen text detection."""

    @abstractmethod
    def extract_text(
        self,
        frames: List[Any],  # List[Sample]
        config: ProviderConfig
    ) -> List[OCRObservation]:
        """Detect and read text visible in video frames."""
        pass


class LLMProvider(BaseProvider):
    """Interface for language model interactions."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str,
        config: ProviderConfig
    ) -> LLMResponse:
        """Generate text using a prompt and return standard LLMResponse."""
        pass


class EmbeddingProvider(BaseProvider):
    """Interface for text embedding generation."""

    @abstractmethod
    def embed_query(
        self,
        text: str,
        config: ProviderConfig
    ) -> EmbeddingResponse:
        """Generate vector embedding for a single text string."""
        pass


class ValidationProvider(BaseProvider):
    """Interface for semantic validation of captions."""

    @abstractmethod
    def validate(
        self,
        captions: Dict[str, Any],  # Dict[str, Caption]
        narrative: Any,  # Narrative
        config: ProviderConfig
    ) -> ValidationResult:
        """Validate generated captions against the source narrative."""
        pass


class ImageProvider(BaseProvider):
    """Future expansion interface for image generation models."""
    pass


class VideoProvider(BaseProvider):
    """Future expansion interface for video generation models."""
    pass
