from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class VideoDescriptor(BaseModel):
    """Represents metadata of a video loaded into the system."""
    path: str
    duration_seconds: float = Field(..., gt=0.0)
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)
    fps: float = Field(..., gt=0.0)
    has_audio: bool
    format: str
    file_size_bytes: int = Field(..., gt=0)


class Sample(BaseModel):
    """Represents a single sampled video frame and optional audio segment."""
    frame_data: bytes
    timestamp: float = Field(..., ge=0.0)
    sample_index: int = Field(..., ge=0)
    audio_segment: Optional[bytes] = None

    @field_validator("frame_data")
    @classmethod
    def frame_data_not_empty(cls, v: bytes) -> bytes:
        if len(v) == 0:
            raise ValueError("frame_data must not be empty")
        return v


class Observation(BaseModel):
    """Unified observation from a single modality at a specific timestamp."""
    content: str
    timestamp: float = Field(..., ge=0.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    source: Literal["visual", "audio", "text"]
    observation_type: str  # object, action, scene, speech, ocr_text, emotion, sound

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content must not be empty")
        return v


class Event(BaseModel):
    """Fused event spanning a duration, derived from observations."""
    description: str
    actors: List[str] = Field(default_factory=list)
    actions: List[str] = Field(default_factory=list)
    objects: List[str] = Field(default_factory=list)
    timestamp_start: float = Field(..., ge=0.0)
    timestamp_end: float = Field(..., ge=0.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence_sources: List[str] = Field(default_factory=list)
    salience: float = Field(..., ge=0.0, le=1.0)

    @field_validator("timestamp_end")
    @classmethod
    def validate_timestamps(cls, v: float, info: Any) -> float:
        if "timestamp_start" in info.data and v < info.data["timestamp_start"]:
            raise ValueError("timestamp_end cannot be before timestamp_start")
        return v


class Timeline(BaseModel):
    """Collection of sorted observations representing the video timeline."""
    observations: List[Observation]
    duration: float = Field(..., ge=0.0)
    modalities_present: List[str] = Field(default_factory=list)

    @field_validator("observations")
    @classmethod
    def sort_observations(cls, v: List[Observation]) -> List[Observation]:
        # Always return observations sorted by timestamp
        return sorted(v, key=lambda x: x.timestamp)


class GraphNode(BaseModel):
    """A node in the semantic representation (entity or event)."""
    id: str
    type: Literal["entity", "event"]
    label: str
    attributes: Dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    """A directed relationship edge in the semantic graph."""
    source: str
    target: str
    relationship: Literal["temporal", "causal", "spatial", "participates_in"]


class SemanticGraph(BaseModel):
    """The graph representation of entities, events and their relationships."""
    nodes: List[GraphNode] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Narrative(BaseModel):
    """A style-neutral, factually grounded textual summary of the video."""
    text: str
    key_events: List[str] = Field(default_factory=list)
    salience_scores: Dict[str, float] = Field(default_factory=dict)
    evidence_mapping: Dict[str, List[str]] = Field(default_factory=dict)


class Caption(BaseModel):
    """A styled caption generated from the narrative."""
    text: str
    style: Literal["formal", "sarcastic", "tech_humor", "non_tech_humor"]
    word_count: int = Field(..., ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CaptionValidation(BaseModel):
    """Validation report for an individual caption with detailed score dimensions."""
    style: str
    passed: bool
    hallucinations: List[str] = Field(default_factory=list)
    missing_facts: List[str] = Field(default_factory=list)
    style_adherence: float = Field(..., ge=0.0, le=1.0)
    semantic_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)
    hallucination_risk: Optional[float] = Field(None, ge=0.0, le=1.0)
    grammar_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    temporal_consistency_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    word_budget_pass: Optional[bool] = None
    overall_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)


class ValidationReport(BaseModel):
    """Full semantic validation report across all generated captions."""
    overall_pass: bool
    per_caption: Dict[str, CaptionValidation] = Field(default_factory=dict)
    hallucination_count: int = Field(..., ge=0)
    consistency_score: float = Field(..., ge=0.0, le=1.0)
