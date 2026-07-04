import pytest
from pydantic import ValidationError
from src.shared.models import (
    VideoDescriptor,
    Sample,
    Observation,
    Event,
    Timeline,
    GraphNode,
    GraphEdge,
    SemanticGraph,
    Narrative,
    Caption,
    CaptionValidation,
    ValidationReport,
)


def test_video_descriptor_validation():
    # Valid model
    desc = VideoDescriptor(
        path="video.mp4",
        duration_seconds=30.0,
        width=1920,
        height=1080,
        fps=30.0,
        has_audio=True,
        format="mp4",
        file_size_bytes=1024 * 1024
    )
    assert desc.width == 1920

    # Invalid: negative duration
    with pytest.raises(ValidationError):
        VideoDescriptor(
            path="video.mp4",
            duration_seconds=-5.0,
            width=1920,
            height=1080,
            fps=30.0,
            has_audio=True,
            format="mp4",
            file_size_bytes=1024
        )


def test_sample_validation():
    # Valid
    sample = Sample(frame_data=b"image_bytes", timestamp=1.5, sample_index=1)
    assert sample.timestamp == 1.5

    # Invalid: empty frame_data
    with pytest.raises(ValidationError):
        Sample(frame_data=b"", timestamp=1.5, sample_index=1)


def test_observation_validation():
    # Valid
    obs = Observation(
        content="a cat sits",
        timestamp=2.0,
        confidence=0.95,
        source="visual",
        observation_type="object"
    )
    assert obs.confidence == 0.95

    # Invalid: confidence > 1.0
    with pytest.raises(ValidationError):
        Observation(
            content="a cat sits",
            timestamp=2.0,
            confidence=1.5,
            source="visual",
            observation_type="object"
        )

    # Invalid: empty content
    with pytest.raises(ValidationError):
        Observation(
            content="   ",
            timestamp=2.0,
            confidence=0.5,
            source="visual",
            observation_type="object"
        )


def test_event_timestamps():
    # Valid
    ev = Event(
        description="cat runs",
        timestamp_start=1.0,
        timestamp_end=3.0,
        confidence=0.9,
        salience=0.8
    )
    assert ev.timestamp_end == 3.0

    # Invalid: end before start
    with pytest.raises(ValidationError):
        Event(
            description="cat runs",
            timestamp_start=4.0,
            timestamp_end=3.0,
            confidence=0.9,
            salience=0.8
        )


def test_timeline_sorting():
    obs1 = Observation(content="A", timestamp=5.0, confidence=0.8, source="visual", observation_type="scene")
    obs2 = Observation(content="B", timestamp=1.0, confidence=0.9, source="audio", observation_type="speech")
    obs3 = Observation(content="C", timestamp=3.0, confidence=0.75, source="text", observation_type="ocr_text")

    timeline = Timeline(observations=[obs1, obs2, obs3], duration=10.0, modalities_present=["visual", "audio", "text"])

    # Verify automatically sorted by timestamp
    assert timeline.observations[0].content == "B"  # 1.0
    assert timeline.observations[1].content == "C"  # 3.0
    assert timeline.observations[2].content == "A"  # 5.0
