import pytest
from src.shared.models import Observation
from src.fusion.timeline_builder import TimelineBuilder


def test_merge_all_modalities():
    obs1 = Observation(content="Person walks", timestamp=2.0, confidence=0.8, source="visual", observation_type="action")
    obs2 = Observation(content="Hello", timestamp=0.5, confidence=0.9, source="audio", observation_type="speech")
    obs3 = Observation(content="Exit sign", timestamp=4.5, confidence=0.95, source="text", observation_type="ocr_text")

    builder = TimelineBuilder()
    timeline = builder.build_timeline([obs1, obs2, obs3], duration=10.0)

    # Check length
    assert len(timeline.observations) == 3
    # Check modalities present
    assert set(timeline.modalities_present) == {"visual", "audio", "text"}
    # Check sorted ordering: 0.5 (audio) -> 2.0 (visual) -> 4.5 (text)
    assert timeline.observations[0].source == "audio"
    assert timeline.observations[1].source == "visual"
    assert timeline.observations[2].source == "text"


def test_single_modality():
    obs1 = Observation(content="A", timestamp=2.0, confidence=0.8, source="visual", observation_type="scene")
    obs2 = Observation(content="B", timestamp=1.0, confidence=0.9, source="visual", observation_type="object")

    builder = TimelineBuilder()
    timeline = builder.build_timeline([obs1, obs2], duration=5.0)

    assert len(timeline.observations) == 2
    assert timeline.modalities_present == ["visual"]
    assert timeline.observations[0].content == "B"  # 1.0
    assert timeline.observations[1].content == "A"  # 2.0


def test_empty_observations():
    builder = TimelineBuilder()
    timeline = builder.build_timeline([], duration=10.0)

    assert len(timeline.observations) == 0
    assert timeline.modalities_present == []
    assert timeline.duration == 10.0


def test_timestamp_ordering_ties():
    # Stable sort for ties
    obs1 = Observation(content="Visual event", timestamp=2.0, confidence=0.8, source="visual", observation_type="scene")
    obs2 = Observation(content="Audio event", timestamp=2.0, confidence=0.9, source="audio", observation_type="speech")

    builder = TimelineBuilder()
    timeline = builder.build_timeline([obs1, obs2], duration=10.0)

    assert len(timeline.observations) == 2
    # Check that ties preserve original input order (stable sort)
    assert timeline.observations[0].content == "Visual event"
    assert timeline.observations[1].content == "Audio event"
