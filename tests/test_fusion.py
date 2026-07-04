import pytest
from src.shared.models import Timeline, Observation, Event
from src.shared.providers import MockLLMProvider, ProviderError, LLMProvider
from src.fusion.fusion_engine import FusionEngine


class BrokenLLMProvider(LLMProvider):
    """An LLM provider that always raises ProviderError."""
    def generate(self, prompt, system_prompt, config):
        raise ProviderError("LLM rate limit hit")


class GarbageLLMProvider(LLMProvider):
    """An LLM provider that returns invalid non-JSON text."""
    def generate(self, prompt, system_prompt, config):
        return "Sorry, I cannot help with that."


def test_group_observations():
    # Observations that span multiple windows
    # Window 1: 0.0 - 2.0s
    obs1 = Observation(content="person enters kitchen", timestamp=0.0, confidence=0.9, source="visual", observation_type="scene")
    obs2 = Observation(content="starts coffee maker", timestamp=1.5, confidence=0.85, source="visual", observation_type="action")
    # Window 2: 5.0 - 7.0s
    obs3 = Observation(content="spoken words: 'good morning'", timestamp=5.0, confidence=0.95, source="audio", observation_type="speech")
    obs4 = Observation(content="exit sign visible", timestamp=6.2, confidence=0.7, source="text", observation_type="ocr_text")

    timeline = Timeline(observations=[obs1, obs2, obs3, obs4], duration=10.0)

    # Use window of 3.0 seconds
    mock_llm = MockLLMProvider()
    engine = FusionEngine(llm_provider=mock_llm, temporal_window_seconds=3.0)
    events = engine.fuse_timeline(timeline)

    # The MockLLMProvider returns canned JSON response of 2 events:
    # 1. enters/inspects (0.0s - 2.0s)
    # 2. speaks (2.0s - 5.0s)
    # Both are within bounds and have confidence >= min_confidence (0.3).
    assert len(events) == 4
    # Chronologically sorted by timestamp_start:
    assert events[0].timestamp_start == 0.0
    assert events[1].timestamp_start == 1.5  # Clamped to t_end of window 1 (1.5)
    assert events[2].timestamp_start == 5.0  # Clamped to t_start of window 2 (5.0)
    assert events[3].timestamp_start == 5.0  # Clamped to t_start of window 2 (5.0)


def test_fallback_mechanism():
    # Test that when the LLM throws an exception, the rule-based fallback kicks in.
    obs1 = Observation(content="person walks", timestamp=1.0, confidence=0.8, source="visual", observation_type="action")
    obs2 = Observation(content="dog barking", timestamp=1.5, confidence=0.9, source="audio", observation_type="speech")

    timeline = Timeline(observations=[obs1, obs2], duration=5.0)

    broken_llm = BrokenLLMProvider()
    engine = FusionEngine(llm_provider=broken_llm, temporal_window_seconds=2.0)
    events = engine.fuse_timeline(timeline)

    # 1 window: 1.0s to 1.5s -> 1 fused event via fallback
    assert len(events) == 1
    fallback_event = events[0]
    assert fallback_event.timestamp_start == 1.0
    assert fallback_event.timestamp_end == 1.5
    assert set(fallback_event.evidence_sources) == {"visual", "audio"}
    assert "person walks" in fallback_event.description
    assert "dog barking" in fallback_event.description
    assert "person" in fallback_event.actors
    assert "walks" in fallback_event.actions
    assert fallback_event.confidence == pytest.approx(0.85)


def test_garbage_response_fallback():
    # Test that when the LLM returns invalid JSON, the rule-based fallback kicks in.
    obs1 = Observation(content="kitchen scene", timestamp=2.0, confidence=0.9, source="visual", observation_type="scene")

    timeline = Timeline(observations=[obs1], duration=5.0)

    garbage_llm = GarbageLLMProvider()
    engine = FusionEngine(llm_provider=garbage_llm)
    events = engine.fuse_timeline(timeline)

    assert len(events) == 1
    assert events[0].timestamp_start == 2.0
    assert "kitchen scene" in events[0].description
    assert "kitchen" in events[0].objects


def test_temporal_ordering():
    # Empty observations list
    timeline_empty = Timeline(observations=[], duration=10.0)
    engine = FusionEngine(llm_provider=MockLLMProvider())
    events = engine.fuse_timeline(timeline_empty)
    assert events == []
