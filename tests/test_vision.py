import pytest
from src.shared.models import Sample, Observation
from src.shared.providers import MockVisionProvider, VisionConfig, ProviderError, VisionProvider
from src.perception.vision_processor import VisionProcessor


class BrokenVisionProvider(VisionProvider):
    """A vision provider that always raises ProviderError."""
    def analyze_frames(self, frames, prompt, config):
        raise ProviderError("API timeout or rate limit exceeded")


def test_valid_frames():
    mock_provider = MockVisionProvider()
    processor = VisionProcessor(provider=mock_provider, confidence_threshold=0.5)

    samples = [
        Sample(frame_data=b"f1", timestamp=0.0, sample_index=0),
        Sample(frame_data=b"f2", timestamp=1.5, sample_index=1),
    ]

    observations = processor.process(samples)

    # MockVisionProvider generates:
    # - action observed at 0.0s
    # - kitchen scene observed at 0.0s
    # - action observed at 1.5s
    assert len(observations) == 3
    assert observations[0].timestamp == 0.0
    assert observations[1].timestamp == 0.0
    assert observations[2].timestamp == 1.5
    assert all(o.source == "visual" for o in observations)
    assert all(o.confidence >= 0.5 for o in observations)


def test_empty_frames():
    mock_provider = MockVisionProvider()
    processor = VisionProcessor(provider=mock_provider)

    observations = processor.process([])
    assert observations == []


def test_api_failure():
    broken_provider = BrokenVisionProvider()
    processor = VisionProcessor(provider=broken_provider)

    samples = [Sample(frame_data=b"f1", timestamp=0.0, sample_index=0)]

    with pytest.raises(ProviderError):
        processor.process(samples)


def test_observation_format():
    mock_provider = MockVisionProvider()
    # High confidence threshold to filter some out
    processor = VisionProcessor(provider=mock_provider, confidence_threshold=0.92)

    samples = [
        Sample(frame_data=b"f1", timestamp=0.0, sample_index=0),
    ]

    observations = processor.process(samples)
    
    # scene has confidence 0.95 (kept), action has 0.90 (filtered out)
    assert len(observations) == 1
    assert observations[0].observation_type == "scene"
    assert observations[0].confidence == 0.95
