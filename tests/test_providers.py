import pytest
from src.shared.models import Sample, Caption, Narrative
from src.shared.providers import (
    VisionConfig,
    AudioConfig,
    OCRConfig,
    LLMConfig,
    ValidatorConfig,
    MockVisionProvider,
    MockAudioProvider,
    MockOCRProvider,
    MockLLMProvider,
    MockValidator,
    ProviderError,
    RateLimitError,
    NoAudioError,
)


def test_mock_vision_provider():
    provider = MockVisionProvider()
    frames = [
        Sample(frame_data=b"f1", timestamp=0.0, sample_index=0),
        Sample(frame_data=b"f2", timestamp=1.0, sample_index=1),
    ]
    config = VisionConfig(provider="mock", model="mock-model")
    obs = provider.analyze_frames(frames, "analyze", config)

    # Check we got observations back
    assert len(obs) > 0
    # Check they have correct source
    for o in obs:
        assert o.source == "visual"
    # Check scene observation on index 0
    scenes = [o for o in obs if o.observation_type == "scene"]
    assert len(scenes) == 1
    assert "kitchen" in scenes[0].content


def test_mock_audio_provider():
    provider = MockAudioProvider()
    frames = [
        Sample(frame_data=b"f1", timestamp=0.0, sample_index=0, audio_segment=None),
        Sample(frame_data=b"f2", timestamp=1.0, sample_index=1, audio_segment=b"audio_bytes"),
    ]
    config = AudioConfig(provider="mock", model="mock-model")
    obs = provider.transcribe(frames, config)

    # Should only transcribe sample with audio_segment
    assert len(obs) == 1
    assert obs[0].source == "audio"
    assert obs[0].timestamp == 1.0


def test_mock_ocr_provider():
    provider = MockOCRProvider()
    frames = [
        Sample(frame_data=b"f1", timestamp=0.0, sample_index=0),
        Sample(frame_data=b"f2", timestamp=1.0, sample_index=1),
        Sample(frame_data=b"f3", timestamp=2.0, sample_index=2),
    ]
    config = OCRConfig(provider="mock", model="mock-model")
    obs = provider.extract_text(frames, config)

    # Should find text in the middle frame (index 1)
    assert len(obs) == 1
    assert obs[0].source == "text"
    assert obs[0].timestamp == 1.0


def test_mock_llm_provider():
    provider = MockLLMProvider()
    config = LLMConfig(provider="mock", model="mock-model")

    formal = provider.generate("formal rewrite of kitchen scene", "system", config)
    assert "formal" in formal.lower()

    sarcastic = provider.generate("sarcastic write", "system", config)
    assert "groundbreaking" in sarcastic.lower()


def test_mock_validator():
    provider = MockValidator()
    captions = {
        "formal": Caption(text="A formal text", style="formal", word_count=3),
        "sarcastic": Caption(text="A sarcastic text", style="sarcastic", word_count=3),
    }
    narrative = Narrative(text="neutral narrative")
    config = ValidatorConfig(provider="mock", model="mock-model")

    report = provider.validate(captions, narrative, config)
    assert report.overall_pass is True
    assert len(report.per_caption) == 2
    assert report.per_caption["formal"].passed is True


def test_custom_exceptions():
    with pytest.raises(ProviderError):
        raise ProviderError("API down")

    with pytest.raises(RateLimitError):
        raise RateLimitError("Rate limit hit")

    with pytest.raises(NoAudioError):
        raise NoAudioError("Missing audio track")
