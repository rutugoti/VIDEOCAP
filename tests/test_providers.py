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
    
    # Check confidence_meta
    for o in obs:
        assert o.confidence_meta is not None
        assert o.confidence_meta.source == "default"
        assert o.confidence_meta.value in (0.90, 0.95)


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
    assert obs[0].confidence_meta is not None
    assert obs[0].confidence_meta.source == "default"
    assert obs[0].confidence_meta.value == 0.88


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
    assert obs[0].confidence_meta is not None
    assert obs[0].confidence_meta.source == "default"
    assert obs[0].confidence_meta.value == 0.92


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


def test_groq_provider():
    from src.providers.registry import ProviderRegistry
    from src.providers.factory import ProviderFactory
    from src.providers.groq_provider import GroqProvider
    from unittest.mock import MagicMock

    # 1. Verify provider is auto-discovered and registered
    groq_cls = ProviderRegistry.get("groq")
    assert groq_cls == GroqProvider

    # 2. Instantiate and verify capabilities
    provider = GroqProvider(api_key="mock_key_for_testing")
    assert provider.get_name() == "groq"
    caps = provider.get_capabilities()
    assert caps.supports_vision is True
    assert caps.supports_json is True

    # 3. Test execution path with mocked client completions API
    from src.providers.base import ProviderConfig
    config = ProviderConfig(provider_name="groq", model_name="llama-3.3-70b-versatile")
    
    # Mock LLM response
    mock_chat_completion_llm = MagicMock()
    mock_chat_completion_llm.choices = [
        MagicMock(message=MagicMock(content="Mocked LLM content response"))
    ]
    mock_chat_completion_llm.usage = MagicMock(prompt_tokens=10, completion_tokens=20)

    # Mock Vision response
    mock_chat_completion_vision = MagicMock()
    mock_chat_completion_vision.choices = [
        MagicMock(message=MagicMock(content='[{"content": "mocked vision obs", "confidence": 0.9, "observation_type": "scene"}]'))
    ]
    mock_chat_completion_vision.usage = MagicMock(prompt_tokens=15, completion_tokens=25)

    provider.client.chat.completions.create = MagicMock(side_effect=[
        mock_chat_completion_llm,
        mock_chat_completion_vision
    ])

    # Test LLM generate
    response = provider.generate("test prompt", "test system", config)
    assert response.text == "Mocked LLM content response"

    # Test Vision analyze_frames
    frames = [Sample(frame_data=b"f1", timestamp=0.0, sample_index=0)]
    obs = provider.analyze_frames(frames, "test prompt", config)
    assert len(obs) > 0
    assert obs[0].content == "mocked vision obs"

