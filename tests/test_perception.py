import pytest
from src.shared.models import Sample, Observation
from src.shared.providers import (
    MockAudioProvider,
    MockOCRProvider,
    AudioConfig,
    OCRConfig,
    ProviderError,
    AudioProvider,
    OCRProvider,
)
from src.perception.speech_processor import SpeechProcessor
from src.perception.ocr_processor import OCRProcessor


class BrokenAudioProvider(AudioProvider):
    def transcribe(self, frames, config):
        raise ProviderError("Whisper model offline")


class BrokenOCRProvider(OCRProvider):
    def extract_text(self, frames, config):
        raise ProviderError("OCR engine crashed")


class CustomOCRProvider(OCRProvider):
    def __init__(self, observations):
        self.observations = observations

    def extract_text(self, frames, config):
        return self.observations


# =====================================================================
# SpeechProcessor Tests
# =====================================================================

def test_speech_clear_speech():
    mock_audio = MockAudioProvider()
    processor = SpeechProcessor(provider=mock_audio)

    samples = [
        Sample(frame_data=b"f1", timestamp=0.0, sample_index=0, audio_segment=None),
        Sample(frame_data=b"f2", timestamp=2.0, sample_index=1, audio_segment=b"wav_bytes"),
    ]

    observations = processor.process(samples, has_audio=True)
    assert len(observations) == 1
    assert observations[0].source == "audio"
    assert observations[0].timestamp == 2.0
    assert "spoken words" in observations[0].content


def test_speech_no_audio():
    mock_audio = MockAudioProvider()
    processor = SpeechProcessor(provider=mock_audio)

    samples = [
        Sample(frame_data=b"f1", timestamp=0.0, sample_index=0, audio_segment=b"bytes"),
    ]

    # has_audio is False
    obs_no_track = processor.process(samples, has_audio=False)
    assert obs_no_track == []

    # samples list is empty
    obs_empty = processor.process([], has_audio=True)
    assert obs_empty == []

    # no sample contains audio segment bytes
    samples_no_bytes = [Sample(frame_data=b"f1", timestamp=0.0, sample_index=0, audio_segment=None)]
    obs_no_bytes = processor.process(samples_no_bytes, has_audio=True)
    assert obs_no_bytes == []


def test_speech_api_failure():
    broken = BrokenAudioProvider()
    processor = SpeechProcessor(provider=broken)
    samples = [Sample(frame_data=b"f1", timestamp=1.0, sample_index=0, audio_segment=b"bytes")]

    with pytest.raises(ProviderError):
        processor.process(samples)


# =====================================================================
# OCRProcessor Tests
# =====================================================================

def test_ocr_clear_text():
    mock_ocr = MockOCRProvider()
    processor = OCRProcessor(provider=mock_ocr)

    samples = [
        Sample(frame_data=b"f1", timestamp=0.0, sample_index=0),
        Sample(frame_data=b"f2", timestamp=1.0, sample_index=1),
        Sample(frame_data=b"f3", timestamp=2.0, sample_index=2),
    ]

    observations = processor.process(samples)
    assert len(observations) == 1
    assert observations[0].source == "text"
    assert observations[0].timestamp == 1.0
    assert "brand logo" in observations[0].content


def test_ocr_no_text():
    # An OCR provider returning empty observations
    empty_ocr = CustomOCRProvider([])
    processor = OCRProcessor(provider=empty_ocr)

    samples = [Sample(frame_data=b"f1", timestamp=0.0, sample_index=0)]
    assert processor.process(samples) == []


def test_ocr_deduplication():
    # Observations containing duplicate text on consecutive frames
    obs1 = Observation(content="  Warning  ", timestamp=1.0, confidence=0.7, source="text", observation_type="ocr_text")
    obs2 = Observation(content="warning", timestamp=2.0, confidence=0.9, source="text", observation_type="ocr_text")
    obs3 = Observation(content="Stop", timestamp=3.0, confidence=0.85, source="text", observation_type="ocr_text")
    obs4 = Observation(content="warning", timestamp=4.0, confidence=0.8, source="text", observation_type="ocr_text")

    custom_ocr = CustomOCRProvider([obs1, obs2, obs3, obs4])
    processor = OCRProcessor(provider=custom_ocr)

    samples = [Sample(frame_data=b"f1", timestamp=0.0, sample_index=0)]
    results = processor.process(samples)

    # obs1 and obs2 are consecutive duplicates. obs2 (confidence 0.9) should merge/replace obs1 (0.7).
    # obs3 ("Stop") is different.
    # obs4 ("warning") is not consecutive to obs2, so it is kept.
    # Resulting order:
    # 1. obs2 (timestamp 2.0, content "warning")
    # 2. obs3 (timestamp 3.0, content "Stop")
    # 3. obs4 (timestamp 4.0, content "warning")
    assert len(results) == 3
    assert results[0].timestamp == 2.0
    assert results[0].confidence == 0.9
    assert results[1].content == "Stop"
    assert results[2].timestamp == 4.0


def test_ocr_api_failure():
    broken = BrokenOCRProvider()
    processor = OCRProcessor(provider=broken)
    samples = [Sample(frame_data=b"f1", timestamp=0.0, sample_index=0)]

    with pytest.raises(ProviderError):
        processor.process(samples)
