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
from unittest.mock import patch, MagicMock
import numpy as np

class MockEasyOCRReader:
    def __init__(self, mock_results):
        self.mock_results = mock_results
        self.call_count = 0

    def readtext(self, img):
        if self.call_count < len(self.mock_results):
            res = self.mock_results[self.call_count]
            self.call_count += 1
            return res
        return []

@patch('cv2.imdecode')
def test_ocr_clear_text(mock_imdecode):
    mock_imdecode.return_value = np.zeros((10,10,3), dtype=np.uint8)
    
    processor = OCRProcessor()
    # Mock the reader to return a specific text on the second frame
    processor._reader = MockEasyOCRReader([
        [], # frame 0: no text
        [ (None, "brand logo", 0.95) ], # frame 1
        [], # frame 2
    ])

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


@patch('cv2.imdecode')
def test_ocr_no_text(mock_imdecode):
    mock_imdecode.return_value = np.zeros((10,10,3), dtype=np.uint8)
    processor = OCRProcessor()
    processor._reader = MockEasyOCRReader([[]])

    samples = [Sample(frame_data=b"f1", timestamp=0.0, sample_index=0)]
    assert processor.process(samples) == []


@patch('cv2.imdecode')
def test_ocr_deduplication(mock_imdecode):
    mock_imdecode.return_value = np.zeros((10,10,3), dtype=np.uint8)
    processor = OCRProcessor(min_confidence=0.5)
    
    # 4 frames:
    # 1. "  Warning  " (0.7)
    # 2. "warning" (0.9)
    # 3. "Stop" (0.85)
    # 4. "warning" (0.8)
    processor._reader = MockEasyOCRReader([
        [ (None, "  Warning  ", 0.7) ],
        [ (None, "warning", 0.9) ],
        [ (None, "Stop", 0.85) ],
        [ (None, "warning", 0.8) ]
    ])

    samples = [
        Sample(frame_data=b"f1", timestamp=1.0, sample_index=0),
        Sample(frame_data=b"f2", timestamp=2.0, sample_index=1),
        Sample(frame_data=b"f3", timestamp=3.0, sample_index=2),
        Sample(frame_data=b"f4", timestamp=4.0, sample_index=3),
    ]
    
    results = processor.process(samples)

    assert len(results) == 3
    assert results[0].timestamp == 2.0
    assert results[0].confidence == 0.9
    assert results[1].content == "Stop"
    assert results[2].timestamp == 4.0


@patch('cv2.imdecode')
def test_ocr_api_failure(mock_imdecode):
    mock_imdecode.return_value = np.zeros((10,10,3), dtype=np.uint8)
    processor = OCRProcessor()
    
    class BrokenReader:
        def readtext(self, img):
            raise Exception("OCR engine crashed")
            
    processor._reader = BrokenReader()
    samples = [Sample(frame_data=b"f1", timestamp=0.0, sample_index=0)]

    with pytest.raises(ProviderError):
        processor.process(samples)
