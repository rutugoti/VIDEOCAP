import pytest
import numpy as np
from src.sampling.adaptive_sampler import AdaptiveSampler
from src.shared.models import VideoDescriptor
from tests.test_video_loader import create_dummy_video

def test_silence_detection():
    sampler = AdaptiveSampler(silence_threshold=-30.0)
    
    # 1. Generate 16-bit silent bytes (all zeros)
    silent_data = b"\x00" * 32000
    assert sampler.is_silent(silent_data, sample_width=2) is True
    
    # 2. Generate 16-bit high amplitude noise
    noisy_samples = (np.sin(np.linspace(0, 1000, 16000)) * 32767).astype(np.int16)
    noisy_data = noisy_samples.tobytes()
    assert sampler.is_silent(noisy_data, sample_width=2) is False


def test_adaptive_sampling_short_video(tmp_path):
    video_file = str(tmp_path / "short_clip.mp4")
    # 4 seconds duration
    create_dummy_video(video_file, duration=4.0, has_audio=False)

    desc = VideoDescriptor(
        path=video_file,
        duration_seconds=4.0,
        width=320,
        height=240,
        fps=30.0,
        has_audio=False,
        format="mp4",
        file_size_bytes=10000
    )

    # Short video -> higher density scaling to make sure we don't miss quick events
    sampler = AdaptiveSampler(method="adaptive", max_frames=30, min_frames=5)
    samples = sampler.sample_video(desc)
    
    # Should yield at least min_frames (5)
    assert len(samples) >= 5
    assert len(samples) <= 30
    assert all(s.audio_segment is None for s in samples)


def test_scene_boundary_segmentation_mock():
    # Verify detect_scene_boundaries doesn't crash on invalid/non-existent files and returns fallback
    sampler = AdaptiveSampler()
    boundaries = sampler.detect_scene_boundaries("non_existent_file.mp4", duration=10.0)
    assert boundaries == [0.0]
