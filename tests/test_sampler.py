import os
import pytest
import av
from src.ingestion.video_loader import VideoLoader
from src.sampling.adaptive_sampler import AdaptiveSampler
from tests.test_video_loader import create_dummy_video


def test_uniform_sampling(tmp_path):
    video_file = str(tmp_path / "test_uniform.mp4")
    # 35 seconds duration, no audio
    create_dummy_video(video_file, duration=35.0, has_audio=False)

    loader = VideoLoader(min_duration=30.0, max_duration=120.0)
    desc = loader.load_video(video_file)

    # Sample at 1 FPS (default), which should result in 35 samples (or capped at max_frames)
    sampler = AdaptiveSampler(fps=1.0, max_frames=50, min_frames=5)
    samples = sampler.sample_video(desc)

    assert len(samples) == 35
    assert samples[0].sample_index == 0
    assert samples[-1].sample_index == 34
    assert all(s.audio_segment is None for s in samples)


def test_timestamps_ordered(tmp_path):
    video_file = str(tmp_path / "test_ordered.mp4")
    create_dummy_video(video_file, duration=35.0, has_audio=False)

    loader = VideoLoader(min_duration=30.0, max_duration=120.0)
    desc = loader.load_video(video_file)

    sampler = AdaptiveSampler(fps=1.0, max_frames=30, min_frames=5)
    samples = sampler.sample_video(desc)

    # Check timestamps are monotonically increasing
    timestamps = [s.timestamp for s in samples]
    assert len(timestamps) > 0
    for i in range(1, len(timestamps)):
        assert timestamps[i] > timestamps[i - 1]
    assert timestamps[-1] < desc.duration_seconds


def test_max_frames_cap(tmp_path):
    video_file = str(tmp_path / "test_cap.mp4")
    # 45 seconds duration
    create_dummy_video(video_file, duration=45.0, has_audio=False)

    loader = VideoLoader(min_duration=30.0, max_duration=120.0)
    desc = loader.load_video(video_file)

    # Capped at max_frames=10
    sampler = AdaptiveSampler(fps=1.0, max_frames=10, min_frames=5)
    samples = sampler.sample_video(desc)

    assert len(samples) == 10
    # Ensure min/max ranges
    assert samples[0].timestamp == 0.0
    assert samples[-1].timestamp > 0.0


def test_short_video_floor(tmp_path):
    video_file = str(tmp_path / "test_short.mp4")
    # 15 seconds duration. Let's load with override bounds so it's valid
    create_dummy_video(video_file, duration=15.0, has_audio=False)

    loader = VideoLoader(min_duration=10.0, max_duration=60.0)
    desc = loader.load_video(video_file)

    # Capped at min_frames=10 even though fps=0.2 (would be 3 frames)
    sampler = AdaptiveSampler(fps=0.2, max_frames=20, min_frames=10)
    samples = sampler.sample_video(desc)

    assert len(samples) == 10


def test_sampling_with_audio(tmp_path):
    video_file = str(tmp_path / "test_audio.mp4")
    # 35 seconds, with audio
    create_dummy_video(video_file, duration=35.0, has_audio=True)

    loader = VideoLoader(min_duration=30.0, max_duration=120.0)
    desc = loader.load_video(video_file)

    sampler = AdaptiveSampler(fps=1.0, max_frames=30, min_frames=5)
    samples = sampler.sample_video(desc)

    # At least some samples should contain non-empty audio segment bytes (WAV files)
    samples_with_audio = [s for s in samples if s.audio_segment is not None]
    assert len(samples_with_audio) > 0
    # The first bytes of a WAV file are "RIFF"
    assert samples_with_audio[0].audio_segment.startswith(b"RIFF")


def test_adaptive_sampling_complexity(tmp_path):
    video_file = str(tmp_path / "test_adaptive.mp4")
    create_dummy_video(video_file, duration=35.0, has_audio=False)

    loader = VideoLoader(min_duration=30.0, max_duration=120.0)
    desc = loader.load_video(video_file)

    # Use adaptive complexity sampling
    sampler = AdaptiveSampler(method="adaptive", max_frames=20, min_frames=5)
    samples = sampler.sample_video(desc)

    # Static dummy video has 0 complexity, so it should scale down to min_frames (5)
    assert len(samples) == 5
