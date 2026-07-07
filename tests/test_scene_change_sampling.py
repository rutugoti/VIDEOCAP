import pytest

from src.sampling.adaptive_sampler import AdaptiveSampler
from src.ingestion.video_loader import VideoLoader
from tests.test_video_loader import create_dummy_video


def test_peak_selector_flat_falls_back_to_uniform():
    # Flat diffs (static video) → uniform spacing
    times = [1.0, 2.0, 3.0, 4.0, 5.0]
    diffs = [0.0, 0.0, 0.0, 0.0, 0.0]
    out = AdaptiveSampler._select_peak_timestamps(times, diffs, num_samples=5, duration=10.0)
    assert out == [0.0, 2.0, 4.0, 6.0, 8.0]


def test_peak_selector_picks_motion_peaks():
    # High change at t=3.0 and t=8.0 should be selected alongside the t=0 anchor
    times = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    diffs = [1.0, 1.0, 50.0, 1.0, 1.0, 1.0, 1.0, 40.0, 1.0]
    out = AdaptiveSampler._select_peak_timestamps(times, diffs, num_samples=3, duration=10.0)
    assert len(out) == 3
    assert out == sorted(out)          # strictly ordered
    assert 0.0 in out                  # anchored at start
    assert 3.0 in out and 8.0 in out   # the two motion peaks


def test_peak_selector_always_returns_exact_count():
    times = [0.5, 1.0, 1.1, 1.2]       # clustered candidates
    diffs = [10.0, 9.0, 8.0, 7.0]
    out = AdaptiveSampler._select_peak_timestamps(times, diffs, num_samples=5, duration=5.0)
    assert len(out) == 5
    assert len(set(out)) == 5          # distinct
    assert out == sorted(out)


def test_peak_selector_insufficient_signal_uniform():
    # Fewer diff samples than requested → uniform
    out = AdaptiveSampler._select_peak_timestamps([1.0], [5.0], num_samples=4, duration=8.0)
    assert out == [0.0, 2.0, 4.0, 6.0]


def test_scene_change_end_to_end_static_video(tmp_path):
    # Static dummy video → detection returns uniform-equivalent, pipeline still yields frames
    video_file = str(tmp_path / "scene.mp4")
    create_dummy_video(video_file, duration=20.0, has_audio=False)

    loader = VideoLoader(min_duration=10.0, max_duration=60.0)
    desc = loader.load_video(video_file)

    sampler = AdaptiveSampler(method="scene_change", max_frames=10, min_frames=5)
    samples = sampler.sample_video(desc)

    assert len(samples) >= sampler.min_frames
    timestamps = [s.timestamp for s in samples]
    assert timestamps == sorted(timestamps)   # ordered
    assert all(s.frame_data for s in samples)  # every sample has frame bytes
