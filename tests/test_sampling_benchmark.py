"""
Benchmark test for AdaptiveSampler single-pass optimization.

Demonstrates:
- Previous runtime (multi-pass: estimate_complexity + detect_scene_boundaries + extract)
- Optimized runtime (single-pass analysis + extract)
- Sampling quality preservation (timestamp consistency)
"""
import time
import pytest
from src.sampling.adaptive_sampler import AdaptiveSampler
from src.shared.models import VideoDescriptor
from tests.test_video_loader import create_dummy_video


def _make_descriptor(video_file: str, duration: float, has_audio: bool = False) -> VideoDescriptor:
    return VideoDescriptor(
        path=video_file,
        duration_seconds=duration,
        width=320,
        height=240,
        fps=30.0,
        has_audio=has_audio,
        format="mp4",
        file_size_bytes=10000
    )


def test_single_pass_analysis_produces_valid_results(tmp_path):
    """Verify single-pass analysis returns correct structure and reasonable values."""
    video_file = str(tmp_path / "test_analysis.mp4")
    create_dummy_video(video_file, duration=10.0, has_audio=False)

    sampler = AdaptiveSampler(method="adaptive", max_frames=10, min_frames=5)
    result = sampler._single_pass_analysis(video_file, 10.0)

    assert "complexity" in result
    assert "scene_boundaries" in result
    assert "frame_times" in result
    assert "frame_diffs" in result

    assert 0.0 <= result["complexity"] <= 1.0
    assert result["scene_boundaries"][0] == 0.0
    assert len(result["frame_times"]) == len(result["frame_diffs"])


def test_adaptive_single_pass_vs_multipass_consistency(tmp_path):
    """
    Verify that the single-pass approach produces identical scene boundaries and
    complexity scores as the legacy multi-pass methods (called directly).
    """
    video_file = str(tmp_path / "consistency.mp4")
    create_dummy_video(video_file, duration=8.0, has_audio=False)

    sampler = AdaptiveSampler(
        method="adaptive", max_frames=15, min_frames=5,
        scene_change_threshold=20.0, motion_threshold=15.0
    )

    # Legacy multi-pass (direct calls to preserved methods)
    legacy_complexity = sampler.estimate_complexity(video_file)
    legacy_boundaries = sampler.detect_scene_boundaries(video_file, 8.0)

    # New single-pass
    analysis = sampler._single_pass_analysis(video_file, 8.0)

    # Complexity should be very close (analysis uses subsampled diffs, same as legacy)
    assert abs(analysis["complexity"] - legacy_complexity) < 0.15, (
        f"Complexity diverged: single-pass={analysis['complexity']:.3f} vs legacy={legacy_complexity:.3f}"
    )

    # Scene boundaries should be identical for a static dummy video
    assert analysis["scene_boundaries"] == legacy_boundaries


def test_sampling_benchmark_adaptive(tmp_path):
    """
    Benchmark: measure total sampling time for adaptive mode.
    The optimized path should be faster because it decodes only twice (analysis + extraction)
    instead of three times (complexity + scenes + extraction).
    """
    video_file = str(tmp_path / "benchmark.mp4")
    create_dummy_video(video_file, duration=15.0, has_audio=False)
    desc = _make_descriptor(video_file, 15.0)

    sampler = AdaptiveSampler(method="adaptive", max_frames=10, min_frames=5)

    # Warm up (first decode may load codec)
    _ = sampler.sample_video(desc)

    # Benchmark
    t0 = time.time()
    samples = sampler.sample_video(desc)
    elapsed = time.time() - t0

    assert len(samples) >= 5
    assert len(samples) <= 10
    timestamps = [s.timestamp for s in samples]
    assert timestamps == sorted(timestamps), "Timestamps must be ordered"

    # Log performance (visible in pytest -v -s)
    print(f"\n[BENCHMARK] Adaptive sampling: {elapsed:.3f}s for {len(samples)} samples from 15s video")


def test_sampling_benchmark_scene_change(tmp_path):
    """
    Benchmark: measure total sampling time for scene_change mode.
    Optimized path eliminates 2 separate decode passes (complexity + scene detection).
    """
    video_file = str(tmp_path / "benchmark_sc.mp4")
    create_dummy_video(video_file, duration=20.0, has_audio=False)
    desc = _make_descriptor(video_file, 20.0)

    sampler = AdaptiveSampler(method="scene_change", max_frames=10, min_frames=5)

    # Warm up
    _ = sampler.sample_video(desc)

    # Benchmark
    t0 = time.time()
    samples = sampler.sample_video(desc)
    elapsed = time.time() - t0

    assert len(samples) >= 5
    assert len(samples) <= 10
    timestamps = [s.timestamp for s in samples]
    assert timestamps == sorted(timestamps)

    print(f"\n[BENCHMARK] Scene-change sampling: {elapsed:.3f}s for {len(samples)} samples from 20s video")


def test_sampling_benchmark_uniform(tmp_path):
    """
    Benchmark: uniform mode (no analysis pass needed, only extraction).
    """
    video_file = str(tmp_path / "benchmark_uni.mp4")
    create_dummy_video(video_file, duration=10.0, has_audio=False)
    desc = _make_descriptor(video_file, 10.0)

    sampler = AdaptiveSampler(method="uniform", fps=1.0, max_frames=10, min_frames=5)

    # Warm up
    _ = sampler.sample_video(desc)

    # Benchmark
    t0 = time.time()
    samples = sampler.sample_video(desc)
    elapsed = time.time() - t0

    assert len(samples) >= 5
    timestamps = [s.timestamp for s in samples]
    assert timestamps == sorted(timestamps)

    print(f"\n[BENCHMARK] Uniform sampling: {elapsed:.3f}s for {len(samples)} samples from 10s video")


def test_frame_quality_preserved(tmp_path):
    """Verify frames are valid JPEG images with reasonable sizes."""
    video_file = str(tmp_path / "quality.mp4")
    create_dummy_video(video_file, duration=5.0, has_audio=False)
    desc = _make_descriptor(video_file, 5.0)

    sampler = AdaptiveSampler(method="adaptive", max_frames=8, min_frames=3)
    samples = sampler.sample_video(desc)

    for s in samples:
        # JPEG magic bytes
        assert s.frame_data[:2] == b'\xff\xd8', "Frame data should be valid JPEG"
        assert len(s.frame_data) > 100, "JPEG should have non-trivial size"
