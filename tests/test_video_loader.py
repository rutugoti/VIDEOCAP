import os
import pytest
import av
from src.ingestion.video_loader import VideoLoader
from src.shared.providers import InvalidVideoError


def create_dummy_video(
    path: str,
    duration: float = 35.0,
    fps: int = 24,
    width: int = 320,
    height: int = 240,
    has_audio: bool = False
) -> None:
    """Helper to generate a real, short dummy video using PyAV."""
    with av.open(path, mode="w") as container:
        # Add video stream
        video_stream = container.add_stream("mpeg4", rate=fps)
        video_stream.width = width
        video_stream.height = height
        video_stream.pix_fmt = "yuv420p"

        # Add audio stream if requested
        if has_audio:
            audio_stream = container.add_stream("aac", rate=16000)
            # Write a silent audio frame to finalize the stream
            audio_frame = av.AudioFrame(format='s16', layout='mono', samples=1024)
            audio_frame.sample_rate = 16000
            # Get plane buffer to write silence
            # For mono s16, 1024 samples * 2 bytes = 2048 bytes of zeros
            audio_frame.planes[0].update(b'\x00' * 2048)
            for packet in audio_stream.encode(audio_frame):
                container.mux(packet)
            for packet in audio_stream.encode():
                container.mux(packet)

        # Write minimum frames to satisfy duration
        # Using a fast generation by writing black frames
        total_frames = int(duration * fps)
        for i in range(total_frames):
            frame = av.VideoFrame(width, height, "yuv420p")
            # Fill with black/dummy pixels
            for packet in video_stream.encode(frame):
                container.mux(packet)

        # Flush encoder
        for packet in video_stream.encode():
            container.mux(packet)


def test_load_valid_video(tmp_path):
    video_file = str(tmp_path / "valid_video.mp4")
    # Create 35s dummy video (within [30, 120] range) with audio
    create_dummy_video(video_file, duration=35.0, has_audio=True)

    loader = VideoLoader(min_duration=30.0, max_duration=120.0)
    desc = loader.load_video(video_file)

    assert desc.has_audio is True
    assert desc.width == 320
    assert desc.height == 240
    assert desc.format == "mp4"
    # Allow small precision range on floating-point duration
    assert 34.0 <= desc.duration_seconds <= 36.0


def test_invalid_format(tmp_path):
    invalid_file = str(tmp_path / "invalid_format.txt")
    with open(invalid_file, "w") as f:
        f.write("not a video")

    loader = VideoLoader()
    with pytest.raises(InvalidVideoError) as exc_info:
        loader.load_video(invalid_file)
    assert "Unsupported video format" in str(exc_info.value)


def test_too_short(tmp_path):
    short_video = str(tmp_path / "too_short.mp4")
    # 10s video (under 30s limit)
    create_dummy_video(short_video, duration=10.0)

    loader = VideoLoader(min_duration=30.0, max_duration=120.0)
    with pytest.raises(InvalidVideoError) as exc_info:
        loader.load_video(short_video)
    assert "outside permitted bounds" in str(exc_info.value)


def test_too_long(tmp_path):
    long_video = str(tmp_path / "too_long.mp4")
    # 130s video (over 120s limit)
    create_dummy_video(long_video, duration=130.0)

    loader = VideoLoader(min_duration=30.0, max_duration=120.0)
    with pytest.raises(InvalidVideoError) as exc_info:
        loader.load_video(long_video)
    assert "outside permitted bounds" in str(exc_info.value)


def test_no_audio(tmp_path):
    silent_video = str(tmp_path / "silent_video.mp4")
    # 35s video, no audio
    create_dummy_video(silent_video, duration=35.0, has_audio=False)

    loader = VideoLoader(min_duration=30.0, max_duration=120.0)
    desc = loader.load_video(silent_video)

    assert desc.has_audio is False
    assert 34.0 <= desc.duration_seconds <= 36.0


def test_corrupted_file(tmp_path):
    corrupt_file = str(tmp_path / "corrupt.mp4")
    with open(corrupt_file, "wb") as f:
        f.write(b"raw garbage data")

    loader = VideoLoader()
    with pytest.raises(InvalidVideoError) as exc_info:
        loader.load_video(corrupt_file)
    assert "Corrupted or invalid" in str(exc_info.value)
