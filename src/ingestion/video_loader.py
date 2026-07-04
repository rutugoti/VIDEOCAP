import os
import logging
from pathlib import Path
from typing import List, Optional
import av

from src.shared.models import VideoDescriptor
from src.shared.providers import InvalidVideoError
from src.config.settings import get_config

logger = logging.getLogger(__name__)


class VideoLoader:
    """Module responsible for loading and validating video files."""

    def __init__(
        self,
        supported_formats: Optional[List[str]] = None,
        min_duration: float = 30.0,
        max_duration: float = 120.0,
    ):
        self.supported_formats = supported_formats or [".mp4", ".mkv", ".mov", ".avi"]
        self.min_duration = min_duration
        self.max_duration = max_duration

    def load_video(self, video_path: str) -> VideoDescriptor:
        """
        Load a video file, validate its format/duration, and extract metadata.

        Args:
            video_path: String path to the video file.

        Returns:
            VideoDescriptor containing video metadata.

        Raises:
            InvalidVideoError: If the video format is unsupported, duration is outside 
                              valid ranges, or the file is corrupted.
        """
        path = Path(video_path)
        if not path.exists():
            raise InvalidVideoError(f"Video file does not exist: {video_path}")

        # Validate format (extension)
        ext = path.suffix.lower()
        if ext not in self.supported_formats:
            raise InvalidVideoError(
                f"Unsupported video format '{ext}'. Supported formats: {self.supported_formats}"
            )

        file_size_bytes = path.stat().st_size

        try:
            # Use PyAV (av) to extract video metadata and check audio presence
            with av.open(str(path)) as container:
                # Get video streams
                video_streams = container.streams.video
                if not video_streams:
                    raise InvalidVideoError(f"No video stream found in: {video_path}")

                video_stream = video_streams[0]
                
                # Check for audio streams
                audio_streams = container.streams.audio
                has_audio = len(audio_streams) > 0

                # Duration estimation: PyAV duration is in microseconds, or check stream duration
                duration_seconds = float(container.duration / av.time_base) if container.duration else 0.0
                
                # Fallback duration from video stream if container duration is unavailable
                if duration_seconds <= 0.0 and video_stream.duration:
                    time_base = float(video_stream.time_base)
                    duration_seconds = float(video_stream.duration * time_base)

                # Frame-level properties
                width = video_stream.width
                height = video_stream.height
                
                # FPS estimation
                fps = float(video_stream.average_rate) if video_stream.average_rate else 30.0
                if fps <= 0.0:
                    fps = 30.0

                # Validate extracted properties
                if width <= 0 or height <= 0:
                    raise InvalidVideoError(f"Invalid video dimensions: {width}x{height}")

                # Validate duration constraints
                if duration_seconds < self.min_duration or duration_seconds > self.max_duration:
                    raise InvalidVideoError(
                        f"Video duration ({duration_seconds:.2f}s) is outside permitted bounds "
                        f"({self.min_duration}s - {self.max_duration}s)"
                    )

                logger.info(
                    f"Successfully loaded video: {path.name} | "
                    f"Duration: {duration_seconds:.2f}s | "
                    f"Resolution: {width}x{height} | "
                    f"FPS: {fps:.2f} | "
                    f"Audio: {has_audio}"
                )

                return VideoDescriptor(
                    path=str(path.resolve()),
                    duration_seconds=duration_seconds,
                    width=width,
                    height=height,
                    fps=fps,
                    has_audio=has_audio,
                    format=ext.lstrip("."),
                    file_size_bytes=file_size_bytes
                )

        except av.FFmpegError as e:
            logger.error(f"PyAV decoding error for {video_path}: {e}")
            raise InvalidVideoError(f"Corrupted or invalid video file: {video_path}") from e
        except InvalidVideoError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading video {video_path}: {e}")
            raise InvalidVideoError(f"Failed to load video due to internal error: {e}") from e
