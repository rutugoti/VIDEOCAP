import io
import logging
import wave
from pathlib import Path
from typing import List, Optional, Tuple
import av
import numpy as np

from src.shared.models import VideoDescriptor, Sample
from src.config.settings import get_config

logger = logging.getLogger(__name__)


class AdaptiveSampler:
    """Module responsible for sampling frames and corresponding audio segments from a video."""

    def __init__(
        self,
        method: str = "uniform",
        fps: float = 1.0,
        max_frames: int = 30,
        min_frames: int = 5,
    ):
        self.method = method
        self.fps = fps
        self.max_frames = max_frames
        self.min_frames = min_frames

    def sample_video(self, video_desc: VideoDescriptor) -> List[Sample]:
        """
        Sample the video into a list of Sample objects (containing frames and audio).

        Args:
            video_desc: The VideoDescriptor of the loaded video.

        Returns:
            List of Sample objects.
        """
        video_path = video_desc.path
        duration = video_desc.duration_seconds

        # 1. Calculate number of samples
        num_samples = int(duration * self.fps)
        num_samples = max(self.min_frames, min(self.max_frames, num_samples))
        interval = duration / num_samples

        logger.info(
            f"Sampling video: {Path(video_path).name} | "
            f"Method: {self.method} | "
            f"Target Samples: {num_samples} | "
            f"Interval: {interval:.2f}s"
        )

        # Target timestamps for each sample
        target_timestamps = [i * interval for i in range(num_samples)]

        samples: List[Sample] = []
        
        try:
            # We open the video container using PyAV
            with av.open(video_path) as container:
                # Video stream setup
                video_streams = container.streams.video
                if not video_streams:
                    raise ValueError(f"No video stream in {video_path}")
                video_stream = video_streams[0]

                # Audio stream setup (optional)
                audio_streams = container.streams.audio
                has_audio = len(audio_streams) > 0 and video_desc.has_audio
                audio_stream = audio_streams[0] if has_audio else None

                # Structures to store frame data and audio samples per target window
                # We do a single sequential pass through the container for efficiency
                frames_data: List[Optional[bytes]] = [None] * num_samples
                audio_buffers: List[List[Tuple[bytes, int]]] = [[] for _ in range(num_samples)]
                
                # Metadata of audio stream if present
                audio_sample_rate = 16000
                audio_channels = 1
                audio_sample_width = 2 # 16-bit

                if audio_stream:
                    audio_sample_rate = audio_stream.rate
                    audio_channels = audio_stream.channels
                    # Determine sample width from format
                    if "s16" in audio_stream.format.name:
                        audio_sample_width = 2
                    elif "flt" in audio_stream.format.name:
                        audio_sample_width = 4
                    elif "u8" in audio_stream.format.name:
                        audio_sample_width = 1
                    else:
                        audio_sample_width = 2

                # 2. Sequential decode pass
                for packet in container.demux():
                    for frame in packet.decode():
                        # Determine timestamp
                        pts_time = float(frame.pts * frame.time_base) if frame.pts is not None else 0.0

                        if isinstance(frame, av.VideoFrame):
                            # Find which sample index this frame is closest to
                            closest_idx = int(round(pts_time / interval))
                            if 0 <= closest_idx < num_samples:
                                # Only keep the first frame found for that slot (or update if closer)
                                if frames_data[closest_idx] is None:
                                    # Convert frame to PIL image and save as JPEG bytes
                                    img = frame.to_image()
                                    buf = io.BytesIO()
                                    img.save(buf, format="JPEG")
                                    frames_data[closest_idx] = buf.getvalue()

                        elif isinstance(frame, av.AudioFrame) and has_audio:
                            # Map audio frame to the appropriate interval window
                            # The window for index i is [i * interval, (i + 1) * interval)
                            idx = int(pts_time / interval)
                            if 0 <= idx < num_samples:
                                # Convert audio plane data to bytes
                                # Note: Whisper expects mono. If stereo, we'll write stereo WAV or mono WAV
                                plane_bytes = frame.to_ndarray().tobytes()
                                audio_buffers[idx].append((plane_bytes, frame.samples))

                # 3. Finalize samples (handling missing frames or empty audio windows)
                for i in range(num_samples):
                    timestamp = target_timestamps[i]

                    # Fallback for missing frames: copy from previous frame or next frame
                    frame_bytes = frames_data[i]
                    if frame_bytes is None:
                        # Find closest available frame
                        for fallback_idx in range(num_samples):
                            if frames_data[fallback_idx] is not None:
                                frame_bytes = frames_data[fallback_idx]
                                break
                        
                        # If still None, create a dummy black frame
                        if frame_bytes is None:
                            import PIL.Image
                            dummy_img = PIL.Image.new("RGB", (320, 240), color="black")
                            buf = io.BytesIO()
                            dummy_img.save(buf, format="JPEG")
                            frame_bytes = buf.getvalue()

                    # Compile audio segment into a WAV byte stream
                    wav_bytes = None
                    if has_audio and audio_buffers[i]:
                        # Combine all plane data bytes for this window
                        combined_audio = b"".join([item[0] for item in audio_buffers[i]])
                        
                        # Write to in-memory WAV file
                        wav_io = io.BytesIO()
                        with wave.open(wav_io, "wb") as wav_file:
                            wav_file.setnchannels(audio_channels)
                            wav_file.setsampwidth(audio_sample_width)
                            wav_file.setframerate(audio_sample_rate)
                            wav_file.writeframes(combined_audio)
                        
                        wav_bytes = wav_io.getvalue()

                    samples.append(
                        Sample(
                            frame_data=frame_bytes,
                            timestamp=timestamp,
                            sample_index=i,
                            audio_segment=wav_bytes
                        )
                    )

            logger.info(f"Successfully sampled {len(samples)} samples from {Path(video_path).name}")
            return samples

        except Exception as e:
            logger.error(f"Error sampling video {video_path}: {e}")
            raise
