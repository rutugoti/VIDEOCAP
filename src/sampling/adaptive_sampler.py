import io
import logging
import time
import wave
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import av
import numpy as np

from src.shared.models import VideoDescriptor, Sample
from src.config.settings import get_config

logger = logging.getLogger(__name__)


class AdaptiveSampler:
    """
    Adaptive Sampler responsible for content-aware frame and audio sampling.
    
    Dynamically adapts frame rates, keyframes, and audio extraction based on:
    - Scene transitions/cuts (scene boundary detection)
    - Visual motion complexity (motion peaks)
    - Dynamic allocation of frame sampling budget
    - Audio silence/speech detection (disabling audio transcription for silent segments)
    
    Optimized to decode the video in a SINGLE pass, performing complexity estimation,
    scene detection, and frame/audio extraction simultaneously.
    """

    # Downscale target for analysis frames (grayscale, small) to speed up numpy ops
    _ANALYSIS_WIDTH = 160
    _ANALYSIS_HEIGHT = 120

    def __init__(
        self,
        method: str = "uniform",
        fps: float = 1.0,
        max_frames: int = 30,
        min_frames: int = 5,
        baseline_fps: float = 1.0,
        min_scene_duration: float = 1.0,
        motion_threshold: float = 15.0,
        scene_change_threshold: float = 20.0,
        silence_threshold: float = -40.0,
    ):
        self.method = method
        self.fps = fps
        self.max_frames = max_frames
        self.min_frames = min_frames
        self.baseline_fps = baseline_fps
        self.min_scene_duration = min_scene_duration
        self.motion_threshold = motion_threshold
        self.scene_change_threshold = scene_change_threshold
        self.silence_threshold = silence_threshold

    def estimate_complexity(self, video_path: str) -> float:
        """
        Estimate video motion complexity using standard deviation of frame differences.
        Returns a value between 0.0 (static) and 1.0 (highly dynamic/complex).
        """
        try:
            with av.open(video_path) as container:
                video_streams = container.streams.video
                if not video_streams:
                    return 0.5
                video_stream = video_streams[0]

                prev_gray = None
                diffs = []
                count = 0
                
                # Decode up to 30 frames spread across the video to estimate complexity
                for i, frame in enumerate(container.decode(video_stream)):
                    if i % 5 != 0:
                        continue
                    img = frame.to_image().convert("L")
                    arr = np.array(img)
                    if prev_gray is not None:
                        diff = float(np.mean(np.abs(arr.astype(np.int16) - prev_gray.astype(np.int16))))
                        diffs.append(diff)
                    prev_gray = arr
                    count += 1
                    if count >= 15:
                        break

                if not diffs:
                    return 0.5

                avg_diff = sum(diffs) / len(diffs)
                complexity = min(1.0, max(0.0, avg_diff / self.motion_threshold))
                logger.info(f"Complexity estimation: avg_diff={avg_diff:.2f}, score={complexity:.2f}")
                return complexity
        except Exception as e:
            logger.warning(f"Failed to estimate complexity, default to 0.5. Error: {e}")
            return 0.5

    def detect_scene_boundaries(self, video_path: str, duration: float) -> List[float]:
        """
        Detect scene boundaries (cuts/fades) using frame differences.
        Ensures boundaries respect min_scene_duration.
        """
        boundaries = [0.0]  # always anchor start
        try:
            with av.open(video_path) as container:
                video_streams = container.streams.video
                if not video_streams:
                    return boundaries
                video_stream = video_streams[0]

                prev_gray = None
                last_boundary_time = 0.0
                decoded = 0

                for frame in container.decode(video_stream):
                    pts_time = float(frame.pts * frame.time_base) if frame.pts is not None else float(decoded / video_stream.average_rate)
                    img = frame.to_image().convert("L")
                    arr = np.array(img, dtype=np.int16)

                    if prev_gray is not None:
                        diff = float(np.mean(np.abs(arr - prev_gray)))
                        if diff > self.scene_change_threshold:
                            if pts_time - last_boundary_time >= self.min_scene_duration:
                                boundaries.append(pts_time)
                                last_boundary_time = pts_time
                    
                    prev_gray = arr
                    decoded += 1
                    # Cap visual analysis to prevent excessive processing on huge files
                    if decoded >= 1200:
                        break

            logger.info(f"Scene detection: found {len(boundaries)} scene segments starting at {boundaries}")
            return boundaries
        except Exception as e:
            logger.warning(f"Scene boundary detection failed: {e}. Falling back to single scene.")
            return [0.0]

    def _scene_change_timestamps(self, video_path: str, num_samples: int, duration: float) -> List[float]:
        """
        Decode a bounded prefix of frames, measure per-frame grayscale change, and pick
        the timestamps at motion peaks. Falls back to uniform spacing on any failure.
        """
        uniform = [i * (duration / num_samples) for i in range(num_samples)]
        try:
            times: List[float] = []
            diffs: List[float] = []
            with av.open(video_path) as container:
                streams = container.streams.video
                if not streams:
                    return uniform
                video_stream = streams[0]
                prev = None
                decoded = 0
                for frame in container.decode(video_stream):
                    pts_time = float(frame.pts * frame.time_base) if frame.pts is not None else float(decoded)
                    arr = np.array(frame.to_image().convert("L"), dtype=np.int16)
                    if prev is not None:
                        times.append(pts_time)
                        diffs.append(float(np.mean(np.abs(arr - prev))))
                    prev = arr
                    decoded += 1
                    if decoded >= 600:
                        break
            return self._select_peak_timestamps(times, diffs, num_samples, duration)
        except Exception as e:
            logger.warning(f"scene_change detection failed ({e}); falling back to uniform sampling.")
            return uniform

    @staticmethod
    def _select_peak_timestamps(
        times: List[float], diffs: List[float], num_samples: int, duration: float
    ) -> List[float]:
        """
        Select timestamps corresponding to motion peaks.
        """
        uniform = [i * (duration / num_samples) for i in range(num_samples)]
        if not diffs or len(times) < num_samples:
            return uniform
        if max(diffs) - min(diffs) < 1e-6:
            return uniform

        min_gap = duration / (num_samples * 2) if num_samples > 0 else 0.0
        selected: List[float] = [0.0]

        for _, t in sorted(zip(diffs, times), key=lambda p: p[0], reverse=True):
            if len(selected) >= num_samples:
                break
            if all(abs(t - s) >= min_gap for s in selected):
                selected.append(t)

        for u in uniform:
            if len(selected) >= num_samples:
                break
            if u not in selected:
                selected.append(u)

        return sorted(set(selected))[:num_samples]

    def is_silent(self, audio_data: bytes, sample_width: int) -> bool:
        """
        Check if audio segment is silent based on RMS dB threshold.
        """
        if not audio_data:
            return True
        try:
            if sample_width == 2:
                samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            elif sample_width == 4:
                samples = np.frombuffer(audio_data, dtype=np.float32)
            elif sample_width == 1:
                samples = (np.frombuffer(audio_data, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
            else:
                samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            if len(samples) == 0:
                return True

            rms = np.sqrt(np.mean(samples ** 2))
            db = 20 * np.log10(rms + 1e-9)
            return bool(db < self.silence_threshold)
        except Exception as e:
            logger.warning(f"Error computing audio silence: {e}")
            return False

    # ==========================================================================
    # Single-Pass Analysis Engine
    # ==========================================================================

    def _single_pass_analysis(self, video_path: str, duration: float) -> Dict[str, Any]:
        """
        Decode the video ONCE and compute all analysis data simultaneously:
        - complexity score
        - scene boundaries
        - per-frame timestamps and motion diffs (for scene_change mode peak selection)

        Returns a dict with keys:
            complexity: float
            scene_boundaries: List[float]
            frame_times: List[float]   (timestamps of analyzed frames)
            frame_diffs: List[float]   (inter-frame motion for each analyzed frame)
        """
        t0 = time.time()
        complexity = 0.5
        boundaries = [0.0]
        frame_times: List[float] = []
        frame_diffs: List[float] = []

        try:
            with av.open(video_path) as container:
                video_streams = container.streams.video
                if not video_streams:
                    logger.warning("No video stream found during single-pass analysis.")
                    return {
                        "complexity": complexity,
                        "scene_boundaries": boundaries,
                        "frame_times": frame_times,
                        "frame_diffs": frame_diffs,
                    }

                video_stream = video_streams[0]
                # Configure the codec context to use multiple threads for decoding
                video_stream.thread_type = "AUTO"

                prev_gray: Optional[np.ndarray] = None
                last_boundary_time = 0.0
                decoded = 0
                complexity_diffs: List[float] = []

                for frame in container.decode(video_stream):
                    pts_time = (
                        float(frame.pts * frame.time_base)
                        if frame.pts is not None
                        else float(decoded / (video_stream.average_rate or 30))
                    )

                    # Convert to small grayscale for fast analysis.
                    # PyAV's reformat is backed by libswscale — much faster than
                    # frame.to_image().convert("L") which goes through PIL.
                    gray_frame = frame.reformat(
                        width=self._ANALYSIS_WIDTH,
                        height=self._ANALYSIS_HEIGHT,
                        format="gray"
                    )
                    arr = np.frombuffer(gray_frame.planes[0], dtype=np.uint8).reshape(
                        self._ANALYSIS_HEIGHT, self._ANALYSIS_WIDTH
                    )

                    if prev_gray is not None:
                        diff = float(np.mean(np.abs(
                            arr.astype(np.int16) - prev_gray.astype(np.int16)
                        )))

                        # Record for scene_change peak selection
                        frame_times.append(pts_time)
                        frame_diffs.append(diff)

                        # Scene boundary detection
                        if diff > self.scene_change_threshold:
                            if pts_time - last_boundary_time >= self.min_scene_duration:
                                boundaries.append(pts_time)
                                last_boundary_time = pts_time

                        # Accumulate complexity samples (subsample for efficiency)
                        if decoded % 5 == 0:
                            complexity_diffs.append(diff)

                    prev_gray = arr
                    decoded += 1
                    # Cap analysis frames (same cap as original detect_scene_boundaries)
                    if decoded >= 1200:
                        break

                # Compute complexity score
                if complexity_diffs:
                    avg_diff = sum(complexity_diffs) / len(complexity_diffs)
                    complexity = min(1.0, max(0.0, avg_diff / self.motion_threshold))
                    logger.info(f"Single-pass complexity: avg_diff={avg_diff:.2f}, score={complexity:.2f}")

        except Exception as e:
            logger.warning(f"Single-pass analysis failed: {e}. Using defaults.")

        elapsed = time.time() - t0
        logger.info(
            f"Single-pass analysis complete in {elapsed:.2f}s | "
            f"Complexity={complexity:.2f} | Scenes={len(boundaries)} | "
            f"Frames analyzed={len(frame_diffs)}"
        )

        return {
            "complexity": complexity,
            "scene_boundaries": boundaries,
            "frame_times": frame_times,
            "frame_diffs": frame_diffs,
        }

    # ==========================================================================
    # Main entry point
    # ==========================================================================

    def sample_video(self, video_desc: VideoDescriptor) -> List[Sample]:
        """
        Perform dynamic content-aware frame and audio sampling.
        
        Optimized to perform analysis in a single decode pass, then extract
        frames and audio in one final pass — reducing total decodes from 3 to 2
        (analysis + extraction), or from 2 to 1 for uniform mode.
        """
        video_path = video_desc.path
        duration = video_desc.duration_seconds
        t_total = time.time()

        # Backward compatibility check for uniform method
        if self.method == "uniform":
            num_samples = int(duration * self.fps)
            num_samples = max(self.min_frames, min(self.max_frames, num_samples))
            interval = duration / num_samples
            target_timestamps = [i * interval for i in range(num_samples)]
            logger.info(f"Uniform sampling: {num_samples} frames.")

        elif self.method == "scene_change":
            # Single pass gets complexity + scene data + diffs for peak selection
            analysis = self._single_pass_analysis(video_path, duration)
            complexity = analysis["complexity"]
            range_frames = self.max_frames - self.min_frames
            num_samples = int(self.min_frames + complexity * range_frames)
            num_samples = max(self.min_frames, min(self.max_frames, num_samples))

            # Use already-computed diffs for peak timestamp selection
            target_timestamps = self._select_peak_timestamps(
                analysis["frame_times"], analysis["frame_diffs"], num_samples, duration
            )
            logger.info(f"Scene change sampling: {num_samples} frames.")

        else:
            # Adaptive mode: single pass for complexity + scene boundaries
            analysis = self._single_pass_analysis(video_path, duration)
            complexity = analysis["complexity"]

            # 1. Determine Dynamic Frame Budget
            range_frames = self.max_frames - self.min_frames
            if duration <= 10.0:
                target_frames = max(self.min_frames, int(duration * 2.0))
            else:
                target_frames = int(self.min_frames + complexity * range_frames)
            target_frames = max(self.min_frames, min(self.max_frames, target_frames))

            logger.info(
                f"AdaptiveSampler: Ingesting '{Path(video_path).name}' ({duration:.2f}s) | "
                f"Complexity={complexity:.2f} | Dynamic Target Frames={target_frames}"
            )

            # 2. Scene Boundary Detection (already computed)
            scene_boundaries = analysis["scene_boundaries"]
            scene_boundaries.append(duration)
            scene_boundaries = sorted(list(set(scene_boundaries)))

            # Construct scene intervals
            intervals = []
            for i in range(len(scene_boundaries) - 1):
                intervals.append((scene_boundaries[i], scene_boundaries[i+1]))

            # 3. Budget Allocation across Scenes
            allocations = []
            total_len = sum(end - start for start, end in intervals)
            for start, end in intervals:
                ratio = (end - start) / (total_len + 1e-9)
                allocated = int(round(ratio * target_frames))
                allocations.append(max(1, allocated))

            while sum(allocations) > target_frames and any(a > 1 for a in allocations):
                idx = allocations.index(max(allocations))
                allocations[idx] -= 1
            while sum(allocations) < target_frames:
                idx = allocations.index(min(allocations))
                allocations[idx] += 1

            # 4. Generate Target Timestamps
            target_timestamps = []
            for (start, end), alloc in zip(intervals, allocations):
                if alloc == 1:
                    target_timestamps.append(start + (end - start) / 2.0)
                else:
                    step = (end - start) / alloc
                    for j in range(alloc):
                        target_timestamps.append(start + j * step + step / 2.0)

            target_timestamps = sorted(list(set(target_timestamps)))
            num_samples = len(target_timestamps)
            logger.info(f"Target keyframe timestamps: {['%.2f' % t for t in target_timestamps]}")

        # 5. Extract Frames and Audio segments (single extraction pass)
        t_extract = time.time()
        samples = self._extract_samples(
            video_path, target_timestamps, num_samples, duration, video_desc.has_audio
        )
        logger.info(f"Frame/audio extraction latency: {time.time() - t_extract:.2f}s")
        logger.info(
            f"Total sampling latency: {time.time() - t_total:.2f}s | "
            f"Samples generated: {len(samples)}"
        )
        return samples

    def _extract_samples(
        self,
        video_path: str,
        target_timestamps: List[float],
        num_samples: int,
        duration: float,
        has_audio_flag: bool,
    ) -> List[Sample]:
        """
        Extract frame images and audio segments in a single demux pass.
        Matches decoded frames to the nearest target timestamp slot.
        """
        interval_val = duration / num_samples if num_samples > 0 else duration
        samples: List[Sample] = []

        try:
            with av.open(video_path) as container:
                video_streams = container.streams.video
                if not video_streams:
                    raise ValueError(f"No video stream found in {video_path}")
                video_stream = video_streams[0]
                # Enable multi-threaded decoding
                video_stream.thread_type = "AUTO"

                audio_streams = container.streams.audio
                has_audio = len(audio_streams) > 0 and has_audio_flag
                audio_stream = audio_streams[0] if has_audio else None

                frames_data: List[Optional[bytes]] = [None] * num_samples
                audio_buffers: List[List[Tuple[bytes, int]]] = [[] for _ in range(num_samples)]

                audio_sample_rate = 16000
                audio_channels = 1
                audio_sample_width = 2

                if audio_stream:
                    audio_sample_rate = audio_stream.rate
                    audio_channels = audio_stream.channels
                    if "s16" in audio_stream.format.name:
                        audio_sample_width = 2
                    elif "flt" in audio_stream.format.name:
                        audio_sample_width = 4
                    elif "u8" in audio_stream.format.name:
                        audio_sample_width = 1

                # Pre-compute a sorted array of target timestamps for fast lookup
                ts_array = np.array(target_timestamps)
                filled_count = 0

                # Demux and decode streams
                for packet in container.demux():
                    if packet.stream.type not in ("video", "audio"):
                        continue
                    if packet.stream != video_stream and (audio_stream is None or packet.stream != audio_stream):
                        continue

                    try:
                        frames = packet.decode()
                    except Exception:
                        continue

                    for frame in frames:
                        pts_time = float(frame.pts * frame.time_base) if frame.pts is not None else 0.0

                        if isinstance(frame, av.VideoFrame):
                            # Fast closest-index lookup via numpy searchsorted
                            if self.method == "uniform":
                                closest_idx = int(round(pts_time / interval_val))
                            else:
                                idx = np.searchsorted(ts_array, pts_time)
                                # Compare with neighbors to find true closest
                                if idx == 0:
                                    closest_idx = 0
                                elif idx >= num_samples:
                                    closest_idx = num_samples - 1
                                else:
                                    if abs(ts_array[idx - 1] - pts_time) <= abs(ts_array[idx] - pts_time):
                                        closest_idx = idx - 1
                                    else:
                                        closest_idx = idx

                            if 0 <= closest_idx < num_samples:
                                if frames_data[closest_idx] is None:
                                    # Resize using PyAV's libswscale (much faster than PIL)
                                    w = frame.width
                                    if w > 600:
                                        new_width = 600
                                        new_height = int(frame.height * (new_width / w))
                                        frame = frame.reformat(width=new_width, height=new_height)

                                    img = frame.to_image()
                                    buf = io.BytesIO()
                                    img.save(buf, format="JPEG")
                                    frames_data[closest_idx] = buf.getvalue()
                                    filled_count += 1

                                    # Early exit if all slots are filled and no audio needed
                                    if filled_count >= num_samples and not has_audio:
                                        break

                        elif isinstance(frame, av.AudioFrame) and has_audio:
                            if self.method == "uniform":
                                idx = int(pts_time / interval_val)
                            else:
                                si = np.searchsorted(ts_array, pts_time)
                                if si == 0:
                                    idx = 0
                                elif si >= num_samples:
                                    idx = num_samples - 1
                                else:
                                    if abs(ts_array[si - 1] - pts_time) <= abs(ts_array[si] - pts_time):
                                        idx = si - 1
                                    else:
                                        idx = si

                            if 0 <= idx < num_samples:
                                plane_bytes = frame.to_ndarray().tobytes()
                                audio_buffers[idx].append((plane_bytes, frame.samples))

                # 6. Finalize Samples
                for i in range(num_samples):
                    timestamp = target_timestamps[i]
                    frame_bytes = frames_data[i]

                    # Fallback for missing frames
                    if frame_bytes is None:
                        for fallback_idx in range(num_samples):
                            if frames_data[fallback_idx] is not None:
                                frame_bytes = frames_data[fallback_idx]
                                break
                        if frame_bytes is None:
                            import PIL.Image
                            dummy_img = PIL.Image.new("RGB", (320, 240), color="black")
                            buf = io.BytesIO()
                            dummy_img.save(buf, format="JPEG")
                            frame_bytes = buf.getvalue()

                    # Compile and filter audio segment
                    wav_bytes = None
                    if has_audio and audio_buffers[i]:
                        combined_audio = b"".join([item[0] for item in audio_buffers[i]])

                        # Apply Silence Detection only if NOT uniform method
                        if self.method != "uniform" and self.is_silent(combined_audio, audio_sample_width):
                            logger.info(f"Silence detected at window {i} (timestamp {timestamp:.2f}s). Skipping audio transcription.")
                            wav_bytes = None
                        else:
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

            logger.info(f"Successfully generated {len(samples)} samples.")
            return samples
        except Exception as e:
            logger.error(f"Error sampling video: {e}")
            raise
