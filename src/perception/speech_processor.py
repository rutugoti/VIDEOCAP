import logging
from typing import List, Optional

from src.shared.models import Sample, Observation
from src.shared.providers import AudioProvider, AudioConfig, ProviderError

logger = logging.getLogger(__name__)


class SpeechProcessor:
    """Module responsible for transcribing speech from audio segments using an AudioProvider."""

    def __init__(
        self,
        provider: AudioProvider,
        config: Optional[AudioConfig] = None,
    ):
        self.provider = provider
        self.config = config or AudioConfig(
            provider="fireworks",
            model="whisper-v3",
            language="en"
        )

    def process(self, samples: List[Sample], has_audio: bool = True) -> List[Observation]:
        """
        Transcribe audio segments from the sample list and return speech observations.

        Args:
            samples: List of Sample objects.
            has_audio: Flag indicating if the video track has audio.

        Returns:
            List of audio Observation objects sorted by timestamp.
        """
        if not has_audio or not samples:
            logger.info("SpeechProcessor: Skipping transcription (no audio track or no samples).")
            return []

        # Filter samples that actually contain audio segment bytes
        samples_with_audio = [s for s in samples if s.audio_segment is not None]
        if not samples_with_audio:
            logger.info("SpeechProcessor: No audio segments found in samples.")
            return []

        try:
            logger.info(f"SpeechProcessor transcribing {len(samples_with_audio)} audio segments.")
            observations = self.provider.transcribe(
                frames=samples_with_audio,
                config=self.config
            )

            # Ensure source and type are correct
            valid_observations = []
            for obs in observations:
                if obs.source != "audio":
                    obs.source = "audio"
                valid_observations.append(obs)

            # Sort observations chronologically
            sorted_observations = sorted(valid_observations, key=lambda x: x.timestamp)
            logger.info(f"SpeechProcessor generated {len(sorted_observations)} observations.")
            return sorted_observations

        except ProviderError as e:
            logger.error(f"Audio provider API failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in SpeechProcessor: {e}")
            raise ProviderError(f"Speech processor internal failure: {e}") from e
