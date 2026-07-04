import logging
from typing import List, Optional

from src.shared.models import Sample, Observation
from src.shared.providers import VisionProvider, VisionConfig, ProviderError

logger = logging.getLogger(__name__)


class VisionProcessor:
    """Module responsible for extracting visual observations from video frames using a VisionProvider."""

    def __init__(
        self,
        provider: VisionProvider,
        config: Optional[VisionConfig] = None,
        prompt: Optional[str] = None,
        confidence_threshold: float = 0.3,
    ):
        self.provider = provider
        # Default fallback config if not supplied
        self.config = config or VisionConfig(
            provider="fireworks",
            model="accounts/fireworks/models/llava-v1.6",
            max_tokens=4096,
            temperature=0.2
        )
        self.prompt = prompt or (
            "Describe what you observe in these video frames.\n"
            "Focus on: people, actions, objects, emotions, sequence of events.\n"
            "Only describe what is visible. Do not infer motivations."
        )
        self.confidence_threshold = confidence_threshold

    def process(self, samples: List[Sample]) -> List[Observation]:
        """
        Analyze a list of sampled frames and return visual observations.

        Args:
            samples: List of Sample frames.

        Returns:
            List of visual Observation objects sorted by timestamp.
        """
        if not samples:
            logger.warning("VisionProcessor received an empty list of samples.")
            return []

        try:
            logger.info(f"VisionProcessor sending {len(samples)} frames to provider for analysis.")
            observations = self.provider.analyze_frames(
                frames=samples,
                prompt=self.prompt,
                config=self.config
            )

            # Filter by confidence and source validity
            valid_observations = []
            for obs in observations:
                if obs.source != "visual":
                    # Force correct source if model returned mismatch
                    obs.source = "visual"
                
                if obs.confidence >= self.confidence_threshold:
                    valid_observations.append(obs)
                else:
                    logger.debug(
                        f"Skipping observation '{obs.content}' due to low confidence: "
                        f"{obs.confidence} < {self.confidence_threshold}"
                    )

            # Sort observations by timestamp
            sorted_observations = sorted(valid_observations, key=lambda x: x.timestamp)
            logger.info(f"VisionProcessor generated {len(sorted_observations)} observations after filtering.")
            return sorted_observations

        except ProviderError as e:
            logger.error(f"Vision provider API failed: {e}")
            # Propagate or return empty depending on severity, standard pipeline raises ProviderError
            raise
        except Exception as e:
            logger.error(f"Unexpected error in VisionProcessor: {e}")
            raise ProviderError(f"Vision processor internal failure: {e}") from e
