import logging
from typing import List, Optional

from src.shared.models import Sample, Observation
from src.shared.providers import OCRProvider, OCRConfig, ProviderError

logger = logging.getLogger(__name__)


class OCRProcessor:
    """Module responsible for extracting on-screen text from video frames using an OCRProvider."""

    def __init__(
        self,
        provider: OCRProvider,
        config: Optional[OCRConfig] = None,
        min_confidence: float = 0.3,
    ):
        self.provider = provider
        self.config = config or OCRConfig(
            provider="fireworks",
            model="accounts/fireworks/models/llava-v1.6",
            max_tokens=1024
        )
        self.min_confidence = min_confidence

    def process(self, samples: List[Sample]) -> List[Observation]:
        """
        Extract text from frames and return OCR text observations.

        Args:
            samples: List of Sample frames.

        Returns:
            List of text Observation objects sorted by timestamp and deduplicated.
        """
        if not samples:
            logger.warning("OCRProcessor received an empty list of samples.")
            return []

        try:
            logger.info(f"OCRProcessor sending {len(samples)} frames to OCR provider.")
            observations = self.provider.extract_text(
                frames=samples,
                config=self.config
            )

            # Filter by confidence and ensure correct source
            filtered_observations = []
            for obs in observations:
                if obs.source != "text":
                    obs.source = "text"
                
                if obs.confidence >= self.min_confidence:
                    filtered_observations.append(obs)
                else:
                    logger.debug(
                        f"Skipping OCR observation '{obs.content}' due to low confidence: "
                        f"{obs.confidence} < {self.min_confidence}"
                    )

            # Sort chronologically by timestamp
            sorted_observations = sorted(filtered_observations, key=lambda x: x.timestamp)

            # Deduplicate same text across consecutive frames
            deduplicated_observations: List[Observation] = []
            for obs in sorted_observations:
                if not deduplicated_observations:
                    deduplicated_observations.append(obs)
                else:
                    prev_obs = deduplicated_observations[-1]
                    # If content is identical (case-insensitive strip) and within close time bounds
                    content_match = obs.content.strip().lower() == prev_obs.content.strip().lower()
                    
                    if content_match:
                        # Duplicate consecutive detection: merge by keeping the one with higher confidence
                        if obs.confidence > prev_obs.confidence:
                            deduplicated_observations[-1] = obs
                        logger.debug(f"Deduplicated OCR observation: '{obs.content}'")
                    else:
                        deduplicated_observations.append(obs)

            logger.info(
                f"OCRProcessor generated {len(deduplicated_observations)} "
                f"observations (filtered from {len(observations)})."
            )
            return deduplicated_observations

        except ProviderError as e:
            logger.error(f"OCR provider API failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in OCRProcessor: {e}")
            raise ProviderError(f"OCR processor internal failure: {e}") from e
