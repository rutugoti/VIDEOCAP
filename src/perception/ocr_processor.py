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

    @staticmethod
    def check_text_heuristics(frame_data: bytes) -> bool:
        """
        Fast local check using OpenCV to detect text-like properties in the frame.
        """
        import cv2
        import numpy as np
        
        if not frame_data or frame_data == b"\x00":
            return False
            
        try:
            nparr = np.frombuffer(frame_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return False
                
            h, w = img.shape
            # Text typically resides in the lower third or upper header regions
            roi_bottom = img[int(h*0.7):h, :]
            roi_top = img[0:int(h*0.25), :]
            
            # Calculate Canny edge density
            edges_bottom = cv2.Canny(roi_bottom, 50, 150)
            edges_top = cv2.Canny(roi_top, 50, 150)
            
            density_bottom = np.sum(edges_bottom > 0) / roi_bottom.size
            density_top = np.sum(edges_top > 0) / roi_top.size
            
            # Heuristic threshold: if either region has sufficient high frequency edges
            return density_bottom > 0.015 or density_top > 0.015
        except Exception as e:
            logger.warning(f"Error in OCR text heuristics check: {e}")
            return True # Fallback to True to avoid false negatives

    def process(self, samples: List[Sample], mode: str = "DEEP") -> List[Observation]:
        """
        Extract text from frames and return OCR text observations.

        Args:
            samples: List of Sample frames.
            mode: Execution mode ('FAST', 'BALANCED', 'DEEP').

        Returns:
            List of text Observation objects sorted by timestamp and deduplicated.
        """
        if mode == "FAST":
            logger.info("OCRProcessor: FAST mode active. Skipping OCR processing completely.")
            return []

        if not samples:
            logger.warning("OCRProcessor received an empty list of samples.")
            return []

        # Filter samples based on heuristics in BALANCED mode
        active_samples = samples
        if mode == "BALANCED":
            active_samples = [s for s in samples if self.check_text_heuristics(s.frame_data)]
            logger.info(
                f"OCRProcessor: BALANCED mode filtered {len(samples)} samples down to "
                f"{len(active_samples)} candidate text frames."
            )
            if not active_samples:
                logger.info("OCRProcessor: No candidate text frames found. Skipping OCR API requests.")
                return []
        else:
            logger.info(f"OCRProcessor: DEEP mode active. Processing all {len(samples)} frames.")

        try:
            logger.info(f"OCRProcessor sending {len(active_samples)} frames to OCR provider.")
            observations = self.provider.extract_text(
                frames=active_samples,
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

