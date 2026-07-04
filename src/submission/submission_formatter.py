import json
import logging
import os
from typing import Dict, List, Optional, Any

from src.shared.models import Caption

logger = logging.getLogger(__name__)


class SubmissionFormatter:
    """Module responsible for packaging validated captions into the competition-required JSON format."""

    STYLE_MAP = {
        "formal": "formal",
        "sarcastic": "sarcastic",
        "tech_humor": "humorous_tech",
        "non_tech_humor": "humorous_non_tech"
    }

    def format_video_captions(self, video_id: str, captions: Dict[str, Caption]) -> Dict[str, str]:
        """
        Format the captions for a single video into the competition dictionary format.

        Args:
            video_id: Unique identifier for the video.
            captions: Dict mapping internal style names to Caption objects.

        Returns:
            A dictionary containing mapped style keys and their caption text values.
        """
        output = {"video_id": video_id}
        for internal_style, target_key in self.STYLE_MAP.items():
            caption = captions.get(internal_style)
            output[target_key] = caption.text if caption else ""
        return output

    def write_submission(self, formatted_videos: List[Dict[str, str]], output_path: str) -> None:
        """
        Write the list of formatted video captions to a JSON file.

        Args:
            formatted_videos: List of formatted video caption dictionaries.
            output_path: Path to write the output JSON file.
        """
        try:
            # Ensure parent directories exist
            parent_dir = os.path.dirname(output_path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(formatted_videos, f, indent=2, ensure_ascii=False)

            logger.info(f"Successfully wrote {len(formatted_videos)} videos to submission file: {output_path}")

        except Exception as e:
            logger.error(f"Failed to write submission JSON file to {output_path}: {e}")
            raise
