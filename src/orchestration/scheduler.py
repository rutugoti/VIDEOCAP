import time
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class BatchScheduler:
    """
    Coordinates batch processing runtimes to stay under the 10-minute limit
    by dynamically adjusting the execution mode (FAST, BALANCED, DEEP).
    """
    def __init__(self, total_time_limit_sec: float = 600.0, target_margin_sec: float = 60.0):
        self.total_limit = total_time_limit_sec - target_margin_sec  # E.g., 540s safe budget
        self.start_time = None
        self.completed_count = 0
        self.total_videos = 0
        self.processing_times: List[float] = []
        
        # Runtime instrumentation metrics
        self.provider_failures = 0
        self.provider_fallbacks = 0
        self.total_api_calls = 0
        self.total_tokens_used = 0

    def start_batch(self, total_videos: int):
        self.start_time = time.time()
        self.total_videos = total_videos
        self.completed_count = 0
        self.processing_times = []
        logger.info(f"Starting batch scheduler: total_videos={total_videos}, limit={self.total_limit}s")

    def register_completed(self, elapsed: float, api_calls: int, tokens: int):
        self.completed_count += 1
        self.processing_times.append(elapsed)
        self.total_api_calls += api_calls
        self.total_tokens_used += tokens
        logger.info(
            f"Registered video completion. Elapsed: {elapsed:.2f}s | "
            f"API calls: {api_calls} | Tokens: {tokens} | "
            f"Progress: {self.completed_count}/{self.total_videos}"
        )

    def register_fallback(self):
        self.provider_fallbacks += 1

    def register_failure(self):
        self.provider_failures += 1

    def decide_execution_mode(self) -> str:
        """
        Before every clip decide: FAST, BALANCED, DEEP based on remaining runtime.
        Never blindly execute the deepest pipeline.
        """
        if self.start_time is None:
            self.start_time = time.time()

        elapsed_so_far = time.time() - self.start_time
        remaining_time = self.total_limit - elapsed_so_far
        remaining_videos = self.total_videos - self.completed_count

        if remaining_videos <= 0:
            return "FAST"

        # If it's the first video, run BALANCED to gather baseline metrics
        if not self.processing_times:
            logger.info("First video in batch. Using BALANCED mode.")
            return "BALANCED"

        avg_time = sum(self.processing_times) / len(self.processing_times)
        projected_completion_time = remaining_videos * avg_time

        logger.info(
            f"Scheduler Status - Elapsed: {elapsed_so_far:.2f}s | Remaining: {remaining_time:.2f}s | "
            f"Avg Video Time: {avg_time:.2f}s | Projected batch time: {projected_completion_time:.2f}s"
        )

        # Check budget thresholds
        if projected_completion_time > remaining_time * 0.9:
            # We are running behind schedule -> switch to FAST mode to catch up
            logger.warning(
                f"Projected completion time ({projected_completion_time:.2f}s) is close to or exceeds "
                f"remaining time ({remaining_time:.2f}s). Demoting to FAST mode."
            )
            return "FAST"
        elif projected_completion_time < remaining_time * 0.5:
            # We are well ahead of schedule (plenty of time buffer) -> upgrade to DEEP mode
            logger.info(
                f"Projected completion time ({projected_completion_time:.2f}s) is well below half of "
                f"remaining time ({remaining_time:.2f}s). Upgrading to DEEP mode."
            )
            return "DEEP"
        else:
            logger.info("Using BALANCED execution mode.")
            return "BALANCED"

    def can_retry(self, mode: str) -> bool:
        """
        Decides if a validation retry is allowed based on remaining time.
        """
        if self.start_time is None:
            return False

        if mode == "FAST":
            # FAST mode never retries
            return False

        elapsed_so_far = time.time() - self.start_time
        remaining_time = self.total_limit - elapsed_so_far
        remaining_videos = self.total_videos - self.completed_count

        avg_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 30.0

        if mode == "BALANCED":
            # Retry only if we are ahead of schedule with enough time for remaining videos plus one extra run
            return (remaining_time - avg_time) > (remaining_videos * avg_time)
        elif mode == "DEEP":
            # Allow retry in DEEP mode if we have at least one average video run worth of time left
            return remaining_time > avg_time
        return False
