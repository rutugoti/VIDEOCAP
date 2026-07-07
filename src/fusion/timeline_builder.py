import logging
from typing import List

from src.shared.models import Observation, Timeline

logger = logging.getLogger(__name__)


class TimelineBuilder:
    """Module responsible for merging and sorting observations into a unified Timeline."""

    def build_timeline(self, observations: List[Observation], duration: float) -> Timeline:
        """
        Merge observations from all modalities into a single, sorted Timeline.

        Args:
            observations: List of all observations from perception layers (vision, speech, text).
            duration: Video duration in seconds.

        Returns:
            A Timeline object containing sorted observations.
        """
        if not observations:
            logger.info("TimelineBuilder: Building timeline from empty observations.")
            return Timeline(observations=[], duration=duration, modalities_present=[])

        # Identify unique modalities present
        modalities_set = set(obs.source for obs in observations)
        modalities_present = sorted(list(modalities_set))

        logger.info(
            f"TimelineBuilder merging {len(observations)} observations across "
            f"modalities: {modalities_present} | Video duration: {duration:.2f}s"
        )

        # The Timeline Pydantic model's field_validator automatically sorts observations
        # chronologically by timestamp (stable sort).
        timeline = Timeline(
            observations=observations,
            duration=duration,
            modalities_present=modalities_present
        )

        # Assign stable ascending ids in chronological order so downstream stages
        # (EvidenceResolutionPolicy contested-pair alternatives, Event provenance
        # back-links) can reference observations. Without this, `Observation.id`
        # stays None and alternative cross-references are silently dropped.
        for idx, obs in enumerate(timeline.observations):
            if obs.id is None:
                obs.id = idx

        return timeline
