import json
import logging
from typing import List, Optional

from src.shared.models import Timeline, Event, Observation
from src.shared.providers import LLMProvider, LLMConfig, ProviderError

logger = logging.getLogger(__name__)


class FusionEngine:
    """Module responsible for grouping observations temporally and semantically into coherent Events."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        llm_config: Optional[LLMConfig] = None,
        temporal_window_seconds: float = 3.0,
        min_confidence: float = 0.3,
    ):
        self.llm_provider = llm_provider
        self.llm_config = llm_config or LLMConfig(
            provider="fireworks",
            model="accounts/fireworks/models/llama-v3-70b-instruct",
            max_tokens=2048,
            temperature=0.1
        )
        self.temporal_window_seconds = temporal_window_seconds
        self.min_confidence = min_confidence

    def fuse_timeline(self, timeline: Timeline) -> List[Event]:
        """
        Group chronologically aligned observations in the Timeline into a list of Events.

        Args:
            timeline: The unified Timeline containing multimodal observations.

        Returns:
            List of Events sorted chronologically.
        """
        observations = timeline.observations
        if not observations:
            logger.info("FusionEngine: Timeline contains no observations. Returning empty event list.")
            return []

        # 1. Temporal Window Grouping
        # We slice observations into groups. Each group starts at the timestamp of the first
        # ungrouped observation and contains all observations within `temporal_window_seconds` from it.
        sorted_obs = sorted(observations, key=lambda x: x.timestamp)
        groups: List[List[Observation]] = []
        
        current_group: List[Observation] = []
        group_start_time = sorted_obs[0].timestamp

        for obs in sorted_obs:
            if obs.timestamp <= group_start_time + self.temporal_window_seconds:
                current_group.append(obs)
            else:
                groups.append(current_group)
                current_group = [obs]
                group_start_time = obs.timestamp
        if current_group:
            groups.append(current_group)

        events: List[Event] = []

        # System prompt instructions
        system_prompt = (
            "You are the Event Fusion Engine. Your task is to merge and resolve redundant or conflicting "
            "multimodal observations in a video segment into a list of coherent events.\n"
            "Each event must be output as a JSON object matching this schema:\n"
            "{\n"
            "  \"description\": \"A brief summary of the event\",\n"
            "  \"actors\": [\"actor1\", ...],\n"
            "  \"actions\": [\"action1\", ...],\n"
            "  \"objects\": [\"object1\", ...],\n"
            "  \"timestamp_start\": float,\n"
            "  \"timestamp_end\": float,\n"
            "  \"confidence\": float (between 0.0 and 1.0),\n"
            "  \"evidence_sources\": [\"visual\", \"audio\", \"text\"],\n"
            "  \"salience\": float (between 0.0 and 1.0)\n"
            "}\n"
            "Rules:\n"
            "1. Group observations of the same action or occurrence across modalities into a single event.\n"
            "2. If modalities conflict (e.g. vision says crying, audio says laughing), prefer the modality with "
            "higher confidence and note the resolution. Prefer multi-modal agreement for higher confidence.\n"
            "3. Return ONLY a valid JSON list of event objects. No explanation, no wrapper markdown."
        )

        for idx, group in enumerate(groups):
            t_start = min(obs.timestamp for obs in group)
            t_end = max(obs.timestamp for obs in group)

            logger.info(f"Fusing window {idx}: {t_start:.1f}s - {t_end:.1f}s containing {len(group)} observations.")

            # Format prompt with observations text
            obs_lines = []
            for o in group:
                obs_lines.append(f"- [{o.source}] at {o.timestamp:.2f}s: {o.content} (type: {o.observation_type}, conf: {o.confidence:.2f})")
            observations_text = "\n".join(obs_lines)

            prompt = (
                f"Fuse the following observations from timestamp {t_start:.2f}s to {t_end:.2f}s:\n"
                f"{observations_text}\n\n"
                f"Group them into a list of coherent events. Filter out low importance events. "
                f"Ensure the event timestamp bounds stay within [{t_start:.2f}, {t_end:.2f}]."
            )

            try:
                # Call LLM Provider
                response = self.llm_provider.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    config=self.llm_config
                )

                # Parse LLM response
                parsed_events = self._parse_llm_response(response)
                
                if parsed_events:
                    # Validate and clamp bounds
                    for evt in parsed_events:
                        evt.timestamp_start = max(t_start, min(t_end, evt.timestamp_start))
                        evt.timestamp_end = max(evt.timestamp_start, min(t_end, evt.timestamp_end))
                        
                        if evt.confidence >= self.min_confidence:
                            events.append(evt)
                else:
                    # Fallback to rule-based generation if parsing yielded nothing
                    logger.warning(f"Window {idx} parsing returned empty results. Falling back to rule-based fusion.")
                    events.append(self._rule_based_fallback(group, t_start, t_end))

            except Exception as e:
                logger.error(f"Failed to fuse window {idx} via LLM due to: {e}. Using rule-based fallback.")
                events.append(self._rule_based_fallback(group, t_start, t_end))

        # Sort all fused events chronologically
        sorted_events = sorted(events, key=lambda x: x.timestamp_start)
        logger.info(f"FusionEngine generated {len(sorted_events)} events total.")
        return sorted_events

    def _parse_llm_response(self, text: str) -> List[Event]:
        """Robust parser to extract a JSON list of events from the raw LLM response."""
        text = text.strip()
        if not text:
            return []

        # Find JSON list bounds
        start_idx = text.find("[")
        end_idx = text.rfind("]")

        if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
            # Maybe it is a single object? Let's check for '{' and '}'
            start_obj = text.find("{")
            end_obj = text.rfind("}")
            if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
                try:
                    obj = json.loads(text[start_obj:end_obj + 1])
                    return [Event(**obj)]
                except Exception:
                    pass
            return []

        json_str = text[start_idx:end_idx + 1]

        try:
            data = json.loads(json_str)
            if not isinstance(data, list):
                data = [data]
            
            events = []
            for item in data:
                # Basic schema alignment/fallbacks
                if "timestamp_start" not in item:
                    item["timestamp_start"] = 0.0
                if "timestamp_end" not in item:
                    item["timestamp_end"] = item["timestamp_start"]
                if "confidence" not in item:
                    item["confidence"] = 0.8
                if "salience" not in item:
                    item["salience"] = 0.5
                if "actors" not in item:
                    item["actors"] = []
                if "actions" not in item:
                    item["actions"] = []
                if "objects" not in item:
                    item["objects"] = []
                if "evidence_sources" not in item:
                    item["evidence_sources"] = []

                events.append(Event(**item))
            return events

        except Exception as e:
            logger.debug(f"Regex JSON parsing failed for: {json_str} | Error: {e}")
            return []

    def _rule_based_fallback(self, group: List[Observation], t_start: float, t_end: float) -> Event:
        """Rule-based backup fusion to extract an event when the LLM/parser fails."""
        descriptions = []
        actors = set()
        actions = set()
        objects = set()
        sources = set()
        
        total_conf = 0.0
        
        for obs in group:
            descriptions.append(obs.content)
            sources.add(obs.source)
            total_conf += obs.confidence

            # Basic keyword heuristic for actors, actions, objects
            words = obs.content.lower().split()
            # Simple keyword matching for common nouns/verbs
            for word in words:
                word_clean = "".join(c for c in word if c.isalnum())
                if word_clean in ["person", "man", "woman", "dog", "cat", "child", "speaker"]:
                    actors.add(word_clean)
                elif word_clean in ["walks", "runs", "talks", "enters", "starts", "plays", "speaks", "laughs"]:
                    actions.add(word_clean)
                elif word_clean in ["kitchen", "guitar", "appliances", "sign", "logo", "door", "coffee"]:
                    objects.add(word_clean)

        description = "; ".join(descriptions)
        avg_confidence = total_conf / len(group) if group else 0.5

        return Event(
            description=description[:250], # Cap length
            actors=list(actors) if actors else ["unknown"],
            actions=list(actions) if actions else ["occurs"],
            objects=list(objects),
            timestamp_start=t_start,
            timestamp_end=t_end,
            confidence=avg_confidence,
            evidence_sources=list(sources),
            salience=0.5
        )
