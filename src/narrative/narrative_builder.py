import json
import logging
from typing import List, Dict, Any, Optional

from src.shared.models import SemanticGraph, Narrative, Event, GraphNode
from src.shared.providers import LLMProvider, LLMConfig, ProviderError
from src.shared.utils import sanitize_untrusted_input

logger = logging.getLogger(__name__)


class NarrativeBuilder:
    """Module responsible for compressing the SemanticGraph into a style-neutral Narrative."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        llm_config: Optional[LLMConfig] = None,
        max_events: int = 5,
        max_words: int = 100,
    ):
        self.llm_provider = llm_provider
        self.llm_config = llm_config or LLMConfig(
            provider="fireworks",
            model="accounts/fireworks/models/gemma-3-27b-it",
            max_tokens=1024,
            temperature=0.1
        )
        self.max_events = max_events
        self.max_words = max_words

    def build_narrative(self, graph: SemanticGraph) -> Narrative:
        """
        Compress the semantic graph into a concise, style-neutral, importance-ranked Narrative.

        Args:
            graph: The SemanticGraph representing video occurrences.

        Returns:
            A Narrative object.
        """
        # Extract events from graph nodes
        event_nodes = [node for node in graph.nodes if node.type == "event"]
        if not event_nodes:
            logger.info("NarrativeBuilder: SemanticGraph contains no events. Returning empty Narrative.")
            return Narrative(text="No events recorded.", key_events=[], salience_scores={}, evidence_mapping={})

        # Reconstruct temporal order based on graph edges or attributes
        # Sort by timestamp_start (stored in attributes)
        sorted_events = sorted(
            event_nodes,
            key=lambda x: x.attributes.get("timestamp_start", 0.0)
        )

        # Rank events by salience
        ranked_events = sorted(
            sorted_events,
            key=lambda x: x.attributes.get("salience", 0.5),
            reverse=True
        )

        # Select top events within max_events limit
        top_events = ranked_events[:self.max_events]
        # Keep top events in chronological order for narrative generation
        top_events_chrono = sorted(
            top_events,
            key=lambda x: x.attributes.get("timestamp_start", 0.0)
        )

        logger.info(
            f"NarrativeBuilder: Selected top {len(top_events_chrono)} salient events "
            f"out of {len(event_nodes)} total events."
        )

        # Build list of evidence mapping for top events
        # We can scan graph participates_in edges to map entities or check source modalities
        # Wait, in the events, evidence_sources were saved.
        # Let's find associated evidence sources. Since attributes might not have full evidence list,
        # we can check if they are in the graph metadata or reconstruct.
        # Actually, let's look at the events context. Since we only have the GraphNodes from the graph,
        # we can query the participates_in relations or retrieve evidence from attributes.
        # Let's see: we can populate a summary for LLM to digest.
        event_details = []
        for idx, node in enumerate(top_events_chrono):
            # Try to get evidence sources from attributes
            sources = node.attributes.get("evidence_sources", ["visual"])
            label = node.label
            t_start = node.attributes.get("timestamp_start", 0.0)
            salience = node.attributes.get("salience", 0.5)
            
            sanitized_label = sanitize_untrusted_input(label)
            event_details.append(
                f"- [ID: {node.id}] at {t_start:.1f}s: <untrusted_input>\"{sanitized_label}\"</untrusted_input> (salience: {salience:.2f}, sources: {sources})"
            )

        events_list_text = "\n".join(event_details)

        system_prompt = (
            "You are the Narrative Builder. Your task is to compress a list of video events into a "
            "style-neutral, factually-grounded, chronological text summary (a narrative).\n"
            "Rules:\n"
            "1. Output ONLY a valid JSON object matching the Narrative schema:\n"
            "{\n"
            "  \"text\": \"concise narrative summary text (under 100 words)\",\n"
            "  \"key_events\": [\"event label 1\", \"event label 2\", ...],\n"
            "  \"salience_scores\": {\n"
            "     \"event label 1\": float (0.0 to 1.0),\n"
            "     ...\n"
            "  },\n"
            "  \"evidence_mapping\": {\n"
            "     \"event label 1\": [\"visual\", \"audio\", ...],\n"
            "     ...\n"
            "  }\n"
            "}\n"
            "2. The text summary must be style-neutral (no humor, no sarcasm, no opinion, just facts).\n"
            "3. Ensure the events described in key_events map to the ones provided.\n"
            "4. Return ONLY the JSON object. Do not wrap in markdown or explain."
        )

        prompt = (
            f"Build a narrative from these events:\n"
            f"{events_list_text}\n\n"
            f"Generate a chronological text summary of these events under {self.max_words} words."
        )

        try:
            # Generate via LLM
            response = self.llm_provider.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                config=self.llm_config
            )

            # Parse LLM response
            narrative = self._parse_llm_response(response, top_events_chrono)
            if narrative:
                # Enforce word limit by cutting if excessive
                words = narrative.text.split()
                if len(words) > self.max_words:
                    narrative.text = " ".join(words[:self.max_words]) + "."
                
                logger.info("NarrativeBuilder: Successfully built narrative via LLM.")
                return narrative
            else:
                logger.warning("NarrativeBuilder: LLM response parsing failed. Using rule-based fallback.")
                return self._rule_based_fallback(top_events_chrono)

        except Exception as e:
            logger.error(f"NarrativeBuilder: LLM generation failed: {e}. Using rule-based fallback.")
            return self._rule_based_fallback(top_events_chrono)

    def _parse_llm_response(self, text: str, events: List[GraphNode]) -> Optional[Narrative]:
        """Parse the JSON response from the LLM into a Narrative object."""
        text = text.strip()
        if not text:
            return None

        # Find JSON boundaries
        start_idx = text.find("{")
        end_idx = text.rfind("}")

        if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
            return None

        json_str = text[start_idx:end_idx + 1]

        try:
            data = json.loads(json_str)
            
            # Validation and alignment
            if "text" not in data or not data["text"].strip():
                return None
            if "key_events" not in data or not isinstance(data["key_events"], list):
                data["key_events"] = [e.label for e in events]
            if "salience_scores" not in data or not isinstance(data["salience_scores"], dict):
                data["salience_scores"] = {e.label: e.attributes.get("salience", 0.5) for e in events}
            if "evidence_mapping" not in data or not isinstance(data["evidence_mapping"], dict):
                data["evidence_mapping"] = {e.label: e.attributes.get("evidence_sources", ["visual"]) for e in events}

            return Narrative(**data)

        except Exception as e:
            logger.debug(f"Narrative JSON parsing error: {e}")
            return None

    def _rule_based_fallback(self, events: List[GraphNode]) -> Narrative:
        """Fallback method to construct a valid Narrative if LLM or JSON parsing fails."""
        descriptions = []
        key_events = []
        salience_scores = {}
        evidence_mapping = {}

        for event in events:
            label = event.label
            descriptions.append(label)
            key_events.append(label)
            
            # Extract from attributes with defaults
            salience_scores[label] = event.attributes.get("salience", 0.5)
            evidence_mapping[label] = event.attributes.get("evidence_sources", ["visual"])

        # Join chronological descriptions into a neutral summary
        text = ". ".join(descriptions) + "."

        return Narrative(
            text=text,
            key_events=key_events,
            salience_scores=salience_scores,
            evidence_mapping=evidence_mapping
        )
