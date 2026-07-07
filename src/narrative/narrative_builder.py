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
        consume_edges: bool = True,
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
        self.consume_edges = consume_edges

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

        if self.consume_edges:
            top_event_ids = {node.id for node in top_events_chrono}
            
            # Map participating entities (actors/objects) to top events
            event_entities = {eid: [] for eid in top_event_ids}
            entity_events = {}
            
            for edge in graph.edges:
                if edge.relationship == "participates_in" and edge.target in top_event_ids:
                    ent = edge.source
                    event_entities[edge.target].append(ent)
                    if ent not in entity_events:
                        entity_events[ent] = []
                    entity_events[ent].append(edge.target)

            # Extract temporal and causal relations between top events
            temporal_relations = []
            causal_relations = []
            for edge in graph.edges:
                if edge.source in top_event_ids and edge.target in top_event_ids:
                    if edge.relationship == "temporal":
                        temporal_relations.append(f"Event {edge.source} happened before Event {edge.target}")
                    elif edge.relationship == "causal":
                        causal_relations.append(f"Event {edge.source} causally triggered Event {edge.target}")

            # Build actor-centric threads
            entity_threads = []
            for ent, eids in sorted(entity_events.items()):
                nodes_for_ent = [n for n in top_events_chrono if n.id in eids]
                sorted_nodes = sorted(nodes_for_ent, key=lambda x: x.attributes.get("timestamp_start", 0.0))
                event_labels_str = " -> ".join([f"[ID: {n.id}] (\"{sanitize_untrusted_input(n.label)}\")" for n in sorted_nodes])
                entity_threads.append(f"- Entity '{ent}' participates in: {event_labels_str}")

            entity_threads_text = "\n".join(entity_threads) if entity_threads else "- No entity participation threads."
            temporal_relations_text = "\n".join(temporal_relations) if temporal_relations else "- No explicit temporal order edges."
            causal_relations_text = "\n".join(causal_relations) if causal_relations else "- No explicit causal relationship edges."

            # Reconstruct details with evidence mapping and participating entities
            event_details = []
            any_uncertain = False
            for idx, node in enumerate(top_events_chrono):
                sources = node.attributes.get("evidence_sources", ["visual"])
                label = node.label
                t_start = node.attributes.get("timestamp_start", 0.0)
                salience = node.attributes.get("salience", 0.5)
                entities_list = event_entities.get(node.id, [])

                # P4.2: surface a genuine, unresolved ambiguity flagged upstream by fusion.
                uncertainty_note = ""
                if node.attributes.get("contested") and node.attributes.get("alternatives"):
                    any_uncertain = True
                    alts = node.attributes.get("alternatives", [])
                    alt_str = " | ".join(sanitize_untrusted_input(str(a)) for a in alts)
                    uncertainty_note = f" [UNCERTAIN — competing interpretations: {alt_str}]"

                sanitized_label = sanitize_untrusted_input(label)
                event_details.append(
                    f"- [ID: {node.id}] at {t_start:.1f}s: <untrusted_input>\"{sanitized_label}\"</untrusted_input> "
                    f"(salience: {salience:.2f}, sources: {sources}, entities: {entities_list}){uncertainty_note}"
                )

            events_list_text = "\n".join(event_details)

            uncertainty_rule = (
                "\n- Some events are marked [UNCERTAIN — competing interpretations: ...]. "
                "For at most ONE such event you MAY express the ambiguity with hedging language "
                "(e.g. 'appears to', 'possibly', 'either ... or ...'), presenting the competing "
                "interpretations. Never fabricate certainty for an uncertain event, and never hedge "
                "an event that is not marked uncertain.\n"
                if any_uncertain else ""
            )

            system_prompt = (
                "You are the Narrative Builder. Your task is to compress a list of video events into a "
                "style-neutral, factually-grounded, chronological text summary (a narrative).\n"
                "You are provided with:\n"
                "1. A list of events with timestamps, evidence sources, and participating entities.\n"
                "2. Entity participation threads showing what sequence of events each entity was involved in.\n"
                "3. Explicit temporal and causal relations between events.\n\n"
                "Your output narrative text must strictly respect these entity participation sequences and temporal/causal ordering. "
                "Do not invent facts or hallucinate details that contradict these constraints."
                + uncertainty_rule +
                "\nYou must return a valid JSON object with the following schema:\n"
                "{\n"
                "  \"text\": \"Your style-neutral chronological narrative summary.\",\n"
                "  \"key_events\": [\"event description 1\", \"event description 2\"],\n"
                "  \"salience_scores\": {\"event description 1\": 0.95, \"event description 2\": 0.8},\n"
                "  \"evidence_mapping\": {\"event description 1\": [\"visual\"], \"event description 2\": [\"visual\", \"audio\"]}\n"
                "}\n"
            )

            prompt = (
                f"Here are the chronological events:\n{events_list_text}\n\n"
                f"Here are the entity participation threads:\n{entity_threads_text}\n\n"
                f"Here are the temporal relationships:\n{temporal_relations_text}\n\n"
                f"Here are the causal relationships:\n{causal_relations_text}\n\n"
                "Construct a style-neutral chronological narrative summarizing these events."
            )
        else:
            # Legacy fallback prompt construction
            event_details = []
            for idx, node in enumerate(top_events_chrono):
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
                "You must return a valid JSON object with the following schema:\n"
                "{\n"
                "  \"text\": \"Your style-neutral chronological narrative summary.\",\n"
                "  \"key_events\": [\"event description 1\", \"event description 2\"],\n"
                "  \"salience_scores\": {\"event description 1\": 0.95, \"event description 2\": 0.8},\n"
                "  \"evidence_mapping\": {\"event description 1\": [\"visual\"], \"event description 2\": [\"visual\", \"audio\"]}\n"
                "}\n"
            )

            prompt = (
                f"Here are the chronological events:\n{events_list_text}\n\n"
                "Construct a style-neutral chronological narrative summarizing these events."
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
