import json
import time
import logging
from typing import List, Dict, Any, Tuple, Optional

from src.shared.models import Timeline, Event, Observation, SemanticGraph, GraphNode, GraphEdge, Narrative
from src.shared.providers import LLMProvider, LLMConfig, ProviderError
from src.shared.utils import sanitize_untrusted_input

logger = logging.getLogger(__name__)

class SemanticContractResolver:
    """
    Unified reasoning center of the pipeline. Replaces Fusion -> Graph -> Narrative builder
    with a single structured reasoning call.
    """
    def __init__(self, llm_provider: LLMProvider, llm_config: Optional[LLMConfig] = None):
        self.llm_provider = llm_provider
        self.llm_config = llm_config or LLMConfig(
            provider="fireworks",
            model="accounts/fireworks/models/gemma-3-27b-it",
            max_tokens=3072,
            temperature=0.1
        )

    def resolve(self, timeline: Timeline) -> Tuple[List[Event], SemanticGraph, Narrative]:
        """
        Execute the Semantic Contract reasoning pass on all observations in a timeline.
        Returns a tuple of (events, graph, narrative).
        """
        t_start = time.time()
        observations = timeline.observations

        if not observations:
            logger.info("SemanticContractResolver: Timeline contains no observations.")
            return [], SemanticGraph(), Narrative(text="No events observed.")

        # 1. Format observations for the prompt
        obs_lines = []
        for o in observations:
            # We explicitly output the unique ID so the LLM can reference it in source_observation_ids
            obs_id_str = f"ID {o.id}" if o.id is not None else "ID unknown"
            sanitized = sanitize_untrusted_input(o.content)
            obs_lines.append(
                f"- {obs_id_str} [{o.source}] at {o.timestamp:.2f}s: \"{sanitized}\" "
                f"(type: {o.observation_type}, conf: {o.confidence:.2f})"
            )
        observations_text = "\n".join(obs_lines)

        # 2. Construct the prompt
        system_prompt = (
            "You are the central video reasoning engine. Your task is to analyze a timeline of raw "
            "multimodal observations (visual, speech, text) and resolve them into a single, cohesive "
            "semantic structure.\n"
            "You MUST output exactly a single JSON object matching this schema:\n"
            "{\n"
            "  \"events\": [\n"
            "    {\n"
            "      \"id\": \"evt_1\",\n"
            "      \"description\": \"A clear, descriptive summary of the event\",\n"
            "      \"actors\": [\"actor1\", ...],\n"
            "      \"actions\": [\"action1\", ...],\n"
            "      \"objects\": [\"object1\", ...],\n"
            "      \"timestamp_start\": float,\n"
            "      \"timestamp_end\": float,\n"
            "      \"confidence\": float,\n"
            "      \"evidence_sources\": [\"visual\", \"audio\", \"text\"],\n"
            "      \"salience\": float,\n"
            "      \"source_observation_ids\": [int, ...]\n"
            "    }\n"
            "  ],\n"
            "  \"relationships\": [\n"
            "    {\n"
            "      \"source\": \"evt_1\",\n"
            "      \"target\": \"evt_2\",\n"
            "      \"relationship\": \"temporal | causal | enables\",\n"
            "      \"confidence\": float,\n"
            "      \"evidence\": \"One sentence explanation of the link\"\n"
            "    }\n"
            "  ],\n"
            "  \"narrative\": \"A cohesive, style-neutral prose description of the video, describing what happens chronologically, incorporating relationships and uncertainty without mentioning event IDs.\"\n"
            "}\n\n"
            "Rules:\n"
            "1. Merge redundant or overlapping observations into a sequence of events.\n"
            "2. Under 'source_observation_ids', list the integer IDs of the observations that serve as direct evidence for that event.\n"
            "3. Ground all statements in the narrative on the listed events. Do not fabricate events.\n"
            "4. Respond ONLY with a valid JSON object. No explanation, no wrapper markdown."
        )

        prompt = (
            f"Video Duration: {timeline.duration:.2f} seconds.\n"
            f"Raw Observations:\n{observations_text}\n\n"
            f"Process the timeline observations and return the structured JSON output."
        )

        try:
            response = self.llm_provider.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                config=self.llm_config
            )

            # Parse and repair JSON
            from src.shared.fireworks_providers import _parse_and_repair_json
            parsed = _parse_and_repair_json(response)
            if not parsed or not isinstance(parsed, dict):
                raise ValueError("Parsed result is empty or not a dictionary")

            # 3. Extract Events
            events: List[Event] = []
            event_id_map: Dict[str, Event] = {}
            raw_events = parsed.get("events", [])
            for item in raw_events:
                try:
                    evt_id = item.get("id", f"evt_{len(events)+1}")
                    # Clamp timestamps
                    t_start_val = max(0.0, min(timeline.duration, item.get("timestamp_start", 0.0)))
                    t_end_val = max(t_start_val, min(timeline.duration, item.get("timestamp_end", t_start_val)))

                    evt = Event(
                        description=item.get("description", ""),
                        actors=item.get("actors", []),
                        actions=item.get("actions", []),
                        objects=item.get("objects", []),
                        timestamp_start=t_start_val,
                        timestamp_end=t_end_val,
                        confidence=item.get("confidence", 0.8),
                        evidence_sources=item.get("evidence_sources", []),
                        salience=item.get("salience", 0.5),
                        source_observation_ids=item.get("source_observation_ids", []),
                        contested=item.get("contested", False),
                        alternatives=item.get("alternatives", [])
                    )
                    events.append(evt)
                    event_id_map[evt_id] = evt
                except Exception as e:
                    logger.warning(f"Error parsing event item {item}: {e}")

            # 4. Extract Relationships & build Semantic Graph
            nodes: List[GraphNode] = []
            edges: List[GraphEdge] = []
            
            # Populate Nodes
            for eid, evt in event_id_map.items():
                nodes.append(GraphNode(
                    id=eid,
                    type="event",
                    label=evt.description,
                    attributes={
                        "evidence_sources": evt.evidence_sources,
                        "actors": evt.actors,
                        "objects": evt.objects,
                        "timestamp_start": evt.timestamp_start,
                        "timestamp_end": evt.timestamp_end
                    }
                ))

            raw_rels = parsed.get("relationships", [])
            for rel in raw_rels:
                try:
                    src = rel.get("source")
                    tgt = rel.get("target")
                    rel_type = rel.get("relationship", "temporal")
                    if rel_type not in ["temporal", "causal", "spatial", "participates_in", "enables"]:
                        rel_type = "temporal"

                    if src in event_id_map and tgt in event_id_map:
                        edges.append(GraphEdge(
                            source=src,
                            target=tgt,
                            relationship=rel_type,
                            confidence=rel.get("confidence", 1.0),
                            weight=rel.get("confidence", 1.0) * 0.5,  # simple mapping
                            evidence=rel.get("evidence")
                        ))
                except Exception as e:
                    logger.warning(f"Error parsing relationship {rel}: {e}")

            graph = SemanticGraph(nodes=nodes, edges=edges)

            # 5. Extract Narrative
            narrative_text = parsed.get("narrative", "")
            if not narrative_text:
                # Fallback narrative compilation
                narrative_text = " ".join([evt.description for evt in sorted(events, key=lambda x: x.timestamp_start)])

            narrative = Narrative(
                text=narrative_text,
                key_events=[evt.description for evt in events],
                salience_scores={evt.description: evt.salience for evt in events},
                evidence_mapping={evt.description: evt.evidence_sources for evt in events}
            )

            latency = time.time() - t_start
            logger.info(
                f"SemanticContractResolver successfully resolved timeline. Latency: {latency:.2f}s | "
                f"Events: {len(events)} | Edges: {len(edges)} | Narrative word count: {len(narrative_text.split())}"
            )
            return events, graph, narrative

        except Exception as e:
            logger.error(f"SemanticContractResolver failed. Falling back to rules. Error: {e}")
            return self._rule_based_fallback(timeline)

    def _rule_based_fallback(self, timeline: Timeline) -> Tuple[List[Event], SemanticGraph, Narrative]:
        """
        Pure python rule-based fallback when VLM or parsing fails. No mock fallbacks.
        """
        from src.fusion.fusion_engine import FusionEngine
        # We can construct a basic set of events chronologically from observations
        events: List[Event] = []
        observations = timeline.observations
        
        # Simple windowing
        window = 3.0
        current_group: List[Observation] = []
        if not observations:
            return [], SemanticGraph(), Narrative(text="No observations.")
            
        group_start = observations[0].timestamp
        for o in observations:
            if o.timestamp <= group_start + window:
                current_group.append(o)
            else:
                events.append(self._create_fallback_event(current_group, len(events)+1))
                current_group = [o]
                group_start = o.timestamp
        if current_group:
            events.append(self._create_fallback_event(current_group, len(events)+1))

        # Build basic sequential graph
        nodes = []
        edges = []
        for i, evt in enumerate(events):
            nodes.append(GraphNode(
                id=f"evt_{i+1}",
                type="event",
                label=evt.description,
                attributes={"evidence_sources": evt.evidence_sources}
            ))
            if i > 0:
                edges.append(GraphEdge(
                    source=f"evt_{i}",
                    target=f"evt_{i+1}",
                    relationship="temporal",
                    confidence=1.0,
                    weight=0.5
                ))

        graph = SemanticGraph(nodes=nodes, edges=edges)
        narrative_text = " ".join([evt.description for evt in events])
        narrative = Narrative(
            text=narrative_text,
            key_events=[evt.description for evt in events],
            salience_scores={evt.description: 0.5 for evt in events},
            evidence_mapping={evt.description: evt.evidence_sources for evt in events}
        )
        return events, graph, narrative

    def _create_fallback_event(self, group: List[Observation], idx: int) -> Event:
        descriptions = [o.content for o in group]
        sources = list(set([o.source for o in group]))
        t_start = min(o.timestamp for o in group)
        t_end = max(o.timestamp for o in group)
        avg_conf = sum(o.confidence for o in group) / len(group) if group else 0.5
        obs_ids = [o.id for o in group if o.id is not None]

        return Event(
            description="; ".join(descriptions),
            timestamp_start=t_start,
            timestamp_end=t_end,
            confidence=avg_conf,
            evidence_sources=sources,
            salience=0.5,
            source_observation_ids=obs_ids
        )
