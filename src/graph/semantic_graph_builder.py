import logging
from typing import List, Dict, Any, Optional, Set

from src.shared.models import Event, SemanticGraph, GraphNode, GraphEdge
from src.shared.providers import LLMProvider, LLMConfig
from src.graph.causal_classifier import (
    generate_candidates,
    classify_candidates,
    verdicts_to_edges,
)

logger = logging.getLogger(__name__)


class SemanticGraphBuilder:
    """Module responsible for constructing a semantic entity-relationship graph from fused events."""

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        llm_config: Optional[LLMConfig] = None,
        causal_edges: bool = False,
        causal_window_seconds: float = 4.0,
    ):
        """
        Args:
            llm_provider: LLM provider for Gemma-based causal classification (P3.1).
            llm_config: LLM config for causal classification calls.
            causal_edges: If True, use Gemma classifier for causal edges.
                          If False, use legacy keyword heuristics.
            causal_window_seconds: Maximum gap between events to consider for causal candidates.
        """
        self.llm_provider = llm_provider
        self.llm_config = llm_config
        self.causal_edges = causal_edges
        self.causal_window_seconds = causal_window_seconds

    def build_graph(self, events: List[Event]) -> SemanticGraph:
        """
        Build an entity-relationship graph representing actors, events, and their links.

        Args:
            events: List of fused Event objects, chronologically sorted.

        Returns:
            A SemanticGraph object.
        """
        if not events:
            logger.info("SemanticGraphBuilder: Building graph from empty events list.")
            return SemanticGraph(nodes=[], edges=[], metadata={})

        nodes: List[GraphNode] = []
        edges: List[GraphEdge] = []
        metadata: Dict[str, Any] = {
            "total_events": len(events),
            "causal_mode": "gemma" if self.causal_edges else "keyword_heuristic"
        }

        # 1. Entity and Event Node Extraction
        # Keep track of unique entities to avoid duplicate nodes
        seen_entities: Set[str] = set()

        for idx, event in enumerate(events):
            event_id = f"ev{idx + 1}"
            
            # Create Event Node
            event_node = GraphNode(
                id=event_id,
                type="event",
                label=event.description,
                attributes={
                    "timestamp_start": event.timestamp_start,
                    "timestamp_end": event.timestamp_end,
                    "confidence": event.confidence,
                    "salience": event.salience,
                    "evidence_sources": event.evidence_sources,
                    "actors": event.actors,
                    "objects": event.objects,
                    "contested": event.contested,
                    "alternatives": event.alternatives
                }
            )
            nodes.append(event_node)

            # Process participating entities (actors + objects)
            participating_entities = set(event.actors + event.objects)
            for entity in participating_entities:
                entity_clean = entity.strip().lower()
                if not entity_clean:
                    continue

                entity_id = entity_clean.replace(" ", "_")

                # If entity has not been created yet, add Entity Node
                if entity_id not in seen_entities:
                    seen_entities.add(entity_id)
                    entity_node = GraphNode(
                        id=entity_id,
                        type="entity",
                        label=entity_clean
                    )
                    nodes.append(entity_node)

                # Add relationship edge: entity participates_in event
                edges.append(
                    GraphEdge(
                        source=entity_id,
                        target=event_id,
                        relationship="participates_in"
                    )
                )

        # Link consecutive events in time: ev1 -> temporal -> ev2 -> ...
        for i in range(len(events) - 1):
            source_id = f"ev{i + 1}"
            target_id = f"ev{i + 2}"
            edges.append(
                GraphEdge(
                    source=source_id,
                    target=target_id,
                    relationship="temporal"
                )
            )

        # 3. Causal Relationship Edges — gated by config
        if self.causal_edges and self.llm_provider and self.llm_config:
            # P3.1: Gemma-based bounded causal classification
            causal_edges_list = self._classify_causal_gemma(events)
            edges.extend(causal_edges_list)
            metadata["causal_edges_count"] = len(causal_edges_list)
        else:
            # Legacy keyword heuristic path (kept for A/B ablation)
            causal_edges_list = self._classify_causal_keyword(events)
            edges.extend(causal_edges_list)
            metadata["causal_edges_count"] = len(causal_edges_list)

        # Calculate edge weights (Phase P2.1)
        node_salience = {n.id: n.attributes.get("salience", 0.5) for n in nodes}
        TYPE_PRIOR = {"causal": 1.0, "enables": 0.9, "temporal": 0.5, "participates_in": 0.3}
        for e in edges:
            s = (node_salience.get(e.source, 0.5) + node_salience.get(e.target, 0.5)) / 2.0
            e.weight = TYPE_PRIOR.get(e.relationship, 0.3) * e.confidence * s

        logger.info(
            f"SemanticGraph built successfully. Nodes: {len(nodes)} (Events: {len(events)}, "
            f"Entities: {len(seen_entities)}), Edges: {len(edges)}, "
            f"Causal mode: {metadata['causal_mode']}"
        )

        return SemanticGraph(nodes=nodes, edges=edges, metadata=metadata)

    def _classify_causal_gemma(self, events: List[Event]) -> List[GraphEdge]:
        """
        P3.1: Generate bounded candidates and classify via Gemma.
        Only causal/enables verdicts become edges.
        """
        candidates = generate_candidates(events, self.causal_window_seconds)
        if not candidates:
            return []

        verdicts = classify_candidates(candidates, self.llm_provider, self.llm_config)
        return verdicts_to_edges(verdicts)

    def _classify_causal_keyword(self, events: List[Event]) -> List[GraphEdge]:
        """
        Legacy keyword heuristic causal detection (original lines 96-131).
        Kept as the A/B ablation baseline for P3.2.
        """
        causal_edges: List[GraphEdge] = []

        for i in range(len(events) - 1):
            ev_current = events[i]
            ev_next = events[i + 1]

            desc_curr = ev_current.description.lower()
            desc_next = ev_next.description.lower()

            is_causal = False

            # Heuristic 1: Explicit causal transition keywords
            causal_triggers = ["cause", "trigger", "make", "start", "because", "force", "lead to"]
            if any(trigger in desc_curr for trigger in causal_triggers) or "because" in desc_next:
                is_causal = True

            # Heuristic 2: Specific audio-visual causal dependencies (e.g. bark -> react/laugh, fall -> cry)
            if "bark" in desc_curr and ("laugh" in desc_next or "scare" in desc_next or "look" in desc_next):
                is_causal = True
            if "fall" in desc_curr and ("cry" in desc_next or "scream" in desc_next or "hurt" in desc_next):
                is_causal = True
            if "empty" in desc_curr and ("order" in desc_next or "buy" in desc_next or "go" in desc_next):
                is_causal = True

            if is_causal:
                source_id = f"ev{i + 1}"
                target_id = f"ev{i + 2}"
                causal_edges.append(
                    GraphEdge(
                        source=source_id,
                        target=target_id,
                        relationship="causal"
                    )
                )
                logger.info(f"Detected heuristic causal relation from {source_id} to {target_id}.")

        return causal_edges
