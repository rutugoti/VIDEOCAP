import logging
from typing import List, Dict, Any, Set

from src.shared.models import Event, SemanticGraph, GraphNode, GraphEdge

logger = logging.getLogger(__name__)


class SemanticGraphBuilder:
    """Module responsible for constructing a semantic entity-relationship graph from fused events."""

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
            "total_events": len(events)
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
                    "salience": event.salience
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

        # 3. Causal Relationship Edges (Heuristic-based)
        # We search for common causal relationships based on descriptions, actors, or keywords.
        # Example: dog barking (ev1) -> person gets scared (ev2).
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
                edges.append(
                    GraphEdge(
                        source=source_id,
                        target=target_id,
                        relationship="causal"
                    )
                )
                logger.info(f"Detected heuristic causal relation from {source_id} to {target_id}.")

        logger.info(
            f"SemanticGraph built successfully. Nodes: {len(nodes)} (Events: {len(events)}, "
            f"Entities: {len(seen_entities)}), Edges: {len(edges)}"
        )

        return SemanticGraph(nodes=nodes, edges=edges, metadata=metadata)
