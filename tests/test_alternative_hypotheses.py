import pytest

from src.shared.models import (
    Observation, Confidence, Timeline, SemanticGraph, GraphNode, Event,
)
from src.shared.providers import MockLLMProvider, LLMProvider
from src.fusion.timeline_builder import TimelineBuilder
from src.fusion.fusion_engine import FusionEngine
from src.graph.semantic_graph_builder import SemanticGraphBuilder
from src.narrative.narrative_builder import NarrativeBuilder


def _measured(value):
    return Confidence(value=value, source="measured")


def test_fusion_marks_contested_when_gap_small_and_measured():
    # Visual vs audio contradiction (low word overlap), both measured, gap < 0.15
    obs = [
        Observation(content="a dog runs across the yard", timestamp=1.0, confidence=0.80,
                    confidence_meta=_measured(0.80), source="visual", observation_type="action"),
        Observation(content="a cat meows on the sofa", timestamp=1.2, confidence=0.72,
                    confidence_meta=_measured(0.72), source="audio", observation_type="speech"),
    ]
    tl = TimelineBuilder().build_timeline(obs, 5.0)
    events = FusionEngine(llm_provider=MockLLMProvider(), temporal_window_seconds=3.0).fuse_timeline(tl)

    assert any(e.contested for e in events)
    contested = next(e for e in events if e.contested)
    assert any("dog" in a for a in contested.alternatives)
    assert any("cat" in a for a in contested.alternatives)


def test_fusion_does_not_hedge_when_confidence_is_default():
    # Same contradiction but audio confidence is a `default` placeholder → no hedge
    obs = [
        Observation(content="a dog runs across the yard", timestamp=1.0, confidence=0.80,
                    confidence_meta=_measured(0.80), source="visual", observation_type="action"),
        Observation(content="a cat meows on the sofa", timestamp=1.2, confidence=0.75,
                    confidence_meta=Confidence(value=0.75, source="default"), source="audio", observation_type="speech"),
    ]
    tl = TimelineBuilder().build_timeline(obs, 5.0)
    events = FusionEngine(llm_provider=MockLLMProvider(), temporal_window_seconds=3.0).fuse_timeline(tl)

    assert not any(e.contested for e in events)


def test_fusion_does_not_hedge_when_gap_large():
    # Both measured but confidence gap >= 0.15 → resolved by confidence, no hedge
    obs = [
        Observation(content="a dog runs across the yard", timestamp=1.0, confidence=0.90,
                    confidence_meta=_measured(0.90), source="visual", observation_type="action"),
        Observation(content="a cat meows on the sofa", timestamp=1.2, confidence=0.50,
                    confidence_meta=_measured(0.50), source="audio", observation_type="speech"),
    ]
    tl = TimelineBuilder().build_timeline(obs, 5.0)
    events = FusionEngine(llm_provider=MockLLMProvider(), temporal_window_seconds=3.0).fuse_timeline(tl)

    assert not any(e.contested for e in events)


def test_graph_propagates_contested_to_node_attributes():
    ev = Event(
        description="ambiguous animal action", actors=["dog"], actions=["runs"], objects=[],
        timestamp_start=1.0, timestamp_end=2.0, confidence=0.8, evidence_sources=["visual", "audio"],
        salience=0.9, contested=True, alternatives=["a dog runs", "a cat meows"],
    )
    graph = SemanticGraphBuilder().build_graph([ev])
    node = next(n for n in graph.nodes if n.type == "event")
    assert node.attributes["contested"] is True
    assert node.attributes["alternatives"] == ["a dog runs", "a cat meows"]


class CapturingLLMProvider(LLMProvider):
    def __init__(self):
        self.captured_prompt = None
        self.captured_system_prompt = None

    def generate(self, prompt, system_prompt, config):
        self.captured_prompt = prompt
        self.captured_system_prompt = system_prompt
        return (
            '{"text": "An animal is present.", "key_events": ["ambiguous action"], '
            '"salience_scores": {"ambiguous action": 0.9}, '
            '"evidence_mapping": {"ambiguous action": ["visual", "audio"]}}'
        )


def test_narrative_surfaces_uncertainty_in_prompt():
    nodes = [
        GraphNode(
            id="ev1", type="event", label="ambiguous action",
            attributes={
                "timestamp_start": 1.0, "salience": 0.9,
                "evidence_sources": ["visual", "audio"],
                "contested": True, "alternatives": ["a dog runs", "a cat meows"],
            },
        ),
    ]
    graph = SemanticGraph(nodes=nodes, edges=[])
    cap = CapturingLLMProvider()
    NarrativeBuilder(llm_provider=cap, consume_edges=True).build_narrative(graph)

    assert "UNCERTAIN" in cap.captured_prompt
    assert "a dog runs" in cap.captured_prompt and "a cat meows" in cap.captured_prompt
    assert "hedging language" in cap.captured_system_prompt


def test_narrative_no_uncertainty_rule_when_not_contested():
    nodes = [
        GraphNode(
            id="ev1", type="event", label="clear action",
            attributes={"timestamp_start": 1.0, "salience": 0.9, "evidence_sources": ["visual"]},
        ),
    ]
    graph = SemanticGraph(nodes=nodes, edges=[])
    cap = CapturingLLMProvider()
    NarrativeBuilder(llm_provider=cap, consume_edges=True).build_narrative(graph)

    assert "UNCERTAIN" not in cap.captured_prompt
    assert "hedging language" not in cap.captured_system_prompt
