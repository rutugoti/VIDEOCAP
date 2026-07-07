import pytest
from src.shared.models import SemanticGraph, GraphNode, GraphEdge, Narrative
from src.shared.providers import MockLLMProvider, ProviderError, LLMProvider
from src.narrative.narrative_builder import NarrativeBuilder


class BrokenLLMProvider(LLMProvider):
    def generate(self, prompt, system_prompt, config):
        raise ProviderError("Network timeout")


class GarbageLLMProvider(LLMProvider):
    def generate(self, prompt, system_prompt, config):
        return "Non-JSON response text."


def test_compression_and_ranking():
    # Construct a graph with 4 events of different salience
    # We want max_events = 2
    nodes = [
        GraphNode(id="ev1", type="event", label="enters room", attributes={"timestamp_start": 0.0, "salience": 0.3}),
        GraphNode(id="ev2", type="event", label="starts playing music", attributes={"timestamp_start": 2.0, "salience": 0.9}),
        GraphNode(id="ev3", type="event", label="spills water", attributes={"timestamp_start": 5.0, "salience": 0.85}),
        GraphNode(id="ev4", type="event", label="exits room", attributes={"timestamp_start": 8.0, "salience": 0.2}),
    ]

    graph = SemanticGraph(nodes=nodes, edges=[])

    # We use a broken provider so we trigger the rule-based fallback and inspect the exact chosen events!
    broken_llm = BrokenLLMProvider()
    builder = NarrativeBuilder(llm_provider=broken_llm, max_events=2)
    narrative = builder.build_narrative(graph)

    # Output should include top 2 events: ev2 (0.9) and ev3 (0.85)
    # Sorted chronologically: ev2 (2.0s) -> ev3 (5.0s)
    assert len(narrative.key_events) == 2
    assert narrative.key_events[0] == "starts playing music"
    assert narrative.key_events[1] == "spills water"
    assert narrative.salience_scores["starts playing music"] == 0.9
    assert narrative.salience_scores["spills water"] == 0.85
    assert narrative.text == "starts playing music. spills water."


def test_temporal_order():
    # 3 events. Salience ranks them: ev3 (0.9), ev1 (0.8), ev2 (0.7).
    # Chronological: ev1 (0.0s) -> ev2 (3.0s) -> ev3 (6.0s).
    # Since max_events = 3, all are kept. They must be chronologically ordered.
    nodes = [
        GraphNode(id="ev1", type="event", label="event A", attributes={"timestamp_start": 0.0, "salience": 0.8}),
        GraphNode(id="ev2", type="event", label="event B", attributes={"timestamp_start": 3.0, "salience": 0.7}),
        GraphNode(id="ev3", type="event", label="event C", attributes={"timestamp_start": 6.0, "salience": 0.9}),
    ]
    graph = SemanticGraph(nodes=nodes, edges=[])

    broken_llm = BrokenLLMProvider()
    builder = NarrativeBuilder(llm_provider=broken_llm, max_events=3)
    narrative = builder.build_narrative(graph)

    assert narrative.key_events == ["event A", "event B", "event C"]


def test_valid_llm_generation():
    nodes = [
        GraphNode(id="ev1", type="event", label="enters kitchen", attributes={"timestamp_start": 0.0, "salience": 0.5}),
    ]
    graph = SemanticGraph(nodes=nodes, edges=[])

    mock_llm = MockLLMProvider()
    builder = NarrativeBuilder(llm_provider=mock_llm)
    narrative = builder.build_narrative(graph)

    # MockLLMProvider returns canned JSON with 3 events:
    # enters kitchen, starts coffee maker, exits kitchen
    assert "starts the coffee maker" in narrative.text
    assert len(narrative.key_events) == 3
    assert narrative.salience_scores["starts coffee maker"] == 0.95
    assert narrative.evidence_mapping["starts coffee maker"] == ["visual", "audio"]


def test_garbage_llm_fallback():
    nodes = [
        GraphNode(id="ev1", type="event", label="starts engine", attributes={"timestamp_start": 1.0, "salience": 0.8, "evidence_sources": ["visual"]}),
    ]
    graph = SemanticGraph(nodes=nodes, edges=[])

    garbage_llm = GarbageLLMProvider()
    builder = NarrativeBuilder(llm_provider=garbage_llm)
    narrative = builder.build_narrative(graph)

    assert narrative.text == "starts engine."
    assert narrative.key_events == ["starts engine"]
    assert narrative.salience_scores["starts engine"] == 0.8
    assert narrative.evidence_mapping["starts engine"] == ["visual"]


def test_evidence_sources_propagation():
    # Construct nodes with custom evidence sources
    nodes = [
        GraphNode(
            id="ev1",
            type="event",
            label="detect siren",
            attributes={
                "timestamp_start": 2.0,
                "salience": 0.9,
                "evidence_sources": ["audio", "text"]
            }
        ),
    ]
    graph = SemanticGraph(nodes=nodes, edges=[])

    broken_llm = BrokenLLMProvider()
    builder = NarrativeBuilder(llm_provider=broken_llm)
    narrative = builder.build_narrative(graph)

    # In rule-based fallback, evidence_mapping should reflect attributes["evidence_sources"]
    assert narrative.evidence_mapping["detect siren"] == ["audio", "text"]


class CapturingLLMProvider(LLMProvider):
    def __init__(self):
        self.captured_prompt = None
        self.captured_system_prompt = None

    def generate(self, prompt, system_prompt, config):
        self.captured_prompt = prompt
        self.captured_system_prompt = system_prompt
        return """{
            "text": "A chronological summary detailing graph events.",
            "key_events": ["event A", "event B"],
            "salience_scores": {"event A": 0.8, "event B": 0.9},
            "evidence_mapping": {"event A": ["visual"], "event B": ["audio"]}
        }"""


def test_consume_edges_prompt_construction():
    nodes = [
        GraphNode(id="ev1", type="event", label="event A", attributes={"timestamp_start": 0.0, "salience": 0.9}),
        GraphNode(id="ev2", type="event", label="event B", attributes={"timestamp_start": 2.0, "salience": 0.8}),
    ]
    edges = [
        GraphEdge(source="dog", target="ev1", relationship="participates_in"),
        GraphEdge(source="ev1", target="ev2", relationship="temporal"),
        GraphEdge(source="ev1", target="ev2", relationship="causal"),
    ]
    graph = SemanticGraph(nodes=nodes, edges=edges)

    capturing_llm = CapturingLLMProvider()
    builder = NarrativeBuilder(llm_provider=capturing_llm, consume_edges=True)
    builder.build_narrative(graph)

    # Assert that edge consumption is reflected in the prompt parameters
    assert capturing_llm.captured_prompt is not None
    assert "Entity 'dog' participates in:" in capturing_llm.captured_prompt
    assert "Event ev1 happened before Event ev2" in capturing_llm.captured_prompt
    assert "Event ev1 causally triggered Event ev2" in capturing_llm.captured_prompt
    assert "respect these entity participation sequences" in capturing_llm.captured_system_prompt


