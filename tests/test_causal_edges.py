"""
Tests for P3.1 — Bounded causal-edge labeling via Gemma classifier.

Covers:
- Candidate generation respects the causal window bound
- Far-apart event pairs are excluded from candidates
- Mock classifier produces correct causal/enables edges
- Temporal/unrelated verdicts do NOT produce edges
- SemanticGraphBuilder integrates gemma path when causal_edges=True
- SemanticGraphBuilder falls back to keyword path when causal_edges=False
"""

import pytest
from src.shared.models import Event, GraphEdge
from src.shared.providers import MockLLMProvider, LLMConfig
from src.graph.causal_classifier import (
    generate_candidates,
    classify_candidates,
    verdicts_to_edges,
    CausalCandidate,
    CausalVerdict,
)
from src.graph.semantic_graph_builder import SemanticGraphBuilder


# ── Helper fixtures ──────────────────────────────────────────────────

def _make_event(desc: str, t_start: float, t_end: float, salience: float = 0.7) -> Event:
    return Event(
        description=desc,
        actors=["person"],
        actions=[],
        objects=[],
        timestamp_start=t_start,
        timestamp_end=t_end,
        confidence=0.9,
        evidence_sources=["visual"],
        salience=salience
    )


def _llm_config() -> LLMConfig:
    return LLMConfig(
        provider="mock",
        model="mock-model",
        max_tokens=256,
        temperature=0.1,
    )


# ── Candidate generation tests ──────────────────────────────────────

def test_candidate_gen_within_window():
    """Events within the causal window should produce candidates."""
    events = [
        _make_event("A happens", 0.0, 1.0),
        _make_event("B happens", 1.5, 2.5),  # gap = 1.5 - 1.0 = 0.5s
        _make_event("C happens", 3.0, 4.0),  # gap = 3.0 - 2.5 = 0.5s
    ]
    candidates = generate_candidates(events, causal_window_seconds=2.0)
    # With window=2.0: (A,B) gap=0.5 ✓, (A,C) gap=3.0-1.0=2.0 ✓, (B,C) gap=0.5 ✓
    assert len(candidates) == 3
    pairs = [(c.source_idx, c.target_idx) for c in candidates]
    assert (0, 1) in pairs
    assert (0, 2) in pairs
    assert (1, 2) in pairs


def test_candidate_gen_excludes_far_apart():
    """Events beyond the causal window must NOT produce candidates."""
    events = [
        _make_event("A happens", 0.0, 1.0),
        _make_event("B happens", 10.0, 11.0),  # gap = 10.0 - 1.0 = 9.0s
    ]
    candidates = generate_candidates(events, causal_window_seconds=4.0)
    assert len(candidates) == 0


def test_candidate_gen_overlapping_intervals():
    """Overlapping intervals (gap < 0) should produce candidates."""
    events = [
        _make_event("A happens", 0.0, 3.0),
        _make_event("B happens", 2.0, 4.0),  # gap = 2.0 - 3.0 = -1.0 (overlap)
    ]
    candidates = generate_candidates(events, causal_window_seconds=1.0)
    assert len(candidates) == 1
    assert candidates[0].source_idx == 0
    assert candidates[0].target_idx == 1


def test_candidate_gen_empty_events():
    """Empty event list produces no candidates."""
    candidates = generate_candidates([], causal_window_seconds=4.0)
    assert len(candidates) == 0


def test_candidate_gen_single_event():
    """Single event produces no candidates."""
    events = [_make_event("A happens", 0.0, 1.0)]
    candidates = generate_candidates(events, causal_window_seconds=4.0)
    assert len(candidates) == 0


# ── Classification tests ────────────────────────────────────────────

def test_classify_causal_keyword_trigger():
    """MockLLMProvider should return 'causal' when 'trigger' is in the event description."""
    events = [
        _make_event("Dog bark triggers a response", 0.0, 1.0),
        _make_event("Person gets scared", 1.0, 2.0),
    ]
    candidates = generate_candidates(events, causal_window_seconds=4.0)
    verdicts = classify_candidates(candidates, MockLLMProvider(), _llm_config())
    
    assert len(verdicts) == 1
    assert verdicts[0].relationship == "causal"
    assert verdicts[0].confidence > 0.0


def test_classify_temporal_fallback():
    """Events with no causal keywords should get 'temporal' classification."""
    events = [
        _make_event("Person walks in the park", 0.0, 1.0),
        _make_event("Bird flies overhead", 1.0, 2.0),
    ]
    candidates = generate_candidates(events, causal_window_seconds=4.0)
    verdicts = classify_candidates(candidates, MockLLMProvider(), _llm_config())
    
    assert len(verdicts) == 1
    assert verdicts[0].relationship == "temporal"


# ── Verdict-to-edge conversion tests ────────────────────────────────

def test_verdicts_to_edges_causal_only():
    """Only causal and enables verdicts should produce edges."""
    verdicts = [
        CausalVerdict(0, 1, "causal", 0.85, "A caused B"),
        CausalVerdict(1, 2, "temporal", 0.60, "Only sequential"),
        CausalVerdict(2, 3, "enables", 0.75, "A enabled B"),
        CausalVerdict(3, 4, "unrelated", 0.50, "No connection"),
    ]
    edges = verdicts_to_edges(verdicts)
    assert len(edges) == 2
    
    causal_edge = edges[0]
    assert causal_edge.source == "ev1"
    assert causal_edge.target == "ev2"
    assert causal_edge.relationship == "causal"
    assert causal_edge.confidence == 0.85
    assert causal_edge.evidence == "A caused B"
    
    enables_edge = edges[1]
    assert enables_edge.source == "ev3"
    assert enables_edge.target == "ev4"
    assert enables_edge.relationship == "enables"


# ── Integration: SemanticGraphBuilder with Gemma causal ─────────────

def test_graph_builder_gemma_mode():
    """When causal_edges=True, the graph builder uses Gemma classification."""
    events = [
        _make_event("Coffee maker triggers steam alarm", 0.0, 1.0),
        _make_event("Person reacts to the alarm", 1.0, 2.0),
        _make_event("Person walks away slowly", 5.0, 6.0),
    ]

    builder = SemanticGraphBuilder(
        llm_provider=MockLLMProvider(),
        llm_config=_llm_config(),
        causal_edges=True,
        causal_window_seconds=3.0,
    )
    graph = builder.build_graph(events)

    assert graph.metadata["causal_mode"] == "gemma"
    
    # (ev1, ev2) gap=0.0 ✓ within window, "trigger" in desc → causal
    # (ev1, ev3) gap=5.0-1.0=4.0 > 3.0 window → excluded
    # (ev2, ev3) gap=5.0-2.0=3.0 ≤ 3.0 → included, no trigger keywords → temporal
    causal_edges = [e for e in graph.edges if e.relationship == "causal"]
    enables_edges = [e for e in graph.edges if e.relationship == "enables"]
    temporal_edges = [e for e in graph.edges if e.relationship == "temporal"]
    
    # At least one causal edge from ev1->ev2
    assert len(causal_edges) >= 1
    assert causal_edges[0].source == "ev1"
    assert causal_edges[0].target == "ev2"
    
    # Temporal edges still present (consecutive linking)
    assert len(temporal_edges) == 2  # ev1->ev2, ev2->ev3


def test_graph_builder_keyword_mode():
    """When causal_edges=False, the graph builder uses keyword heuristics (legacy)."""
    events = [
        _make_event("The refrigerator milk carton is empty", 0.0, 1.0),
        _make_event("Person orders groceries on their phone", 1.0, 2.0),
    ]

    builder = SemanticGraphBuilder(
        causal_edges=False,
    )
    graph = builder.build_graph(events)

    assert graph.metadata["causal_mode"] == "keyword_heuristic"
    
    # The keyword heuristic detects "empty" -> "order" as causal
    causal_edges = [e for e in graph.edges if e.relationship == "causal"]
    assert len(causal_edges) == 1
    assert causal_edges[0].source == "ev1"
    assert causal_edges[0].target == "ev2"


def test_graph_builder_gemma_no_causal_for_unrelated():
    """Gemma mode should NOT add causal edges for unrelated events."""
    events = [
        _make_event("Person walks in park", 0.0, 1.0),
        _make_event("Bird flies overhead", 1.5, 2.5),
    ]

    builder = SemanticGraphBuilder(
        llm_provider=MockLLMProvider(),
        llm_config=_llm_config(),
        causal_edges=True,
        causal_window_seconds=4.0,
    )
    graph = builder.build_graph(events)

    # No trigger/cause/enable keywords → temporal verdict → no causal edges
    causal_edges = [e for e in graph.edges if e.relationship in ("causal", "enables")]
    assert len(causal_edges) == 0


def test_graph_builder_causal_edge_has_evidence():
    """Gemma-classified causal edges should carry an evidence rationale string."""
    events = [
        _make_event("Dog bark triggers a loud response", 0.0, 1.0),
        _make_event("Person gets startled", 1.0, 2.0),
    ]

    builder = SemanticGraphBuilder(
        llm_provider=MockLLMProvider(),
        llm_config=_llm_config(),
        causal_edges=True,
        causal_window_seconds=4.0,
    )
    graph = builder.build_graph(events)

    causal_edges = [e for e in graph.edges if e.relationship == "causal"]
    assert len(causal_edges) >= 1
    assert causal_edges[0].evidence is not None
    assert len(causal_edges[0].evidence) > 0
