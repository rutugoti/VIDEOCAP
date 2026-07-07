import pytest
from src.shared.models import Event
from src.graph.semantic_graph_builder import SemanticGraphBuilder


def test_entity_extraction():
    # 2 events with overlapping and distinct actors/objects
    ev1 = Event(
        description="Person enters the kitchen",
        actors=["Person"],
        actions=["enters"],
        objects=["Kitchen"],
        timestamp_start=0.0,
        timestamp_end=2.0,
        confidence=0.9,
        evidence_sources=["visual"],
        salience=0.8
    )
    ev2 = Event(
        description="Person opens the refrigerator to check milk",
        actors=["Person"],
        actions=["opens", "check"],
        objects=["Refrigerator", "Milk"],
        timestamp_start=2.0,
        timestamp_end=4.0,
        confidence=0.85,
        evidence_sources=["visual"],
        salience=0.7
    )

    builder = SemanticGraphBuilder()
    graph = builder.build_graph([ev1, ev2])

    # Check node types
    entities = [n for n in graph.nodes if n.type == "entity"]
    events = [n for n in graph.nodes if n.type == "event"]

    assert len(events) == 2
    assert events[0].id == "ev1"
    assert events[1].id == "ev2"

    # Entities should be normalized: "person", "kitchen", "refrigerator", "milk"
    entity_labels = [e.label for e in entities]
    assert len(entities) == 4
    assert set(entity_labels) == {"person", "kitchen", "refrigerator", "milk"}

    # participates_in edges
    part_edges = [edge for edge in graph.edges if edge.relationship == "participates_in"]
    # ev1: person participates, kitchen participates
    # ev2: person participates, refrigerator participates, milk participates
    assert len(part_edges) == 5
    
    # Check that person is linked to both events
    person_links = [e.target for e in part_edges if e.source == "person"]
    assert set(person_links) == {"ev1", "ev2"}


def test_temporal_edges():
    ev1 = Event(description="A", actors=[], actions=[], objects=[], timestamp_start=0.0, timestamp_end=1.0, confidence=0.8, evidence_sources=[], salience=0.5)
    ev2 = Event(description="B", actors=[], actions=[], objects=[], timestamp_start=1.0, timestamp_end=2.0, confidence=0.8, evidence_sources=[], salience=0.5)
    ev3 = Event(description="C", actors=[], actions=[], objects=[], timestamp_start=2.0, timestamp_end=3.0, confidence=0.8, evidence_sources=[], salience=0.5)

    builder = SemanticGraphBuilder()
    graph = builder.build_graph([ev1, ev2, ev3])

    temporal_edges = [edge for edge in graph.edges if edge.relationship == "temporal"]
    assert len(temporal_edges) == 2
    assert temporal_edges[0].source == "ev1" and temporal_edges[0].target == "ev2"
    assert temporal_edges[1].source == "ev2" and temporal_edges[1].target == "ev3"


def test_single_event():
    ev1 = Event(
        description="Dog barks loud",
        actors=["Dog"],
        actions=["barks"],
        objects=[],
        timestamp_start=2.0,
        timestamp_end=3.0,
        confidence=0.9,
        evidence_sources=["audio"],
        salience=0.6
    )

    builder = SemanticGraphBuilder()
    graph = builder.build_graph([ev1])

    assert len(graph.nodes) == 2  # 1 event node, 1 entity node (dog)
    assert len(graph.edges) == 1  # participates_in
    assert graph.edges[0].source == "dog"
    assert graph.edges[0].target == "ev1"
    assert graph.edges[0].relationship == "participates_in"


def test_causal_heuristics():
    # ev1: Empty milk -> ev2: orders grocery on phone
    ev1 = Event(
        description="The refrigerator milk carton is empty",
        actors=["Person"],
        actions=[],
        objects=["milk carton", "refrigerator"],
        timestamp_start=1.0,
        timestamp_end=2.0,
        confidence=0.9,
        evidence_sources=["visual"],
        salience=0.8
    )
    ev2 = Event(
        description="Person orders groceries on their phone",
        actors=["Person"],
        actions=["orders"],
        objects=["phone"],
        timestamp_start=2.0,
        timestamp_end=3.0,
        confidence=0.85,
        evidence_sources=["visual"],
        salience=0.7
    )

    builder = SemanticGraphBuilder()
    graph = builder.build_graph([ev1, ev2])

    causal_edges = [edge for edge in graph.edges if edge.relationship == "causal"]
    assert len(causal_edges) == 1
    assert causal_edges[0].source == "ev1"
    assert causal_edges[0].target == "ev2"


def test_edge_weight_ranking():
    # ev1: person enters kitchen (salience: 0.5)
    ev1 = Event(
        description="person enters kitchen because of hunger",
        actors=["person"],
        actions=["enters"],
        objects=["kitchen"],
        timestamp_start=0.0,
        timestamp_end=1.0,
        confidence=1.0,
        evidence_sources=["visual"],
        salience=0.5
    )
    # ev2: coffee maker triggers steam (salience: 0.9)
    ev2 = Event(
        description="coffee maker triggers steam",
        actors=["coffee"],
        actions=["triggers"],
        objects=["steam"],
        timestamp_start=1.0,
        timestamp_end=2.0,
        confidence=1.0,
        evidence_sources=["visual"],
        salience=0.9
    )

    builder = SemanticGraphBuilder()
    graph = builder.build_graph([ev1, ev2])

    # Find the causal edge and the participates_in edge
    causal_edge = next(e for e in graph.edges if e.relationship == "causal")
    part_edge = next(e for e in graph.edges if e.relationship == "participates_in" and e.source == "person")

    # Assert that causal edge weight outranks participates_in edge weight
    # Causal type prior is 1.0; participates_in is 0.3.
    # Node saliences: ev1=0.5, ev2=0.9, person=0.5, kitchen=0.5, coffee=0.5, steam=0.5 (defaults)
    # Causal: source="ev1", target="ev2", s=(0.5+0.9)/2 = 0.7. Weight = 1.0 * 1.0 * 0.7 = 0.7
    # Part_edge: source="person", target="ev1", s=(0.5+0.5)/2 = 0.5. Weight = 0.3 * 1.0 * 0.5 = 0.15
    assert causal_edge.weight > part_edge.weight
    assert causal_edge.weight == 0.7
    assert part_edge.weight == 0.15
