import pytest
from src.shared.models import Timeline, Observation
from src.fusion.semantic_contract import SemanticContractResolver
from src.shared.providers import LLMProvider, LLMConfig

class SimpleMockLLMProvider(LLMProvider):
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.last_prompt = None

    def generate(self, prompt: str, system_prompt: str = "", config: LLMConfig = None) -> str:
        self.last_prompt = prompt
        return self.response_text

def test_semantic_contract_parsing():
    mock_json = """
    {
      "events": [
        {
          "id": "evt_1",
          "description": "A man starts preparing coffee in the kitchen.",
          "actors": ["man"],
          "actions": ["preparing coffee"],
          "objects": ["coffee machine", "mug"],
          "timestamp_start": 0.0,
          "timestamp_end": 5.2,
          "confidence": 0.95,
          "evidence_sources": ["visual"],
          "salience": 0.9,
          "source_observation_ids": [101]
        },
        {
          "id": "evt_2",
          "description": "The man pours hot coffee into a mug.",
          "actors": ["man"],
          "actions": ["pouring coffee"],
          "objects": ["mug", "coffee pot"],
          "timestamp_start": 5.5,
          "timestamp_end": 10.0,
          "confidence": 0.85,
          "evidence_sources": ["visual", "audio"],
          "salience": 0.8,
          "source_observation_ids": [102, 103]
        }
      ],
      "relationships": [
        {
          "source": "evt_1",
          "target": "evt_2",
          "relationship": "temporal",
          "confidence": 0.9,
          "evidence": "Coffee preparation precedes pouring."
        }
      ],
      "narrative": "A man is in the kitchen preparing coffee, and then pours the freshly brewed coffee into a mug."
    }
    """
    
    provider = SimpleMockLLMProvider(mock_json)
    resolver = SemanticContractResolver(llm_provider=provider)
    
    obs = [
        Observation(id=101, content="A man is standing near a coffee maker", timestamp=1.0, confidence=0.9, source="visual", observation_type="scene"),
        Observation(id=102, content="Water boiling sound", timestamp=6.0, confidence=0.8, source="audio", observation_type="sound"),
        Observation(id=103, content="Pouring liquid into a cup", timestamp=7.0, confidence=0.9, source="visual", observation_type="action")
    ]
    timeline = Timeline(observations=obs, duration=12.0)
    
    events, graph, narrative = resolver.resolve(timeline)
    
    assert len(events) == 2
    assert events[0].description == "A man starts preparing coffee in the kitchen."
    assert events[0].source_observation_ids == [101]
    assert events[1].source_observation_ids == [102, 103]
    
    assert len(graph.nodes) == 2
    assert len(graph.edges) == 1
    assert graph.edges[0].source == "evt_1"
    assert graph.edges[0].target == "evt_2"
    assert graph.edges[0].relationship == "temporal"
    
    assert narrative.text == "A man is in the kitchen preparing coffee, and then pours the freshly brewed coffee into a mug."

def test_semantic_contract_fallback():
    provider = SimpleMockLLMProvider("invalid json response")
    resolver = SemanticContractResolver(llm_provider=provider)
    
    obs = [
        Observation(id=101, content="Visual detection of coffee cup", timestamp=2.0, confidence=0.9, source="visual", observation_type="object")
    ]
    timeline = Timeline(observations=obs, duration=5.0)
    
    events, graph, narrative = resolver.resolve(timeline)
    
    # Fallback should succeed without throwing
    assert len(events) == 1
    assert "coffee cup" in events[0].description
    assert events[0].source_observation_ids == [101]
