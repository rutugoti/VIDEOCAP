import pytest
import time
from typing import List, Dict, Any

from src.shared.fireworks_providers import _parse_and_repair_json, _call_openai_with_retry
from src.shared.models import Narrative, Caption, Observation, Event
from src.shared.providers import LLMProvider, LLMConfig, ProviderError
from src.generation.style_generator import StyleGenerator
from src.validation.semantic_validator import SemanticValidator


# 1. Test JSON Repair
def test_json_repair():
    # Test stripping markdown wrappers
    markdown_json = "```json\n{\"key\": \"value\"}\n```"
    assert _parse_and_repair_json(markdown_json) == {"key": "value"}

    # Test extracting boundaries
    nested_json = "Here is some prefix text { \"status\": \"success\" } and some suffix text"
    assert _parse_and_repair_json(nested_json) == {"status": "success"}

    # Test Python values replacement
    py_values = "{'item': None, 'active': True, 'failed': False}"
    # Replace single quotes for json compatibility
    py_values_fixed = py_values.replace("'", "\"")
    repaired = _parse_and_repair_json(py_values_fixed)
    assert repaired == {"item": None, "active": True, "failed": False}


# 2. Test Retry Logic
def test_retry_logic():
    call_count = 0

    def mock_api_call_success():
        nonlocal call_count
        call_count += 1
        return "success"

    # Direct success
    assert _call_openai_with_retry(mock_api_call_success) == "success"
    assert call_count == 1

    # Success after transient failures
    import openai
    import httpx
    fail_count = 0
    dummy_req = httpx.Request("POST", "https://api.fireworks.ai")
    dummy_res = httpx.Response(status_code=429, request=dummy_req)

    def mock_api_call_transient():
        nonlocal fail_count
        fail_count += 1
        if fail_count < 3:
            raise openai.RateLimitError("Rate limit exceeded", response=dummy_res, body=None)
        return "recovered"

    result = _call_openai_with_retry(mock_api_call_transient, max_retries=3, base_delay=0.01)
    assert result == "recovered"
    assert fail_count == 3

    # Non-transient failure (should raise immediately without retry)
    def mock_api_call_fatal():
        raise openai.AuthenticationError("Invalid Auth", response=dummy_res, body=None)

    with pytest.raises(ProviderError):
        _call_openai_with_retry(mock_api_call_fatal)


# 3. Test Style Generator self-critique loop and sentence boundary word trimming
class MockLLMForGenerator(LLMProvider):
    def __init__(self):
        self.call_log = []

    def generate(self, prompt: str, system_prompt: str, config: LLMConfig) -> str:
        self.call_log.append(prompt)
        # First call is drafting, second is critique, third is rewrite.
        # Let's return responses appropriate to each phase.
        if "Critique this draft caption" in prompt:
            # Critique phase
            return '{"missing_facts": [], "hallucinations": [], "style_drift": [], "grammar_issues": [], "word_count_issue": true}'
        elif "Rewrite the caption to fix all critique" in prompt:
            # Rewrite phase - make it too long to trigger sentence boundary trimming test
            return "This is a very long caption designed to exceed the maximum word count constraint so that the sentence boundary trimmer is triggered. However, the second sentence is shorter. This is the third sentence."
        else:
            # Draft phase
            return "A simple draft caption for the video."


def test_generator_self_critique_and_trimming():
    llm = MockLLMForGenerator()
    config = LLMConfig(provider="mock", model="mock-model", max_tokens=100)
    generator = StyleGenerator(llm_provider=llm, llm_config=config, min_caption_words=5, max_caption_words=20)
    
    narrative = Narrative(text="A user opens a laptop and starts coding.", key_events=[], salience_scores={})
    captions = generator.generate_captions(narrative)
    
    # We should have all four styles
    assert len(captions) == 4
    for style, caption in captions.items():
        assert isinstance(caption, Caption)
        # Ensure it was trimmed at sentence boundaries and is under 20 words
        assert len(caption.text.split()) <= 20
        # Ensure it contains a valid sentence boundary
        assert caption.text.endswith(".")
        # Metadata must contain draft & critique
        assert "draft" in caption.metadata
        assert "critique" in caption.metadata


# 4. Test Semantic Validator Scoring Report
class MockLLMForValidator(LLMProvider):
    def generate(self, prompt: str, system_prompt: str, config: LLMConfig) -> str:
        # Return a fully scored JSON report
        return """
        {
          "hallucinations": [],
          "missing_facts": [],
          "fact_drift": [],
          "style_match": true,
          "overall_pass": true,
          "semantic_accuracy": 0.95,
          "hallucination_risk": 0.05,
          "grammar_score": 0.98,
          "temporal_consistency_score": 0.90,
          "word_budget_pass": true,
          "overall_confidence": 0.92
        }
        """

def test_validator_scoring_report():
    llm = MockLLMForValidator()
    config = LLMConfig(provider="mock", model="mock-model", max_tokens=100)
    validator = SemanticValidator(llm_provider=llm, llm_config=config)

    captions = {
        "formal": Caption(text="This is a valid formal caption.", style="formal", word_count=6)
    }
    narrative = Narrative(text="A ground-truth narrative text.", key_events=[], salience_scores={})

    report = validator.validate(captions, narrative)
    assert report.overall_pass is True
    assert "formal" in report.per_caption
    fc = report.per_caption["formal"]
    
    # Assert all detailed scores are correctly populated
    assert fc.semantic_accuracy == 0.95
    assert fc.hallucination_risk == 0.05
    assert fc.grammar_score == 0.98
    assert fc.temporal_consistency_score == 0.90
    assert fc.word_budget_pass is True
    assert fc.overall_confidence == 0.92
