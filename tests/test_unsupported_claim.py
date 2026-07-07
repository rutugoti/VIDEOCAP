import pytest

from src.shared.models import Narrative, Caption
from src.shared.providers import MockLLMProvider, LLMProvider, ProviderError
from src.validation.unsupported_claim_detector import UnsupportedClaimDetector
from src.validation.semantic_validator import SemanticValidator


class BrokenLLMProvider(LLMProvider):
    def generate(self, prompt, system_prompt, config):
        raise ProviderError("boom")


def test_candidate_additions_are_term_diff():
    det = UnsupportedClaimDetector(llm_provider=MockLLMProvider())
    cands = det.candidate_additions(
        caption_text="A person happily drinks coffee.",
        narrative_text="A person drinks coffee.",
    )
    # "happily" is in caption but not narrative; stopwords/shared terms excluded
    assert "happily" in cands
    assert "coffee" not in cands
    assert "person" not in cands


def test_no_candidates_means_no_llm_call():
    class CountingLLM(LLMProvider):
        def __init__(self):
            self.calls = 0
        def generate(self, prompt, system_prompt, config):
            self.calls += 1
            return "{}"
    llm = CountingLLM()
    det = UnsupportedClaimDetector(llm_provider=llm)
    # Caption terms are a subset of narrative terms → zero candidates → zero calls
    out = det.detect("A person drinks coffee.", "A person happily drinks coffee at home.")
    assert out == []
    assert llm.calls == 0


def test_detects_unsupported_affect_term():
    det = UnsupportedClaimDetector(llm_provider=MockLLMProvider())
    out = det.detect(
        caption_text="A person happily drinks coffee.",
        narrative_text="A person drinks coffee.",
    )
    assert out == ["happily"]


def test_detector_fails_open_on_llm_error():
    det = UnsupportedClaimDetector(llm_provider=BrokenLLMProvider())
    # Even with a real candidate, an LLM failure returns no flags (fail-open)
    out = det.detect("A person happily drinks coffee.", "A person drinks coffee.")
    assert out == []


def test_validator_flags_unsupported_claim_when_enabled():
    narrative = Narrative(text="A person drinks coffee.", key_events=[], salience_scores={}, evidence_mapping={})
    captions = {
        "formal": Caption(text="A person happily drinks coffee today.", style="formal", word_count=6),
    }
    validator = SemanticValidator(llm_provider=MockLLMProvider(), unsupported_claim_check=True)
    report = validator.validate(captions, narrative)

    assert report.per_caption["formal"].passed is False
    assert any("happily" in h for h in report.per_caption["formal"].hallucinations)


def test_validator_ignores_unsupported_claim_when_disabled():
    narrative = Narrative(text="A person drinks coffee.", key_events=[], salience_scores={}, evidence_mapping={})
    captions = {
        "formal": Caption(text="A person happily drinks coffee today.", style="formal", word_count=6),
    }
    # Default off: the affect word must NOT fail the caption
    validator = SemanticValidator(llm_provider=MockLLMProvider(), unsupported_claim_check=False)
    report = validator.validate(captions, narrative)

    assert not any("happily" in h for h in report.per_caption["formal"].hallucinations)
