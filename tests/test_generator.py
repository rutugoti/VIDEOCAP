import os
import pytest
from src.shared.models import Narrative
from src.shared.providers import MockLLMProvider, LLMProvider
from src.generation.style_generator import StyleGenerator


class LengthyLLMProvider(LLMProvider):
    """An LLM provider that returns outputs violating the word budget on first try, but fits on retry."""
    def __init__(self):
        self.attempts = {}

    def generate(self, prompt, system_prompt, config):
        # Detect style from prompt
        style = "generic"
        for s in ["formal", "sarcastic", "tech", "humor"]:
            if s in prompt.lower():
                style = s
                break

        if style not in self.attempts:
            self.attempts[style] = 1
            # Return long response (exceeding 35 words)
            return "This is a extremely long and verbose response designed specifically to exceed the maximum word count limit of thirty five words on the very first attempt to trigger the retry validation block and it contains extra words to exceed the threshold easily."
        else:
            # Return acceptable response (between 15 and 35 words)
            return "This is a shorter and concise caption designed specifically to satisfy the word count limit on retry."


def test_four_styles():
    narrative = Narrative(
        text="A person enters the kitchen, starts the coffee maker, and exits.",
        key_events=["enters kitchen", "starts coffee maker", "exits kitchen"],
        salience_scores={"enters kitchen": 0.5, "starts coffee maker": 0.95, "exits kitchen": 0.4},
        evidence_mapping={"enters kitchen": ["visual"], "starts coffee maker": ["visual", "audio"], "exits kitchen": ["visual"]}
    )

    mock_llm = MockLLMProvider()
    generator = StyleGenerator(llm_provider=mock_llm)
    captions = generator.generate_captions(narrative)

    assert set(captions.keys()) == {"formal", "sarcastic", "tech_humor", "non_tech_humor"}
    for style, cap in captions.items():
        assert cap.style == style
        assert len(cap.text) > 0
        assert cap.word_count == len(cap.text.split())
        assert 15 <= cap.word_count <= 35


def test_style_distinction():
    narrative = Narrative(
        text="A person enters the kitchen, starts the coffee maker, and exits.",
        key_events=["enters kitchen", "starts coffee maker", "exits kitchen"],
        salience_scores={"enters kitchen": 0.5, "starts coffee maker": 0.95, "exits kitchen": 0.4},
        evidence_mapping={"enters kitchen": ["visual"], "starts coffee maker": ["visual", "audio"], "exits kitchen": ["visual"]}
    )

    mock_llm = MockLLMProvider()
    generator = StyleGenerator(llm_provider=mock_llm)
    captions = generator.generate_captions(narrative)

    # Styles must be distinct
    assert captions["formal"].text != captions["sarcastic"].text
    assert captions["tech_humor"].text != captions["non_tech_humor"].text
    assert "EventLoop" in captions["tech_humor"].text
    assert "formal rewrite" in captions["formal"].text.lower()


def test_word_limits_retry_loop():
    narrative = Narrative(
        text="A person enters the kitchen, starts the coffee maker, and exits.",
        key_events=["enters kitchen", "starts coffee maker", "exits kitchen"],
        salience_scores={"enters kitchen": 0.5, "starts coffee maker": 0.95, "exits kitchen": 0.4},
        evidence_mapping={"enters kitchen": ["visual"], "starts coffee maker": ["visual", "audio"], "exits kitchen": ["visual"]}
    )

    lengthy_llm = LengthyLLMProvider()
    generator = StyleGenerator(llm_provider=lengthy_llm)
    captions = generator.generate_captions(narrative)

    # All captions should have been corrected on retry to have 17 words (within 15-35 range)
    for style, cap in captions.items():
        assert cap.word_count == 17
        assert "shorter and concise" in cap.text


def test_prompt_loading(tmp_path):
    # Create temp prompt files to verify parsing from disk
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()

    style_file = prompts_dir / "formal.md"
    style_file.write_text("""# Prompt: Formal Style
## Version
v1.0
## System Prompt
System prompt for formal style content.
## User Prompt Template
```
User prompt template for {narrative}.
```
""")

    mock_llm = MockLLMProvider()
    generator = StyleGenerator(llm_provider=mock_llm, prompts_dir=str(prompts_dir))
    
    sys_p, usr_p = generator._load_prompt("formal")
    assert sys_p == "System prompt for formal style content."
    assert usr_p == "User prompt template for {narrative}."

    # Fallback check
    sys_fallback, usr_fallback = generator._load_prompt("sarcastic")
    assert sys_fallback == generator.FALLBACK_PROMPTS["sarcastic"]["system"]


def test_single_pass_success():
    narrative = Narrative(
        text="A person enters the kitchen, starts the coffee maker, and exits.",
        key_events=["enters kitchen", "starts coffee maker", "exits kitchen"],
        salience_scores={"enters kitchen": 0.5, "starts coffee maker": 0.95, "exits kitchen": 0.4},
        evidence_mapping={"enters kitchen": ["visual"], "starts coffee maker": ["visual", "audio"], "exits kitchen": ["visual"]}
    )
    mock_llm = MockLLMProvider()
    generator = StyleGenerator(llm_provider=mock_llm, single_pass=True, style_separation_min=0.4)
    captions = generator.generate_captions(narrative)
    assert set(captions.keys()) == {"formal", "sarcastic", "tech_humor", "non_tech_humor"}
    assert captions["formal"].metadata.get("regenerated") is False
    assert captions["sarcastic"].metadata.get("regenerated") is False


def test_single_pass_missing_key():
    # Trigger trigger phrase for test_missing_key mock behavior
    narrative = Narrative(
        text="test_missing_key: A person enters the kitchen.",
        key_events=[],
        salience_scores={},
        evidence_mapping={}
    )
    mock_llm = MockLLMProvider()
    generator = StyleGenerator(llm_provider=mock_llm, single_pass=True)
    captions = generator.generate_captions(narrative)
    # The missing key "formal" should have been regenerated in isolation
    assert set(captions.keys()) == {"formal", "sarcastic", "tech_humor", "non_tech_humor"}
    assert "formal rewrite" in captions["formal"].text.lower()


def test_single_pass_over_budget_deterministic_trim():
    # Trigger trigger phrase for test_over_budget mock behavior
    narrative = Narrative(
        text="test_over_budget: A person enters the kitchen.",
        key_events=[],
        salience_scores={},
        evidence_mapping={}
    )
    mock_llm = MockLLMProvider()
    generator = StyleGenerator(llm_provider=mock_llm, single_pass=True)
    captions = generator.generate_captions(narrative)
    # "formal" style returned is extremely long, so it gets trimmed
    assert set(captions.keys()) == {"formal", "sarcastic", "tech_humor", "non_tech_humor"}
    # The word count must be within limit (35 words)
    assert len(captions["formal"].text.split()) <= 35


def test_single_pass_low_separation_regeneration():
    # Trigger trigger phrase for test_low_separation mock behavior
    # This will make tech_humor and non_tech_humor identical, triggering regeneration for both
    narrative = Narrative(
        text="test_low_separation: A person enters.",
        key_events=[],
        salience_scores={},
        evidence_mapping={}
    )
    mock_llm = MockLLMProvider()
    generator = StyleGenerator(llm_provider=mock_llm, single_pass=True, style_separation_min=0.5)
    captions = generator.generate_captions(narrative)
    # Regeneration flag should be true for tech_humor and non_tech_humor
    assert captions["tech_humor"].metadata.get("regenerated") is True
    assert captions["non_tech_humor"].metadata.get("regenerated") is True

