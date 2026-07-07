import json
import logging
from typing import Any, Dict, List, Optional

from src.providers.base import (
    BaseProvider,
    VisionProvider,
    SpeechProvider,
    OCRProvider,
    LLMProvider,
    ValidationProvider,
    ProviderCapabilities,
    ProviderConfig,
    VisionObservation,
    SpeechObservation,
    OCRObservation,
    LLMResponse,
    ValidationResult,
)
from src.providers.registry import ProviderRegistry

logger = logging.getLogger(__name__)


class MockProvider(VisionProvider, SpeechProvider, OCRProvider, LLMProvider, ValidationProvider):
    """Unified Offline Mock Provider for offline testing and backward compatibility."""

    PROVIDER_NAME = "mock"

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
        self.api_key = api_key
        self.base_url = base_url

    def get_name(self) -> str:
        return "mock"

    def get_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_vision=True,
            supports_audio=True,
            supports_json=True,
            supports_multimodal=True,
        )

    # 1. Vision Mock
    def analyze_frames(
        self,
        frames: List[Any],
        prompt: str,
        config: ProviderConfig
    ) -> List[VisionObservation]:
        observations = []
        for f in frames:
            observations.append(
                VisionObservation(
                    content=f"action observed at timestamp {f.timestamp}s",
                    timestamp=f.timestamp,
                    confidence=0.90,
                    observation_type="action"
                )
            )
            if f.sample_index == 0:
                observations.append(
                    VisionObservation(
                        content="kitchen scene with modern appliances",
                        timestamp=f.timestamp,
                        confidence=0.95,
                        observation_type="scene"
                    )
                )
        return observations

    # 2. Speech Mock
    def transcribe(
        self,
        frames: List[Any],
        config: ProviderConfig
    ) -> List[SpeechObservation]:
        observations = []
        for f in frames:
            if getattr(f, "audio_segment", None) is not None:
                observations.append(
                    SpeechObservation(
                        content=f"spoken words recorded at {f.timestamp}s",
                        timestamp=f.timestamp,
                        confidence=0.88
                    )
                )
        return observations

    # 3. OCR Mock
    def extract_text(
        self,
        frames: List[Any],
        config: ProviderConfig
    ) -> List[OCRObservation]:
        observations = []
        for f in frames:
            if f.sample_index == len(frames) // 2:
                observations.append(
                    OCRObservation(
                        content="mock brand logo text",
                        timestamp=f.timestamp,
                        confidence=0.92
                    )
                )
        return observations

    # 4. LLM Mock
    def generate(
        self,
        prompt: str,
        system_prompt: str,
        config: ProviderConfig
    ) -> LLMResponse:
        prompt_lower = prompt.lower()
        sys_lower = system_prompt.lower()

        # P3.1: Causal relationship classifier mock
        if "causal relationship classifier" in sys_lower:
            if "cause" in prompt_lower or "trigger" in prompt_lower or "start" in prompt_lower:
                text = '{"relationship": "causal", "confidence": 0.85, "rationale": "Direct cause-effect from description keywords."}'
            elif "enable" in prompt_lower or "condition" in prompt_lower or "allow" in prompt_lower:
                text = '{"relationship": "enables", "confidence": 0.75, "rationale": "Enabling condition detected."}'
            else:
                text = '{"relationship": "temporal", "confidence": 0.60, "rationale": "Only temporal sequence observed."}'
            return LLMResponse(text=text)

        is_single_pass = (
            ("formal" in prompt_lower and "sarcastic" in prompt_lower and "tech_humor" in prompt_lower and "non_tech_humor" in prompt_lower) or
            ("formal" in sys_lower and "sarcastic" in sys_lower and "tech_humor" in sys_lower and "non_tech_humor" in sys_lower)
        )
        if is_single_pass:
            if "test_missing_key" in prompt_lower:
                text = """{
                    "sarcastic": "Oh fantastic, another video where events happen. Absolutely groundbreaking.",
                    "tech_humor": "EventLoop returned status code 200 after resolving kitchen tasks successfully.",
                    "non_tech_humor": "Well, the kitchen is clean but the coffee is gone."
                }"""
            elif "test_over_budget" in prompt_lower:
                text = """{
                    "formal": "This is a extremely long and verbose response designed specifically to exceed the maximum word count limit of thirty five words on the very first attempt to trigger the retry validation block and it contains extra words to exceed the threshold easily.",
                    "sarcastic": "Oh fantastic, another video where events happen. Absolutely groundbreaking.",
                    "tech_humor": "EventLoop returned status code 200 after resolving kitchen tasks successfully.",
                    "non_tech_humor": "Well, the kitchen is clean but the coffee is gone."
                }"""
            elif "test_low_separation" in prompt_lower:
                text = """{
                    "formal": "A person enters the kitchen, starts the coffee maker, and exits.",
                    "sarcastic": "Oh fantastic, another video where events happen.",
                    "tech_humor": "EventLoop returned status code 200 after resolving kitchen tasks successfully.",
                    "non_tech_humor": "EventLoop returned status code 200 after resolving kitchen tasks successfully."
                }"""
            else:
                text = """{
                    "formal": "A formal rewrite of the video narrative detailing events sequentially, ensuring all facts are preserved.",
                    "sarcastic": "Oh fantastic, another video where events happen. Absolutely groundbreaking, and I am completely thrilled by this development.",
                    "tech_humor": "EventLoop returned status code 200 after resolving kitchen tasks successfully and flushing the local cache.",
                    "non_tech_humor": "Well, the kitchen is clean but the coffee is gone. Classic morning tragedy that happens to the best of us."
                }"""
            return LLMResponse(text=text)

        # Check system prompt first for validator
        if "fact-checker" in sys_lower or "validator" in sys_lower:
            caption_part = ""
            if "caption:" in prompt_lower:
                parts = prompt_lower.split("caption:")
                if len(parts) > 1:
                    after_caption = parts[1]
                    if "report your findings" in after_caption:
                        caption_part = after_caption.split("report your findings")[0].strip()
                    else:
                        caption_part = after_caption.strip()
            else:
                caption_part = prompt_lower

            if "hallucinate" in caption_part:
                text = """{
                    "hallucinations": ["caption mentions a red car not in the source"],
                    "missing_facts": [],
                    "fact_drift": [],
                    "style_match": true,
                    "overall_pass": false
                }"""
            elif "drift" in caption_part:
                text = """{
                    "hallucinations": [],
                    "missing_facts": [],
                    "fact_drift": ["coffee maker changed to toaster"],
                    "style_match": true,
                    "overall_pass": false
                }"""
            elif "leakage" in caption_part or "leak" in caption_part:
                text = """{
                    "hallucinations": [],
                    "missing_facts": [],
                    "fact_drift": [],
                    "style_match": false,
                    "overall_pass": false
                }"""
            else:
                text = """{
                    "hallucinations": [],
                    "missing_facts": [],
                    "fact_drift": [],
                    "style_match": true,
                    "overall_pass": true
                }"""
            return LLMResponse(text=text)

        # P4.1: Unsupported claim detector mock
        if "unsupported-claim entailment detector" in sys_lower or "entailment detector" in sys_lower:
            # Check if any claim in prompt contains 'hallucination' or 'unsupported'
            if "hallucination" in prompt_lower or "unsupported" in prompt_lower or "red car" in prompt_lower:
                text = '{"status": "unsupported", "rationales": {"claim": "unsupported fact"}}'
            else:
                text = '{"status": "supported", "rationales": {}}'
            return LLMResponse(text=text)

        if "software engineer" in sys_lower or "tech_humor" in sys_lower:
            return LLMResponse(text="EventLoop returned status code 200 after resolving kitchen tasks successfully and flushing the local cache.")
        elif "funny social media" in sys_lower or "non_tech_humor" in sys_lower or "non-tech" in sys_lower:
            return LLMResponse(text="Well, the kitchen is clean but the coffee is gone. Classic morning tragedy that happens to the best of us.")
        elif "sarcastic" in sys_lower:
            return LLMResponse(text="Oh fantastic, another video where events happen. Absolutely groundbreaking, and I am completely thrilled by this development.")
        elif "technical writer" in sys_lower or "formal" in sys_lower:
            return LLMResponse(text="A formal rewrite of the video narrative detailing events sequentially, ensuring all facts are preserved.")

        # Fallback to prompt-based checks
        if "narrative" in prompt_lower or "compress" in prompt_lower or "summary" in prompt_lower:
            text = """{
                "text": "A person enters the kitchen, starts the coffee maker, and exits.",
                "key_events": [
                    "enters kitchen",
                    "starts coffee maker",
                    "exits kitchen"
                ],
                "salience_scores": {
                    "enters kitchen": 0.50,
                    "starts coffee maker": 0.95,
                    "exits kitchen": 0.40
                },
                "evidence_mapping": {
                    "enters kitchen": ["visual"],
                    "starts coffee maker": ["visual", "audio"],
                    "exits kitchen": ["visual"]
                }
            }"""
            return LLMResponse(text=text)
        elif "fuse" in prompt_lower or "fusion" in prompt_lower:
            text = """[
                {
                    "description": "A person enters the kitchen and inspects the modern appliances.",
                    "actors": ["person"],
                    "actions": ["enters", "inspects"],
                    "objects": ["appliances"],
                    "timestamp_start": 0.0,
                    "timestamp_end": 2.0,
                    "confidence": 0.95,
                    "evidence_sources": ["visual"],
                    "salience": 0.8
                },
                {
                    "description": "A person speaks and transcribes audio context.",
                    "actors": ["person"],
                    "actions": ["speaks"],
                    "objects": [],
                    "timestamp_start": 2.0,
                    "timestamp_end": 5.0,
                    "confidence": 0.90,
                    "evidence_sources": ["audio"],
                    "salience": 0.7
                }
            ]"""
            return LLMResponse(text=text)
        elif "formal" in prompt_lower:
            return LLMResponse(text="A formal rewrite of the video narrative detailing events sequentially, ensuring all facts are preserved.")
        elif "sarcastic" in prompt_lower:
            return LLMResponse(text="Oh fantastic, another video where events happen. Absolutely groundbreaking, and I am completely thrilled by this development.")
        elif "tech_humor" in prompt_lower or "software engineering" in prompt_lower:
            return LLMResponse(text="EventLoop returned status code 200 after resolving kitchen tasks successfully and flushing the local cache.")
        elif "humor" in prompt_lower or "funny" in prompt_lower:
            return LLMResponse(text="Well, the kitchen is clean but the coffee is gone. Classic morning tragedy that happens to the best of us.")

        return LLMResponse(text="Generic mock LLM text response.")

    # 5. Validation Mock
    def validate(
        self,
        captions: Dict[str, Any],
        narrative: Any,
        config: ProviderConfig
    ) -> ValidationResult:
        # Cross-style mock validation
        hallucinations = []
        missing_facts = []
        
        # Check if any caption contains 'hallucinate' or 'drift' or 'leakage'
        has_hallucination = False
        has_drift = False
        has_leakage = False
        for c in captions.values():
            text_lower = c.text.lower()
            if "hallucinate" in text_lower:
                has_hallucination = True
            if "drift" in text_lower:
                has_drift = True
            if "leakage" in text_lower or "leak" in text_lower:
                has_leakage = True
                
        if has_hallucination:
            return ValidationResult(
                passed=False,
                status="FAIL",
                hallucinations=["caption mentions a red car not in the source"]
            )
        elif has_drift:
            return ValidationResult(
                passed=False,
                status="FAIL",
                missing_facts=["coffee maker changed to toaster"]
            )
        elif has_leakage:
            return ValidationResult(
                passed=False,
                status="FAIL",
                style_adherence=0.3
            )
            
        return ValidationResult(passed=True, status="PASS")


# Auto-register mock provider
ProviderRegistry.register("mock", MockProvider)
