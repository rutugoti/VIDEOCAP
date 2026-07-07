import logging
from typing import Dict, Any, Optional

from src.providers.base import (
    BaseProvider,
    ValidationProvider,
    LLMProvider,
    ProviderCapabilities,
    ProviderConfig,
    ProviderError,
    ValidationResult,
)
from src.providers.registry import ProviderRegistry
from src.validation.semantic_validator import SemanticValidator
from src.shared.models import Caption, Narrative
from src.shared.providers import LLMConfig

logger = logging.getLogger(__name__)


class LLMValidationProvider(ValidationProvider):
    """Validation provider delegating validation rules to a configured LLMProvider."""

    PROVIDER_NAME = "llm_validator"

    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        self.llm_provider = llm_provider

    def get_name(self) -> str:
        return "llm_validator"

    def get_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_vision=False,
            supports_audio=False,
            supports_json=True,
            supports_multimodal=False,
        )

    def validate(
        self,
        captions: Dict[str, Any],
        narrative: Any,
        config: ProviderConfig
    ) -> ValidationResult:
        # Delegate to SemanticValidator but wrap the LLMProvider to be backward-compatible
        # We need a backward-compatible adapter from src/providers/base/LLMProvider to src/shared/providers/LLMProvider
        from src.shared.providers import LLMProvider as LegacyLLMProvider
        
        class LegacyLLMProviderAdapter(LegacyLLMProvider):
            def __init__(self, new_llm: LLMProvider):
                self.new_llm = new_llm
            def generate(self, prompt: str, system_prompt: str, legacy_cfg: Any) -> str:
                # Map legacy LLMConfig properties to ProviderConfig
                p_cfg = ProviderConfig(
                    provider_name=self.new_llm.get_name(),
                    model_name=legacy_cfg.model,
                    max_tokens=legacy_cfg.max_tokens,
                    temperature=legacy_cfg.temperature
                )
                res = self.new_llm.generate(prompt, system_prompt, p_cfg)
                return res.text

        llm_instance = self.llm_provider
        if not llm_instance:
            raise ProviderError(
                "LLMValidationProvider requires an underlying LLM provider; none was configured."
            )

        legacy_llm = LegacyLLMProviderAdapter(llm_instance)
        
        legacy_config = LLMConfig(
            provider=llm_instance.get_name(),
            model=config.model_name,
            max_tokens=config.max_tokens or 1024,
            temperature=config.temperature or 0.1
        )

        validator = SemanticValidator(
            llm_provider=legacy_llm,
            llm_config=legacy_config
        )

        # Execute existing validation logic
        report = validator.validate(captions, narrative)

        # Map details to ValidationResult
        hallucinations = []
        missing_facts = []
        style_scores = []
        for style_name, cap_val in report.per_caption.items():
            hallucinations.extend(cap_val.hallucinations)
            missing_facts.extend(cap_val.missing_facts)
            style_scores.append(cap_val.style_adherence)
            
        style_adherence = sum(style_scores) / max(1, len(style_scores))

        return ValidationResult(
            passed=report.overall_pass,
            status=report.status,
            hallucinations=list(set(hallucinations)),
            missing_facts=list(set(missing_facts)),
            style_adherence=style_adherence,
            overall_confidence=report.consistency_score
        )


# Auto-register LLM Validation provider
ProviderRegistry.register("llm_validator", LLMValidationProvider)
