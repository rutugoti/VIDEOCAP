import logging
import os
from typing import Any, Dict, List, Optional, Union

from src.providers.base import (
    ProviderConfig,
    BaseProvider,
    VisionProvider,
    SpeechProvider,
    OCRProvider,
    LLMProvider,
    EmbeddingProvider,
    ValidationProvider,
    ProviderError,
    AuthenticationError,
    RateLimitError,
    TimeoutError,
    ProviderUnavailableError,
    InvalidResponseError,
    VisionObservation,
    SpeechObservation,
    OCRObservation,
    LLMResponse,
    EmbeddingResponse,
    ValidationResult,
)
from src.providers.registry import ProviderRegistry

logger = logging.getLogger(__name__)


# =====================================================================
# Fallback Proxy Implementations
# =====================================================================

class FallbackVisionProviderProxy(VisionProvider):
    """Wraps a list of VisionProviders and falls back sequentially on failures."""

    def __init__(self, providers: List[VisionProvider]):
        self.providers = providers

    def get_name(self) -> str:
        return f"FallbackVisionProxy({', '.join(p.get_name() for p in self.providers)})"

    def get_capabilities(self):
        # Merge capabilities: if any support, it supports
        from src.providers.base import ProviderCapabilities
        caps = ProviderCapabilities()
        for p in self.providers:
            p_caps = p.get_capabilities()
            caps.supports_vision = caps.supports_vision or p_caps.supports_vision
            caps.supports_json = caps.supports_json or p_caps.supports_json
            caps.supports_multimodal = caps.supports_multimodal or p_caps.supports_multimodal
        return caps

    def analyze_frames(
        self,
        frames: List[Any],
        prompt: str,
        config: ProviderConfig
    ) -> List[VisionObservation]:
        last_error = None
        for i, provider in enumerate(self.providers):
            logger.info(f"FallbackVisionProxy: Trying provider '{provider.get_name()}' ({i+1}/{len(self.providers)})")
            try:
                # Update config to match provider specific details if needed
                config.provider_name = provider.get_name()
                return provider.analyze_frames(frames, prompt, config)
            except ProviderError as e:
                logger.warning(f"FallbackVisionProxy: Provider '{provider.get_name()}' failed: {e}")
                last_error = e
            except Exception as e:
                logger.warning(f"FallbackVisionProxy: Unexpected error on provider '{provider.get_name()}': {e}")
                last_error = ProviderError(f"Unexpected error: {e}")
        
        raise ProviderError(f"All vision fallback providers failed. Last error: {last_error}")


class FallbackSpeechProviderProxy(SpeechProvider):
    """Wraps a list of SpeechProviders and falls back sequentially on failures."""

    def __init__(self, providers: List[SpeechProvider]):
        self.providers = providers

    def get_name(self) -> str:
        return f"FallbackSpeechProxy({', '.join(p.get_name() for p in self.providers)})"

    def get_capabilities(self):
        from src.providers.base import ProviderCapabilities
        caps = ProviderCapabilities()
        for p in self.providers:
            p_caps = p.get_capabilities()
            caps.supports_audio = caps.supports_audio or p_caps.supports_audio
        return caps

    def transcribe(
        self,
        frames: List[Any],
        config: ProviderConfig
    ) -> List[SpeechObservation]:
        last_error = None
        for i, provider in enumerate(self.providers):
            logger.info(f"FallbackSpeechProxy: Trying provider '{provider.get_name()}' ({i+1}/{len(self.providers)})")
            try:
                config.provider_name = provider.get_name()
                return provider.transcribe(frames, config)
            except ProviderError as e:
                logger.warning(f"FallbackSpeechProxy: Provider '{provider.get_name()}' failed: {e}")
                last_error = e
            except Exception as e:
                logger.warning(f"FallbackSpeechProxy: Unexpected error on provider '{provider.get_name()}': {e}")
                last_error = ProviderError(f"Unexpected error: {e}")
        
        raise ProviderError(f"All speech fallback providers failed. Last error: {last_error}")


class FallbackOCRProviderProxy(OCRProvider):
    """Wraps a list of OCRProviders and falls back sequentially on failures."""

    def __init__(self, providers: List[OCRProvider]):
        self.providers = providers

    def get_name(self) -> str:
        return f"FallbackOCRProxy({', '.join(p.get_name() for p in self.providers)})"

    def get_capabilities(self):
        from src.providers.base import ProviderCapabilities
        caps = ProviderCapabilities()
        for p in self.providers:
            p_caps = p.get_capabilities()
            caps.supports_vision = caps.supports_vision or p_caps.supports_vision
        return caps

    def extract_text(
        self,
        frames: List[Any],
        config: ProviderConfig
    ) -> List[OCRObservation]:
        last_error = None
        for i, provider in enumerate(self.providers):
            logger.info(f"FallbackOCRProxy: Trying provider '{provider.get_name()}' ({i+1}/{len(self.providers)})")
            try:
                config.provider_name = provider.get_name()
                return provider.extract_text(frames, config)
            except ProviderError as e:
                logger.warning(f"FallbackOCRProxy: Provider '{provider.get_name()}' failed: {e}")
                last_error = e
            except Exception as e:
                logger.warning(f"FallbackOCRProxy: Unexpected error on provider '{provider.get_name()}': {e}")
                last_error = ProviderError(f"Unexpected error: {e}")
        
        raise ProviderError(f"All OCR fallback providers failed. Last error: {last_error}")


class FallbackLLMProviderProxy(LLMProvider):
    """Wraps a list of LLMProviders and falls back sequentially on failures."""

    def __init__(self, providers: List[LLMProvider]):
        self.providers = providers

    def get_name(self) -> str:
        return f"FallbackLLMProxy({', '.join(p.get_name() for p in self.providers)})"

    def get_capabilities(self):
        from src.providers.base import ProviderCapabilities
        caps = ProviderCapabilities()
        for p in self.providers:
            p_caps = p.get_capabilities()
            caps.supports_json = caps.supports_json or p_caps.supports_json
            caps.supports_streaming = caps.supports_streaming or p_caps.supports_streaming
        return caps

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        config: ProviderConfig
    ) -> LLMResponse:
        last_error = None
        for i, provider in enumerate(self.providers):
            logger.info(f"FallbackLLMProxy: Trying provider '{provider.get_name()}' ({i+1}/{len(self.providers)})")
            try:
                config.provider_name = provider.get_name()
                return provider.generate(prompt, system_prompt, config)
            except ProviderError as e:
                logger.warning(f"FallbackLLMProxy: Provider '{provider.get_name()}' failed: {e}")
                last_error = e
            except Exception as e:
                logger.warning(f"FallbackLLMProxy: Unexpected error on provider '{provider.get_name()}': {e}")
                last_error = ProviderError(f"Unexpected error: {e}")
        
        raise ProviderError(f"All LLM fallback providers failed. Last error: {last_error}")


class FallbackValidationProviderProxy(ValidationProvider):
    """Wraps a list of ValidationProviders and falls back sequentially on failures."""

    def __init__(self, providers: List[ValidationProvider]):
        self.providers = providers

    def get_name(self) -> str:
        return f"FallbackValidationProxy({', '.join(p.get_name() for p in self.providers)})"

    def get_capabilities(self):
        from src.providers.base import ProviderCapabilities
        caps = ProviderCapabilities()
        for p in self.providers:
            p_caps = p.get_capabilities()
            caps.supports_json = caps.supports_json or p_caps.supports_json
        return caps

    def validate(
        self,
        captions: Dict[str, Any],
        narrative: Any,
        config: ProviderConfig
    ) -> ValidationResult:
        last_error = None
        for i, provider in enumerate(self.providers):
            logger.info(f"FallbackValidationProxy: Trying provider '{provider.get_name()}' ({i+1}/{len(self.providers)})")
            try:
                config.provider_name = provider.get_name()
                return provider.validate(captions, narrative, config)
            except ProviderError as e:
                logger.warning(f"FallbackValidationProxy: Provider '{provider.get_name()}' failed: {e}")
                last_error = e
            except Exception as e:
                logger.warning(f"FallbackValidationProxy: Unexpected error on provider '{provider.get_name()}': {e}")
                last_error = ProviderError(f"Unexpected error: {e}")
        
        raise ProviderError(f"All validation fallback providers failed. Last error: {last_error}")


# =====================================================================
# Factory Implementation
# =====================================================================

class ProviderFactory:
    """Dependency Injection and instantiation container with Fallback support."""

    @staticmethod
    def get_env_api_key(provider_name: str) -> Optional[str]:
        """Resolves API key from environment variables based on provider name."""
        p_clean = provider_name.strip().lower()
        if p_clean == "fireworks":
            return os.environ.get("FIREWORKS_API_KEY") or os.environ.get("FIREWORKS_API_KEY_ENV")
        elif p_clean == "openai":
            return os.environ.get("OPENAI_API_KEY")
        elif p_clean in ("google", "gemini"):
            return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        elif p_clean == "anthropic":
            return os.environ.get("ANTHROPIC_API_KEY")
        elif p_clean == "groq":
            return os.environ.get("GROQ_API_KEY")
        elif p_clean == "together":
            return os.environ.get("TOGETHER_API_KEY")
        return None

    @classmethod
    def create_provider(
        cls,
        provider_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> BaseProvider:
        """Instantiate a single provider by looking up from registry."""
        p_clean = provider_name.strip().lower()
        provider_class = ProviderRegistry.get(p_clean)
        
        # Resolve API Key & Base URL
        resolved_key = api_key or cls.get_env_api_key(p_clean)
        
        # For testing compatibility / offline fallback
        if not resolved_key and p_clean not in ("mock", "ollama"):
            resolved_key = "mock_key_for_testing"

        # Instantiate provider
        if p_clean in ("mock", "ollama"):
            # Mock / Ollama doesn't strictly require key
            return provider_class(base_url=base_url) if base_url else provider_class()
        else:
            return provider_class(api_key=resolved_key, base_url=base_url)

    @classmethod
    def get_vision(
        cls,
        providers: Union[str, List[str]],
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> VisionProvider:
        """Get VisionProvider (single or fallback proxy)."""
        provider_list = [providers] if isinstance(providers, str) else providers
        instances = [cls.create_provider(p, api_key, base_url) for p in provider_list]
        
        # Verify types
        for inst in instances:
            if not isinstance(inst, VisionProvider):
                raise TypeError(f"Provider '{inst.get_name()}' does not implement VisionProvider interface.")
                
        if len(instances) == 1:
            return instances[0]
        return FallbackVisionProviderProxy(instances)

    @classmethod
    def get_speech(
        cls,
        providers: Union[str, List[str]],
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> SpeechProvider:
        """Get SpeechProvider (single or fallback proxy)."""
        provider_list = [providers] if isinstance(providers, str) else providers
        instances = [cls.create_provider(p, api_key, base_url) for p in provider_list]
        
        for inst in instances:
            if not isinstance(inst, SpeechProvider):
                raise TypeError(f"Provider '{inst.get_name()}' does not implement SpeechProvider interface.")
                
        if len(instances) == 1:
            return instances[0]
        return FallbackSpeechProviderProxy(instances)

    @classmethod
    def get_ocr(
        cls,
        providers: Union[str, List[str]],
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> OCRProvider:
        """Get OCRProvider (single or fallback proxy)."""
        provider_list = [providers] if isinstance(providers, str) else providers
        instances = [cls.create_provider(p, api_key, base_url) for p in provider_list]
        
        for inst in instances:
            if not isinstance(inst, OCRProvider):
                raise TypeError(f"Provider '{inst.get_name()}' does not implement OCRProvider interface.")
                
        if len(instances) == 1:
            return instances[0]
        return FallbackOCRProviderProxy(instances)

    @classmethod
    def get_llm(
        cls,
        providers: Union[str, List[str]],
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> LLMProvider:
        """Get LLMProvider (single or fallback proxy)."""
        provider_list = [providers] if isinstance(providers, str) else providers
        instances = [cls.create_provider(p, api_key, base_url) for p in provider_list]
        
        for inst in instances:
            if not isinstance(inst, LLMProvider):
                raise TypeError(f"Provider '{inst.get_name()}' does not implement LLMProvider interface.")
                
        if len(instances) == 1:
            return instances[0]
        return FallbackLLMProviderProxy(instances)

    @classmethod
    def get_validator(
        cls,
        providers: Union[str, List[str]],
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> ValidationProvider:
        """Get ValidationProvider (single or fallback proxy)."""
        provider_list = [providers] if isinstance(providers, str) else providers
        instances = [cls.create_provider(p, api_key, base_url) for p in provider_list]
        
        for inst in instances:
            if not isinstance(inst, ValidationProvider):
                raise TypeError(f"Provider '{inst.get_name()}' does not implement ValidationProvider interface.")
                
        if len(instances) == 1:
            return instances[0]
        return FallbackValidationProviderProxy(instances)
