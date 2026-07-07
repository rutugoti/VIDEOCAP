import importlib
import inspect
import logging
import pkgutil
from typing import Dict, Type, List

from src.providers.base import BaseProvider

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """Registry managing available provider classes and auto-discovering plugins."""

    _providers: Dict[str, Type[BaseProvider]] = {}
    _discovered = False

    @classmethod
    def register(cls, name: str, provider_class: Type[BaseProvider]) -> None:
        """Register a provider class with a unique name."""
        name_clean = name.strip().lower()
        cls._providers[name_clean] = provider_class
        logger.info(f"ProviderRegistry: Registered provider '{name_clean}' ({provider_class.__name__})")

    @classmethod
    def unregister(cls, name: str) -> None:
        """Unregister a provider by name."""
        name_clean = name.strip().lower()
        if name_clean in cls._providers:
            del cls._providers[name_clean]
            logger.info(f"ProviderRegistry: Unregistered provider '{name_clean}'")

    @classmethod
    def get(cls, name: str) -> Type[BaseProvider]:
        """Retrieve a registered provider class."""
        name_clean = name.strip().lower()
        if not cls._discovered:
            cls.discover()
        if name_clean not in cls._providers:
            raise KeyError(f"Provider '{name}' is not registered.")
        return cls._providers[name_clean]

    @classmethod
    def list_providers(cls) -> List[str]:
        """List names of all registered providers."""
        if not cls._discovered:
            cls.discover()
        return list(cls._providers.keys())

    @classmethod
    def discover(cls) -> None:
        """Dynamically discover and register providers in the src.providers package."""
        cls._discovered = True
        import src.providers
        prefix = src.providers.__name__ + "."
        
        for _, module_name, _ in pkgutil.iter_modules(src.providers.__path__, prefix):
            # Avoid importing modules that are currently being defined
            if module_name in (f"{prefix}base", f"{prefix}registry", f"{prefix}factory"):
                continue
            
            try:
                module = importlib.import_module(module_name)
                for name, obj in inspect.getmembers(module):
                    if (
                        inspect.isclass(obj) 
                        and issubclass(obj, BaseProvider) 
                        and obj is not BaseProvider
                        and not inspect.isabstract(obj)
                    ):
                        # Instantiate temporary or check metadata to get provider name
                        try:
                            # Use class method get_name or inspect metadata
                            # For simplicity we register it under the module name suffix or class name lowercase
                            provider_name = getattr(obj, "PROVIDER_NAME", obj.__name__.lower().replace("provider", ""))
                            cls.register(provider_name, obj)
                        except Exception as e:
                            logger.warning(f"Failed to auto-register class {obj.__name__} in module {module_name}: {e}")
            except Exception as e:
                logger.error(f"Failed to discover/import module {module_name}: {e}")
