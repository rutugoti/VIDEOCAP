import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env, overriding any existing shell placeholders
load_dotenv(override=True)

# Root directory of the project
ROOT_DIR = Path(__file__).resolve().parent.parent.parent


from typing import Any, Dict, List, Optional, Union

class ModelParams(BaseModel):
    provider: Union[str, List[str]]
    model: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    language: Optional[str] = None



class ModelsConfig(BaseModel):
    vision: ModelParams
    speech: ModelParams
    ocr: ModelParams
    llm: ModelParams
    validator: ModelParams


class AdaptiveLimits(BaseModel):
    max_samples: int = 30
    min_samples: int = 5


class SamplingConfig(BaseModel):
    method: str
    fps: float
    max_frames: int
    min_frames: int
    baseline_fps: float = 1.0
    min_scene_duration: float = 1.0
    motion_threshold: float = 15.0
    scene_change_threshold: float = 20.0
    silence_threshold: float = -40.0
    adaptive_sampling_limits: Optional[AdaptiveLimits] = None


class PerceptionDetails(BaseModel):
    enabled: bool


class PerceptionConfig(BaseModel):
    parallel: bool
    timeout_seconds: int
    max_workers: Optional[int] = 3
    vision: PerceptionDetails
    speech: PerceptionDetails
    ocr: PerceptionDetails


class FusionConfig(BaseModel):
    min_confidence: float
    temporal_window_seconds: float


class GraphConfig(BaseModel):
    causal_edges: bool
    causal_window_seconds: float
    edge_weight_threshold: float
    max_edges_to_narrative: int


class NarrativeConfig(BaseModel):
    max_events: int
    max_words: int
    consume_edges: bool


class GenerationConfig(BaseModel):
    styles: List[str]
    max_caption_words: int
    min_caption_words: int
    single_pass: bool
    style_separation_min: float


class EvidencePolicyConfig(BaseModel):
    escalate_conflict_min_conf: float


class ValidationConfig(BaseModel):
    enabled: bool
    hallucination_check: bool
    style_leakage_check: bool
    fact_drift_check: bool
    retry_on_failure: bool
    max_retries: int
    unsupported_claim_check: bool = False


class PipelineConfig(BaseModel):
    sampling: SamplingConfig
    perception: PerceptionConfig
    fusion: FusionConfig
    graph: GraphConfig
    narrative: NarrativeConfig
    generation: GenerationConfig
    evidence_policy: EvidencePolicyConfig
    validation: ValidationConfig


class PromptsConfig(BaseModel):
    generation: Dict[str, str]
    validation: Dict[str, str]
    vision: Dict[str, str]


class AppDetailsConfig(BaseModel):
    name: str
    version: str
    log_level: str
    output_dir: str


class APIDetailsConfig(BaseModel):
    fireworks_base_url: str
    timeout_seconds: int
    max_retries: int
    retry_delay_seconds: int


class LoggingDetailsConfig(BaseModel):
    format: str
    file: str
    console: bool


class SettingsConfig(BaseModel):
    app: AppDetailsConfig
    api: APIDetailsConfig
    logging: LoggingDetailsConfig


class FullConfig(BaseModel):
    settings: SettingsConfig
    models: ModelsConfig
    pipeline: PipelineConfig
    prompts: PromptsConfig
    fireworks_api_key: Optional[str] = Field(None, exclude=True)



def load_yaml(file_path: Path) -> Dict[str, Any]:
    """Helper function to load a YAML file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def setup_logging(config: SettingsConfig) -> None:
    """Setup logging configuration."""
    log_level_str = os.environ.get("LOG_LEVEL", config.app.log_level).upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    log_format = config.logging.format
    log_file = ROOT_DIR / config.logging.file

    # Create logs directory if it doesn't exist
    log_file.parent.mkdir(parents=True, exist_ok=True)

    handlers: List[logging.Handler] = []
    if config.logging.console:
        handlers.append(logging.StreamHandler())
    if config.logging.file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers,
        force=True
    )
    logging.info(f"Logging initialized with level {log_level_str}")


def get_config() -> FullConfig:
    """Load, merge and validate all configurations."""
    # Surfaced on FullConfig for callers that want the Fireworks key directly. Real
    # key resolution/validation happens per-provider in ProviderFactory, which raises
    # if no usable key is configured — there is no offline-mock fallback.
    fireworks_api_key = os.environ.get("FIREWORKS_API_KEY")

    config_dir = ROOT_DIR / "configs"

    settings_dict = load_yaml(config_dir / "settings.yaml")
    models_dict = load_yaml(config_dir / "models.yaml")
    pipeline_dict = load_yaml(config_dir / "pipeline.yaml")
    prompts_dict = load_yaml(config_dir / "prompts.yaml")

    # Handle overrides from environment variables
    output_dir_override = os.environ.get("OUTPUT_DIR")
    if output_dir_override and "app" in settings_dict:
        settings_dict["app"]["output_dir"] = output_dir_override

    # Construct the FullConfig model which validates the input types
    config = FullConfig(
        settings=SettingsConfig(**settings_dict),
        models=ModelsConfig(**models_dict),
        pipeline=PipelineConfig(**pipeline_dict),
        prompts=PromptsConfig(**prompts_dict),
        fireworks_api_key=fireworks_api_key
    )

    # Initialize the logging system
    setup_logging(config.settings)

    return config
