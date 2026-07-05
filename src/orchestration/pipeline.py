import logging
import os
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

from src.config.settings import FullConfig, get_config
from src.shared.models import (
    VideoDescriptor,
    Sample,
    Observation,
    Timeline,
    Event,
    SemanticGraph,
    Narrative,
    Caption,
    ValidationReport,
)
from src.shared.providers import (
    LLMProvider,
    VisionProvider,
    AudioProvider,
    OCRProvider,
    MockLLMProvider,
    MockVisionProvider,
    MockAudioProvider,
    MockOCRProvider,
    ProviderError,
    ValidationError,
    VisionConfig,
    AudioConfig,
    OCRConfig,
    LLMConfig,
)

from src.ingestion.video_loader import VideoLoader
from src.sampling.adaptive_sampler import AdaptiveSampler
from src.perception.vision_processor import VisionProcessor
from src.perception.speech_processor import SpeechProcessor
from src.perception.ocr_processor import OCRProcessor
from src.fusion.timeline_builder import TimelineBuilder
from src.fusion.fusion_engine import FusionEngine
from src.graph.semantic_graph_builder import SemanticGraphBuilder
from src.narrative.narrative_builder import NarrativeBuilder
from src.generation.style_generator import StyleGenerator
from src.validation.semantic_validator import SemanticValidator
from src.submission.submission_formatter import SubmissionFormatter
from src.shared.utils import get_cache_key, load_from_cache, save_to_cache

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Execution coordinator running video processing steps load-to-validate sequentially."""

    def __init__(
        self,
        config: Optional[FullConfig] = None,
        llm_provider: Optional[LLMProvider] = None,
        vision_provider: Optional[VisionProvider] = None,
        audio_provider: Optional[AudioProvider] = None,
        ocr_provider: Optional[OCRProvider] = None,
    ):
        # 1. Load configuration
        self.config = config or get_config()

        # 2. Assign/Fallback Providers
        self.llm_provider = llm_provider or MockLLMProvider()
        self.vision_provider = vision_provider or MockVisionProvider()
        self.audio_provider = audio_provider or MockAudioProvider()
        self.ocr_provider = ocr_provider or MockOCRProvider()

        # 3. Initialize Ingestion & Sampling Modules
        # Disable strict duration checks in loader if the configuration allows or bypass via defaults
        self.video_loader = VideoLoader(min_duration=0.0, max_duration=10000.0)
        
        sampling_cfg = self.config.pipeline.sampling
        self.adaptive_sampler = AdaptiveSampler(
            method=sampling_cfg.method,
            fps=sampling_cfg.fps,
            max_frames=sampling_cfg.max_frames,
            min_frames=sampling_cfg.min_frames
        )

        # 4. Initialize Perception Processors
        # Convert configuration parameters to internal models/configs
        vis_model_cfg = VisionConfig(
            provider=self.config.models.vision.provider,
            model=self.config.models.vision.model,
            max_tokens=self.config.models.vision.max_tokens or 4096,
            temperature=self.config.models.vision.temperature or 0.2
        )
        self.vision_processor = VisionProcessor(
            provider=self.vision_provider,
            config=vis_model_cfg,
            prompt=self.config.prompts.vision.get("frame_analysis")
        )

        speech_model_cfg = AudioConfig(
            provider=self.config.models.speech.provider,
            model=self.config.models.speech.model,
            language=self.config.models.speech.language or "en"
        )
        self.speech_processor = SpeechProcessor(
            provider=self.audio_provider,
            config=speech_model_cfg
        )

        ocr_model_cfg = OCRConfig(
            provider=self.config.models.ocr.provider,
            model=self.config.models.ocr.model,
            max_tokens=self.config.models.ocr.max_tokens or 1024
        )
        self.ocr_processor = OCRProcessor(
            provider=self.ocr_provider,
            config=ocr_model_cfg
        )

        # 5. Initialize Reasoning & Generation Modules
        self.timeline_builder = TimelineBuilder()

        fusion_cfg = self.config.pipeline.fusion
        fusion_llm_cfg = LLMConfig(
            provider=self.config.models.llm.provider,
            model=self.config.models.llm.model,
            max_tokens=2048,
            temperature=0.1
        )
        self.fusion_engine = FusionEngine(
            llm_provider=self.llm_provider,
            llm_config=fusion_llm_cfg,
            temporal_window_seconds=fusion_cfg.temporal_window_seconds,
            min_confidence=fusion_cfg.min_confidence
        )

        self.graph_builder = SemanticGraphBuilder()

        narrative_cfg = self.config.pipeline.narrative
        narrative_llm_cfg = LLMConfig(
            provider=self.config.models.llm.provider,
            model=self.config.models.llm.model,
            max_tokens=1024,
            temperature=0.1
        )
        self.narrative_builder = NarrativeBuilder(
            llm_provider=self.llm_provider,
            llm_config=narrative_llm_cfg,
            max_events=narrative_cfg.max_events,
            max_words=narrative_cfg.max_words
        )

        generation_cfg = self.config.pipeline.generation
        gen_llm_cfg = LLMConfig(
            provider=self.config.models.llm.provider,
            model=self.config.models.llm.model,
            max_tokens=512,
            temperature=0.7
        )
        # Note prompts directory mapping
        prompts_dir = "docs/prompts"
        self.style_generator = StyleGenerator(
            llm_provider=self.llm_provider,
            llm_config=gen_llm_cfg,
            prompts_dir=prompts_dir,
            min_caption_words=generation_cfg.min_caption_words,
            max_caption_words=generation_cfg.max_caption_words
        )

        # 6. Initialize Validator and Formatter
        val_cfg = self.config.pipeline.validation
        val_llm_cfg = LLMConfig(
            provider=self.config.models.validator.provider,
            model=self.config.models.validator.model,
            max_tokens=512,
            temperature=0.1
        )
        self.validator = SemanticValidator(
            llm_provider=self.llm_provider,
            llm_config=val_llm_cfg,
            prompts_dir=prompts_dir,
            hallucination_check=val_cfg.hallucination_check,
            style_leakage_check=val_cfg.style_leakage_check,
            fact_drift_check=val_cfg.fact_drift_check
        )

        self.submission_formatter = SubmissionFormatter()

    def process_video(self, video_path: str) -> Dict[str, Caption]:
        """
        Execute the pipeline on a single video file, returning the validated captions.

        Args:
            video_path: Path to the video file.

        Returns:
            Dict mapping caption style name to Caption objects.
        """
        import time
        logger.info(f"Starting pipeline orchestration for: {video_path}")
        pipeline_start = time.time()

        # Step 1: Ingestion
        t0 = time.time()
        descriptor = self.video_loader.load_video(video_path)
        logger.info(f"Ingestion latency: {time.time() - t0:.2f}s | Format: {descriptor.format} | Duration: {descriptor.duration_seconds}s")

        # Step 2: Sampling
        t0 = time.time()
        samples = self.adaptive_sampler.sample_video(descriptor)
        logger.info(f"Sampling latency: {time.time() - t0:.2f}s | Frames sampled: {len(samples)}")

        # Step 3: Perception (Parallel / Sequential / Cached)
        t0 = time.time()
        observations: List[Observation] = []
        perception_cfg = self.config.pipeline.perception

        perception_extra_cfg = {
            "parallel": perception_cfg.parallel,
            "vision_enabled": perception_cfg.vision.enabled,
            "vision_model": self.config.models.vision.model,
            "speech_enabled": perception_cfg.speech.enabled,
            "speech_model": self.config.models.speech.model,
            "ocr_enabled": perception_cfg.ocr.enabled,
            "ocr_model": self.config.models.ocr.model,
            "sampling_frames": len(samples)
        }
        perception_key = get_cache_key(video_path, "perception", perception_extra_cfg)
        cached_obs = load_from_cache(perception_key)

        if cached_obs is not None:
            logger.info("Retrieved observations from cache.")
            observations = [Observation(**o) for o in cached_obs]
        else:
            if perception_cfg.parallel:
                logger.info(f"Executing perception layers in parallel with max_workers={perception_cfg.max_workers}...")
                futures = {}
                with ThreadPoolExecutor(max_workers=perception_cfg.max_workers) as executor:
                    if perception_cfg.vision.enabled:
                        futures[executor.submit(self.vision_processor.process, samples)] = "vision"
                    if perception_cfg.speech.enabled:
                        futures[executor.submit(self.speech_processor.process, samples, descriptor.has_audio)] = "speech"
                    if perception_cfg.ocr.enabled:
                        futures[executor.submit(self.ocr_processor.process, samples)] = "ocr"

                    try:
                        for future in as_completed(futures, timeout=perception_cfg.timeout_seconds):
                            proc_name = futures[future]
                            try:
                                result = future.result()
                                observations.extend(result)
                                logger.info(f"Perception module '{proc_name}' finished with {len(result)} observations.")
                            except Exception as e:
                                logger.error(f"Perception module '{proc_name}' failed with error: {e}")
                                raise
                    except TimeoutError as e:
                        logger.error(f"Parallel perception execution timed out after {perception_cfg.timeout_seconds}s.")
                        raise ProviderError(f"Perception processing timed out: {e}") from e
            else:
                logger.info("Executing perception layers sequentially...")
                if perception_cfg.vision.enabled:
                    observations.extend(self.vision_processor.process(samples))
                if perception_cfg.speech.enabled:
                    observations.extend(self.speech_processor.process(samples, descriptor.has_audio))
                if perception_cfg.ocr.enabled:
                    observations.extend(self.ocr_processor.process(samples))

            save_to_cache(perception_key, [o.model_dump() for o in observations])

        logger.info(f"Perception latency: {time.time() - t0:.2f}s | Total raw observations: {len(observations)}")

        # Clear large binary data from samples to release memory immediately
        for s in samples:
            s.frame_data = b"\x00"
            s.audio_segment = None

        # Step 4: Timeline Merging
        t0 = time.time()
        timeline = self.timeline_builder.build_timeline(observations, descriptor.duration_seconds)
        logger.info(f"Timeline building latency: {time.time() - t0:.2f}s")

        # Step 5: Event Fusion
        t0 = time.time()
        events = self.fusion_engine.fuse_timeline(timeline)
        logger.info(f"Fusion engine latency: {time.time() - t0:.2f}s | Fused events: {len(events)}")

        # Step 6: Semantic Graph Building
        t0 = time.time()
        graph = self.graph_builder.build_graph(events)
        logger.info(f"Graph building latency: {time.time() - t0:.2f}s | Graph nodes: {len(graph.nodes)} | Edges: {len(graph.edges)}")

        # Step 7: Narrative Builder (Cached)
        t0 = time.time()
        narrative_extra_cfg = {
            "events": [e.model_dump() for e in events],
            "model": self.config.models.llm.model
        }
        narrative_key = get_cache_key(video_path, "narrative", narrative_extra_cfg)
        cached_narrative = load_from_cache(narrative_key)

        if cached_narrative is not None:
            logger.info("Retrieved narrative from cache.")
            narrative = Narrative(**cached_narrative)
        else:
            narrative = self.narrative_builder.build_narrative(graph)
            save_to_cache(narrative_key, narrative.model_dump())

        logger.info(f"Narrative building latency: {time.time() - t0:.2f}s | Words: {len(narrative.text.split())}")

        # Step 8: Styled Caption Generation & Semantic Validation Retry Loop
        t0 = time.time()
        val_cfg = self.config.pipeline.validation
        max_attempts = val_cfg.max_retries + 1 if val_cfg.enabled and val_cfg.retry_on_failure else 1

        captions: Dict[str, Caption] = {}
        for attempt in range(1, max_attempts + 1):
            logger.info(f"Caption generation attempt {attempt}/{max_attempts}")
            captions = self.style_generator.generate_captions(narrative)

            if val_cfg.enabled:
                logger.info("Running SemanticValidator checks...")
                report = self.validator.validate(captions, narrative)
                
                # Enrich metadata with validation and input context for explainability
                for style, cap in captions.items():
                    cap.metadata["narrative"] = narrative.text
                    cap.metadata["events_used"] = [e.model_dump() for e in events]
                    cap.metadata["observations_used"] = [o.model_dump() for o in observations]
                    if style in report.per_caption:
                        cap.metadata["validation_result"] = report.per_caption[style].model_dump()

                if report.overall_pass:
                    logger.info("Semantic validation passed.")
                    break
                else:
                    logger.warning(
                        f"Semantic validation failed on attempt {attempt}. "
                        f"Hallucination count: {report.hallucination_count}, Consistency score: {report.consistency_score}"
                    )
                    if attempt < max_attempts:
                        logger.info("Retrying caption generation...")
                    else:
                        logger.error("Max validation attempts reached. Fail-closed: raising ValidationError.")
                        raise ValidationError(
                            f"Semantic validation failed with status {report.status} after {max_attempts} attempts. "
                            f"Hallucinations: {report.hallucination_count}, consistency score: {report.consistency_score}"
                        )
            else:
                # Still enrich basic narrative/event explainability when validation is disabled
                for style, cap in captions.items():
                    cap.metadata["narrative"] = narrative.text
                    cap.metadata["events_used"] = [e.model_dump() for e in events]
                    cap.metadata["observations_used"] = [o.model_dump() for o in observations]
                break

        logger.info(f"Generation/Validation stage latency: {time.time() - t0:.2f}s")
        logger.info(f"Pipeline orchestration successfully completed. Total Latency: {time.time() - pipeline_start:.2f}s")
        return captions

    def process_batch(self, video_paths: List[str], output_path: str) -> List[Dict[str, str]]:
        """
        Process a list of video files, aggregate final competition JSON results, and save.
        Implements error recovery, checkpoint progress saving, and resume capability.

        Args:
            video_paths: List of string paths to videos.
            output_path: Path to write the final submission JSON.

        Returns:
            List of formatted competition dictionaries.
        """
        import json
        checkpoint_path = output_path + ".checkpoint"
        formatted_results: List[Dict[str, str]] = []
        completed_ids = set()

        # Load existing progress from checkpoint if available
        if os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, "r", encoding="utf-8") as f:
                    checkpoint_data = json.load(f)
                    if isinstance(checkpoint_data, list):
                        formatted_results = checkpoint_data
                        completed_ids = {item["video_id"] for item in formatted_results}
                        logger.info(f"Resumed from checkpoint. Found {len(completed_ids)} already completed videos.")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint file {checkpoint_path}: {e}. Starting fresh.")

        for video_path in video_paths:
            video_id = os.path.splitext(os.path.basename(video_path))[0]
            if video_id in completed_ids:
                logger.info(f"Skipping video {video_id} (already present in checkpoint).")
                continue

            logger.info(f"Batch processing video: {video_id}")
            try:
                captions = self.process_video(video_path)
                formatted = self.submission_formatter.format_video_captions(video_id, captions)
                formatted_results.append(formatted)
                completed_ids.add(video_id)

                # Save checkpoint after each successful process
                try:
                    parent_dir = os.path.dirname(checkpoint_path)
                    if parent_dir and not os.path.exists(parent_dir):
                        os.makedirs(parent_dir, exist_ok=True)
                    with open(checkpoint_path, "w", encoding="utf-8") as f:
                        json.dump(formatted_results, f, indent=2, ensure_ascii=False)
                except Exception as ce:
                    logger.warning(f"Failed to write checkpoint update: {ce}")

            except Exception as e:
                logger.error(f"Pipeline failed to process video {video_path}: {e}. Continuing with remaining videos in batch.")

        # Save all accumulated results to the final submission file
        self.submission_formatter.write_submission(formatted_results, output_path)

        # Cleanup checkpoint if processing was successful
        if os.path.exists(checkpoint_path):
            try:
                os.remove(checkpoint_path)
                logger.info("Cleared temporary batch checkpoint file.")
            except Exception as e:
                logger.warning(f"Could not remove checkpoint file: {e}")

        return formatted_results
