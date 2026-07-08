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

        # 2. Require Providers (No mock fallbacks)
        if llm_provider is None:
            raise ValueError("llm_provider is required.")
        if vision_provider is None:
            raise ValueError("vision_provider is required.")
        if audio_provider is None:
            raise ValueError("audio_provider is required.")
        if ocr_provider is None:
            raise ValueError("ocr_provider is required.")

        self.llm_provider = llm_provider
        self.vision_provider = vision_provider
        self.audio_provider = audio_provider
        self.ocr_provider = ocr_provider

        # 3. Initialize Ingestion & Sampling Modules
        # Disable strict duration checks in loader if the configuration allows or bypass via defaults
        self.video_loader = VideoLoader(min_duration=0.0, max_duration=10000.0)
        
        sampling_cfg = self.config.pipeline.sampling
        self.adaptive_sampler = AdaptiveSampler(
            method=sampling_cfg.method,
            fps=sampling_cfg.fps,
            max_frames=sampling_cfg.max_frames,
            min_frames=sampling_cfg.min_frames,
            baseline_fps=getattr(sampling_cfg, "baseline_fps", 1.0),
            min_scene_duration=getattr(sampling_cfg, "min_scene_duration", 1.0),
            motion_threshold=getattr(sampling_cfg, "motion_threshold", 15.0),
            scene_change_threshold=getattr(sampling_cfg, "scene_change_threshold", 20.0),
            silence_threshold=getattr(sampling_cfg, "silence_threshold", -40.0)
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
            min_confidence=fusion_cfg.min_confidence,
            escalate_conflict_min_conf=self.config.pipeline.evidence_policy.escalate_conflict_min_conf
        )

        graph_cfg = self.config.pipeline.graph
        graph_llm_cfg = LLMConfig(
            provider=self.config.models.llm.provider,
            model=self.config.models.llm.model,
            max_tokens=256,
            temperature=0.1
        )
        self.graph_builder = SemanticGraphBuilder(
            llm_provider=self.llm_provider,
            llm_config=graph_llm_cfg,
            causal_edges=graph_cfg.causal_edges,
            causal_window_seconds=graph_cfg.causal_window_seconds,
        )

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
            max_words=narrative_cfg.max_words,
            consume_edges=narrative_cfg.consume_edges
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
            max_caption_words=generation_cfg.max_caption_words,
            single_pass=generation_cfg.single_pass,
            style_separation_min=generation_cfg.style_separation_min
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
            fact_drift_check=val_cfg.fact_drift_check,
            unsupported_claim_check=val_cfg.unsupported_claim_check
        )

        self.submission_formatter = SubmissionFormatter()

    def process_video(
        self,
        video_path: str,
        mode: str = "BALANCED",
        scheduler: Optional[Any] = None
    ) -> Dict[str, Caption]:
        """
        Execute the pipeline on a single video file, returning the validated captions.

        Args:
            video_path: Path to the video file.
            mode: Execution mode ('FAST', 'BALANCED', 'DEEP').
            scheduler: Optional BatchScheduler instance to check retry authorization.

        Returns:
            Dict mapping caption style name to Caption objects.
        """
        import time
        logger.info(f"Starting pipeline orchestration for: {video_path} in mode {mode}")
        pipeline_start = time.time()

        # Step 1: Ingestion
        t0 = time.time()
        descriptor = self.video_loader.load_video(video_path)
        logger.info(f"Ingestion latency: {time.time() - t0:.2f}s | Format: {descriptor.format} | Duration: {descriptor.duration_seconds}s")

        # Step 2: Sampling
        t0 = time.time()
        if mode == "FAST":
            # FAST mode: temporarily restrict sampling density to 4-5 frames
            original_max = self.adaptive_sampler.max_frames
            original_min = self.adaptive_sampler.min_frames
            self.adaptive_sampler.max_frames = 5
            self.adaptive_sampler.min_frames = 4
            samples = self.adaptive_sampler.sample_video(descriptor)
            self.adaptive_sampler.max_frames = original_max
            self.adaptive_sampler.min_frames = original_min
        elif mode == "BALANCED":
            # BALANCED mode: scale target sampling to 6-10 frames
            original_max = self.adaptive_sampler.max_frames
            original_min = self.adaptive_sampler.min_frames
            self.adaptive_sampler.max_frames = 10
            self.adaptive_sampler.min_frames = 6
            samples = self.adaptive_sampler.sample_video(descriptor)
            self.adaptive_sampler.max_frames = original_max
            self.adaptive_sampler.min_frames = original_min
        else:
            # DEEP mode: full sampling capability (up to 30 frames)
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
            "sampling_frames": len(samples),
            "mode": mode
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
                        futures[executor.submit(self.speech_processor.process, samples, descriptor.has_audio, mode)] = "speech"
                    if perception_cfg.ocr.enabled:
                        futures[executor.submit(self.ocr_processor.process, samples, mode)] = "ocr"

                    failed_modules = []
                    try:
                        for future in as_completed(futures, timeout=perception_cfg.timeout_seconds):
                            proc_name = futures[future]
                            try:
                                result = future.result()
                                observations.extend(result)
                                logger.info(f"Perception module '{proc_name}' finished with {len(result)} observations.")
                            except Exception as e:
                                logger.error(f"Perception module '{proc_name}' failed with error: {e}. Continuing with other modalities.")
                                failed_modules.append(proc_name)
                    except TimeoutError as e:
                        logger.error(f"Parallel perception execution timed out after {perception_cfg.timeout_seconds}s; using partial observations.")
                        failed_modules.append("timeout")

                    if not observations:
                        raise ProviderError(
                            f"All perception modalities failed ({', '.join(failed_modules) or 'unknown'}); "
                            f"no observations produced."
                        )
            else:
                logger.info("Executing perception layers sequentially...")
                seq_modules = []
                if perception_cfg.vision.enabled:
                    seq_modules.append(("vision", lambda: self.vision_processor.process(samples)))
                if perception_cfg.speech.enabled:
                    seq_modules.append(("speech", lambda: self.speech_processor.process(samples, descriptor.has_audio, mode)))
                if perception_cfg.ocr.enabled:
                    seq_modules.append(("ocr", lambda: self.ocr_processor.process(samples, mode)))

                failed_modules = []
                for proc_name, run in seq_modules:
                    try:
                        observations.extend(run())
                    except Exception as e:
                        logger.error(f"Perception module '{proc_name}' failed with error: {e}. Continuing with other modalities.")
                        failed_modules.append(proc_name)

                if not observations:
                    raise ProviderError(
                        f"All perception modalities failed ({', '.join(failed_modules) or 'unknown'}); "
                        f"no observations produced."
                    )

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

        # Step 5: Semantic Contract (Fusion + Graph + Narrative in a single pass)
        t0 = time.time()
        contract_extra_cfg = {
            "observations": [o.model_dump() for o in observations],
            "model": self.config.models.llm.model
        }
        contract_key = get_cache_key(video_path, "contract", contract_extra_cfg)
        cached_contract = load_from_cache(contract_key)

        if cached_contract is not None:
            logger.info("Retrieved Semantic Contract output from cache.")
            events = [Event(**e) for e in cached_contract["events"]]
            graph = SemanticGraph(**cached_contract["graph"])
            narrative = Narrative(**cached_contract["narrative"])
        else:
            if not hasattr(self, "semantic_contract_resolver"):
                from src.fusion.semantic_contract import SemanticContractResolver
                self.semantic_contract_resolver = SemanticContractResolver(llm_provider=self.llm_provider)
            
            events, graph, narrative = self.semantic_contract_resolver.resolve(timeline)
            save_to_cache(contract_key, {
                "events": [e.model_dump() for e in events],
                "graph": graph.model_dump(),
                "narrative": narrative.model_dump()
            })

        logger.info(f"Semantic Contract resolution latency: {time.time() - t0:.2f}s | Events: {len(events)}")

        # Step 6: Styled Caption Generation & Semantic Validation Retry Loop
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
                
                # Build one centralized Evidence Justification Record (EJR)
                ejr_markdown = self.build_ejr(events, observations)

                # Enrich metadata with validation and EJR context for explainability without duplicating observations
                for style, cap in captions.items():
                    cap.metadata["narrative"] = narrative.text
                    cap.metadata["ejr"] = ejr_markdown
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
                    
                    # Verify if validation retry is permitted by BatchScheduler
                    allow_retry = True
                    if scheduler is not None:
                        allow_retry = scheduler.can_retry(mode)

                    if attempt < max_attempts and allow_retry:
                        logger.info("Retrying caption generation...")
                    else:
                        if scheduler is not None and not allow_retry:
                            logger.warning("Retry budget exhausted or denied by BatchScheduler. Accepting current captions.")
                        else:
                            logger.error("Max validation attempts reached. Fail-closed: raising ValidationError.")
                            raise ValidationError(
                                f"Semantic validation failed with status {report.status} after {max_attempts} attempts. "
                                f"Hallucinations: {report.hallucination_count}, consistency score: {report.consistency_score}"
                            )
                        break
            else:
                ejr_markdown = self.build_ejr(events, observations)
                # Still enrich basic narrative/EJR explainability when validation is disabled
                for style, cap in captions.items():
                    cap.metadata["narrative"] = narrative.text
                    cap.metadata["ejr"] = ejr_markdown
                break

        logger.info(f"Generation/Validation stage latency: {time.time() - t0:.2f}s")
        logger.info(f"Pipeline orchestration successfully completed. Total Latency: {time.time() - pipeline_start:.2f}s")
        return captions

    def build_ejr(self, events: List[Event], observations: List[Observation]) -> str:
        """
        Build a centralized Evidence Justification Record (EJR) as a markdown table.
        """
        lines = [
            "| Event Description | Time Window | Evidence Sources | Confidence | Conflict Resolution Summary |",
            "| :--- | :--- | :--- | :--- | :--- |"
        ]
        
        for e in events:
            # Gather conflict resolution notes for observations in this event's window
            resolution_notes = []
            
            # Find any conflicts in self.fusion_engine.all_conflicts
            conflicts = getattr(self.fusion_engine, "all_conflicts", [])
            for c in conflicts:
                is_relevant = False
                c_type = c.get("type")
                if c_type in ("ocr_downweighted", "audio_only_downweighted"):
                    obs_content = c.get("observation", "")
                    for o in observations:
                        if o.content == obs_content and e.timestamp_start - 0.5 <= o.timestamp <= e.timestamp_end + 0.5:
                            is_relevant = True
                            break
                elif c_type in ("modality_agreement", "modality_contradiction"):
                    v_content = c.get("visual", "")
                    a_content = c.get("audio", "")
                    for o in observations:
                        if o.content in (v_content, a_content) and e.timestamp_start - 0.5 <= o.timestamp <= e.timestamp_end + 0.5:
                            is_relevant = True
                            break
                
                if is_relevant:
                    res_str = c.get("resolution", "")
                    if c_type == "modality_contradiction" and c.get("escalate_to_llm"):
                        verdict = c.get("gemma_verdict")
                        if verdict:
                            res_str += f" (Gemma resolved: {verdict.get('resolution')})"
                    resolution_notes.append(f"{c_type}: {res_str}")
            
            notes_str = "; ".join(set(resolution_notes)) if resolution_notes else "No conflicts"
            sources = ", ".join(e.evidence_sources)
            time_win = f"{e.timestamp_start:.2f}s - {e.timestamp_end:.2f}s"
            lines.append(f"| {e.description} | {time_win} | {sources} | {e.confidence:.2f} | {notes_str} |")
            
        return "\n".join(lines)

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
        import time
        from src.orchestration.scheduler import BatchScheduler

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

        # Initialize Batch Scheduler
        scheduler = BatchScheduler()
        scheduler.start_batch(total_videos=len(video_paths))
        # Account for resumed completed videos in scheduler
        for _ in completed_ids:
            scheduler.register_completed(elapsed=30.0, api_calls=2, tokens=200)

        for video_path in video_paths:
            video_id = os.path.splitext(os.path.basename(video_path))[0]
            if video_id in completed_ids:
                logger.info(f"Skipping video {video_id} (already present in checkpoint).")
                continue

            # Decide mode based on remaining runtime
            mode = scheduler.decide_execution_mode()
            logger.info(f"Batch processing video: {video_id} using mode {mode}")
            
            t_start = time.time()
            try:
                import inspect
                sig = inspect.signature(self.process_video)
                kwargs = {}
                if "mode" in sig.parameters:
                    kwargs["mode"] = mode
                if "scheduler" in sig.parameters:
                    kwargs["scheduler"] = scheduler
                captions = self.process_video(video_path, **kwargs)
                formatted = self.submission_formatter.format_video_captions(video_id, captions)
                formatted_results.append(formatted)
                completed_ids.add(video_id)

                elapsed = time.time() - t_start
                # Fallback telemetry registration
                api_calls = getattr(self.llm_provider, "api_calls_count", 2)
                tokens = getattr(self.llm_provider, "tokens_count", 200)
                scheduler.register_completed(elapsed, api_calls, tokens)

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
                scheduler.register_failure()

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
