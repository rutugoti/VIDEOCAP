import argparse
import os
import sys
import logging
import json
import urllib.request
import shutil
import tempfile

from src.config.settings import get_config
from src.orchestration.pipeline import PipelineOrchestrator
from src.providers.factory import ProviderFactory
from src.shared.providers import (
    LegacyVisionProviderAdapter,
    LegacyAudioProviderAdapter,
    LegacyOCRProviderAdapter,
    LegacyLLMProviderAdapter,
)

logger = logging.getLogger("video_captioner_cli")


def download_video(url: str) -> str:
    """Download video from URL streaming to a temporary file on disk."""
    temp_fd, temp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(temp_fd)
    
    logger.info(f"Downloading video from {url} to {temp_path}...")
    try:
        req = urllib.request.Request(
            url, 
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with urllib.request.urlopen(req, timeout=120) as response, open(temp_path, "wb") as out_file:
            shutil.copyfileobj(response, out_file)
        logger.info(f"Successfully downloaded to {temp_path}")
        return temp_path
    except Exception as e:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        raise IOError(f"Failed to download video from {url}: {e}")


def run_tasks_json_pipeline(input_path: str, output_path: str, orchestrator: PipelineOrchestrator):
    """Execution loop processing /input/tasks.json and producing /output/results.json."""
    logger.info(f"Running in submission mode. Reading tasks from: {input_path}")
    
    if not os.path.exists(input_path):
        logger.error(f"Input file not found at: {input_path}")
        print(f"Error: Input tasks file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
        
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            tasks = json.load(f)
    except Exception as e:
        logger.error(f"Failed to parse tasks JSON: {e}")
        print(f"Error: Failed to parse tasks JSON from {input_path}: {e}", file=sys.stderr)
        sys.exit(1)
        
    if not isinstance(tasks, list):
        logger.error("Tasks JSON is not a list")
        print("Error: Tasks JSON must be a list/array of tasks", file=sys.stderr)
        sys.exit(1)
        
    results = []
    checkpoint_path = output_path + ".checkpoint"
    completed_task_ids = set()
    
    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                checkpoint_data = json.load(f)
                if isinstance(checkpoint_data, list):
                    results = checkpoint_data
                    completed_task_ids = {item["task_id"] for item in results}
                    logger.info(f"Resumed from checkpoint. Found {len(completed_task_ids)} already completed tasks.")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint file {checkpoint_path}: {e}. Starting fresh.")

    for task in tasks:
        task_id = task.get("task_id")
        video_url = task.get("video_url")
        requested_styles = task.get("styles", ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"])
        
        if not task_id or not video_url:
            logger.warning(f"Skipping invalid task: {task}")
            continue
            
        if task_id in completed_task_ids:
            logger.info(f"Skipping task {task_id} (already completed).")
            continue
            
        logger.info(f"Processing task {task_id} with video: {video_url}")
        
        temp_video_path = None
        try:
            # 1. Download
            temp_video_path = download_video(video_url)
            
            # 2. Run pipeline
            captions_generated = orchestrator.process_video(temp_video_path)
            
            # 3. Format output
            captions_out = {}
            for req_style in requested_styles:
                # Map to internal style name
                canonical_style = req_style
                if req_style == "humorous_tech":
                    canonical_style = "tech_humor"
                elif req_style == "humorous_non_tech":
                    canonical_style = "non_tech_humor"
                    
                cap_obj = captions_generated.get(canonical_style)
                captions_out[req_style] = cap_obj.text if cap_obj else ""
                
            results.append({
                "task_id": task_id,
                "captions": captions_out
            })
            completed_task_ids.add(task_id)
            
            # Save checkpoint
            try:
                parent_dir = os.path.dirname(checkpoint_path)
                if parent_dir and not os.path.exists(parent_dir):
                    os.makedirs(parent_dir, exist_ok=True)
                with open(checkpoint_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
            except Exception as ce:
                logger.warning(f"Failed to write checkpoint update: {ce}")
                
        except Exception as e:
            logger.error(f"Failed to process task {task_id}: {e}")
            # Do not crash the entire batch run, continue with other tasks
        finally:
            # 4. Clean up downloaded video file immediately
            if temp_video_path and os.path.exists(temp_video_path):
                try:
                    os.remove(temp_video_path)
                    logger.info(f"Cleaned up temporary video: {temp_video_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete temp video {temp_video_path}: {e}")
                    
    # Write final results
    try:
        parent_dir = os.path.dirname(output_path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Successfully wrote {len(results)} task results to: {output_path}")
        print(f"Pipeline executed successfully. Processed {len(results)} tasks.")
        print(f"Final output written to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to write results file to {output_path}: {e}")
        print(f"Error: Failed to write results: {e}", file=sys.stderr)
        sys.exit(1)
        
    # Cleanup checkpoint
    if os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
        except Exception as e:
            logger.warning(f"Could not remove checkpoint file: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Video Captioning Pipeline CLI - Ingestion to Semantic Validation"
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--video",
        type=str,
        help="Path to a single video file to process."
    )
    group.add_argument(
        "--input-dir",
        type=str,
        help="Path to a directory containing video files to process in batch."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/submission.json",
        help="Path to write the final competition submission JSON (when running in video mode)."
    )
    parser.add_argument(
        "--use-mocks",
        action="store_true",
        help="Force use of mock providers rather than sending live API calls."
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = get_config()
    except ValueError as e:
        # If API key is missing and they didn't ask for mocks, report failure
        if not args.use_mocks:
            print(f"Configuration Error: {e}", file=sys.stderr)
            print("To run in mock/offline mode, please set '--use-mocks'.", file=sys.stderr)
            sys.exit(1)
        # Mock key placeholder for config validator to load settings
        os.environ["FIREWORKS_API_KEY"] = "mock_key_for_testing"
        config = get_config()

    # Setup appropriate providers
    if args.use_mocks:
        vision_provs = "mock"
        speech_provs = "mock"
        ocr_provs = "mock"
        llm_provs = "mock"
    else:
        vision_provs = config.models.vision.provider
        speech_provs = config.models.speech.provider
        ocr_provs = config.models.ocr.provider
        llm_provs = config.models.llm.provider

    print(f"Initializing providers via Factory...")
    raw_vision = ProviderFactory.get_vision(vision_provs)
    raw_speech = ProviderFactory.get_speech(speech_provs)
    raw_ocr = ProviderFactory.get_ocr(ocr_provs)
    raw_llm = ProviderFactory.get_llm(llm_provs)

    # Wrap in compatibility adapters for legacy pipeline code
    vision = LegacyVisionProviderAdapter(raw_vision)
    audio = LegacyAudioProviderAdapter(raw_speech)
    ocr = LegacyOCRProviderAdapter(raw_ocr)
    llm = LegacyLLMProviderAdapter(raw_llm)

    # Initialize the orchestrator
    orchestrator = PipelineOrchestrator(
        config=config,
        llm_provider=llm,
        vision_provider=vision,
        audio_provider=audio,
        ocr_provider=ocr
    )

    # If both --video and --input-dir are omitted, run the submission tasks.json flow
    if not args.video and not args.input-dir:
        input_path = os.environ.get("INPUT_PATH", "/input/tasks.json")
        output_path = os.environ.get("OUTPUT_PATH", "/output/results.json")
        run_tasks_json_pipeline(input_path, output_path, orchestrator)
        return

    # Otherwise, process input files in standard CLI mode
    video_paths = []
    if args.video:
        if not os.path.exists(args.video):
            print(f"Error: Video file not found: {args.video}", file=sys.stderr)
            sys.exit(1)
        video_paths.append(args.video)
    elif args.input_dir:
        if not os.path.isdir(args.input_dir):
            print(f"Error: Input directory not found: {args.input_dir}", file=sys.stderr)
            sys.exit(1)
        for filename in os.listdir(args.input_dir):
            if filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                video_paths.append(os.path.join(args.input_dir, filename))
        if not video_paths:
            print(f"Error: No supported video files found in: {args.input_dir}", file=sys.stderr)
            sys.exit(1)

    print(f"Found {len(video_paths)} video file(s) to process.")
    
    try:
        # Run processing batch pipeline
        results = orchestrator.process_batch(video_paths, args.output)
        print(f"Pipeline executed successfully. Processed {len(results)} videos.")
        print(f"Final output written to: {args.output}")
    except Exception as e:
        print(f"Pipeline execution failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
