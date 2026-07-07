import argparse
import os
import sys
import logging

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


def main():
    parser = argparse.ArgumentParser(
        description="Video Captioning Pipeline CLI - Ingestion to Semantic Validation"
    )
    group = parser.add_mutually_exclusive_group(required=True)
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
        help="Path to write the final competition submission JSON."
    )
    parser.add_argument(
        "--use-mocks",
        action="store_true",
        help="Force use of mock providers rather than sending live Fireworks AI API calls."
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

    # Process input files
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
