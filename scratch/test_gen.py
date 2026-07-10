import logging
import os
import sys

sys.path.insert(0, os.path.abspath("."))

from src.config.settings import get_config
from src.orchestration.pipeline import PipelineOrchestrator
from src.providers.factory import ProviderFactory
from src.shared.providers import (
    LegacyVisionProviderAdapter,
    LegacyAudioProviderAdapter,
    LegacyOCRProviderAdapter,
    LegacyLLMProviderAdapter,
)

# Setup basic logging to stdout
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

config = get_config()

vision_provs = config.models.vision.provider
speech_provs = config.models.speech.provider
ocr_provs = config.models.ocr.provider
llm_provs = config.models.llm.provider

raw_vision = ProviderFactory.get_vision(vision_provs)
raw_speech = ProviderFactory.get_speech(speech_provs)
raw_ocr = ProviderFactory.get_ocr(ocr_provs)
raw_llm = ProviderFactory.get_llm(llm_provs)

vision = LegacyVisionProviderAdapter(raw_vision)
audio = LegacyAudioProviderAdapter(raw_speech)
ocr = LegacyOCRProviderAdapter(raw_ocr)
llm = LegacyLLMProviderAdapter(raw_llm)

orchestrator = PipelineOrchestrator(
    config=config,
    llm_provider=llm,
    vision_provider=vision,
    audio_provider=audio,
    ocr_provider=ocr
)

video_path = "v1.mp4"
logger = logging.getLogger("test_gen")
logger.info(f"Running pipeline on {video_path}...")

captions = orchestrator.process_video(video_path)

print("\n--- GENERATED CAPTIONS ---")
for style, cap in captions.items():
    print(f"\nStyle: {style} (Word Count: {cap.word_count})")
    print(f"Text: {cap.text}")
    print(f"Metadata: {cap.metadata}")
