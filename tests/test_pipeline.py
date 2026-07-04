import json
import os
import pytest
import av
from pathlib import Path

from src.config.settings import get_config
from src.orchestration.pipeline import PipelineOrchestrator
from src.submission.submission_formatter import SubmissionFormatter
from src.shared.providers import (
    MockLLMProvider,
    MockVisionProvider,
    MockAudioProvider,
    MockOCRProvider,
    Caption,
)


def create_test_dummy_video(
    path: str,
    duration: float = 5.0,
    fps: int = 10,
    width: int = 160,
    height: int = 120,
    has_audio: bool = True
) -> None:
    """Helper to generate a real, short dummy video using PyAV for testing the orchestrator."""
    with av.open(path, mode="w") as container:
        video_stream = container.add_stream("mpeg4", rate=fps)
        video_stream.width = width
        video_stream.height = height
        video_stream.pix_fmt = "yuv420p"

        if has_audio:
            audio_stream = container.add_stream("aac", rate=16000)
            audio_frame = av.AudioFrame(format='s16', layout='mono', samples=1024)
            audio_frame.sample_rate = 16000
            audio_frame.planes[0].update(b'\x00' * 2048)
            for packet in audio_stream.encode(audio_frame):
                container.mux(packet)
            for packet in audio_stream.encode():
                container.mux(packet)

        total_frames = int(duration * fps)
        for i in range(total_frames):
            frame = av.VideoFrame(width, height, "yuv420p")
            for packet in video_stream.encode(frame):
                container.mux(packet)

        for packet in video_stream.encode():
            container.mux(packet)


def test_submission_formatter_style_mapping():
    formatter = SubmissionFormatter()
    captions = {
        "formal": Caption(text="This is formal.", style="formal", word_count=3),
        "sarcastic": Caption(text="Oh, great.", style="sarcastic", word_count=2),
        "tech_humor": Caption(text="Code resolved.", style="tech_humor", word_count=2),
        "non_tech_humor": Caption(text="Hilarious joke.", style="non_tech_humor", word_count=2),
    }

    formatted = formatter.format_video_captions("test_id", captions)
    assert formatted["video_id"] == "test_id"
    assert formatted["formal"] == "This is formal."
    assert formatted["sarcastic"] == "Oh, great."
    assert formatted["humorous_tech"] == "Code resolved."
    assert formatted["humorous_non_tech"] == "Hilarious joke."


def test_pipeline_single_video_sequential(tmp_path):
    video_path = str(tmp_path / "test_video.mp4")
    create_test_dummy_video(video_path, duration=3.0, has_audio=True)

    config = get_config()
    # Force sequential for this test
    config.pipeline.perception.parallel = False

    orchestrator = PipelineOrchestrator(
        config=config,
        llm_provider=MockLLMProvider(),
        vision_provider=MockVisionProvider(),
        audio_provider=MockAudioProvider(),
        ocr_provider=MockOCRProvider(),
    )

    captions = orchestrator.process_video(video_path)
    assert isinstance(captions, dict)
    assert "formal" in captions
    assert "sarcastic" in captions
    assert "tech_humor" in captions
    assert "non_tech_humor" in captions
    assert all(isinstance(c, Caption) for c in captions.values())


def test_pipeline_single_video_parallel(tmp_path):
    video_path = str(tmp_path / "test_video_parallel.mp4")
    create_test_dummy_video(video_path, duration=3.0, has_audio=True)

    config = get_config()
    # Force parallel execution of perception layers
    config.pipeline.perception.parallel = True
    config.pipeline.perception.timeout_seconds = 5

    orchestrator = PipelineOrchestrator(
        config=config,
        llm_provider=MockLLMProvider(),
        vision_provider=MockVisionProvider(),
        audio_provider=MockAudioProvider(),
        ocr_provider=MockOCRProvider(),
    )

    captions = orchestrator.process_video(video_path)
    assert "formal" in captions
    assert all(isinstance(c, Caption) for c in captions.values())


def test_pipeline_batch_processing(tmp_path):
    vid1 = str(tmp_path / "vid1.mp4")
    vid2 = str(tmp_path / "vid2.mp4")
    create_test_dummy_video(vid1, duration=2.0, has_audio=False)
    create_test_dummy_video(vid2, duration=2.0, has_audio=False)

    output_json = str(tmp_path / "submission.json")

    config = get_config()
    orchestrator = PipelineOrchestrator(
        config=config,
        llm_provider=MockLLMProvider(),
        vision_provider=MockVisionProvider(),
        audio_provider=MockAudioProvider(),
        ocr_provider=MockOCRProvider(),
    )

    results = orchestrator.process_batch([vid1, vid2], output_json)
    
    assert len(results) == 2
    assert results[0]["video_id"] == "vid1"
    assert results[1]["video_id"] == "vid2"
    assert "formal" in results[0]
    assert "humorous_tech" in results[1]

    # Verify JSON file structure
    assert os.path.exists(output_json)
    with open(output_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["video_id"] == "vid1"


def test_pipeline_validation_retry_flow(tmp_path):
    video_path = str(tmp_path / "retry_test.mp4")
    create_test_dummy_video(video_path, duration=3.0, has_audio=False)

    class FlakyMockLLMProvider(MockLLMProvider):
        def __init__(self):
            super().__init__()
            self.attempt_count = 0

        def generate(self, prompt, system_prompt, config) -> str:
            # Let's count how many times generator system prompt is called
            if "technical writer" in system_prompt.lower() or "formal" in system_prompt.lower() or "sarcastic" in system_prompt.lower() or "funny" in system_prompt.lower():
                self.attempt_count += 1
            return super().generate(prompt, system_prompt, config)

    flaky_llm = FlakyMockLLMProvider()

    config = get_config()
    # Enable validation and retry policy
    config.pipeline.validation.enabled = True
    config.pipeline.validation.retry_on_failure = True
    config.pipeline.validation.max_retries = 2

    # Instruct validator to mock fail (since we check for keyword "hallucinate" or similar, 
    # but since our MockLLMProvider validator returns overall_pass=True for clean captions,
    # let's inject a hallucination into the first generated caption via a small provider override,
    # or let's override validate in a custom Validator).
    # Even simpler: since we have the base validator, we can just let it run.
    # To test that the retry loop actually occurs when validation fails:
    # Let's override the `generate` function to return a failing validation response on the first validation call, and pass on the second!
    class FlakyValidatorLLM(MockLLMProvider):
        def __init__(self):
            super().__init__()
            self.validation_calls = 0
            self.generator_calls = 0

        def generate(self, prompt, system_prompt, config) -> str:
            sys_lower = system_prompt.lower()
            if "fact-checker" in sys_lower or "validator" in sys_lower:
                self.validation_calls += 1
                if self.validation_calls == 1:
                    # Return failure on first attempt
                    return """{
                        "hallucinations": ["caption mentions a red car"],
                        "missing_facts": [],
                        "fact_drift": [],
                        "style_match": true,
                        "overall_pass": false
                    }"""
                else:
                    # Return success on subsequent attempts
                    return """{
                        "hallucinations": [],
                        "missing_facts": [],
                        "fact_drift": [],
                        "style_match": true,
                        "overall_pass": true
                    }"""
            elif "formal" in sys_lower or "sarcastic" in sys_lower or "tech_humor" in sys_lower or "non_tech_humor" in sys_lower or "funny" in sys_lower:
                self.generator_calls += 1

            return super().generate(prompt, system_prompt, config)

    flaky_validator_llm = FlakyValidatorLLM()
    orchestrator = PipelineOrchestrator(
        config=config,
        llm_provider=flaky_validator_llm,
        vision_provider=MockVisionProvider(),
        audio_provider=MockAudioProvider(),
        ocr_provider=MockOCRProvider(),
    )

    captions = orchestrator.process_video(video_path)
    
    # Assert validation retried!
    # Generator should have been called twice for each style because the first validation failed!
    # With 4 styles, generator_calls should be 4 (first attempt) + 4 (second attempt) = 8
    assert flaky_validator_llm.validation_calls == 8 # 4 styles * 2 attempts
    assert "formal" in captions
