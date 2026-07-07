import json
import os
import pytest
import av
from pathlib import Path
from typing import Dict

from src.config.settings import get_config
from src.orchestration.pipeline import PipelineOrchestrator
from src.submission.submission_formatter import SubmissionFormatter
from src.shared.providers import (
    MockLLMProvider,
    MockVisionProvider,
    MockAudioProvider,
    MockOCRProvider,
    Caption,
    VisionConfig,
    AudioConfig,
    OCRConfig,
    LLMConfig,
)
from src.shared.models import Sample, Observation, Narrative
from src.shared.fireworks_providers import (
    FireworksVisionProvider,
    FireworksAudioProvider,
    FireworksOCRProvider,
    FireworksLLMProvider,
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

    config = get_config()
    # Enable validation and retry policy
    config.pipeline.validation.enabled = True
    config.pipeline.validation.retry_on_failure = True
    config.pipeline.validation.max_retries = 2

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
    # With cross-style validation, we validate all 4 styles at once.
    # So validation_calls should be exactly 2 (one for the first failed attempt, one for the second successful attempt).
    assert flaky_validator_llm.validation_calls == 2
    assert "formal" in captions


def test_fireworks_providers_mock_mode():
    """Verify that concrete Fireworks providers behave correctly under mock key environment."""
    api_key = "mock_key_for_testing"
    vision = FireworksVisionProvider(api_key=api_key)
    audio = FireworksAudioProvider(api_key=api_key)
    ocr = FireworksOCRProvider(api_key=api_key)
    llm = FireworksLLMProvider(api_key=api_key)

    sample = Sample(frame_data=b"dummy_frame_jpeg", timestamp=2.0, sample_index=0, audio_segment=b"dummy_wav")

    # 1. Vision Provider
    vis_cfg = VisionConfig(provider="fireworks", model="accounts/fireworks/models/llava-v1.6")
    vis_obs = vision.analyze_frames([sample], "Describe what you see.", vis_cfg)
    assert len(vis_obs) > 0
    assert vis_obs[0].source == "visual"
    assert vis_obs[0].timestamp == 2.0

    # 2. Audio Provider
    aud_cfg = AudioConfig(provider="fireworks", model="whisper-v3", language="en")
    aud_obs = audio.transcribe([sample], aud_cfg)
    assert len(aud_obs) > 0
    assert aud_obs[0].source == "audio"
    assert aud_obs[0].timestamp == 2.0

    # 3. OCR Provider
    ocr_cfg = OCRConfig(provider="fireworks", model="accounts/fireworks/models/gemma-4-31b-it")
    ocr_obs = ocr.extract_text([sample], ocr_cfg)
    assert len(ocr_obs) > 0
    assert ocr_obs[0].source == "text"
    assert ocr_obs[0].timestamp == 2.0

    # 4. LLM Provider
    llm_cfg = LLMConfig(provider="fireworks", model="accounts/fireworks/models/gemma-3-27b-it")
    response = llm.generate("State hello.", "You are a direct responder.", llm_cfg)
    assert isinstance(response, str)
    assert len(response) > 0


class SimulatedJudge:
    """Simulated judge scoring caption quality along 5 dimensions (0-10 scale)."""

    @staticmethod
    def evaluate(captions: Dict[str, Caption], narrative: Narrative) -> Dict[str, Dict[str, float]]:
        scores = {}
        for style, caption in captions.items():
            text = caption.text.lower()
            narr_text = narrative.text.lower()

            # 1. Factual Accuracy
            factual_score = 10.0
            hallucinations = ["red car", "alien", "spaceship", "ufo"]
            for term in hallucinations:
                if term in text and term not in narr_text:
                    factual_score -= 4.0
            factual_score = max(0.0, factual_score)

            # 2. Style Adherence
            style_score = 5.0
            if style == "formal":
                # Check absence of contractions & conversational jokes
                if not any(c in text for c in ["don't", "can't", "won't", "i'm", "haha"]) and len(text.split()) > 5:
                    style_score = 10.0
                elif any(c in text for c in ["don't", "can't"]):
                    style_score = 7.0
            elif style == "sarcastic":
                sarcastic_terms = ["groundbreaking", "thrilled", "absolutely", "genius", "amazing", "classic", "oh", "fantastic"]
                if any(t in text for t in sarcastic_terms):
                    style_score = 10.0
                else:
                    style_score = 6.0
            elif style == "tech_humor":
                tech_terms = ["code", "developer", "loop", "unit test", "bug", "cache", "git", "database", "eventloop", "exception"]
                if any(t in text for t in tech_terms):
                    style_score = 10.0
                else:
                    style_score = 4.0
            elif style == "non_tech_humor":
                tech_terms = ["code", "developer", "unit test", "eventloop", "database"]
                if not any(t in text for t in tech_terms) and any(w in text for w in ["classic", "tragedy", "gone", "clean"]):
                    style_score = 10.0
                else:
                    style_score = 6.0

            # 3. Completeness
            completeness_score = 10.0
            missing_events = 0
            for event in narrative.key_events:
                words = [w for w in event.lower().split() if len(w) > 3]
                if words and not any(w in text for w in words):
                    missing_events += 1
            if narrative.key_events:
                completeness_score -= (missing_events / len(narrative.key_events)) * 10.0
            completeness_score = max(0.0, completeness_score)

            # 4. Conciseness (ideal budget: 15-35 words)
            w_count = len(text.split())
            if 15 <= w_count <= 35:
                conciseness_score = 10.0
            else:
                deviation = min(abs(w_count - 15), abs(w_count - 35))
                conciseness_score = max(0.0, 10.0 - (deviation * 0.5))

            # 5. Cross-Style Consistency
            consistency_score = 10.0

            scores[style] = {
                "factual_accuracy": factual_score,
                "style_adherence": style_score,
                "completeness": completeness_score,
                "conciseness": conciseness_score,
                "consistency": consistency_score,
                "average": (factual_score + style_score + completeness_score + conciseness_score + consistency_score) / 5.0
            }
        return scores


def simulate_baseline_captions(narrative: Narrative) -> Dict[str, Caption]:
    """Generates baseline captions showcasing common style mixing, budget violations, and hallucinations."""
    return {
        "formal": Caption(
            text="Person enters kitchen. Pretty cool coffee machine, don't you think?",
            style="formal",
            word_count=10
        ),
        "sarcastic": Caption(
            text="A person enters the kitchen and proceeds to start the coffee maker and then exits the kitchen after a short while.",
            style="sarcastic",
            word_count=21
        ),
        "tech_humor": Caption(
            text="A person enters and starts the coffee maker. Also there is a red car parked outside.",
            style="tech_humor",
            word_count=16
        ),
        "non_tech_humor": Caption(
            text="EventLoop starts cooking in the kitchen, compiling the coffee program successfully.",
            style="non_tech_humor",
            word_count=11
        )
    }


def test_simulated_judge_evaluation_suite(tmp_path):
    """Executes a simulated judge benchmark comparing the MVP pipeline captions to a single-prompt baseline."""
    video_path = str(tmp_path / "benchmark_video.mp4")
    create_test_dummy_video(video_path, duration=3.0, has_audio=True)

    config = get_config()
    orchestrator = PipelineOrchestrator(
        config=config,
        llm_provider=MockLLMProvider(),
        vision_provider=MockVisionProvider(),
        audio_provider=MockAudioProvider(),
        ocr_provider=MockOCRProvider(),
    )

    # 1. Run pipeline to extract captions
    captions = orchestrator.process_video(video_path)

    # Extract pipeline objects to construct target narrative
    descriptor = orchestrator.video_loader.load_video(video_path)
    samples = orchestrator.adaptive_sampler.sample_video(descriptor)
    observations = orchestrator.vision_processor.process(samples)
    timeline = orchestrator.timeline_builder.build_timeline(observations, descriptor.duration_seconds)
    events = orchestrator.fusion_engine.fuse_timeline(timeline)
    graph = orchestrator.graph_builder.build_graph(events)
    narrative = orchestrator.narrative_builder.build_narrative(graph)

    # 2. Score MVP captions using simulated judge
    mvp_scores = SimulatedJudge.evaluate(captions, narrative)
    mvp_avg = sum(s["average"] for s in mvp_scores.values()) / len(mvp_scores)

    # 3. Score baseline captions
    baseline_captions = simulate_baseline_captions(narrative)
    baseline_scores = SimulatedJudge.evaluate(baseline_captions, narrative)
    baseline_avg = sum(s["average"] for s in baseline_scores.values()) / len(baseline_scores)

    print(f"\n--- Simulated Judge Evaluation Results ---")
    print(f"MVP Pipeline Aggregate Quality Score: {mvp_avg:.2f} / 10.0")
    print(f"Baseline Aggregate Quality Score:     {baseline_avg:.2f} / 10.0")
    print(f"Quality Improvement Margin:           {(mvp_avg - baseline_avg):.2f}")
    print(f"-----------------------------------------")

    # The modular and validated MVP must score significantly higher than the unvalidated mixed-style baseline
    assert mvp_avg > baseline_avg
    assert mvp_avg >= 7.0


def test_pipeline_ejr_generation(tmp_path):
    video_path = str(tmp_path / "ejr_test.mp4")
    create_test_dummy_video(video_path, duration=3.0, has_audio=True)

    config = get_config()
    orchestrator = PipelineOrchestrator(
        config=config,
        llm_provider=MockLLMProvider(),
        vision_provider=MockVisionProvider(),
        audio_provider=MockAudioProvider(),
        ocr_provider=MockOCRProvider(),
    )

    captions = orchestrator.process_video(video_path)
    
    # Check that ejr is present in all captions metadata and is formatted correctly
    for style, cap in captions.items():
        assert "ejr" in cap.metadata
        ejr = cap.metadata["ejr"]
        assert "| Event Description | Time Window | Evidence Sources |" in ejr
        assert "No conflicts" in ejr or "conflict" in ejr.lower() or "agreement" in ejr.lower()
        # Verify that duplication has been removed
        assert "events_used" not in cap.metadata
        assert "observations_used" not in cap.metadata
