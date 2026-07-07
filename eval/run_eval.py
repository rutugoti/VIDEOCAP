"""
Evaluation harness — Phase P0.3 of docs/IMPLEMENT.md.

Purpose: produce the **baseline vector** the improvement plan's hypotheses (H1-H9)
are measured against. It instruments an LLM-call/token counter and computes every
metric that does NOT require hand-labeled ground truth. Metrics that DO require a
labeled probe set are emitted as "TBD_requires_ground_truth" rather than fabricated.

Run modes
---------
1. Synthetic (default, offline, no API key or videos needed):
       python -m eval.run_eval
   Exercises the reasoning half (narrative -> captions -> cross-style validation)
   from a canned narrative using Mock providers. Establishes the generation-side
   baseline: LLM calls, latency, style separation, cross-style contradiction rate.

2. Real videos (full pipeline; needs FIREWORKS_API_KEY + real providers):
       python -m eval.run_eval --videos eval/probe_set/videos

Output: eval/baseline.json  (freeze this BEFORE landing behavior changes).
"""

import argparse
import json
import logging
import os
import re
import statistics
import time
from typing import Dict, List, Optional

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("eval")

# --- Metrics that need a hand-labeled probe set; never fabricated here ---
GROUND_TRUTH_METRICS = {
    "hallucination_rate": "TBD_requires_ground_truth",
    "factual_consistency": "TBD_requires_ground_truth",
    "temporal_correctness": "TBD_requires_ground_truth",
}

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "to",
    "of", "in", "on", "at", "for", "with", "by", "about", "this", "that",
}


def _terms(text: str) -> set:
    cleaned = re.sub(r"[^\w\s]", "", text.lower())
    return {w for w in cleaned.split() if len(w) > 2 and w not in _STOPWORDS}


def _jaccard_distance(a: str, b: str) -> float:
    ta, tb = _terms(a), _terms(b)
    if not ta and not tb:
        return 1.0
    return 1.0 - len(ta & tb) / len(ta | tb)


class CallCounter:
    """Shared tally of provider calls and (when available) token usage."""

    def __init__(self):
        self.llm_calls = 0
        self.vision_calls = 0
        self.audio_calls = 0
        self.ocr_calls = 0
        self.tokens = 0  # stays 0 for mocks; real providers can add usage

    def as_dict(self) -> Dict[str, int]:
        return {
            "llm_calls": self.llm_calls,
            "vision_calls": self.vision_calls,
            "audio_calls": self.audio_calls,
            "ocr_calls": self.ocr_calls,
            "tokens": self.tokens if self.tokens else "TBD_requires_real_provider",
        }


class CountingLLMProvider:
    """Wraps any LLMProvider, counting generate() calls."""

    def __init__(self, inner, counter: CallCounter):
        self._inner = inner
        self._counter = counter

    def generate(self, prompt, system_prompt, config):
        self._counter.llm_calls += 1
        return self._inner.generate(prompt, system_prompt, config)


def _cross_style_contradiction_rate(captions: Dict[str, str], narrative: str) -> float:
    """Fraction of narrative fact-terms present in all-but-one caption (a contradiction)."""
    styles = list(captions.keys())
    if len(styles) < 3:
        return 0.0
    n_terms = _terms(narrative)
    if not n_terms:
        return 0.0
    cap_terms = {s: _terms(t) for s, t in captions.items()}
    contradictions = 0
    for term in n_terms:
        present = [s for s in styles if term in cap_terms[s]]
        if len(present) == len(styles) - 1:
            contradictions += 1
    return contradictions / len(n_terms)


def _mean_style_separation(captions: Dict[str, str]) -> float:
    styles = list(captions.keys())
    dists = []
    for i in range(len(styles)):
        for j in range(i + 1, len(styles)):
            dists.append(_jaccard_distance(captions[styles[i]], captions[styles[j]]))
    return statistics.mean(dists) if dists else 1.0


def run_synthetic(runs: int) -> dict:
    """Generation-side baseline using Mock providers — no API key or videos required."""
    from src.shared.providers import MockLLMProvider
    from src.shared.models import Narrative
    from src.generation.style_generator import StyleGenerator
    from src.validation.semantic_validator import SemanticValidator

    counter = CallCounter()
    llm = CountingLLMProvider(MockLLMProvider(), counter)

    narrative = Narrative(
        text="A person enters the kitchen, starts the coffee maker, and exits.",
        key_events=["enters kitchen", "starts coffee maker", "exits kitchen"],
        salience_scores={"enters kitchen": 0.5, "starts coffee maker": 0.95, "exits kitchen": 0.4},
        evidence_mapping={"enters kitchen": ["visual"], "starts coffee maker": ["visual", "audio"]},
    )

    gen = StyleGenerator(llm_provider=llm, single_pass=True)
    validator = SemanticValidator(llm_provider=llm)

    latencies: List[float] = []
    separations: List[float] = []
    contradiction_rates: List[float] = []

    for _ in range(runs):
        t0 = time.time()
        captions = gen.generate_captions(narrative)
        validator.validate(captions, narrative)
        latencies.append(time.time() - t0)

        cap_texts = {s: c.text for s, c in captions.items()}
        separations.append(_mean_style_separation(cap_texts))
        contradiction_rates.append(_cross_style_contradiction_rate(cap_texts, narrative.text))

    return {
        "mode": "synthetic",
        "runs": runs,
        "llm_calls_per_run": counter.llm_calls / runs,
        "provider_calls_total": counter.as_dict(),
        "latency_p50_s": round(statistics.median(latencies), 4),
        "latency_max_s": round(max(latencies), 4),
        "mean_style_separation": round(statistics.mean(separations), 4),
        "cross_style_contradiction_rate": round(statistics.mean(contradiction_rates), 4),
        **GROUND_TRUTH_METRICS,
    }


def run_real(videos_dir: str) -> dict:
    """Full-pipeline baseline over real videos. Requires FIREWORKS_API_KEY + providers."""
    from src.config.settings import get_config
    from src.orchestration.pipeline import PipelineOrchestrator

    video_paths = [
        os.path.join(videos_dir, f)
        for f in sorted(os.listdir(videos_dir))
        if f.lower().endswith((".mp4", ".mov", ".mkv", ".avi"))
    ]
    if not video_paths:
        raise SystemExit(f"No videos found under {videos_dir}")

    counter = CallCounter()
    config = get_config()
    orch = PipelineOrchestrator(config=config)
    orch.llm_provider = CountingLLMProvider(orch.llm_provider, counter)

    latencies, separations, contradiction_rates = [], [], []
    for vp in video_paths:
        t0 = time.time()
        captions = orch.process_video(vp)
        latencies.append(time.time() - t0)
        cap_texts = {s: c.text for s, c in captions.items()}
        separations.append(_mean_style_separation(cap_texts))
        narr = next(iter(captions.values())).metadata.get("narrative", "")
        contradiction_rates.append(_cross_style_contradiction_rate(cap_texts, narr))

    return {
        "mode": "real",
        "videos": len(video_paths),
        "llm_calls_per_video": counter.llm_calls / len(video_paths),
        "provider_calls_total": counter.as_dict(),
        "latency_p50_s": round(statistics.median(latencies), 4),
        "latency_p95_s": round(sorted(latencies)[max(0, int(0.95 * len(latencies)) - 1)], 4),
        "mean_style_separation": round(statistics.mean(separations), 4),
        "cross_style_contradiction_rate": round(statistics.mean(contradiction_rates), 4),
        **GROUND_TRUTH_METRICS,
    }


def main():
    ap = argparse.ArgumentParser(description="Track 2 evaluation / baseline harness")
    ap.add_argument("--videos", help="Directory of real videos (full pipeline).")
    ap.add_argument("--runs", type=int, default=5, help="Synthetic runs (default 5).")
    ap.add_argument("--out", default="eval/baseline.json", help="Output path.")
    args = ap.parse_args()

    result = run_real(args.videos) if args.videos else run_synthetic(args.runs)
    result["note"] = (
        "GROUND_TRUTH metrics require a hand-labeled probe set (eval/probe_set/ground_truth.jsonl); "
        "they are intentionally left TBD, not fabricated. See docs/IMPLEMENT.md P0.3."
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"\nBaseline written to {args.out}")


if __name__ == "__main__":
    main()
