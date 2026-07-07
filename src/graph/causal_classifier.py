"""
Causal Edge Classifier — Phase P3.1

Generates candidate event pairs within a bounded temporal window,
then classifies each candidate via Gemma into:
  causal | enables | temporal | unrelated

Only `causal` and `enables` classifications produce graph edges.
The classifier is gated by config `graph.causal_edges` (default false).
"""

import json
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.shared.models import Event, GraphEdge
from src.shared.providers import LLMProvider, LLMConfig

logger = logging.getLogger(__name__)


@dataclass
class CausalCandidate:
    """An ordered pair of events considered for causal classification."""
    source_idx: int
    target_idx: int
    source_event: Event
    target_event: Event


@dataclass
class CausalVerdict:
    """Result of classifying a single candidate pair."""
    source_idx: int
    target_idx: int
    relationship: str  # causal | enables | temporal | unrelated
    confidence: float
    rationale: str


def generate_candidates(
    events: List[Event],
    causal_window_seconds: float
) -> List[CausalCandidate]:
    """
    Generate bounded candidate pairs for causal classification.

    Only pairs (A, B) where A precedes B and B.timestamp_start - A.timestamp_end <= causal_window
    (or intervals overlap) are emitted. This avoids N² explosion for long videos.
    """
    candidates: List[CausalCandidate] = []
    n = len(events)

    for i in range(n):
        a = events[i]
        for j in range(i + 1, n):
            b = events[j]
            # Check if b starts within the causal window after a ends,
            # or if intervals overlap (b starts before a ends)
            gap = b.timestamp_start - a.timestamp_end
            if gap <= causal_window_seconds:
                candidates.append(CausalCandidate(
                    source_idx=i,
                    target_idx=j,
                    source_event=a,
                    target_event=b
                ))
            else:
                # Events are sorted chronologically; once the gap exceeds
                # the window, all subsequent j values will also exceed it.
                break

    logger.info(
        f"CausalClassifier: Generated {len(candidates)} candidate pairs "
        f"from {n} events (window={causal_window_seconds}s)"
    )
    return candidates


def classify_candidates(
    candidates: List[CausalCandidate],
    llm_provider: LLMProvider,
    llm_config: LLMConfig
) -> List[CausalVerdict]:
    """
    Classify each candidate pair using Gemma.

    Returns a CausalVerdict for each candidate.
    """
    verdicts: List[CausalVerdict] = []

    for cand in candidates:
        a_desc = cand.source_event.description
        b_desc = cand.target_event.description

        system_prompt = (
            "You are a causal relationship classifier for video events. "
            "Given two sequential events from a video, classify their relationship as one of:\n"
            "- causal: Event A directly caused Event B\n"
            "- enables: Event A created conditions that made Event B possible\n"
            "- temporal: Events are only related by time sequence\n"
            "- unrelated: Events have no meaningful connection\n\n"
            "Respond with a valid JSON object:\n"
            '{"relationship": "causal|enables|temporal|unrelated", '
            '"confidence": 0.0-1.0, "rationale": "≤12-word explanation"}'
        )

        prompt = (
            f'Event A (at {cand.source_event.timestamp_start:.1f}s): "{a_desc}"\n'
            f'Event B (at {cand.target_event.timestamp_start:.1f}s): "{b_desc}"\n\n'
            "Classify the relationship between Event A and Event B."
        )

        try:
            raw = llm_provider.generate(prompt, system_prompt, llm_config)
            parsed = _parse_verdict(raw, cand)
            verdicts.append(parsed)
        except Exception as e:
            logger.warning(
                f"CausalClassifier: LLM call failed for pair "
                f"({cand.source_idx}, {cand.target_idx}): {e}. Falling back to 'temporal'."
            )
            verdicts.append(CausalVerdict(
                source_idx=cand.source_idx,
                target_idx=cand.target_idx,
                relationship="temporal",
                confidence=0.5,
                rationale="LLM classification failed; default temporal."
            ))

    return verdicts


def _parse_verdict(raw: str, cand: CausalCandidate) -> CausalVerdict:
    """Parse the JSON verdict from the LLM response."""
    try:
        data = json.loads(raw.strip())
        rel = data.get("relationship", "temporal")
        if rel not in ("causal", "enables", "temporal", "unrelated"):
            rel = "temporal"
        conf = float(data.get("confidence", 0.5))
        conf = max(0.0, min(1.0, conf))
        rationale = str(data.get("rationale", ""))[:100]
    except (json.JSONDecodeError, ValueError, TypeError):
        logger.warning(
            f"CausalClassifier: Failed to parse LLM response for pair "
            f"({cand.source_idx}, {cand.target_idx}). Raw: {raw[:200]}"
        )
        rel = "temporal"
        conf = 0.5
        rationale = "Parse error; default temporal."

    return CausalVerdict(
        source_idx=cand.source_idx,
        target_idx=cand.target_idx,
        relationship=rel,
        confidence=conf,
        rationale=rationale
    )


def verdicts_to_edges(verdicts: List[CausalVerdict]) -> List[GraphEdge]:
    """
    Convert CausalVerdicts into GraphEdge objects.

    Only 'causal' and 'enables' verdicts become edges.
    """
    edges: List[GraphEdge] = []
    for v in verdicts:
        if v.relationship in ("causal", "enables"):
            source_id = f"ev{v.source_idx + 1}"
            target_id = f"ev{v.target_idx + 1}"
            edges.append(GraphEdge(
                source=source_id,
                target=target_id,
                relationship=v.relationship,
                confidence=v.confidence,
                evidence=v.rationale
            ))
            logger.info(
                f"CausalClassifier: {v.relationship} edge {source_id} -> {target_id} "
                f"(conf={v.confidence:.2f}, rationale={v.rationale})"
            )
    return edges
