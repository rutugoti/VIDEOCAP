import logging
from typing import List, Tuple, Dict, Any, Optional
from src.shared.models import Observation, Confidence

logger = logging.getLogger(__name__)


class ResolvedGroup:
    def __init__(self):
        self.kept: List[Observation] = []
        self.dropped: List[Tuple[Observation, str]] = []  # List of (Observation, reason)
        self.contested_pairs: List[Tuple[Observation, Observation]] = []
        self.conflicts: List[Dict[str, Any]] = []


class EvidenceResolutionPolicy:
    """Deterministic rules to resolve contradictions and check evidence reliability before LLM fusion."""

    def __init__(self, escalate_conflict_min_conf: float = 0.7, min_confidence: float = 0.3):
        self.escalate_conflict_min_conf = escalate_conflict_min_conf
        self.min_confidence = min_confidence

    def resolve(self, group: List[Observation]) -> ResolvedGroup:
        res = ResolvedGroup()
        if not group:
            return res

        # 1. Handle missing confidence values first: default to 0.5 with default source
        for o in group:
            if o.confidence_meta is None:
                o.confidence_meta = Confidence(value=0.5, source="default")
            # If o.confidence is not set or zero, align with confidence_meta
            if not o.confidence:
                o.confidence = o.confidence_meta.value

        # Filter out observations below min_confidence
        filtered_group = []
        for o in group:
            if o.confidence < self.min_confidence:
                res.dropped.append((o, f"Confidence {o.confidence:.2f} is below min_confidence {self.min_confidence}"))
            else:
                filtered_group.append(o)

        if not filtered_group:
            return res

        # 2. Extract modality presence and highest visual confidence
        has_visual = any(o.source == "visual" for o in filtered_group)
        highest_visual_conf = max((o.confidence for o in filtered_group if o.source == "visual"), default=0.0)

        # 3. Apply OCR priority (text modality): downweight OCR if visual exists and OCR conf is not > visual + 0.25
        for o in filtered_group:
            if o.source == "text":
                if has_visual and o.confidence <= highest_visual_conf + 0.25:
                    old_conf = o.confidence
                    o.confidence = min(o.confidence, highest_visual_conf * 0.8)
                    o.confidence_meta.value = o.confidence
                    res.conflicts.append({
                        "type": "ocr_downweighted",
                        "observation": o.content,
                        "old_confidence": old_conf,
                        "new_confidence": o.confidence,
                        "resolution": "OCR downweighted (visual priority)"
                    })

        # 4. Audio/Speech Only reliability:
        # If Speech only (audio source, no visual source in group):
        # - keep iff confidence >= 0.6, else discard
        # - if kept, downweight by 0.5
        for o in list(filtered_group):
            if o.source == "audio":
                if not has_visual:
                    if o.confidence < 0.6:
                        res.dropped.append((o, f"Speech only observation {o.content} confidence {o.confidence:.2f} < 0.6"))
                        filtered_group.remove(o)
                    else:
                        old_conf = o.confidence
                        o.confidence *= 0.5
                        o.confidence_meta.value = o.confidence
                        res.conflicts.append({
                            "type": "audio_only_downweighted",
                            "observation": o.content,
                            "old_confidence": old_conf,
                            "new_confidence": o.confidence,
                            "resolution": "Audio only downweighted by 0.5"
                        })

        # 5. Vision contradicts/agrees with Speech:
        # Check Vision + Speech interactions
        visual_obs = [o for o in filtered_group if o.source == "visual"]
        audio_obs = [o for o in filtered_group if o.source == "audio"]

        for v in visual_obs:
            for a in audio_obs:
                dist = self._jaccard_distance(v.content, a.content)
                if dist < 0.6:
                    # They AGREE (high word overlap) -> Boost confidence
                    old_v = v.confidence
                    old_a = a.confidence
                    v.confidence = min(1.0, v.confidence + 0.05)
                    v.confidence_meta.value = v.confidence
                    a.confidence = min(1.0, a.confidence + 0.05)
                    a.confidence_meta.value = a.confidence
                    res.conflicts.append({
                        "type": "modality_agreement",
                        "visual": v.content,
                        "audio": a.content,
                        "old_visual_conf": old_v,
                        "new_visual_conf": v.confidence,
                        "old_audio_conf": old_a,
                        "new_audio_conf": a.confidence,
                        "resolution": "Agreement boost applied"
                    })
                else:
                    # They CONTRADICT -> Preserve both as alternatives, set contested flag
                    v.contested = True
                    a.contested = True
                    if a.id is not None and a.id not in v.alternatives:
                        v.alternatives.append(a.id)
                    if v.id is not None and v.id not in a.alternatives:
                        a.alternatives.append(v.id)

                    res.contested_pairs.append((v, a))
                    
                    escalate = (v.confidence >= self.escalate_conflict_min_conf and 
                                a.confidence >= self.escalate_conflict_min_conf)
                    
                    res.conflicts.append({
                        "type": "modality_contradiction",
                        "visual": v.content,
                        "audio": a.content,
                        "visual_conf": v.confidence,
                        "audio_conf": a.confidence,
                        "escalate_to_llm": escalate,
                        "resolution": "Preserved both as alternatives" + (" (Escalated to LLM)" if escalate else "")
                    })

        res.kept = filtered_group
        return res

    def _jaccard_distance(self, s1: str, s2: str) -> float:
        import re
        words1 = set(re.findall(r'\w+', s1.lower()))
        words2 = set(re.findall(r'\w+', s2.lower()))
        if not words1 and not words2:
            return 1.0
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return 1.0 - (len(intersection) / len(union))
