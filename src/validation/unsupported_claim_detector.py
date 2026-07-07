"""
Unsupported-Claim Entailment Detector — Phase P4.1 (docs/IMPLEMENT.md, plan Step 11).

Detects semantic additions a caption makes that the narrative does not support —
e.g. narrative "person drinks coffee" vs caption "person *happily* drinks coffee".

Two-stage, honest about what each stage can do:

1. Deterministic pre-filter (cheap, high recall, low precision): candidate additions
   are content terms in the caption that do not appear in the narrative fact set.
   This over-flags synonyms/paraphrase and under-flags reordered facts — it only
   *narrows* the work for the LLM. If there are no candidates, NO LLM call is made.

2. Gemma adjudication (the part that genuinely needs a language model): classify each
   candidate as supported | entailed | unsupported given the narrative. Only
   `unsupported` terms are returned. There is no exact symbolic algorithm for
   open-domain entailment; this is a bounded LLM check, not a proof.
"""

import logging
import re
from typing import List, Optional

from src.shared.providers import LLMProvider, LLMConfig
from src.shared.utils import sanitize_untrusted_input

logger = logging.getLogger(__name__)

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "to", "of",
    "in", "on", "at", "for", "with", "by", "about", "this", "that", "it", "as",
    "then", "into", "from", "their", "its", "his", "her", "they", "he", "she",
}


def _content_terms(text: str) -> set:
    """Lowercased content words (len>2, non-stopword), punctuation stripped."""
    cleaned = re.sub(r"[^\w\s]", "", text.lower())
    return {w for w in cleaned.split() if len(w) > 2 and w not in _STOPWORDS}


class UnsupportedClaimDetector:
    """Flags caption terms that add meaning the narrative does not support."""

    def __init__(self, llm_provider: LLMProvider, llm_config: Optional[LLMConfig] = None):
        self.llm_provider = llm_provider
        self.llm_config = llm_config or LLMConfig(
            provider="fireworks",
            model="accounts/fireworks/models/gemma-3-27b-it",
            max_tokens=256,
            temperature=0.0,
        )

    def candidate_additions(self, caption_text: str, narrative_text: str) -> List[str]:
        """Stage 1: deterministic term-diff. Terms in caption but not in narrative."""
        cap_terms = _content_terms(caption_text)
        narr_terms = _content_terms(narrative_text)
        return sorted(cap_terms - narr_terms)

    def detect(self, caption_text: str, narrative_text: str) -> List[str]:
        """
        Return caption terms judged `unsupported` by Gemma.
        No candidates → no LLM call → empty list.
        """
        candidates = self.candidate_additions(caption_text, narrative_text)
        if not candidates:
            return []

        system_prompt = (
            "You are an entailment adjudicator. Given a source NARRATIVE and a list of "
            "candidate terms taken from a caption, classify each candidate term as:\n"
            "- supported: the term or its clear synonym appears in / is directly stated by the narrative\n"
            "- entailed: not stated verbatim but logically implied by the narrative\n"
            "- unsupported: adds meaning (emotion, intent, manner, new entities) the narrative does NOT support\n\n"
            "Function words, articles, and stylistic filler are NOT unsupported claims; classify them as supported.\n"
            "Return ONLY a valid JSON object:\n"
            '{"supported": [...], "entailed": [...], "unsupported": [...]}'
        )
        user_prompt = (
            f"NARRATIVE:\n<untrusted_input>\"{sanitize_untrusted_input(narrative_text)}\"</untrusted_input>\n\n"
            f"CANDIDATE TERMS: {candidates}\n\n"
            "Classify every candidate term. Return the JSON object only."
        )

        try:
            raw = self.llm_provider.generate(
                prompt=user_prompt, system_prompt=system_prompt, config=self.llm_config
            )
            from src.shared.fireworks_providers import _parse_and_repair_json
            data = _parse_and_repair_json(raw)
        except Exception as e:
            logger.warning(f"UnsupportedClaimDetector: LLM call failed: {e}. Returning no flags (fail-open).")
            return []

        if not isinstance(data, dict):
            return []

        unsupported = data.get("unsupported", [])
        if not isinstance(unsupported, list):
            return []

        # Only keep terms that were actually candidates (guard against LLM inventing terms)
        candidate_set = set(candidates)
        result = [str(t) for t in unsupported if str(t).lower() in candidate_set]
        if result:
            logger.info(f"UnsupportedClaimDetector: flagged unsupported additions {result}")
        return result
