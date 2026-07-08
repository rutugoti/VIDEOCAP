import os
import re
import json
import logging
from typing import Dict, List, Optional, Tuple, Any

from src.shared.models import Narrative, Caption, ValidationReport, CaptionValidation
from src.shared.providers import Validator, LLMProvider, LLMConfig, ValidatorConfig
from src.shared.utils import sanitize_untrusted_input

logger = logging.getLogger(__name__)


class SemanticValidator(Validator):
    """Module responsible for verifying captions against a Narrative to catch hallucinations and drift."""

    FALLBACK_SYSTEM_PROMPT = (
        "You are a strict fact-checker. Your task is to compare generated captions against "
        "a source description and identify discrepancies."
    )

    FALLBACK_USER_TEMPLATE = (
        "Compare the following caption against the source description. Identify:\n\n"
        "1. HALLUCINATIONS: Events, objects, or people mentioned in the caption but NOT in the source description.\n"
        "2. MISSING FACTS: Important events in the source description NOT mentioned in the caption.\n"
        "3. FACT DRIFT: Facts that changed between source and caption.\n"
        "4. STYLE CHECK: Does the caption match the intended style?\n"
        "5. TEMPORAL CONSISTENCY: Does the caption follow the exact chronological ordering of events as described in the source? The caption MUST NOT reorder events.\n"
        "6. WORD BUDGET: Is the caption strictly between 15 and 35 words?\n\n"
        "Source Description:\n{narrative}\n\n"
        "Intended Style: {style}\n\n"
        "Caption:\n{caption}\n\n"
        "Report your findings as JSON:\n"
        "{\n"
        "  \"hallucinations\": [\"list of ungrounded claims\"],\n"
        "  \"missing_facts\": [\"list of omitted important events\"],\n"
        "  \"fact_drift\": [\"list of changed facts\"],\n"
        "  \"style_match\": true/false,\n"
        "  \"overall_pass\": true/false,\n"
        "  \"semantic_accuracy\": float (0.0 to 1.0),\n"
        "  \"hallucination_risk\": float (0.0 to 1.0),\n"
        "  \"grammar_score\": float (0.0 to 1.0),\n"
        "  \"temporal_consistency_score\": float (0.0 to 1.0),\n"
        "  \"word_budget_pass\": true/false,\n"
        "  \"overall_confidence\": float (0.0 to 1.0)\n"
        "}"
    )

    def __init__(
        self,
        llm_provider: LLMProvider,
        llm_config: Optional[LLMConfig] = None,
        prompts_dir: str = "docs/prompts",
        hallucination_check: bool = True,
        style_leakage_check: bool = True,
        fact_drift_check: bool = True,
        unsupported_claim_check: bool = False,
    ):
        self.llm_provider = llm_provider
        self.llm_config = llm_config or LLMConfig(
            provider="fireworks",
            model="accounts/fireworks/models/gemma-3-27b-it",
            max_tokens=512,
            temperature=0.1
        )
        self.prompts_dir = prompts_dir
        self.hallucination_check = hallucination_check
        self.style_leakage_check = style_leakage_check
        self.fact_drift_check = fact_drift_check
        self.unsupported_claim_check = unsupported_claim_check
        self.prompt_cache: Optional[Tuple[str, str]] = None
        # P4.1: lazily-built entailment detector (only used when enabled)
        self._unsupported_detector = None
        if self.unsupported_claim_check:
            from src.validation.unsupported_claim_detector import UnsupportedClaimDetector
            self._unsupported_detector = UnsupportedClaimDetector(
                llm_provider=self.llm_provider,
                llm_config=self.llm_config,
            )

    def validate(
        self,
        captions: Dict[str, Caption],
        narrative: Narrative,
        config: Optional[ValidatorConfig] = None
    ) -> ValidationReport:
        """
        Validate a dictionary of Captions against the source Narrative.

        Args:
            captions: Dict mapping style name to Caption object.
            narrative: The ground-truth Narrative object.
            config: Optional config parameter (not used, interface compatibility).

        Returns:
            A ValidationReport object.
        """
        import time
        start_time = time.time()
        if not captions:
            logger.warning("SemanticValidator: No captions provided for validation.")
            return ValidationReport(
                overall_pass=True,
                per_caption={},
                hallucination_count=0,
                consistency_score=1.0
            )

        # 1. Load prompts
        system_prompt, user_template = self._load_validator_prompt()

        # 2. Deterministic fact term checks and leakage checks
        STOPWORDS = {"the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "to", "of", "in", "on", "at", "for", "with", "by", "about"}
        
        def get_fact_terms(text: str) -> set:
            cleaned = re.sub(r'[^\w\s]', '', text.lower())
            return {w for w in cleaned.split() if len(w) > 2 and w not in STOPWORDS}

        narrative_terms = get_fact_terms(narrative.text)
        
        # Check per-style terms
        caption_terms = {}
        for style, cap in captions.items():
            caption_terms[style] = get_fact_terms(cap.text)

        # Detect contradictions (fact present in N-1 captions but missing in 1)
        det_contradictions = {style: [] for style in captions.keys()}
        if len(captions) >= 3:
            for term in narrative_terms:
                present_styles = [style for style, terms in caption_terms.items() if term in terms]
                if len(present_styles) == len(captions) - 1:
                    missing_style = [style for style in captions.keys() if style not in present_styles][0]
                    if missing_style not in ("sarcastic", "tech_humor"):
                        det_contradictions[missing_style].append(
                            f"Fact term '{term}' is present in other style captions but missing in '{missing_style}'."
                        )

        # Check pairwise Jaccard similarity between captions to detect leakage
        det_leakages = {style: False for style in captions.keys()}
        styles_list = list(captions.keys())
        for i in range(len(styles_list)):
            for j in range(i + 1, len(styles_list)):
                s1, s2 = styles_list[i], styles_list[j]
                t1, t2 = caption_terms[s1], caption_terms[s2]
                if not t1 and not t2:
                    sim = 1.0
                else:
                    sim = len(t1.intersection(t2)) / len(t1.union(t2))
                if sim > 0.8:
                    det_leakages[s1] = True
                    det_leakages[s2] = True

        # Single Gemma cross-style validation call
        cross_style_system_prompt = (
            "You are a strict cross-style caption validator. Your task is to validate a set of styled captions "
            "against a ground-truth narrative. Check for factual consistency, style separation, and style match.\n"
            "Return a JSON object conforming exactly to this schema:\n"
            "{\n"
            "  \"consistency_score\": float (0.0 to 1.0),\n"
            "  \"style_separation_check\": {\n"
            "    \"pass\": true/false,\n"
            "    \"reason\": \"explanation of style separation\"\n"
            "  },\n"
            "  \"captions\": {\n"
            "    \"formal\": {\n"
            "      \"hallucinations\": [\"list of ungrounded details or actions not in narrative\"],\n"
            "      \"missing_facts\": [\"list of omitted major facts from narrative\"],\n"
            "      \"style_match\": true/false,\n"
            "      \"semantic_accuracy\": float (0.0 to 1.0),\n"
            "      \"hallucination_risk\": float (0.0 to 1.0),\n"
            "      \"grammar_score\": float (0.0 to 1.0),\n"
            "      \"temporal_consistency_score\": float (0.0 to 1.0),\n"
            "      \"word_budget_pass\": true/false,\n"
            "      \"overall_confidence\": float (0.0 to 1.0)\n"
            "    }\n"
            "  }\n"
            "}"
        )

        captions_details = []
        for style, cap in captions.items():
            captions_details.append(f"- Style '{style}': <untrusted_input>\"{sanitize_untrusted_input(cap.text)}\"</untrusted_input>")
        captions_text = "\n".join(captions_details)
        
        user_prompt = (
            f"Source Narrative:\n<untrusted_input>\"{sanitize_untrusted_input(narrative.text)}\"</untrusted_input>\n\n"
            f"Captions to validate:\n{captions_text}\n\n"
            "Validate all captions at once. Ensure no caption contains ungrounded details (unsupported claims) "
            "or misses critical facts from the narrative. Ensure each caption matches its target style and "
            "that there is clear separation between the styles without overlap."
        )

        try:
            # 3. Call LLM cross-style checker
            response = self.llm_provider.generate(
                prompt=user_prompt,
                system_prompt=cross_style_system_prompt,
                config=self.llm_config
            )

            # 4. Parse validation response with repair fallback
            val_data = self._parse_validator_response(response)
            
            # Adapt legacy/mock single-caption JSON to the new cross-style schema
            if "captions" not in val_data:
                adapted = {
                    "consistency_score": val_data.get("overall_pass", True) and 1.0 or 0.5,
                    "style_separation_check": {
                        "pass": val_data.get("style_match", True),
                        "reason": "Adapted from legacy/mock response"
                    },
                    "captions": {}
                }
                # Copy to all queried styles
                for style in captions.keys():
                    adapted["captions"][style] = {
                        "hallucinations": val_data.get("hallucinations", []),
                        "missing_facts": val_data.get("missing_facts", []) + val_data.get("fact_drift", []),
                        "style_match": val_data.get("style_match", True),
                        "semantic_accuracy": val_data.get("semantic_accuracy", 1.0),
                        "hallucination_risk": val_data.get("hallucination_risk", 0.0),
                        "grammar_score": val_data.get("grammar_score", 1.0),
                        "temporal_consistency_score": val_data.get("temporal_consistency_score", 1.0),
                        "word_budget_pass": val_data.get("word_budget_pass", True),
                        "overall_confidence": val_data.get("overall_confidence", 0.9)
                    }
                val_data = adapted

        except Exception as e:
            logger.error(f"SemanticValidator: LLM validation call failed: {e}. Using fail-closed UNKNOWN fallback.")
            val_data = {
                "consistency_score": 0.0,
                "style_separation_check": {"pass": False, "reason": "Failed call"},
                "captions": {
                    style: {
                        "hallucinations": [],
                        "missing_facts": [],
                        "style_match": False,
                        "semantic_accuracy": 0.0,
                        "hallucination_risk": 1.0,
                        "grammar_score": 0.0,
                        "temporal_consistency_score": 0.0,
                        "word_budget_pass": False,
                        "overall_confidence": 0.0
                    } for style in captions.keys()
                }
            }

        per_caption: Dict[str, CaptionValidation] = {}
        total_hallucinations = 0
        total_missing = 0

        for style, cap in captions.items():
            cap_val = val_data.get("captions", {}).get(style, {})
            
            # Extract results with active-checks masking
            halls = cap_val.get("hallucinations", []) if self.hallucination_check else []
            missings = cap_val.get("missing_facts", []) if self.fact_drift_check else []
            style_match = cap_val.get("style_match", True) if self.style_leakage_check else True

            # P4.1: unsupported-claim entailment check (deterministic pre-filter → Gemma).
            # Flagged terms are treated as hallucinations so the pipeline retry regenerates.
            unsupported_found = False
            if self.unsupported_claim_check and self._unsupported_detector is not None:
                unsupported = self._unsupported_detector.detect(cap.text, narrative.text)
                for term in unsupported:
                    claim = f"unsupported addition: '{term}' not grounded in narrative"
                    if claim not in halls:
                        halls.append(claim)
                        unsupported_found = True

            # Merge deterministic contradiction check
            det_contra = det_contradictions.get(style, [])
            if det_contra:
                missings.extend(det_contra)

            # Merge deterministic leakage check
            det_leak = det_leakages.get(style, False)
            if det_leak:
                style_match = False

            # Determine passed based on active checks
            passed = True
            if self.hallucination_check and halls:
                passed = False
            if self.fact_drift_check and missings:
                passed = False
            if self.style_leakage_check and not style_match:
                passed = False
            if unsupported_found:
                passed = False

            status = "PASS" if passed else "FAIL"
            if cap_val.get("semantic_accuracy") == 0.0 and cap_val.get("grammar_score") == 0.0:
                status = "UNKNOWN"

            total_hallucinations += len(halls)
            total_missing += len(missings)

            # Populate detailed scores
            per_caption[style] = CaptionValidation(
                style=style,
                passed=passed,
                status=status,
                hallucinations=halls,
                missing_facts=missings,
                style_adherence=1.0 if style_match else 0.0,
                semantic_accuracy=cap_val.get("semantic_accuracy", 1.0 - (len(missings) * 0.2)),
                hallucination_risk=cap_val.get("hallucination_risk", len(halls) * 0.2),
                grammar_score=cap_val.get("grammar_score", 1.0),
                temporal_consistency_score=cap_val.get("temporal_consistency_score", 1.0),
                word_budget_pass=cap_val.get("word_budget_pass", True),
                overall_confidence=cap_val.get("overall_confidence", 0.9)
            )

        # 5. Global consistency aggregation
        overall_pass = all(v.passed for v in per_caption.values())
        
        overall_status = "PASS"
        if any(v.status == "FAIL" for v in per_caption.values()):
            overall_status = "FAIL"
        elif any(v.status == "UNKNOWN" for v in per_caption.values()):
            overall_status = "UNKNOWN"

        # Calculate consistency score: 1.0 minus penalty for issues, clamped to [0.0, 1.0]
        issue_penalty = (total_hallucinations + total_missing) * 0.1
        consistency_score = max(0.0, min(1.0, 1.0 - issue_penalty))

        latency = time.time() - start_time
        logger.info(
            f"Validation complete. Latency: {latency:.2f}s | Overall Pass: {overall_pass}, Status: {overall_status}, "
            f"Hallucination Count: {total_hallucinations}, Consistency Score: {consistency_score:.2f}"
        )

        return ValidationReport(
            overall_pass=overall_pass,
            status=overall_status,
            per_caption=per_caption,
            hallucination_count=total_hallucinations,
            consistency_score=consistency_score
        )

    def _load_validator_prompt(self) -> Tuple[str, str]:
        """Load system and user validator prompts from validator.md, falling back if not found."""
        if self.prompt_cache:
            return self.prompt_cache

        file_path = os.path.join(self.prompts_dir, "validator.md")
        if not os.path.exists(file_path):
            logger.warning(f"Validator prompt file not found: {file_path}. Using fallbacks.")
            return self.FALLBACK_SYSTEM_PROMPT, self.FALLBACK_USER_TEMPLATE

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            system_match = re.search(r'## System Prompt\s*\n([^#]+)', content)
            system_prompt = system_match.group(1).strip() if system_match else self.FALLBACK_SYSTEM_PROMPT

            user_match = re.search(r'## User Prompt Template\s*\n\s*```\s*\n(.*?)\n\s*```', content, re.DOTALL)
            user_prompt = user_match.group(1).strip() if user_match else self.FALLBACK_USER_TEMPLATE

            self.prompt_cache = (system_prompt, user_prompt)
            return self.prompt_cache

        except Exception as e:
            logger.error(f"Error reading validator prompt file {file_path}: {e}. Using fallbacks.")
            return self.FALLBACK_SYSTEM_PROMPT, self.FALLBACK_USER_TEMPLATE

    def _parse_validator_response(self, text: str) -> Dict[str, Any]:
        """Parse LLM JSON response for fact checker report with automatic repair."""
        text = text.strip()
        if not text:
            return {}

        from src.shared.fireworks_providers import _parse_and_repair_json
        parsed = _parse_and_repair_json(text)
        if isinstance(parsed, dict):
            return parsed
        return {}
