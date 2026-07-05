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
        self.prompt_cache: Optional[Tuple[str, str]] = None

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

        per_caption: Dict[str, CaptionValidation] = {}
        total_hallucinations = 0
        total_drift = 0
        total_missing = 0

        for style, caption in captions.items():
            logger.info(f"Validating caption style: {style}")

            # 2. Substitute template variables (sanitized to prevent prompt injection)
            user_prompt = (
                user_template
                .replace("{narrative}", f"<untrusted_input>\"{sanitize_untrusted_input(narrative.text)}\"</untrusted_input>")
                .replace("{style}", style)
                .replace("{caption}", f"<untrusted_input>\"{sanitize_untrusted_input(caption.text)}\"</untrusted_input>")
            )

            try:
                # 3. Call LLM fact checker
                response = self.llm_provider.generate(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    config=self.llm_config
                )

                # 4. Parse validation response with repair fallback
                val_data = self._parse_validator_response(response)
                
                # Extract results with active-checks masking
                halls = val_data.get("hallucinations", []) if self.hallucination_check else []
                drifts = val_data.get("fact_drift", []) if self.fact_drift_check else []
                missings = val_data.get("missing_facts", []) if self.fact_drift_check else []
                style_match = val_data.get("style_match", True) if self.style_leakage_check else True

                # Determine passed based on active checks
                passed = True
                if self.hallucination_check and halls:
                    passed = False
                if self.fact_drift_check and (drifts or missings):
                    passed = False
                if self.style_leakage_check and not style_match:
                    passed = False

                status = "PASS" if passed else "FAIL"

                total_hallucinations += len(halls)
                total_drift += len(drifts)
                total_missing += len(missings)

                # Report fact drift in missing_facts so it's captured in the schema
                reported_missing = missings + drifts if self.fact_drift_check else []

                # Populate detailed scores
                per_caption[style] = CaptionValidation(
                    style=style,
                    passed=passed,
                    status=status,
                    hallucinations=halls,
                    missing_facts=reported_missing,
                    style_adherence=1.0 if style_match else 0.0,
                    semantic_accuracy=val_data.get("semantic_accuracy", 1.0 - (len(drifts) * 0.2)),
                    hallucination_risk=val_data.get("hallucination_risk", len(halls) * 0.2),
                    grammar_score=val_data.get("grammar_score", 1.0),
                    temporal_consistency_score=val_data.get("temporal_consistency_score", 1.0),
                    word_budget_pass=val_data.get("word_budget_pass", True),
                    overall_confidence=val_data.get("overall_confidence", 0.9)
                )

            except Exception as e:
                logger.error(f"SemanticValidator: LLM validation call failed for style '{style}': {e}. Using fail-closed UNKNOWN fallback.")
                per_caption[style] = CaptionValidation(
                    style=style,
                    passed=False,
                    status="UNKNOWN",
                    hallucinations=[],
                    missing_facts=[],
                    style_adherence=0.0,
                    semantic_accuracy=0.0,
                    hallucination_risk=1.0,
                    grammar_score=0.0,
                    temporal_consistency_score=0.0,
                    word_budget_pass=False,
                    overall_confidence=0.0
                )

        # 5. Global consistency aggregation
        overall_pass = all(v.passed for v in per_caption.values())
        
        overall_status = "PASS"
        if any(v.status == "FAIL" for v in per_caption.values()):
            overall_status = "FAIL"
        elif any(v.status == "UNKNOWN" for v in per_caption.values()):
            overall_status = "UNKNOWN"

        # Calculate consistency score: 1.0 minus penalty for issues, clamped to [0.0, 1.0]
        issue_penalty = (total_hallucinations + total_drift) * 0.1
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
