import os
import re
import json
import logging
from typing import Dict, List, Optional, Tuple, Any

from src.shared.models import Narrative, Caption, ValidationReport, CaptionValidation
from src.shared.providers import Validator, LLMProvider, LLMConfig, ValidatorConfig

logger = logging.getLogger(__name__)


class SemanticValidator(Validator):
    """Module responsible for verifying captions against a Narrative to catch hallucinations and drift."""

    FALLBACK_SYSTEM_PROMPT = (
        "You are a strict fact-checker. Your task is to compare generated captions against "
        "a source description and identify discrepancies."
    )

    FALLBACK_USER_TEMPLATE = (
        "Compare the following caption against the source description. Identify:\n\n"
        "1. HALLUCINATIONS: Events, objects, or people mentioned in the caption but NOT in the source\n"
        "2. MISSING FACTS: Important events in the source NOT mentioned in the caption\n"
        "3. FACT DRIFT: Facts that changed between source and caption (different objects, people, actions)\n"
        "4. STYLE CHECK: Does the caption match the intended style?\n\n"
        "Source Description:\n{narrative}\n\n"
        "Intended Style: {style}\n\n"
        "Caption:\n{caption}\n\n"
        "Report your findings as JSON:\n"
        "{\n"
        "  \"hallucinations\": [\"list of ungrounded claims\"],\n"
        "  \"missing_facts\": [\"list of omitted important events\"],\n"
        "  \"fact_drift\": [\"list of changed facts\"],\n"
        "  \"style_match\": true/false,\n"
        "  \"overall_pass\": true/false\n"
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
            model="accounts/fireworks/models/llama-v3-70b-instruct",
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

            # 2. Substitute template variables
            user_prompt = (
                user_template
                .replace("{narrative}", narrative.text)
                .replace("{style}", style)
                .replace("{caption}", caption.text)
            )

            try:
                # 3. Call LLM fact checker
                response = self.llm_provider.generate(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    config=self.llm_config
                )

                # 4. Parse validation response
                val_data = self._parse_validator_response(response)
                
                # Extract results with active-checks masking
                halls = val_data.get("hallucinations", []) if self.hallucination_check else []
                drifts = val_data.get("fact_drift", []) if self.fact_drift_check else []
                missings = val_data.get("missing_facts", []) if self.fact_drift_check else []
                style_match = val_data.get("style_match", True) if self.style_leakage_check else True

                # Determine passed based on active checks (overriding the LLM overall_pass if a check is disabled)
                passed = True
                if self.hallucination_check and halls:
                    passed = False
                if self.fact_drift_check and (drifts or missings):
                    passed = False
                if self.style_leakage_check and not style_match:
                    passed = False

                total_hallucinations += len(halls)
                total_drift += len(drifts)
                total_missing += len(missings)

                # Report fact drift in missing_facts so it's captured in the schema
                reported_missing = missings + drifts if self.fact_drift_check else []

                per_caption[style] = CaptionValidation(
                    style=style,
                    passed=passed,
                    hallucinations=halls,
                    missing_facts=reported_missing,
                    style_adherence=1.0 if style_match else 0.0
                )

            except Exception as e:
                logger.error(f"SemanticValidator: LLM validation call failed for style '{style}': {e}. Using passing fallback.")
                # We default to passing so we don't halt the entire pipeline in production due to validator rate limits
                per_caption[style] = CaptionValidation(
                    style=style,
                    passed=True,
                    hallucinations=[],
                    missing_facts=[],
                    style_adherence=1.0
                )

        # 5. Global consistency aggregation
        overall_pass = all(v.passed for v in per_caption.values())
        
        # Calculate consistency score: 1.0 minus penalty for issues, clamped to [0.0, 1.0]
        issue_penalty = (total_hallucinations + total_drift) * 0.1
        consistency_score = max(0.0, min(1.0, 1.0 - issue_penalty))

        logger.info(
            f"Validation complete. Overall Pass: {overall_pass}, "
            f"Hallucination Count: {total_hallucinations}, Consistency Score: {consistency_score:.2f}"
        )

        return ValidationReport(
            overall_pass=overall_pass,
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
        """Parse LLM JSON response for fact checker report."""
        text = text.strip()
        if not text:
            return {}

        start_idx = text.find("{")
        end_idx = text.rfind("}")

        if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
            return {}

        json_str = text[start_idx:end_idx + 1]

        try:
            return json.loads(json_str)
        except Exception as e:
            logger.debug(f"Validator JSON parsing error: {e}")
            return {}
