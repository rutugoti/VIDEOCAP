import os
import re
import logging
from typing import Dict, List, Optional, Tuple

from src.shared.models import Narrative, Caption
from src.shared.providers import LLMProvider, LLMConfig

logger = logging.getLogger(__name__)


class StyleGenerator:
    """Module responsible for translating a factually-grounded Narrative into four styled captions."""

    FALLBACK_PROMPTS = {
        "formal": {
            "system": "You are a professional technical writer. Your task is to rewrite video descriptions in a formal, neutral style.",
            "user": "Given the following video description, rewrite it as a formal caption.\n\nRules:\n- Use professional, neutral language\n- Preserve all factual content exactly\n- No humor, sarcasm, or informal language\n- Use third person\n- No contractions\n- Keep between 15-35 words\n- Do not add events not mentioned in the description\n- Do not remove any mentioned events\n\nVideo Description:\n{narrative}\n\nFormal Caption:"
        },
        "sarcastic": {
            "system": "You are a witty commentator known for sarcastic observations. Your task is to rewrite video descriptions sarcastically while preserving all facts.",
            "user": "Given the following video description, rewrite it with a sarcastic tone.\n\nRules:\n- Use ironic, dismissive, or sardonic language\n- Preserve ALL factual content — do not change what happened\n- Do not invent new events\n- Attitude changes, facts don't\n- Keep between 15-35 words\n- Use rhetorical questions or understatement where appropriate\n\nVideo Description:\n{narrative}\n\nSarcastic Caption:"
        },
        "tech_humor": {
            "system": "You are a software engineer who sees everything through the lens of programming and technology. Your task is to rewrite video descriptions using tech humor.",
            "user": "Given the following video description, rewrite it using software engineering or technology humor.\n\nRules:\n- Use programming concepts: debugging, bugs, exceptions, memory leaks, unit tests, production, deployment, etc.\n- Map real-world events to tech metaphors\n- Preserve ALL factual content — do not change what happened\n- Do not invent events not in the description\n- The humor should come from the tech metaphor, not from changing facts\n- Keep between 15-35 words\n- Avoid non-tech humor\n\nVideo Description:\n{narrative}\n\nTech Humor Caption:"
        },
        "non_tech_humor": {
            "system": "You are a funny social media content creator. Your task is to rewrite video descriptions with general humor that anyone can enjoy.",
            "user": "Given the following video description, rewrite it as a funny, relatable caption.\n\nRules:\n- Use everyday humor — no technology, programming, or engineering jokes\n- Think social media captions, stand-up comedy, observational humor\n- Preserve ALL factual content — do not change what happened\n- Do not invent events not in the description\n- Keep between 15-35 words\n- Must be clearly different from tech humor\n\nVideo Description:\n{narrative}\n\nFunny Caption:"
        }
    }

    def __init__(
        self,
        llm_provider: LLMProvider,
        llm_config: Optional[LLMConfig] = None,
        prompts_dir: str = "docs/prompts",
        min_caption_words: int = 15,
        max_caption_words: int = 35,
    ):
        self.llm_provider = llm_provider
        self.llm_config = llm_config or LLMConfig(
            provider="fireworks",
            model="accounts/fireworks/models/llama-v3-70b-instruct",
            max_tokens=256,
            temperature=0.7
        )
        self.prompts_dir = prompts_dir
        self.min_caption_words = min_caption_words
        self.max_caption_words = max_caption_words
        self.prompt_cache: Dict[str, Tuple[str, str]] = {}

    def generate_captions(self, narrative: Narrative) -> Dict[str, Caption]:
        """
        Generate Formal, Sarcastic, Tech Humor, and Non-Tech Humor captions from the Narrative.

        Args:
            narrative: The style-neutral, factually grounded input Narrative.

        Returns:
            Dict mapping style name to Caption object.
        """
        captions: Dict[str, Caption] = {}
        styles = ["formal", "sarcastic", "tech_humor", "non_tech_humor"]

        for style in styles:
            # 1. Load System and User prompt templates
            system_prompt, user_template = self._load_prompt(style)

            # 2. Insert Narrative
            user_prompt = user_template.replace("{narrative}", narrative.text)

            # 3. Generate with LLM (incorporating word count retry loop)
            caption_text, metadata = self._generate_with_retry(
                style=style,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                narrative_text=narrative.text
            )

            # Word count validation
            words_count = len(caption_text.split())

            captions[style] = Caption(
                text=caption_text,
                style=style,
                word_count=words_count,
                metadata=metadata
            )

        return captions

    def _load_prompt(self, style: str) -> Tuple[str, str]:
        """Load the versioned prompt templates from markdown files, falling back if not found."""
        if style in self.prompt_cache:
            return self.prompt_cache[style]

        file_name = f"{style}.md"
        file_path = os.path.join(self.prompts_dir, file_name)

        if not os.path.exists(file_path):
            logger.warning(f"Prompt file not found: {file_path}. Using fallback prompts.")
            return self.FALLBACK_PROMPTS[style]["system"], self.FALLBACK_PROMPTS[style]["user"]

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Regex search for System Prompt
            system_match = re.search(r'## System Prompt\s*\n([^#]+)', content)
            system_prompt = system_match.group(1).strip() if system_match else self.FALLBACK_PROMPTS[style]["system"]

            # Regex search for User Prompt Template inside code block
            user_match = re.search(r'## User Prompt Template\s*\n\s*```\s*\n(.*?)\n\s*```', content, re.DOTALL)
            user_prompt = user_match.group(1).strip() if user_match else self.FALLBACK_PROMPTS[style]["user"]

            self.prompt_cache[style] = (system_prompt, user_prompt)
            logger.info(f"Successfully loaded and cached prompt for style: {style}")
            return system_prompt, user_prompt

        except Exception as e:
            logger.error(f"Error reading prompt file {file_path}: {e}. Using fallback.")
            return self.FALLBACK_PROMPTS[style]["system"], self.FALLBACK_PROMPTS[style]["user"]

    def _generate_with_retry(self, style: str, system_prompt: str, user_prompt: str, narrative_text: str) -> Tuple[str, dict]:
        """Draft, critique, rewrite, and enforce word limits using Gemma self-critique loop."""
        import json
        prompt_version = "v2.0"
        model_version = self.llm_config.model
        temp = self.llm_config.temperature

        # 1. Draft caption
        try:
            draft_text = self.llm_provider.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                config=self.llm_config
            )
            draft_text = self._clean_caption(draft_text)
        except Exception as e:
            logger.error(f"Drafting caption failed for style '{style}' on attempt 1: {e}")
            draft_text = f"A backup caption representing the video narrative under {style} styling."

        # 2. Critique caption
        critique_system = (
            "You are a critical quality assurance judge. Your job is to critique draft video captions for:\n"
            "- Factual accuracy (check for hallucinations/invented facts)\n"
            "- Style fidelity (ensure tone perfectly matches requested style)\n"
            "- Grammar and spelling\n"
            "- Word limits (strictly between 15 and 35 words)\n\n"
            "Respond in valid JSON format only, matching this schema:\n"
            "{\n"
            "  \"missing_facts\": [\"missing fact 1\", ...],\n"
            "  \"hallucinations\": [\"hallucinated fact 1\", ...],\n"
            "  \"style_drift\": [\"style drift notes\", ...],\n"
            "  \"grammar_issues\": [\"grammar issue 1\", ...],\n"
            "  \"word_count_issue\": bool\n"
            "}"
        )
        
        critique_prompt = (
            f"Narrative:\n{narrative_text}\n\n"
            f"Draft Caption ({style} style):\n{draft_text}\n\n"
            f"Critique this draft caption based on the narrative and style rules. Return the JSON object."
        )

        critique_data = {}
        try:
            critique_response = self.llm_provider.generate(
                prompt=critique_prompt,
                system_prompt=critique_system,
                config=self.llm_config
            )
            from src.shared.fireworks_providers import _parse_and_repair_json
            critique_data = _parse_and_repair_json(critique_response)
        except Exception as e:
            logger.warning(f"Critique generation failed for style '{style}': {e}")

        # 3. Rewrite caption using critique feedback
        rewrite_prompt = (
            f"Narrative:\n{narrative_text}\n\n"
            f"Draft Caption:\n{draft_text}\n\n"
            f"Critique Feedback:\n{json.dumps(critique_data)}\n\n"
            f"Rewrite the caption to fix all critique issues. Ensure it is strictly between {self.min_caption_words} and {self.max_caption_words} words."
        )

        try:
            rewrite_response = self.llm_provider.generate(
                prompt=rewrite_prompt,
                system_prompt=system_prompt,
                config=self.llm_config
            )
            final_text = self._clean_caption(rewrite_response)
        except Exception as e:
            logger.warning(f"Rewrite generation failed for style '{style}': {e}. Falling back to draft.")
            final_text = draft_text

        # 4. Word count limits check & sentence boundary truncation
        words = final_text.split()
        if len(words) > self.max_caption_words:
            logger.warning(f"Style '{style}' output word count ({len(words)}) is out of bounds. Requesting shorter rewrite.")
            shorter_prompt = (
                f"Narrative:\n{narrative_text}\n\n"
                f"Caption to shorten:\n{final_text}\n\n"
                f"Rewrite this caption to be strictly under {self.max_caption_words} words. Do not invent any facts."
            )
            try:
                shorter_response = self.llm_provider.generate(
                    prompt=shorter_prompt,
                    system_prompt=system_prompt,
                    config=self.llm_config
                )
                final_text = self._clean_caption(shorter_response)
                words = final_text.split()
            except Exception as e:
                logger.error(f"Shorter rewrite failed: {e}")

            # If still too long, trim at sentence boundaries, NEVER cut sentences in half
            if len(words) > self.max_caption_words:
                logger.warning(f"Caption is still too long after rewrite. Trimming strictly at sentence boundaries.")
                sentences = re.split(r'(?<=[.!?])\s+', final_text)
                trimmed_parts = []
                current_count = 0
                for sentence in sentences:
                    sentence_words = sentence.split()
                    if current_count + len(sentence_words) <= self.max_caption_words:
                        trimmed_parts.append(sentence)
                        current_count += len(sentence_words)
                    else:
                        break
                if trimmed_parts:
                    final_text = " ".join(trimmed_parts)
                else:
                    first_sentence = sentences[0]
                    first_words = first_sentence.split()
                    final_text = " ".join(first_words[:self.max_caption_words])
                    if not final_text.endswith("."):
                        final_text += "."

        metadata = {
            "prompt_version": prompt_version,
            "model_version": model_version,
            "temperature": temp,
            "draft": draft_text,
            "critique": critique_data
        }
        return final_text, metadata

    def _clean_caption(self, text: str) -> str:
        """Strip enclosing quotes and extraneous formatting prefixes from the generated caption."""
        text = text.strip()
        text = text.replace("**", "").replace("*", "")
        if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
            text = text[1:-1].strip()
        prefixes = [
            "formal caption:", "sarcastic caption:", "tech humor caption:", "funny caption:",
            "formal:", "sarcastic:", "tech humor:", "humor:"
        ]
        for p in prefixes:
            if text.lower().startswith(p):
                text = text[len(p):].strip()
        return text
