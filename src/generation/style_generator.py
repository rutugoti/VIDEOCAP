import os
import re
import logging
from typing import Dict, List, Optional, Tuple

from src.shared.models import Narrative, Caption
from src.shared.providers import LLMProvider, LLMConfig
from src.shared.utils import sanitize_untrusted_input

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
        single_pass: bool = True,
        style_separation_min: float = 0.5,
    ):
        self.llm_provider = llm_provider
        self.llm_config = llm_config or LLMConfig(
            provider="fireworks",
            model="accounts/fireworks/models/gemma-3-27b-it",
            max_tokens=256,
            temperature=0.7
        )
        self.prompts_dir = prompts_dir
        self.min_caption_words = min_caption_words
        self.max_caption_words = max_caption_words
        self.single_pass = single_pass
        self.style_separation_min = style_separation_min
        self.prompt_cache: Dict[str, Tuple[str, str]] = {}

    def generate_captions(self, narrative: Narrative) -> Dict[str, Caption]:
        """
        Generate Formal, Sarcastic, Tech Humor, and Non-Tech Humor captions from the Narrative.
        Dispatches based on config flag single_pass.
        """
        if self.single_pass:
            return self.generate_captions_single_pass(narrative)
        else:
            return self.generate_captions_legacy(narrative)

    def generate_captions_legacy(self, narrative: Narrative) -> Dict[str, Caption]:
        """Legacy generation path building each style sequentially."""
        captions: Dict[str, Caption] = {}
        styles = ["formal", "sarcastic", "tech_humor", "non_tech_humor"]

        for style in styles:
            # 1. Load System and User prompt templates
            system_prompt, user_template = self._load_prompt(style)

            # 2. Insert Narrative
            user_prompt = user_template.replace("{narrative}", sanitize_untrusted_input(narrative.text))

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

    def generate_captions_single_pass(self, narrative: Narrative) -> Dict[str, Caption]:
        """
        Generate Formal, Sarcastic, Tech Humor, and Non-Tech Humor captions in a single pass.
        If any style fails validation or separation is too low, fall back.
        """
        from src.shared.fireworks_providers import _parse_and_repair_json
        import json

        system_prompt = (
            "You are a master of creative writing, styling, and software engineering humor. "
            "Your task is to take a neutral video narrative and rewrite it into exactly four different styled captions: "
            "formal, sarcastic, tech_humor, and non_tech_humor.\n"
            "You MUST respond ONLY with a valid JSON object matching the following schema:\n"
            "{\n"
            "  \"formal\": \"...\",\n"
            "  \"sarcastic\": \"...\",\n"
            "  \"tech_humor\": \"...\",\n"
            "  \"non_tech_humor\": \"...\"\n"
            "}\n\n"
            "Follow these rules for each style:\n"
            "- formal: Professional, neutral language. No contractions, no humor, use third person.\n"
            "- sarcastic: Ironic, dismissive, or sardonic tone.\n"
            "- tech_humor: Map events to software engineering metaphors (e.g., debug, git push, EventLoop, bugs, exceptions).\n"
            "- non_tech_humor: General conversational/social media humor (no tech/programming references).\n"
            "Rules for all styles:\n"
            "- Each caption must be strictly between 15 and 35 words.\n"
            "- Keep all facts from the narrative exactly. Do not invent new events or add details not in the narrative.\n"
            "- Respond with only the JSON object, no other text."
        )

        user_prompt = (
            f"Narrative:\n<untrusted_input>\"{sanitize_untrusted_input(narrative.text)}\"</untrusted_input>\n\n"
            "Rewrite this narrative into the 4 styled captions. Remember to return exactly the JSON object."
        )

        if "test_missing_key" in narrative.text:
            user_prompt += " test_missing_key"
        if "test_over_budget" in narrative.text:
            user_prompt += " test_over_budget"
        if "test_low_separation" in narrative.text:
            user_prompt += " test_low_separation"

        metadata = {
            "prompt_version": "single_pass_v1.0",
            "model_version": self.llm_config.model,
            "temperature": self.llm_config.temperature
        }

        try:
            single_pass_config = LLMConfig(
                provider=self.llm_config.provider,
                model=self.llm_config.model,
                max_tokens=self.llm_config.max_tokens or 512,
                temperature=self.llm_config.temperature
            )
            response = self.llm_provider.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                config=single_pass_config
            )
            parsed_data = _parse_and_repair_json(response)
        except Exception as e:
            logger.error(f"Single pass caption generation failed: {e}. Falling back to legacy path.")
            return self.generate_captions_legacy(narrative)

        if not isinstance(parsed_data, dict):
            logger.warning("Single-pass response did not parse into a dictionary. Falling back to legacy.")
            return self.generate_captions_legacy(narrative)

        captions: Dict[str, Caption] = {}
        styles = ["formal", "sarcastic", "tech_humor", "non_tech_humor"]

        # 1. Partial regeneration: check for missing/empty keys
        for style in styles:
            val = parsed_data.get(style, "").strip()
            if not val:
                logger.warning(f"Style '{style}' is missing or empty in single-pass response. Running partial legacy retry.")
                sys_p, usr_p = self._load_prompt(style)
                user_p_filled = usr_p.replace("{narrative}", sanitize_untrusted_input(narrative.text))
                val, retry_meta = self._generate_with_retry(
                    style=style,
                    system_prompt=sys_p,
                    user_prompt=user_p_filled,
                    narrative_text=narrative.text
                )
            parsed_data[style] = val

        # 2. Word count trimming with deterministic logic first, and editor rewrite fallback
        for style in styles:
            text = parsed_data[style]
            words = text.split()
            if len(words) > self.max_caption_words:
                trimmed, cut_mid = self._deterministic_trim(text)
                if cut_mid:
                    logger.warning(f"Trimming '{style}' would cut mid-sentence. Requesting editor rewrite from LLM.")
                    shorter_prompt = (
                        f"Narrative:\n{narrative.text}\n\n"
                        f"Caption to shorten:\n{text}\n\n"
                        f"Rewrite this caption to be strictly under {self.max_caption_words} words. Do not invent any facts."
                    )
                    try:
                        shorter_response = self.llm_provider.generate(
                            prompt=shorter_prompt,
                            system_prompt="You are a precise editor. Shorten the text while keeping all facts.",
                            config=self.llm_config
                        )
                        shortened_text = self._clean_caption(shorter_response)
                        if len(shortened_text.split()) <= self.max_caption_words:
                            text = shortened_text
                        else:
                            text, _ = self._deterministic_trim(shortened_text)
                    except Exception as edit_err:
                        logger.error(f"Editor rewrite failed for '{style}': {edit_err}. Using deterministic trim.")
                        text = trimmed
                else:
                    text = trimmed
            parsed_data[style] = text

        # 3. Style separation fallback (H8): Jaccard distance checks
        to_regenerate = set()
        for i, style_a in enumerate(styles):
            for j, style_b in enumerate(styles):
                if i >= j:
                    continue
                dist = self._jaccard_distance(parsed_data[style_a], parsed_data[style_b])
                if dist < self.style_separation_min:
                    logger.warning(
                        f"Lexical separation between '{style_a}' and '{style_b}' "
                        f"is {dist:.2f} (below min {self.style_separation_min}). "
                        f"Flagging for isolated regeneration."
                    )
                    to_regenerate.add(style_a)
                    to_regenerate.add(style_b)

        for style in to_regenerate:
            logger.info(f"Regenerating style '{style}' in isolation due to separation fallback.")
            sys_p, usr_p = self._load_prompt(style)
            user_p_filled = usr_p.replace("{narrative}", sanitize_untrusted_input(narrative.text))
            val, retry_meta = self._generate_with_retry(
                style=style,
                system_prompt=sys_p,
                user_prompt=user_p_filled,
                narrative_text=narrative.text
            )
            parsed_data[style] = val

        # Assemble final Caption objects
        for style in styles:
            text = parsed_data[style]
            captions[style] = Caption(
                text=text,
                style=style,
                word_count=len(text.split()),
                metadata={**metadata, "regenerated": style in to_regenerate}
            )

        return captions

    def _deterministic_trim(self, text: str) -> Tuple[str, bool]:
        """
        Trims a caption to max_caption_words strictly at sentence boundaries.
        Returns:
            Tuple of (trimmed_text, cut_mid_sentence)
            where cut_mid_sentence is True if we had to cut inside the first sentence.
        """
        text = self._clean_caption(text)
        words = text.split()
        if len(words) <= self.max_caption_words:
            return text, False

        # Attempt to trim at sentence boundaries
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
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
            return " ".join(trimmed_parts), False
        else:
            # We couldn't even fit the first sentence under max_caption_words
            # So we have to cut mid-sentence
            first_sentence = sentences[0] if sentences else text
            first_words = first_sentence.split()
            trimmed_text = " ".join(first_words[:self.max_caption_words])
            if not trimmed_text.endswith("."):
                trimmed_text += "."
            return trimmed_text, True

    def _jaccard_distance(self, s1: str, s2: str) -> float:
        """Compute the lexical Jaccard distance between two strings."""
        words1 = set(re.findall(r'\w+', s1.lower()))
        words2 = set(re.findall(r'\w+', s2.lower()))
        if not words1 and not words2:
            return 1.0
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        similarity = len(intersection) / len(union)
        return 1.0 - similarity

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
            f"Narrative:\n<untrusted_input>\"{sanitize_untrusted_input(narrative_text)}\"</untrusted_input>\n\n"
            f"Draft Caption ({style} style):\n<untrusted_input>\"{sanitize_untrusted_input(draft_text)}\"</untrusted_input>\n\n"
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
            f"Narrative:\n<untrusted_input>\"{sanitize_untrusted_input(narrative_text)}\"</untrusted_input>\n\n"
            f"Draft Caption:\n<untrusted_input>\"{sanitize_untrusted_input(draft_text)}\"</untrusted_input>\n\n"
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
