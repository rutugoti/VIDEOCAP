import re
import os
import json
import hashlib
import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)

CACHE_DIR = ".cache"

def sanitize_untrusted_input(text: str) -> str:
    """
    Sanitizes untrusted text input (OCR, audio transcripts, user metadata) to prevent prompt injection.
    It redacts known instructions override attempts and escapes characters used in template formatting.
    """
    if not text:
        return ""
        
    # Standard injection terms/phrases to intercept
    injection_patterns = [
        r"ignore\s+(?:all\s+)?previous\s+instructions",
        r"ignore\s+(?:all\s+)?instructions",
        r"forget\s+(?:all\s+)?previous\s+instructions",
        r"forget\s+(?:all\s+)?instructions",
        r"system\s+prompt",
        r"you\s+are\s+now\s+a",
        r"do\s+not\s+follow",
        r"overwrite\s+rules",
        r"ignore\s+rules",
        r"ignore\s+above"
    ]
    
    cleaned = text
    for pattern in injection_patterns:
        cleaned = re.sub(pattern, "[REDACTED_INJECTION_ATTEMPT]", cleaned, flags=re.IGNORECASE)
    
    # Escape curly braces for template safety and escape quotes
    cleaned = cleaned.replace("{", "{{").replace("}", "}}")
    # Clean up double XML tag sequences that could attempt to close delimiters
    cleaned = cleaned.replace("</untrusted_input>", "[CLEANED_TAG]")
    cleaned = cleaned.replace("<untrusted_input>", "[CLEANED_TAG]")
    
    return cleaned.strip()

def get_cache_key(video_path: str, phase: str, extra_config: dict) -> str:
    """Generates a unique cache key based on video path, mod time, and extra config."""
    try:
        mtime = os.path.getmtime(video_path)
    except Exception:
        mtime = 0.0
    
    config_str = json.dumps(extra_config, sort_keys=True)
    hash_input = f"{video_path}_{mtime}_{phase}_{config_str}"
    return hashlib.md5(hash_input.encode("utf-8")).hexdigest()

def load_from_cache(key: str) -> Optional[Any]:
    """Loads cached data if available."""
    cache_path = os.path.join(CACHE_DIR, f"{key}.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                logger.info(f"Cache hit for key: {key}")
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read cache file {cache_path}: {e}")
    return None

def save_to_cache(key: str, data: Any) -> None:
    """Saves data to cache."""
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        cache_path = os.path.join(CACHE_DIR, f"{key}.json")
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved cache for key: {key}")
    except Exception as e:
        logger.warning(f"Failed to write cache file: {e}")
