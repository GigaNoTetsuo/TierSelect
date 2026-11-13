"""Text cleaning and normalization utilities for dialogue turns."""
import re
import unicodedata

def normalize_text(text: str) -> str:
    """Normalize unicode, whitespace, and strip formatting artifacts."""
    if not text:
        return ""
    # Normalize unicode
    text = unicodedata.normalize("NFKD", str(text))
    # Replace multiple whitespaces/newlines with single space
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    # Strip leading/trailing whitespaces
    return text.strip()

def clean_dialogue_turn(raw_text: str) -> str:
    """Clean turn text, removing noisy prefix markers like 'User:', 'Assistant:', etc."""
    text = normalize_text(raw_text)
    # Remove common prefix artifacts
    text = re.sub(r"^(User|Human|Assistant|AI|Agent|Speaker\s*[A-Za-z0-9]+):\s*", "", text, flags=re.IGNORECASE)
    return text.strip()

