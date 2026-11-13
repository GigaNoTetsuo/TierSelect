"""Tier B Digest Compressor.

Merges mid-scoring Tier B chunks into a compact digest of at most L_c = 150 tokens,
preserving names, dates, numbers, stated preferences, and chunk ID back-pointers.

Uses LLMClient for real compression when available, otherwise falls back to
regex-based entity extraction.
"""
import re
from typing import List, Optional
from src.config import DialogueChunk
from src.models.llm_client import LLMClient


class TierBCompressor:
    """Compresses Tier B chunks into an entity-dense digest with attached chunk IDs."""

    def __init__(self, max_tokens: int = 150, llm_client: Optional[LLMClient] = None):
        self.max_tokens = max_tokens
        self.llm = llm_client

    def compress(self, chunks: List[DialogueChunk]) -> str:
        """Compress a list of Tier B dialogue chunks into a compact digest."""
        if not chunks:
            return ""

        # Fast pass-through: if total content is already within the token budget (<= max_tokens),
        # no LLM compression call is necessary, saving ~600-800ms of latency!
        total_words = sum(len(c.text.split()) for c in chunks)
        if total_words <= self.max_tokens:
            return " ".join([f"[{c.chunk_id}]: {c.text}" for c in chunks])

        # Try LLM-based compression when chunks exceed max_tokens
        if self.llm and self.llm.is_available:
            result = self._compress_with_llm(chunks)
            if result:
                return result

        # Fallback: regex-based entity extraction
        return self._compress_extractive(chunks)

    def _compress_with_llm(self, chunks: List[DialogueChunk]) -> str:
        """Use LLM to produce an entity-preserving digest."""
        chunk_texts = "\n".join([f"[{c.chunk_id}]: {c.text}" for c in chunks])

        prompt = (
            f"Compress the following conversation chunks into a concise digest of at most {self.max_tokens} words.\n"
            f"RULES:\n"
            f"- Preserve ALL names, numbers, dates, and stated preferences exactly.\n"
            f"- Keep chunk ID references like [chunk_id] for traceability.\n"
            f"- Remove filler words and redundant phrasing.\n"
            f"- Output ONLY the digest, no explanations.\n\n"
            f"CHUNKS:\n{chunk_texts}\n\n"
            f"COMPRESSED DIGEST:"
        )

        result = self.llm.chat(prompt, temperature=0.0, max_tokens=self.max_tokens + 50)
        if result:
            # Enforce token limit
            words = result.split()
            if len(words) > self.max_tokens:
                result = " ".join(words[:self.max_tokens])
            return result
        return ""

    def _compress_extractive(self, chunks: List[DialogueChunk]) -> str:
        """Fallback: regex-based entity extraction when no LLM is available."""
        digest_items = []
        total_tokens_approx = 0

        # Preserve key entities: dates, numbers, preferences, proper nouns
        entity_pattern = re.compile(
            r'(\b\d{4}\b|\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\b|\b(?:prefer|like|want|dislike|interested|love|booked|traveled|stayed)\b[^\.\\,\;\n]+|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            re.IGNORECASE
        )

        for chunk in chunks:
            text = chunk.text
            # Extract key informative phrases
            matches = entity_pattern.findall(text)
            clean_matches = [m.strip() for m in matches if len(m.strip()) > 3]

            if clean_matches:
                salient_snippet = "; ".join(clean_matches[:3])
            else:
                # Take first 12 words of turn
                words = text.split()
                salient_snippet = " ".join(words[:12])

            item_text = f"[{chunk.chunk_id}]: {salient_snippet}."
            item_token_count = len(item_text.split())

            if total_tokens_approx + item_token_count > self.max_tokens:
                break

            digest_items.append(item_text)
            total_tokens_approx += item_token_count

        return " ".join(digest_items)
