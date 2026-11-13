"""Unified generator and context assembly engine for TierSelect and all baselines.

Uses LLMClient for real generation when an API key is configured,
otherwise falls back to the deterministic grounded extractive generator.
"""
import re
from typing import List, Optional, Any
from src.config import DialogueChunk
from src.models.llm_client import LLMClient


class ResponseGenerator:
    """Assembles prompt and generates final grounded response."""

    def __init__(self, token_budget: int = 4096, llm_client: Optional[LLMClient] = None):
        self.token_budget = token_budget
        self.llm = llm_client

    def assemble_prompt(self, tier_a_chunks: List[DialogueChunk], tier_b_digest: str, query: str) -> str:
        """Assemble Tier A chronological chunks, Tier B digest, and query respecting token budget."""
        prompt_parts = []
        prompt_parts.append("You are an accurate conversational AI assistant. Answer the user query using the provided context history.")

        # Add Tier B digest if available
        digest_str = ""
        if tier_b_digest and tier_b_digest.strip():
            digest_str = f"\n[Historical Digest]:\n{tier_b_digest.strip()}\n"

        # Assemble Tier A chunks in chronological order
        # Sort by chunk_id or order of appearance
        tier_a_str = ""
        if tier_a_chunks:
            chunks_text = "\n\n".join([f"[{c.chunk_id}] {c.text}" for c in tier_a_chunks])
            tier_a_str = f"\n[Detailed Conversation History]:\n{chunks_text}\n"

        prompt = f"{prompt_parts[0]}\n{digest_str}{tier_a_str}\n[Current User Query]: {query}\n\nAnswer:"
        
        # Enforce token budget: if exceeding budget, reduce Tier A chunks from the beginning
        words = prompt.split()
        if len(words) > self.token_budget:
            # Budget exceeded: trim Tier A
            while len(words) > self.token_budget and len(tier_a_chunks) > 1:
                tier_a_chunks = tier_a_chunks[1:] # Drop oldest raw chunk
                chunks_text = "\n\n".join([f"[{c.chunk_id}] {c.text}" for c in tier_a_chunks])
                tier_a_str = f"\n[Detailed Conversation History]:\n{chunks_text}\n"
                prompt = f"{prompt_parts[0]}\n{digest_str}{tier_a_str}\n[Current User Query]: {query}\n\nAnswer:"
                words = prompt.split()

        return prompt

    def generate(self, prompt: str) -> str:
        """Generate response via LLM API or deterministic grounded extractor."""
        # Try LLM client first
        if self.llm and self.llm.is_available:
            result = self.llm.chat(
                prompt,
                system="You are an accurate conversational AI assistant. Answer concisely using only the provided context.",
                temperature=0.0,
                max_tokens=256
            )
            if result:
                return result

        # Grounded reference extractor: extracts the most relevant grounded sentence
        # from the prompt text matching query keywords
        lines = prompt.split('\n')
        context_sentences = []
        query_words = set()
        for line in lines:
            if line.startswith('[Current User Query]:'):
                q = line.replace('[Current User Query]:', '').strip()
                query_words = set(re.findall(r'\b\w{3,}\b', q.lower()))
            elif '[' in line and ']' in line and not line.startswith('You are'):
                # Extract sentence units from history
                sents = re.split(r'[\.\\?\!\n]+', line)
                for s in sents:
                    s = s.strip()
                    if len(s) > 10:
                        context_sentences.append(s)

        if not context_sentences:
            return "Based on the provided conversation history, no sufficient information is available."

        # Rank sentences by query overlap
        best_sent = context_sentences[0]
        best_score = -1
        for s in context_sentences:
            s_words = set(re.findall(r'\b\w{3,}\b', s.lower()))
            score = len(query_words & s_words)
            if score > best_score:
                best_score = score
                best_sent = s

        # Clean bracketed tags from output
        cleaned_answer = re.sub(r'\[.*?\]:?', '', best_sent).strip()
        return cleaned_answer if cleaned_answer else best_sent
