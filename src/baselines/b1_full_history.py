"""Baseline B1: Full History in Prompt (Long-Context)."""
import time
from typing import List, Dict, Any, Optional
from src.config import DialogueChunk
from src.models.generator import ResponseGenerator
from src.models.llm_client import LLMClient

class FullHistoryBaseline:
    """Concatenates all conversation chunks directly into prompt up to token budget."""

    def __init__(self, chunks: List[DialogueChunk], token_budget: int = 4096, llm_client: Optional[LLMClient] = None):
        self.chunks = list(chunks)
        self.generator = ResponseGenerator(token_budget=token_budget, llm_client=llm_client)

    def run_query(self, query: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        # Feed all history chunks verbatim in chronological order
        prompt = self.generator.assemble_prompt(self.chunks, tier_b_digest="", query=query)
        answer = self.generator.generate(prompt)
        total_lat = (time.perf_counter() - t0) * 1000.0

        return {
            "baseline_id": "B1",
            "name": "Full History",
            "query": query,
            "answer": answer,
            "context_tokens": len(prompt.split()),
            "retrieved_chunk_ids": [c.chunk_id for c in self.chunks],
            "total_latency_ms": total_lat
        }

