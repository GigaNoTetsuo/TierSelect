"""Baseline B4: Recursive Summarization (Rsum)."""
import time
from typing import List, Dict, Any, Optional
from src.config import DialogueChunk
from src.models.compressor import TierBCompressor
from src.models.generator import ResponseGenerator
from src.models.llm_client import LLMClient

class RecursiveSummarizationBaseline:
    """Summary-memory baseline maintaining a running hierarchical digest of dialogue history."""

    def __init__(self, chunks: List[DialogueChunk], token_budget: int = 4096, llm_client: Optional[LLMClient] = None):
        self.chunks = list(chunks)
        self.compressor = TierBCompressor(max_tokens=250, llm_client=llm_client)
        self.generator = ResponseGenerator(token_budget=token_budget, llm_client=llm_client)
        # Compute summary digest of history
        self.running_summary = self.compressor.compress(self.chunks)

    def run_query(self, query: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        # Feed running summary digest + most recent turns
        recent_chunks = self.chunks[-2:] if len(self.chunks) >= 2 else self.chunks
        prompt = self.generator.assemble_prompt(recent_chunks, tier_b_digest=self.running_summary, query=query)
        answer = self.generator.generate(prompt)
        total_lat = (time.perf_counter() - t0) * 1000.0

        return {
            "baseline_id": "B4",
            "name": "Recursive Summarization (Rsum)",
            "query": query,
            "answer": answer,
            "context_tokens": len(prompt.split()),
            "retrieved_chunk_ids": [c.chunk_id for c in recent_chunks],
            "total_latency_ms": total_lat
        }

