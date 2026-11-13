"""Baseline B2: Vanilla Top-k Semantic RAG."""
import time
from typing import List, Dict, Any, Optional
from src.config import DialogueChunk
from src.preprocessing.indexer import ChunkVectorIndex
from src.models.generator import ResponseGenerator
from src.models.llm_client import LLMClient

class VanillaRAGBaseline:
    """Standard semantic retrieval baseline selecting top-k chunks by cosine similarity."""

    def __init__(self, index: ChunkVectorIndex, k: int = 5, token_budget: int = 4096, llm_client: Optional[LLMClient] = None):
        self.index = index
        self.k = k
        self.generator = ResponseGenerator(token_budget=token_budget, llm_client=llm_client)

    def run_query(self, query: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        # Retrieve top-k without threshold filtering or goal awareness
        scored = self.index.search(query, tau_loose=0.0, k_max=self.k)
        retrieved_chunks = [c for c, _ in scored]

        prompt = self.generator.assemble_prompt(retrieved_chunks, tier_b_digest="", query=query)
        answer = self.generator.generate(prompt)
        total_lat = (time.perf_counter() - t0) * 1000.0

        return {
            "baseline_id": "B2",
            "name": "Vanilla Top-k RAG",
            "query": query,
            "answer": answer,
            "context_tokens": len(prompt.split()),
            "retrieved_chunk_ids": [c.chunk_id for c in retrieved_chunks],
            "total_latency_ms": total_lat
        }

