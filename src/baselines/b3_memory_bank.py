"""Baseline B3: MemoryBank."""
import time
from typing import List, Dict, Any, Optional
from src.config import DialogueChunk
from src.preprocessing.indexer import ChunkVectorIndex
from src.models.generator import ResponseGenerator
from src.models.llm_client import LLMClient

class MemoryBankBaseline:
    """Memory baseline from long-term memory studies using fact-extracted memories."""

    def __init__(self, chunks: List[DialogueChunk], k: int = 5, token_budget: int = 4096, llm_client: Optional[LLMClient] = None):
        self.k = k
        self.generator = ResponseGenerator(token_budget=token_budget, llm_client=llm_client)
        
        # Build memory bank entries from chunks
        self.memory_chunks = []
        for c in chunks:
            # MemoryBank extracts salient user facts and preferences
            mem_text = f"[Memory item]: {c.user_turn} -> {c.assistant_turn}"
            self.memory_chunks.append(DialogueChunk(
                chunk_id=c.chunk_id,
                session_id=c.session_id,
                user_turn=c.user_turn,
                assistant_turn=c.assistant_turn,
                text=mem_text,
                metadata=c.metadata
            ))
        self.index = ChunkVectorIndex()
        self.index.build(self.memory_chunks)

    def run_query(self, query: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        scored = self.index.search(query, tau_loose=0.0, k_max=self.k)
        retrieved_chunks = [c for c, _ in scored]

        prompt = self.generator.assemble_prompt(retrieved_chunks, tier_b_digest="", query=query)
        answer = self.generator.generate(prompt)
        total_lat = (time.perf_counter() - t0) * 1000.0

        return {
            "baseline_id": "B3",
            "name": "MemoryBank",
            "query": query,
            "answer": answer,
            "context_tokens": len(prompt.split()),
            "retrieved_chunk_ids": [c.chunk_id for c in retrieved_chunks],
            "total_latency_ms": total_lat
        }

