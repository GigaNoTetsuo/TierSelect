"""Baseline B5: Hard Reset on Goal Shift."""
import time
from typing import List, Dict, Any, Optional
from src.config import DialogueChunk
from src.models.jev_classifier import JevClassifier
from src.models.generator import ResponseGenerator
from src.models.llm_client import LLMClient
from src.preprocessing.indexer import ChunkVectorIndex

class HardResetBaseline:
    """Purges dialogue history upon detected goal shift, retaining only active-goal chunks."""

    def __init__(self, classifier: JevClassifier, chunks: List[DialogueChunk], token_budget: int = 4096, llm_client: Optional[LLMClient] = None):
        self.classifier = classifier
        self.all_chunks = list(chunks)
        self.generator = ResponseGenerator(token_budget=token_budget, llm_client=llm_client)

    def run_query(self, query: str, shift_turn_index: Optional[int] = None) -> Dict[str, Any]:
        t0 = time.perf_counter()
        
        # If shift occurred, hard-reset deletes all chunks prior to shift
        if shift_turn_index is not None and shift_turn_index < len(self.all_chunks):
            retained_chunks = self.all_chunks[shift_turn_index:]
        else:
            # Check classifier prediction
            recent_texts = [c.text for c in self.all_chunks[-4:]]
            dec, _ = self.classifier.predict_goal_shift(query, recent_texts)
            if dec == "shift":
                retained_chunks = [] # Complete wipe
            else:
                retained_chunks = self.all_chunks

        # Build index over retained chunks only
        sub_index = ChunkVectorIndex()
        sub_index.build(retained_chunks)
        scored = sub_index.search(query, tau_loose=0.0, k_max=5)
        retrieved_chunks = [c for c, _ in scored]

        prompt = self.generator.assemble_prompt(retrieved_chunks, tier_b_digest="", query=query)
        answer = self.generator.generate(prompt)
        total_lat = (time.perf_counter() - t0) * 1000.0

        return {
            "baseline_id": "B5",
            "name": "Hard Reset",
            "query": query,
            "answer": answer,
            "context_tokens": len(prompt.split()),
            "retained_chunks_count": len(retained_chunks),
            "retrieved_chunk_ids": [c.chunk_id for c in retrieved_chunks],
            "total_latency_ms": total_lat
        }

