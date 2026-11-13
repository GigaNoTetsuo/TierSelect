"""Complete TierSelect End-to-End Execution Pipeline."""
import time
from typing import List, Dict, Any, Optional
from src.config import DialogueChunk, TierSelectConfig
from src.models.jev_classifier import JevClassifier
from src.models.compressor import TierBCompressor
from src.models.generator import ResponseGenerator
from src.models.llm_client import LLMClient
from src.preprocessing.indexer import ChunkVectorIndex
from src.storage.postgres_store import GoalStore
from src.tierselect.goal_tracker import GoalTracker
from src.tierselect.retriever import LooseRetriever
from src.tierselect.tier_allocator import TierAllocator

class TierSelectPipeline:
    """End-to-End Goal-Aware Three-Tier Context Selection and Generation Pipeline."""

    def __init__(
        self,
        config: TierSelectConfig,
        index: ChunkVectorIndex,
        classifier: JevClassifier,
        llm_client: Optional[LLMClient] = None,
        goal_store: Optional[GoalStore] = None,
    ):
        self.config = config
        self.index = index
        self.classifier = classifier
        self.llm = llm_client or LLMClient()
        self.store = goal_store or GoalStore()
        self.goal_tracker = GoalTracker(classifier, store=self.store)
        self.retriever = LooseRetriever(index, tau_loose=config.tau_loose, k_max=config.k_max)
        self.allocator = TierAllocator(classifier, config)
        self.compressor = TierBCompressor(max_tokens=config.max_digest_tokens, llm_client=self.llm)
        self.generator = ResponseGenerator(token_budget=config.token_budget, llm_client=self.llm)

    def run_query(self, query: str) -> Dict[str, Any]:
        """Execute TierSelect for a user query, returning the final answer and inspectable telemetry."""
        latencies: Dict[str, float] = {}

        # 1 & 2. Concurrent Goal Tracking and Loose Semantic Retrieval
        from concurrent.futures import ThreadPoolExecutor

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as executor:
            fut_shift = executor.submit(self.goal_tracker.process_query, query)
            fut_ret = executor.submit(self.retriever.retrieve, query)
            shift_decision, p_shift = fut_shift.result()
            raw_candidates = fut_ret.result()

        goal_context = self.goal_tracker.get_goal_context()
        parallel_lat = (time.perf_counter() - t0) * 1000.0
        latencies["goal_tracking_ms"] = parallel_lat
        latencies["retrieval_ms"] = parallel_lat

        # 3. Relevance Classification & Tier Allocation (Parallel Candidate Scoring)
        t0 = time.perf_counter()
        tier_a, tier_b, tier_c, scored_candidates = self.allocator.allocate(raw_candidates, query, goal_context)
        latencies["relevance_tiering_ms"] = (time.perf_counter() - t0) * 1000.0

        # 4. Compression of Tier B
        t0 = time.perf_counter()
        digest = self.compressor.compress(tier_b)
        latencies["compression_ms"] = (time.perf_counter() - t0) * 1000.0

        # 5. Context Assembly and Generation
        t0 = time.perf_counter()
        prompt = self.generator.assemble_prompt(tier_a, digest, query)
        answer = self.generator.generate(prompt)
        latencies["generation_ms"] = (time.perf_counter() - t0) * 1000.0
        latencies["total_latency_ms"] = sum(latencies.values())

        # Inspectable per-chunk telemetry
        decisions_log = [
            {
                "chunk_id": c.chunk.chunk_id,
                "is_active": c.chunk.is_active,
                "retrieval_score": round(c.retrieval_score, 4),
                "calibrated_p": round(c.relevance_probability, 4),
                "decision_type": c.decision_type,
                "assigned_tier": c.assigned_tier
            }
            for c in scored_candidates
        ]

        context_tokens = len(prompt.split())

        return {
            "query": query,
            "answer": answer,
            "prompt": prompt,
            "goal_shift_decision": shift_decision,
            "p_shift": round(p_shift, 4),
            "tier_a_chunks": [c.chunk_id for c in tier_a],
            "tier_b_chunks": [c.chunk_id for c in tier_b],
            "tier_c_chunks": [c.chunk_id for c in tier_c],
            "digest_text": digest,
            "context_tokens": context_tokens,
            "latencies": latencies,
            "decisions_log": decisions_log,
            "storage_backend": "postgres" if self.store.is_persistent else "in-memory",
            "llm_provider": repr(self.llm),
        }

    def record_turn(self, user_turn: str, assistant_turn: str, session_id: str = "default_session") -> DialogueChunk:
        """Record a completed turn pair into active goal segment and vector index."""
        chunk_id = f"chk_{session_id}_{len(self.index.chunks) + 1:04d}"
        text = f"User: {user_turn}\nAssistant: {assistant_turn}"
        chunk = DialogueChunk(
            chunk_id=chunk_id,
            session_id=session_id,
            user_turn=user_turn,
            assistant_turn=assistant_turn,
            text=text,
            is_active=True
        )
        self.goal_tracker.add_chunk_to_active_segment(chunk)
        self.index.add_chunk(chunk)
        return chunk
