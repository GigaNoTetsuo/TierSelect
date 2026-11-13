"""Three-Tier Context Allocation with Goal Margins."""
from typing import List, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from src.config import DialogueChunk, CandidateChunk, TierSelectConfig
from src.models.jev_classifier import JevClassifier

class TierAllocator:
    """Classifies candidates and partitions them into Tier A (raw), Tier B (compressed), and Tier C (dropped)."""

    def __init__(self, classifier: JevClassifier, config: TierSelectConfig):
        self.classifier = classifier
        self.config = config

    def allocate(
        self,
        candidates: List[Tuple[DialogueChunk, float]],
        query: str,
        goal_context: str
    ) -> Tuple[List[DialogueChunk], List[DialogueChunk], List[DialogueChunk], List[CandidateChunk]]:
        """Score and partition retrieved candidates into Tier A, Tier B, and Tier C.
        
        Returns:
            (tier_a_chunks, tier_b_chunks, tier_c_chunks, all_scored_candidates)
        """
        if not candidates:
            return [], [], [], []

        # 1. Score candidates in parallel
        def _score_one(item):
            chunk, ret_score = item
            is_active = chunk.is_active
            typed_dec, p = self.classifier.score_relevance(chunk, query, goal_context, is_active)
            return CandidateChunk(
                chunk=chunk,
                retrieval_score=ret_score,
                relevance_probability=p,
                decision_type=typed_dec
            )

        max_workers = min(8, len(candidates))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            scored_candidates = list(executor.map(_score_one, candidates))

        # 2. Sort descending by calibrated probability p
        scored_candidates.sort(key=lambda x: x.relevance_probability, reverse=True)

        tier_a: List[DialogueChunk] = []
        tier_b: List[DialogueChunk] = []
        tier_c: List[DialogueChunk] = []

        # 3. Partition with goal margin delta
        theta_hi = self.config.theta_hi
        theta_lo = self.config.theta_lo
        delta = self.config.delta
        n_raw = self.config.n_raw

        for cand in scored_candidates:
            p = cand.relevance_probability
            is_active = cand.chunk.is_active
            
            # Archived goal chunks must clear higher bar: theta_hi + delta
            effective_hi = theta_hi if is_active else (theta_hi + delta)

            if p >= effective_hi and len(tier_a) < n_raw:
                cand.assigned_tier = "A"
                tier_a.append(cand.chunk)
            elif p >= theta_lo:
                # Includes candidates meeting theta_lo and high scorers exceeding n_raw cap
                cand.assigned_tier = "B"
                tier_b.append(cand.chunk)
            else:
                cand.assigned_tier = "C"
                tier_c.append(cand.chunk)

        return tier_a, tier_b, tier_c, scored_candidates

