"""Experiment E5: Efficiency Analysis (RQ3).

Profiles latency and cost breakdown across each stage of TierSelect:
- Stage 1: Goal Tracking
- Stage 2: Loose Retrieval
- Stage 3: Relevance Classification & Tiering
- Stage 4: Tier B Compression
- Stage 5: Generation
"""
from typing import Dict, Any, List
from src.config import TierSelectConfig, Episode
from src.preprocessing.indexer import ChunkVectorIndex
from src.models.jev_classifier import JevClassifier
from src.tierselect.pipeline import TierSelectPipeline

def run_experiment_e5(episodes: List[Episode], config: TierSelectConfig) -> Dict[str, Any]:
    print("\n================== RUNNING EXPERIMENT E5: EFFICIENCY ANALYSIS ==================")
    classifier = JevClassifier(theta_shift=config.theta_shift)

    stage_latencies = {
        "goal_tracking_ms": [],
        "retrieval_ms": [],
        "relevance_tiering_ms": [],
        "compression_ms": [],
        "generation_ms": [],
        "total_latency_ms": []
    }

    for ep in episodes[:20]:
        chunks = ep.goal_a_chunks + ep.goal_b_chunks
        index = ChunkVectorIndex()
        index.build(chunks)
        pipeline = TierSelectPipeline(config, index, classifier)
        res = pipeline.run_query(ep.test_query)
        for k, v in res["latencies"].items():
            if k in stage_latencies:
                stage_latencies[k].append(v)

    avg_lat = {
        k: round(sum(v) / max(1, len(v)), 2)
        for k, v in stage_latencies.items()
    }

    # Cost breakdown per 1,000 queries
    # Stage 1: Jev Goal Tracking (0 tokens API, zero-gradient = $0.00)
    # Stage 2: FAISS inner-product retrieval = $0.00
    # Stage 3: Jev Relevance classification = $0.00
    # Stage 4: Tier B LLM compression (if using small 3B/7B or gpt-4o-mini ~ 150 tokens = $0.02)
    # Stage 5: Generation (~500 tokens input, ~50 tokens output = $0.08)
    cost_breakdown = {
        "Stage_1_Goal_Tracking": "$0.00",
        "Stage_2_Loose_Retrieval": "$0.00",
        "Stage_3_Relevance_Tiering": "$0.00",
        "Stage_4_Digest_Compression": "$0.02 / 1K queries",
        "Stage_5_Answer_Generation": "$0.08 / 1K queries",
        "Total_Estimated_Cost_per_1k_queries": "$0.10"
    }

    return {
        "Stage_Latencies_ms": avg_lat,
        "Cost_Breakdown_per_1K_Queries": cost_breakdown
    }

