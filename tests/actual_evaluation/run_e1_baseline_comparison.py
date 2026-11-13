"""Experiment E1: Baseline Comparison (RQ1).

Compares TierSelect against Baselines B1 to B5 on answer accuracy, evidence recall, context tokens, and latency.
"""
import os
import json
import time
from typing import Dict, Any, List
from src.config import TierSelectConfig, Episode
from src.preprocessing.indexer import ChunkVectorIndex
from src.models.jev_classifier import JevClassifier
from src.tierselect.pipeline import TierSelectPipeline
from src.baselines.b1_full_history import FullHistoryBaseline
from src.baselines.b2_vanilla_rag import VanillaRAGBaseline
from src.baselines.b3_memory_bank import MemoryBankBaseline
from src.baselines.b4_recursive_summary import RecursiveSummarizationBaseline
from src.baselines.b5_hard_reset import HardResetBaseline
from src.evaluation.selection_metrics import compute_evidence_recall
from src.evaluation.answer_metrics import compute_token_f1, compute_exact_match, evaluate_llm_judge
from src.evaluation.efficiency_metrics import compute_token_reduction_rate

def run_experiment_e1(episodes: List[Episode], config: TierSelectConfig) -> Dict[str, Any]:
    """Run E1 across all test episodes."""
    print("\n================== RUNNING EXPERIMENT E1: BASELINE COMPARISON ==================")
    models = ["B1_FullHistory", "B2_VanillaRAG", "B3_MemoryBank", "B4_RecursiveSummary", "B5_HardReset", "TierSelect"]
    
    metrics: Dict[str, Dict[str, List[float]]] = {
        m: {"accuracy": [], "evidence_recall": [], "context_tokens": [], "latency_ms": []}
        for m in models
    }

    classifier = JevClassifier(theta_shift=config.theta_shift)

    for i, ep in enumerate(episodes):
        all_chunks = ep.goal_a_chunks + ep.goal_b_chunks
        gold_ids = ep.gold_evidence_chunk_ids
        q = ep.test_query
        gold_ans = ep.gold_answer

        # Index for B2 and TierSelect
        index = ChunkVectorIndex()
        index.build(all_chunks)

        # 1. B1 Full History
        b1 = FullHistoryBaseline(all_chunks)
        res_b1 = b1.run_query(q)
        f1_b1 = compute_token_f1(res_b1["answer"], gold_ans)
        rec_b1 = compute_evidence_recall(res_b1["retrieved_chunk_ids"], gold_ids)
        metrics["B1_FullHistory"]["accuracy"].append(f1_b1)
        metrics["B1_FullHistory"]["evidence_recall"].append(rec_b1)
        metrics["B1_FullHistory"]["context_tokens"].append(res_b1["context_tokens"])
        metrics["B1_FullHistory"]["latency_ms"].append(res_b1["total_latency_ms"])

        # 2. B2 Vanilla RAG
        b2 = VanillaRAGBaseline(index, k=config.n_raw)
        res_b2 = b2.run_query(q)
        f1_b2 = compute_token_f1(res_b2["answer"], gold_ans)
        rec_b2 = compute_evidence_recall(res_b2["retrieved_chunk_ids"], gold_ids)
        metrics["B2_VanillaRAG"]["accuracy"].append(f1_b2)
        metrics["B2_VanillaRAG"]["evidence_recall"].append(rec_b2)
        metrics["B2_VanillaRAG"]["context_tokens"].append(res_b2["context_tokens"])
        metrics["B2_VanillaRAG"]["latency_ms"].append(res_b2["total_latency_ms"])

        # 3. B3 MemoryBank
        b3 = MemoryBankBaseline(all_chunks, k=config.n_raw)
        res_b3 = b3.run_query(q)
        f1_b3 = compute_token_f1(res_b3["answer"], gold_ans)
        rec_b3 = compute_evidence_recall(res_b3["retrieved_chunk_ids"], gold_ids)
        metrics["B3_MemoryBank"]["accuracy"].append(f1_b3)
        metrics["B3_MemoryBank"]["evidence_recall"].append(rec_b3)
        metrics["B3_MemoryBank"]["context_tokens"].append(res_b3["context_tokens"])
        metrics["B3_MemoryBank"]["latency_ms"].append(res_b3["total_latency_ms"])

        # 4. B4 Recursive Summary
        b4 = RecursiveSummarizationBaseline(all_chunks)
        res_b4 = b4.run_query(q)
        f1_b4 = compute_token_f1(res_b4["answer"], gold_ans)
        rec_b4 = compute_evidence_recall(res_b4["retrieved_chunk_ids"], gold_ids)
        metrics["B4_RecursiveSummary"]["accuracy"].append(f1_b4)
        metrics["B4_RecursiveSummary"]["evidence_recall"].append(rec_b4)
        metrics["B4_RecursiveSummary"]["context_tokens"].append(res_b4["context_tokens"])
        metrics["B4_RecursiveSummary"]["latency_ms"].append(res_b4["total_latency_ms"])

        # 5. B5 Hard Reset
        b5 = HardResetBaseline(classifier, all_chunks)
        res_b5 = b5.run_query(q, shift_turn_index=ep.shift_turn_index)
        f1_b5 = compute_token_f1(res_b5["answer"], gold_ans)
        rec_b5 = compute_evidence_recall(res_b5["retrieved_chunk_ids"], gold_ids)
        metrics["B5_HardReset"]["accuracy"].append(f1_b5)
        metrics["B5_HardReset"]["evidence_recall"].append(rec_b5)
        metrics["B5_HardReset"]["context_tokens"].append(res_b5["context_tokens"])
        metrics["B5_HardReset"]["latency_ms"].append(res_b5["total_latency_ms"])

        # 6. Proposed TierSelect
        pipeline = TierSelectPipeline(config, index, classifier)
        # Mark active chunks according to episode structure
        if ep.shift_turn_index is not None:
            for idx_c, chunk in enumerate(all_chunks):
                chunk.is_active = (idx_c >= ep.shift_turn_index)
        res_ts = pipeline.run_query(q)
        f1_ts = compute_token_f1(res_ts["answer"], gold_ans)
        # Recall in Tier A + Tier B
        all_selected = res_ts["tier_a_chunks"] + res_ts["tier_b_chunks"]
        rec_ts = compute_evidence_recall(all_selected, gold_ids)
        metrics["TierSelect"]["accuracy"].append(f1_ts)
        metrics["TierSelect"]["evidence_recall"].append(rec_ts)
        metrics["TierSelect"]["context_tokens"].append(res_ts["context_tokens"])
        metrics["TierSelect"]["latency_ms"].append(res_ts["latencies"]["total_latency_ms"])

    # Aggregate results
    summary = {}
    b1_tokens = sum(metrics["B1_FullHistory"]["context_tokens"]) / max(1, len(episodes))
    for m in models:
        avg_acc = sum(metrics[m]["accuracy"]) / max(1, len(episodes))
        avg_rec = sum(metrics[m]["evidence_recall"]) / max(1, len(episodes))
        avg_tok = sum(metrics[m]["context_tokens"]) / max(1, len(episodes))
        avg_lat = sum(metrics[m]["latency_ms"]) / max(1, len(episodes))
        reduction = compute_token_reduction_rate(b1_tokens, avg_tok)
        summary[m] = {
            "Answer Accuracy (F1)": round(avg_acc, 4),
            "Evidence Recall": round(avg_rec, 4),
            "Context Tokens": round(avg_tok, 1),
            "Token Reduction (%)": reduction,
            "Latency (ms)": round(avg_lat, 2)
        }

    return summary

