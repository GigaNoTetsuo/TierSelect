"""Experiment E2: Goal-Shift Dynamics & Margin Analysis (RQ2).

Evaluates shift-detection F1, recovery of old-goal evidence, and stale-context rate across varying distractor lengths and margin delta.
"""
from typing import Dict, Any, List
from src.config import TierSelectConfig, Episode
from src.preprocessing.indexer import ChunkVectorIndex
from src.models.jev_classifier import JevClassifier
from src.tierselect.pipeline import TierSelectPipeline
from src.baselines.b5_hard_reset import HardResetBaseline
from src.baselines.b2_vanilla_rag import VanillaRAGBaseline
from src.evaluation.goal_shift_metrics import (
    compute_goal_shift_metrics,
    compute_old_goal_evidence_recovery,
    compute_stale_context_rate
)

def run_experiment_e2(
    shift_episodes: List[Episode],
    topi_shift_pairs: List[Dict[str, Any]],
    config: TierSelectConfig
) -> Dict[str, Any]:
    print("\n================== RUNNING EXPERIMENT E2: GOAL-SHIFT ANALYSIS ==================")
    classifier = JevClassifier(theta_shift=config.theta_shift)

    # 1. Topic-Shift Detection F1 on TopiOCQA
    y_true = [p["ground_truth_shift"] for p in topi_shift_pairs]
    y_pred = []
    for p in topi_shift_pairs:
        dec, _ = classifier.predict_goal_shift(p["query"], p["history"])
        y_pred.append(1 if dec == "shift" else 0)

    shift_metrics = compute_goal_shift_metrics(y_true, y_pred)

    # 2. Compare TierSelect vs HardReset vs VanillaRAG on Old-Goal Recovery and Stale Contamination
    ts_recoveries = []
    hr_recoveries = []
    vr_recoveries = []
    ts_stale_rates = []
    vr_stale_rates = []

    for ep in shift_episodes:
        all_chunks = ep.goal_a_chunks + ep.goal_b_chunks
        goal_a_ids = [c.chunk_id for c in ep.goal_a_chunks]
        goal_b_ids = [c.chunk_id for c in ep.goal_b_chunks]
        q = ep.test_query

        index = ChunkVectorIndex()
        index.build(all_chunks)

        # Mark active vs archived
        if ep.shift_turn_index is not None:
            for idx_c, chunk in enumerate(all_chunks):
                chunk.is_active = (idx_c >= ep.shift_turn_index)

        # TierSelect
        pipeline = TierSelectPipeline(config, index, classifier)
        res_ts = pipeline.run_query(q)
        selected_ts = res_ts["tier_a_chunks"] + res_ts["tier_b_chunks"]
        ts_recoveries.append(compute_old_goal_evidence_recovery(selected_ts, goal_a_ids))
        ts_stale_rates.append(compute_stale_context_rate(res_ts["tier_a_chunks"], goal_b_ids))

        # Hard Reset
        b5 = HardResetBaseline(classifier, all_chunks)
        res_hr = b5.run_query(q, shift_turn_index=ep.shift_turn_index)
        hr_recoveries.append(compute_old_goal_evidence_recovery(res_hr["retrieved_chunk_ids"], goal_a_ids))

        # Vanilla RAG (No goal tracking)
        b2 = VanillaRAGBaseline(index, k=config.n_raw)
        res_vr = b2.run_query(q)
        vr_recoveries.append(compute_old_goal_evidence_recovery(res_vr["retrieved_chunk_ids"], goal_a_ids))
        vr_stale_rates.append(compute_stale_context_rate(res_vr["retrieved_chunk_ids"], goal_b_ids))

    # 3. Margin Delta Sensitivity Study: delta in [0.0, 0.10, 0.20, 0.30]
    delta_results = {}
    for d in [0.0, 0.10, 0.20, 0.30]:
        cfg_d = TierSelectConfig(**config.model_dump())
        cfg_d.delta = d
        d_stale = []
        d_rec = []
        for ep in shift_episodes[:15]:
            all_chunks = ep.goal_a_chunks + ep.goal_b_chunks
            goal_a_ids = [c.chunk_id for c in ep.goal_a_chunks]
            goal_b_ids = [c.chunk_id for c in ep.goal_b_chunks]
            index = ChunkVectorIndex()
            index.build(all_chunks)
            if ep.shift_turn_index is not None:
                for idx_c, chunk in enumerate(all_chunks):
                    chunk.is_active = (idx_c >= ep.shift_turn_index)
            pipe = TierSelectPipeline(cfg_d, index, classifier)
            res = pipe.run_query(ep.test_query)
            d_stale.append(compute_stale_context_rate(res["tier_a_chunks"], goal_b_ids))
            d_rec.append(compute_old_goal_evidence_recovery(res["tier_a_chunks"] + res["tier_b_chunks"], goal_a_ids))

        delta_results[f"delta={d}"] = {
            "Old-Goal Recovery": round(sum(d_rec) / max(1, len(d_rec)), 4),
            "Tier-A Stale-Context Rate": round(sum(d_stale) / max(1, len(d_stale)), 4)
        }

    return {
        "TopiOCQA_Shift_Detection": shift_metrics,
        "Old_Goal_Recovery": {
            "TierSelect (Demotion)": round(sum(ts_recoveries) / max(1, len(ts_recoveries)), 4),
            "Hard Reset (B5)": round(sum(hr_recoveries) / max(1, len(hr_recoveries)), 4),
            "Vanilla RAG (No Goal Tracking)": round(sum(vr_recoveries) / max(1, len(vr_recoveries)), 4)
        },
        "Stale_Context_Rate_Tier_A": {
            "TierSelect (with Goal Margin)": round(sum(ts_stale_rates) / max(1, len(ts_stale_rates)), 4),
            "Vanilla RAG (uncontrolled)": round(sum(vr_stale_rates) / max(1, len(vr_stale_rates)), 4)
        },
        "Margin_Delta_Sensitivity": delta_results
    }

