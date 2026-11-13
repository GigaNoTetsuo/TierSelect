"""Short baseline comparative evaluation using live Jev, live LLM, and PostgreSQL."""
import os
import time
import json
from dotenv import load_dotenv

load_dotenv()

from src.config import TierSelectConfig, Episode, DialogueChunk, DATA_DIR
from src.preprocessing.chunker import DialogueChunker
from src.preprocessing.episode_builder import EpisodeBuilder
from src.preprocessing.indexer import ChunkVectorIndex
from src.models.jev_classifier import JevClassifier
from src.models.llm_client import LLMClient
from src.storage.postgres_store import GoalStore
from src.tierselect.pipeline import TierSelectPipeline
from src.baselines.b1_full_history import FullHistoryBaseline
from src.baselines.b2_vanilla_rag import VanillaRAGBaseline
from src.baselines.b3_memory_bank import MemoryBankBaseline
from src.baselines.b4_recursive_summary import RecursiveSummarizationBaseline
from src.baselines.b5_hard_reset import HardResetBaseline
from src.evaluation.selection_metrics import compute_evidence_recall
from src.evaluation.answer_metrics import compute_token_f1, compute_exact_match, compute_bert_f1
from src.evaluation.goal_shift_metrics import compute_old_goal_evidence_recovery, compute_stale_context_rate
from src.evaluation.efficiency_metrics import compute_token_reduction_rate


def build_short_episodes():
    """Build controlled shift-and-return test episodes."""
    print("Loading 3 controlled shift-and-return evaluation episodes...", flush=True)
    episodes = []
    # Episode 1: Travel -> Cooking -> Travel
    g1 = [
        DialogueChunk("e1_01", "s1", "Planning a trip to Rome for July 15-22.", "Rome is lovely in July.", "user: Planning a trip to Rome for July 15-22. asst: Rome is lovely in July.", is_active=False),
        DialogueChunk("e1_02", "s1", "I booked the Colosseum Boutique Hotel.", "Colosseum Boutique Hotel is booked.", "user: I booked the Colosseum Boutique Hotel. asst: Colosseum Boutique Hotel is booked.", is_active=False)
    ]
    g2 = [
        DialogueChunk("e1_03", "s2", "How to make vegan lasagna?", "Layer zucchini and cashew cheese.", "user: How to make vegan lasagna? asst: Layer zucchini and cashew cheese.", is_active=True),
        DialogueChunk("e1_04", "s2", "What baking temperature?", "Bake at 375 F for 30 minutes.", "user: What baking temperature? asst: Bake at 375 F for 30 minutes.", is_active=True)
    ]
    episodes.append(Episode(
        episode_id="ep_short_1",
        goal_a_chunks=g1,
        goal_b_chunks=g2,
        test_query="Which hotel did I book in Rome and for what dates?",
        gold_answer="Colosseum Boutique Hotel for July 15-22",
        gold_evidence_chunk_ids=["e1_01", "e1_02"],
        episode_type="shift_and_return",
        shift_turn_index=2
    ))

    # Episode 2: Doctor/Health -> Finance -> Health
    g2_1 = [
        DialogueChunk("e2_01", "s1", "My doctor prescribed Amoxicillin 500mg twice daily.", "Got it, Amoxicillin 500mg twice daily with meals.", "user: prescribed Amoxicillin 500mg twice daily. asst: Amoxicillin 500mg twice daily.", is_active=False),
        DialogueChunk("e2_02", "s1", "I have a slight penicillin allergy history.", "Noted penicillin sensitivity in health profile.", "user: penicillin allergy history. asst: Noted penicillin sensitivity.", is_active=False)
    ]
    g2_2 = [
        DialogueChunk("e2_03", "s2", "Should I invest in high yield savings or index funds?", "Index funds for long term growth, savings for emergency.", "user: invest in savings or index funds? asst: Index funds for growth.", is_active=True),
        DialogueChunk("e2_04", "s2", "What is the S&P 500 average return?", "Historically around 10% before inflation.", "user: S&P 500 average return? asst: Around 10%.", is_active=True)
    ]
    episodes.append(Episode(
        episode_id="ep_short_2",
        goal_a_chunks=g2_1,
        goal_b_chunks=g2_2,
        test_query="What medication dose did my doctor prescribe and what allergy was noted?",
        gold_answer="Amoxicillin 500mg twice daily with penicillin allergy noted",
        gold_evidence_chunk_ids=["e2_01", "e2_02"],
        episode_type="shift_and_return",
        shift_turn_index=2
    ))

    # Episode 3: Coding -> Gaming -> Coding
    g3_1 = [
        DialogueChunk("e3_01", "s1", "I am building a web app using FastAPI and PostgreSQL on port 5435.", "Configured FastAPI with port 5435 PostgreSQL connection.", "user: FastAPI and PostgreSQL on port 5435. asst: Configured.", is_active=False),
        DialogueChunk("e3_02", "s1", "The database password is set to tierselect123.", "Saved DB password tierselect123.", "user: DB password tierselect123. asst: Saved.", is_active=False)
    ]
    g3_2 = [
        DialogueChunk("e3_03", "s2", "What are top RPG games on Steam right now?", "Baldur's Gate 3 and Elden Ring are top rated.", "user: top RPG games on Steam? asst: Baldur's Gate 3.", is_active=True),
        DialogueChunk("e3_04", "s2", "Does Elden Ring have co-op?", "Yes, Elden Ring supports multiplayer summoning.", "user: Elden Ring co-op? asst: Yes.", is_active=True)
    ]
    episodes.append(Episode(
        episode_id="ep_short_3",
        goal_a_chunks=g3_1,
        goal_b_chunks=g3_2,
        test_query="What port and password did we configure for our PostgreSQL database?",
        gold_answer="Port 5435 with password tierselect123",
        gold_evidence_chunk_ids=["e3_01", "e3_02"],
        episode_type="shift_and_return",
        shift_turn_index=2
    ))

    return episodes


def main():
    print("=" * 80)
    print("🔬 RUNNING COMPARATIVE EVALUATION: TIERSELECT VS. 5 BASELINES")
    print("   (Using Live Jev Decision Model + Live LLM + PostgreSQL)")
    print("=" * 80)

    # Loosened thresholds experiment: theta_hi 0.65->0.50, theta_lo 0.40->0.20, delta 0.15->0.05, tau_loose 0.30->0.15
    config = TierSelectConfig(theta_shift=0.50, delta=0.05, theta_hi=0.50, theta_lo=0.20, tau_loose=0.15)
    classifier = JevClassifier(theta_shift=config.theta_shift)
    llm = LLMClient()
    goal_store = GoalStore()
    goal_store.reset()

    episodes = build_short_episodes()
    print(f"Running evaluation over {len(episodes)} test episodes...\n", flush=True)

    models = ["B1_FullHistory", "B2_VanillaRAG", "B3_MemoryBank", "B4_RecursiveSummary", "B5_HardReset", "TierSelect"]

    results = {
        m: {
            "evidence_recall": [],
            "old_goal_recovery": [],
            "stale_rate": [],
            "context_tokens": [],
            "accuracy_f1": [],
            "bert_f1": [],
            "latency_ms": []
        }
        for m in models
    }

    for ep_idx, ep in enumerate(episodes):
        all_chunks = ep.goal_a_chunks + ep.goal_b_chunks
        gold_ids = ep.gold_evidence_chunk_ids
        goal_a_ids = [c.chunk_id for c in ep.goal_a_chunks]
        goal_b_ids = [c.chunk_id for c in ep.goal_b_chunks]
        q = ep.test_query
        gold_ans = ep.gold_answer

        # Mark active status based on shift index
        if ep.shift_turn_index is not None:
            for idx_c, chunk in enumerate(all_chunks):
                chunk.is_active = (idx_c >= ep.shift_turn_index)

        index = ChunkVectorIndex()
        index.build(all_chunks)

        print(f"[{ep_idx+1}/{len(episodes)}] Testing Episode '{ep.episode_id}': \"{q[:45]}...\"", flush=True)

        # -------------------------------------------------------------
        # 1. B1 Full History
        # -------------------------------------------------------------
        b1 = FullHistoryBaseline(all_chunks, llm_client=llm)
        t0 = time.perf_counter()
        res_b1 = b1.run_query(q)
        lat_b1 = (time.perf_counter() - t0) * 1000
        rec_b1 = compute_evidence_recall(res_b1["retrieved_chunk_ids"], gold_ids)
        recov_b1 = compute_old_goal_evidence_recovery(res_b1["retrieved_chunk_ids"], goal_a_ids)
        stale_b1 = compute_stale_context_rate(res_b1["retrieved_chunk_ids"], goal_b_ids)
        f1_b1 = compute_token_f1(res_b1["answer"], gold_ans)
        bert_b1 = compute_bert_f1(res_b1["answer"], gold_ans)
        results["B1_FullHistory"]["evidence_recall"].append(rec_b1)
        results["B1_FullHistory"]["old_goal_recovery"].append(recov_b1)
        results["B1_FullHistory"]["stale_rate"].append(stale_b1)
        results["B1_FullHistory"]["context_tokens"].append(res_b1["context_tokens"])
        results["B1_FullHistory"]["accuracy_f1"].append(f1_b1)
        results["B1_FullHistory"]["bert_f1"].append(bert_b1)
        results["B1_FullHistory"]["latency_ms"].append(lat_b1)

        # -------------------------------------------------------------
        # 2. B2 Vanilla RAG
        # -------------------------------------------------------------
        b2 = VanillaRAGBaseline(index, k=config.n_raw, llm_client=llm)
        t0 = time.perf_counter()
        res_b2 = b2.run_query(q)
        lat_b2 = (time.perf_counter() - t0) * 1000
        rec_b2 = compute_evidence_recall(res_b2["retrieved_chunk_ids"], gold_ids)
        recov_b2 = compute_old_goal_evidence_recovery(res_b2["retrieved_chunk_ids"], goal_a_ids)
        stale_b2 = compute_stale_context_rate(res_b2["retrieved_chunk_ids"], goal_b_ids)
        f1_b2 = compute_token_f1(res_b2["answer"], gold_ans)
        bert_b2 = compute_bert_f1(res_b2["answer"], gold_ans)
        results["B2_VanillaRAG"]["evidence_recall"].append(rec_b2)
        results["B2_VanillaRAG"]["old_goal_recovery"].append(recov_b2)
        results["B2_VanillaRAG"]["stale_rate"].append(stale_b2)
        results["B2_VanillaRAG"]["context_tokens"].append(res_b2["context_tokens"])
        results["B2_VanillaRAG"]["accuracy_f1"].append(f1_b2)
        results["B2_VanillaRAG"]["bert_f1"].append(bert_b2)
        results["B2_VanillaRAG"]["latency_ms"].append(lat_b2)

        # -------------------------------------------------------------
        # 3. B3 MemoryBank
        # -------------------------------------------------------------
        b3 = MemoryBankBaseline(all_chunks, k=config.n_raw, llm_client=llm)
        t0 = time.perf_counter()
        res_b3 = b3.run_query(q)
        lat_b3 = (time.perf_counter() - t0) * 1000
        rec_b3 = compute_evidence_recall(res_b3["retrieved_chunk_ids"], gold_ids)
        recov_b3 = compute_old_goal_evidence_recovery(res_b3["retrieved_chunk_ids"], goal_a_ids)
        stale_b3 = compute_stale_context_rate(res_b3["retrieved_chunk_ids"], goal_b_ids)
        f1_b3 = compute_token_f1(res_b3["answer"], gold_ans)
        bert_b3 = compute_bert_f1(res_b3["answer"], gold_ans)
        results["B3_MemoryBank"]["evidence_recall"].append(rec_b3)
        results["B3_MemoryBank"]["old_goal_recovery"].append(recov_b3)
        results["B3_MemoryBank"]["stale_rate"].append(stale_b3)
        results["B3_MemoryBank"]["context_tokens"].append(res_b3["context_tokens"])
        results["B3_MemoryBank"]["accuracy_f1"].append(f1_b3)
        results["B3_MemoryBank"]["bert_f1"].append(bert_b3)
        results["B3_MemoryBank"]["latency_ms"].append(lat_b3)

        # -------------------------------------------------------------
        # 4. B4 Recursive Summary
        # -------------------------------------------------------------
        b4 = RecursiveSummarizationBaseline(all_chunks, llm_client=llm)
        t0 = time.perf_counter()
        res_b4 = b4.run_query(q)
        lat_b4 = (time.perf_counter() - t0) * 1000
        rec_b4 = compute_evidence_recall(res_b4["retrieved_chunk_ids"], gold_ids)
        recov_b4 = compute_old_goal_evidence_recovery(res_b4["retrieved_chunk_ids"], goal_a_ids)
        stale_b4 = compute_stale_context_rate(res_b4["retrieved_chunk_ids"], goal_b_ids)
        f1_b4 = compute_token_f1(res_b4["answer"], gold_ans)
        bert_b4 = compute_bert_f1(res_b4["answer"], gold_ans)
        results["B4_RecursiveSummary"]["evidence_recall"].append(rec_b4)
        results["B4_RecursiveSummary"]["old_goal_recovery"].append(recov_b4)
        results["B4_RecursiveSummary"]["stale_rate"].append(stale_b4)
        results["B4_RecursiveSummary"]["context_tokens"].append(res_b4["context_tokens"])
        results["B4_RecursiveSummary"]["accuracy_f1"].append(f1_b4)
        results["B4_RecursiveSummary"]["bert_f1"].append(bert_b4)
        results["B4_RecursiveSummary"]["latency_ms"].append(lat_b4)

        # -------------------------------------------------------------
        # 5. B5 Hard Reset
        # -------------------------------------------------------------
        b5 = HardResetBaseline(classifier, all_chunks, llm_client=llm)
        t0 = time.perf_counter()
        res_b5 = b5.run_query(q, shift_turn_index=ep.shift_turn_index)
        lat_b5 = (time.perf_counter() - t0) * 1000
        rec_b5 = compute_evidence_recall(res_b5["retrieved_chunk_ids"], gold_ids)
        recov_b5 = compute_old_goal_evidence_recovery(res_b5["retrieved_chunk_ids"], goal_a_ids)
        stale_b5 = compute_stale_context_rate(res_b5["retrieved_chunk_ids"], goal_b_ids)
        f1_b5 = compute_token_f1(res_b5["answer"], gold_ans)
        bert_b5 = compute_bert_f1(res_b5["answer"], gold_ans)
        results["B5_HardReset"]["evidence_recall"].append(rec_b5)
        results["B5_HardReset"]["old_goal_recovery"].append(recov_b5)
        results["B5_HardReset"]["stale_rate"].append(stale_b5)
        results["B5_HardReset"]["context_tokens"].append(res_b5["context_tokens"])
        results["B5_HardReset"]["accuracy_f1"].append(f1_b5)
        results["B5_HardReset"]["bert_f1"].append(bert_b5)
        results["B5_HardReset"]["latency_ms"].append(lat_b5)

        # -------------------------------------------------------------
        # 6. TierSelect (Proposed)
        # -------------------------------------------------------------
        pipeline = TierSelectPipeline(config, index, classifier, llm_client=llm, goal_store=goal_store)
        t0 = time.perf_counter()
        res_ts = pipeline.run_query(q)
        lat_ts = (time.perf_counter() - t0) * 1000
        sel_ts = res_ts["tier_a_chunks"] + res_ts["tier_b_chunks"]
        rec_ts = compute_evidence_recall(sel_ts, gold_ids)
        recov_ts = compute_old_goal_evidence_recovery(sel_ts, goal_a_ids)
        stale_ts = compute_stale_context_rate(res_ts["tier_a_chunks"], goal_b_ids)
        f1_ts = compute_token_f1(res_ts["answer"], gold_ans)
        bert_ts = compute_bert_f1(res_ts["answer"], gold_ans)
        results["TierSelect"]["evidence_recall"].append(rec_ts)
        results["TierSelect"]["old_goal_recovery"].append(recov_ts)
        results["TierSelect"]["stale_rate"].append(stale_ts)
        results["TierSelect"]["context_tokens"].append(res_ts["context_tokens"])
        results["TierSelect"]["accuracy_f1"].append(f1_ts)
        results["TierSelect"]["bert_f1"].append(bert_ts)
        results["TierSelect"]["latency_ms"].append(lat_ts)

    # -------------------------------------------------------------------------
    # Aggregate and Format Output Table
    # -------------------------------------------------------------------------
    N = max(1, len(episodes))
    b1_tokens = sum(results["B1_FullHistory"]["context_tokens"]) / N

    print("\n" + "=" * 118)
    print("🏆 FINAL COMPARATIVE BENCHMARK RESULTS TABLE")
    print("=" * 118)
    header = f"{'Architecture':<22} | {'Evidence Rec':<13} | {'Old-Goal Recov':<15} | {'Stale Rate':<11} | {'Tokens':<8} | {'Reduction':<10} | {'Token F1':<9} | {'BERTScore':<10} | {'Latency'}"
    print(header)
    print("-" * 118)

    table_rows = []
    for m in models:
        rec = sum(results[m]["evidence_recall"]) / N
        recov = sum(results[m]["old_goal_recovery"]) / N
        stale = sum(results[m]["stale_rate"]) / N
        tok = sum(results[m]["context_tokens"]) / N
        f1 = sum(results[m]["accuracy_f1"]) / N
        bert_f1 = sum(results[m]["bert_f1"]) / N
        lat = sum(results[m]["latency_ms"]) / N
        red = compute_token_reduction_rate(b1_tokens, tok)

        disp_name = {
            "B1_FullHistory": "B1: Full History",
            "B2_VanillaRAG": "B2: Vanilla Top-k RAG",
            "B3_MemoryBank": "B3: MemoryBank",
            "B4_RecursiveSummary": "B4: Recursive Summary",
            "B5_HardReset": "B5: Hard Reset",
            "TierSelect": "Proposed TierSelect"
        }[m]

        row_str = f"{disp_name:<22} | {rec:<13.4f} | {recov:<15.4f} | {stale:<11.4f} | {tok:<8.1f} | {red:<10} | {f1:<9.4f} | {bert_f1:<10.4f} | {lat:.1f} ms"
        print(row_str)
        table_rows.append({
            "name": disp_name,
            "evidence_recall": rec,
            "old_goal_recovery": recov,
            "stale_rate": stale,
            "tokens": tok,
            "reduction": red,
            "token_f1": f1,
            "bert_f1": bert_f1,
            "latency_ms": lat
        })

    print("=" * 118)

    # Save summary JSON
    provider = os.environ.get("LLM_PROVIDER", "default").lower().strip()
    out_path = os.path.join(DATA_DIR, "..", "outputs", f"short_baseline_comparison_{provider}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(table_rows, f, indent=2)
    # Also save as latest
    latest_path = os.path.join(DATA_DIR, "..", "outputs", "short_baseline_comparison.json")
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(table_rows, f, indent=2)
    print(f"\nDetailed metrics saved to: {out_path} and {latest_path}\n", flush=True)


if __name__ == "__main__":
    main()
