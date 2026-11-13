"""Demonstration of TierSelect pipeline over 2-3 conversational goal shifts with PostgreSQL persistence."""
import os
import json
import psycopg2
from src.config import TierSelectConfig, DialogueChunk
from src.models.jev_classifier import JevClassifier
from src.models.llm_client import LLMClient
from src.preprocessing.indexer import ChunkVectorIndex
from src.storage.postgres_store import GoalStore
from src.tierselect.pipeline import TierSelectPipeline


def inspect_postgres():
    """Print the actual contents of the PostgreSQL database."""
    db_host = os.environ.get("DB_HOST", "localhost")
    db_port = int(os.environ.get("DB_PORT", "5435"))
    dbname = os.environ.get("DB_NAME", "tierselect")
    user = os.environ.get("DB_USER", "tierselect")
    password = os.environ.get("DB_PASSWORD", "tierselect123")

    try:
        conn = psycopg2.connect(host=db_host, port=db_port, dbname=dbname, user=user, password=password)
        cur = conn.cursor()
        print("\n" + "=" * 80)
        print("📊 LIVE POSTGRESQL DATABASE STATE (tierselect-db)")
        print("=" * 80)

        # Segments
        cur.execute("SELECT segment_id, segment_number, is_active, created_at, archived_at FROM goal_segments ORDER BY segment_number")
        rows = cur.fetchall()
        print(f"\n[Table: goal_segments] Total: {len(rows)} segments")
        print(f"{'Segment ID':<15} {'#':<4} {'Active?':<10} {'Created':<20} {'Archived'}")
        print("-" * 75)
        for r in rows:
            created = str(r[3])[:19] if r[3] else "None"
            archived = str(r[4])[:19] if r[4] else "—"
            print(f"{r[0]:<15} {r[1]:<4} {str(r[2]):<10} {created:<20} {archived}")

        # Chunks
        cur.execute("SELECT chunk_id, segment_id, is_active, text FROM dialogue_chunks ORDER BY created_at")
        c_rows = cur.fetchall()
        print(f"\n[Table: dialogue_chunks] Total: {len(c_rows)} chunks")
        print(f"{'Chunk ID':<22} {'Segment':<15} {'Active?':<10} {'Text Preview'}")
        print("-" * 75)
        for r in c_rows:
            preview = r[3].replace('\n', ' ')[:45] + "..." if len(r[3]) > 45 else r[3]
            print(f"{r[0]:<22} {str(r[1]):<15} {str(r[2]):<10} {preview}")
        print("=" * 80 + "\n")
        conn.close()
    except Exception as e:
        print(f"[Warning] Could not inspect PostgreSQL: {e}")


def main():
    print("=" * 80)
    print("🚀 RUNNING TIERSELECT PIPELINE DEMO: 3 CONVERSATIONAL GOAL SHIFTS")
    print("=" * 80)

    # 1. Initialize Pipeline with live PostgreSQL and Jev
    config = TierSelectConfig(theta_shift=0.50, delta=0.15, theta_hi=0.65, theta_lo=0.40)
    store = GoalStore()
    store.reset()  # Clean start for demo
    
    classifier = JevClassifier(theta_shift=config.theta_shift)
    llm = LLMClient()
    index = ChunkVectorIndex()

    pipeline = TierSelectPipeline(
        config=config,
        index=index,
        classifier=classifier,
        llm_client=llm,
        goal_store=store
    )

    print(f"• Storage Backend: {pipeline.store.is_persistent} (PostgreSQL on port {os.environ.get('DB_PORT', '5435')})")
    print(f"• LLM Provider: {repr(llm)}")
    print(f"• Jev Decision Gate: {classifier.mode} mode")
    print(f"• Tier Thresholds: Tier A >= {config.theta_hi} (Active) / >= {config.theta_hi + config.delta:.2f} (Archived)")
    print(f"• Tier B Threshold: >= {config.theta_lo} | Tier C: < {config.theta_lo}\n")

    # Conversation sequence designed with 3 explicit goal shifts:
    # Goal 1: Italy Summer Vacation Planning
    # Goal 2: Vegan Cooking & Nutrition
    # Goal 3: Laptop Hardware for Deep Learning
    # Goal 4: Return to Goal 1 (Travel Logistics)
    conversation = [
        # --- GOAL 1: TRAVEL ---
        {
            "turn": 1,
            "goal": "Goal 1: Travel Planning",
            "user": "I am planning a summer vacation to Italy in July. What cities should I visit?",
            "assistant": "Rome and Florence are iconic for first-time visitors, offering historic landmarks and art."
        },
        {
            "turn": 2,
            "goal": "Goal 1: Travel Planning",
            "user": "I choose Rome. I booked the Colosseum Boutique Hotel for July 15-22. What are top sights?",
            "assistant": "Great choice! The Colosseum Boutique Hotel is ideal. Top sights include the Pantheon, Vatican City, and Roman Forum."
        },

        # --- GOAL SHIFT 1 -> GOAL 2: COOKING ---
        {
            "turn": 3,
            "goal": "Goal 2: Vegan Cooking (GOAL SHIFT #1)",
            "user": "Let us switch topics completely. How do I make a vegan cashew cheese sauce for lasagna?",
            "assistant": "Soak raw cashews in boiling water, blend with nutritional yeast, garlic powder, lemon juice, and unsweetened plant milk."
        },
        {
            "turn": 4,
            "goal": "Goal 2: Vegan Cooking",
            "user": "Can I use zucchini slices as gluten-free lasagna noodles with that cashew cheese?",
            "assistant": "Yes! Slice zucchini thinly lengthwise and salt them first to draw out excess moisture."
        },

        # --- GOAL SHIFT 2 -> GOAL 3: LAPTOP HARDWARE ---
        {
            "turn": 5,
            "goal": "Goal 3: Laptop Hardware (GOAL SHIFT #2)",
            "user": "Change of subject again. Which laptop should I buy for local PyTorch deep learning under $2000?",
            "assistant": "Look for a gaming laptop with an NVIDIA RTX 4070 or 4080 GPU and at least 32GB RAM."
        },

        # --- GOAL SHIFT 3 / RETURN TO GOAL 1 ---
        {
            "turn": 6,
            "goal": "Goal 1: Travel Return (GOAL SHIFT #3 / OLD-GOAL RETURN)",
            "user": "Back to my Italy trip: what were my travel dates and which hotel in Rome did I book?",
            "assistant": ""  # To be generated by pipeline
        }
    ]

    for step in conversation:
        t_num = step["turn"]
        goal_desc = step["goal"]
        q = step["user"]

        print("-" * 80)
        print(f"▶ TURN {t_num} [{goal_desc}]")
        print(f"  User Query: \"{q}\"")

        # Run TierSelect on query
        res = pipeline.run_query(q)

        shift_dec = res["goal_shift_decision"]
        p_shift = res["p_shift"]
        tier_a = res["tier_a_chunks"]
        tier_b = res["tier_b_chunks"]
        tier_c = res["tier_c_chunks"]
        digest = res["digest_text"]
        ans = res["answer"]
        tokens = res["context_tokens"]
        lat = res["latencies"]

        # Print detailed telemetry
        shift_icon = "🚨 GOAL SHIFT DETECTED!" if shift_dec == "shift" else "➡️ Continues Active Goal"
        print(f"  • Shift Detection: {shift_icon} (P(shift)={p_shift:.3f})")
        print(f"  • Candidates Tiering: Tier A (Raw)={tier_a} | Tier B (Compressed)={tier_b} | Tier C (Dropped)={tier_c}")
        if digest:
            print(f"  • Tier B Digest (<=150 tokens): \"{digest}\"")
        print(f"  • Generated Answer: \"{ans}\"")
        print(f"  • Telemetry: Context Tokens={tokens} | Total Latency={lat['total_latency_ms']:.2f}ms (Retrieval={lat['retrieval_ms']:.2f}ms, Jev Tiering={lat['relevance_tiering_ms']:.2f}ms)")

        # Record this turn pair into the pipeline state and Postgres database
        asst_text = step["assistant"] if step["assistant"] else ans
        pipeline.record_turn(user_turn=q, assistant_turn=asst_text, session_id="demo_session")

    # Final DB Inspection
    inspect_postgres()


if __name__ == "__main__":
    main()

