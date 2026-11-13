"""Dedicated test script to send live queries to TypeSafe AI's Jev model.

Logs request payloads, response models, token usage, and answers so you can
verify the API hits in your TypeSafe AI dashboard.
"""
import os
import time
from dotenv import load_dotenv

# Load .env
load_dotenv()

from typesafe_sdk import TypeSafeClient, Noul, Choice, Score

def main():
    api_key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    print("=" * 80)
    print("🚀 TESTING LIVE TYPESAFE AI (JEV) API CALLS")
    print("=" * 80)
    print(f"API Key: {api_key[:15]}...{api_key[-8:]} (Length: {len(api_key)})")

    client = TypeSafeClient(api_key=api_key)

    # -------------------------------------------------------------------------
    # TEST 1: Goal Shift Detection using Noul Primitive (Binary 0.0 - 1.0)
    # -------------------------------------------------------------------------
    print("\n[Call 1/3] Testing Goal Shift Detection (Noul Primitive)...")
    state_shift = (
        "Active Goal Segment (Travel Planning):\n"
        "- User: I am planning a trip to Italy for July 15-22.\n"
        "- Assistant: Rome and Florence are great choices.\n"
        "- User: I booked the Colosseum Boutique Hotel in Rome.\n\n"
        "Incoming User Query:\n"
        "How do I repair a leaking bathroom pipe under the sink?"
    )
    q_shift = Noul(
        instructions="Does the incoming user query represent a shift or change of conversational goal away from the active goal segment?"
    )

    t0 = time.perf_counter()
    resp_shift = client.system_one(
        state=state_shift,
        questions={"goal_shift": q_shift}
    )
    lat_shift = (time.perf_counter() - t0) * 1000

    print(f"  • Model Used: {resp_shift.model}")
    print(f"  • Token Usage: Input={resp_shift.usage.input_tokens}, Output={resp_shift.usage.output_tokens}")
    print(f"  • Latency: {lat_shift:.2f} ms")
    ans_shift = resp_shift.answers.get("goal_shift")
    print(f"  • Noul Shift Answer: {ans_shift.noul} (Decision: {'SHIFT' if ans_shift.noul >= 0.50 else 'CONTINUE'})")

    # -------------------------------------------------------------------------
    # TEST 2: Candidate Relevance Classification using Choice Primitive
    # -------------------------------------------------------------------------
    print("\n[Call 2/3] Testing Candidate Relevance Tiering (Choice Primitive)...")
    state_rel = (
        "Goal Context: Summer trip to Italy in July 2026\n"
        "Query: Which hotel did I pick in Rome and what were the dates?\n\n"
        "Candidate Conversation Chunk:\n"
        "User: I chose Rome. I booked the Colosseum Boutique Hotel for July 15-22.\n"
        "Assistant: Excellent! You will be within walking distance of the Colosseum."
    )
    q_rel = Choice(
        instructions="How relevant is this candidate chunk to answering the user query in the context of the goal?",
        criteria={
            "relevant": "Directly provides factual evidence, specific hotel name, or dates needed to answer the query",
            "partially_relevant": "Provides related background or general preferences",
            "irrelevant": "Completely unrelated topic or irrelevant turn"
        }
    )

    t0 = time.perf_counter()
    resp_rel = client.system_one(
        state=state_rel,
        questions={"relevance": q_rel}
    )
    lat_rel = (time.perf_counter() - t0) * 1000

    print(f"  • Model Used: {resp_rel.model}")
    print(f"  • Token Usage: Input={resp_rel.usage.input_tokens}, Output={resp_rel.usage.output_tokens}")
    print(f"  • Latency: {lat_rel:.2f} ms")
    ans_rel = resp_rel.answers.get("relevance")
    print(f"  • Choice Value: '{ans_rel.choice}' (Confidence: {ans_rel.confidence})")
    print(f"  • Probability Distribution: {ans_rel.probabilities}")

    # -------------------------------------------------------------------------
    # TEST 3: Multi-Question System One Call (Shift + Relevance + Agreement)
    # -------------------------------------------------------------------------
    print("\n[Call 3/3] Testing Multi-Question Batch System One Call...")
    state_batch = (
        "Reference Gold Answer: The user booked the Colosseum Boutique Hotel for July 15-22.\n"
        "Candidate Generated Answer: Your Rome vacation is at the Colosseum Boutique Hotel from July 15 to July 22.\n"
        "Query: When is the trip and where am I staying?"
    )
    q_agree = Noul(
        instructions="Does the candidate generated answer factually agree with and accurately represent the reference gold answer?"
    )
    q_score = Score(
        instructions="Score the factual completeness of the candidate answer.",
        criteria=[
            "Completely missing or incorrect facts",
            "Mentions only partial facts (e.g. hotel only or dates only)",
            "Fully accurate, mentioning both the Colosseum Boutique Hotel and July 15-22"
        ]
    )

    t0 = time.perf_counter()
    resp_batch = client.system_one(
        state=state_batch,
        questions={"factual_agreement": q_agree, "completeness_score": q_score}
    )
    lat_batch = (time.perf_counter() - t0) * 1000

    print(f"  • Model Used: {resp_batch.model}")
    print(f"  • Token Usage: Input={resp_batch.usage.input_tokens}, Output={resp_batch.usage.output_tokens}")
    print(f"  • Latency: {lat_batch:.2f} ms")
    ans_agree = resp_batch.answers.get("factual_agreement")
    ans_score = resp_batch.answers.get("completeness_score")
    print(f"  • Factual Agreement (Noul): {ans_agree.noul}")
    print(f"  • Completeness (Score 1-5): {ans_score.score}")

    print("\n" + "=" * 80)
    print("✅ ALL 3 LIVE JEV API CALLS COMPLETED SUCCESSFULLY!")
    print("Check your TypeSafe AI dashboard now to see the 3 recorded API requests")
    print(f"Total tokens consumed across tests: ~{resp_shift.usage.input_tokens + resp_rel.usage.input_tokens + resp_batch.usage.input_tokens} input tokens")
    print("=" * 80)

if __name__ == "__main__":
    main()
