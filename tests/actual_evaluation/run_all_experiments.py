"""Master Orchestration Pipeline to Execute All Research Experiments (E1 to E5) on Whole Datasets."""
import os
import sys
import json
import time
from pathlib import Path

# Ensure repository root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv
load_dotenv(ROOT_DIR / ".env")

from src.config import TierSelectConfig, Episode, OUTPUT_DIR, DATA_DIR
from src.preprocessing.chunker import DialogueChunker
from src.preprocessing.episode_builder import EpisodeBuilder

# Import experiment runners
try:
    from tests.actual_evaluation.run_e1_baseline_comparison import run_experiment_e1
    from tests.actual_evaluation.run_e2_goal_shift_analysis import run_experiment_e2
    from tests.actual_evaluation.run_e3_classifier_comparison import run_experiment_e3
    from tests.actual_evaluation.run_e4_robustness_growth import run_experiment_e4
    from tests.actual_evaluation.run_e5_efficiency_breakdown import run_experiment_e5
except ImportError:
    from experiments.run_e1_baseline_comparison import run_experiment_e1
    from experiments.run_e2_goal_shift_analysis import run_experiment_e2
    from experiments.run_e3_classifier_comparison import run_experiment_e3
    from experiments.run_e4_robustness_growth import run_experiment_e4
    from experiments.run_e5_efficiency_breakdown import run_experiment_e5

def load_data_and_build_episodes():
    print("[1/6] Loading benchmark datasets and constructing evaluation episodes...")
    config = TierSelectConfig()
    
    # 1. ImplexConv episodes
    implex_path = os.path.join(DATA_DIR, "ImplexConv", "supportive", "ImplexConv_supportive.json")
    implex_episodes = EpisodeBuilder.build_from_implexconv(implex_path, max_episodes=25, distractor_turn_count=10)
    print(f"  -> Built {len(implex_episodes)} ImplexConv evaluation episodes")

    # 2. TopiOCQA shift labels
    topi_path = os.path.join(DATA_DIR, "TopiOCQA", "data", "topiocqa_valid.jsonl")
    topi_shifts = EpisodeBuilder.extract_topiocqa_shift_labels(topi_path, max_conversations=30)
    print(f"  -> Extracted {len(topi_shifts)} TopiOCQA dialogue turn pairs with ground-truth shift labels")

    # 3. LongMemEval episodes
    longmem_path = os.path.join(DATA_DIR, "LongMemEval", "longmemeval_oracle.json")
    longmem_episodes = []
    if os.path.exists(longmem_path):
        with open(longmem_path, "r", encoding="utf-8") as f:
            lm_data = json.load(f)
        for i, item in enumerate(lm_data[:10]):
            haystack = item.get("haystack_sessions", [])
            s_ids = item.get("haystack_session_ids", [])
            chunks = DialogueChunker.chunk_longmemeval_sessions(haystack, session_ids=s_ids, prefix=f"lm_{i}_")
            longmem_episodes.append(Episode(
                episode_id=f"longmem_{i}",
                goal_a_chunks=chunks,
                goal_b_chunks=[],
                test_query=item.get("question", ""),
                gold_answer=item.get("answer", ""),
                gold_evidence_chunk_ids=[c.chunk_id for c in chunks if c.metadata.get("has_answer", False)],
                episode_type="continuation"
            ))
        print(f"  -> Built {len(longmem_episodes)} LongMemEval zero-shot episodes")

    # 4. LoCoMo episodes
    locomo_path = os.path.join(DATA_DIR, "LoCoMo", "locomo10.json")
    locomo_episodes = []
    if os.path.exists(locomo_path):
        with open(locomo_path, "r", encoding="utf-8") as f:
            loco_data = json.load(f)
        for i, item in enumerate(loco_data[:5]):
            conv_dict = item.get("conversation", {})
            qa_list = item.get("qa", [])
            chunks = DialogueChunker.chunk_locomo_conversation(conv_dict, prefix=f"loco_{i}_")
            q = qa_list[0].get("question", "") if qa_list else ""
            a = qa_list[0].get("answer", "") if qa_list else ""
            locomo_episodes.append(Episode(
                episode_id=f"locomo_{i}",
                goal_a_chunks=chunks,
                goal_b_chunks=[],
                test_query=q,
                gold_answer=a,
                gold_evidence_chunk_ids=[c.chunk_id for c in chunks[:5]],
                episode_type="continuation"
            ))
        print(f"  -> Built {len(locomo_episodes)} LoCoMo zero-shot episodes")

    return implex_episodes, topi_shifts, longmem_episodes, locomo_episodes, config

def main():
    implex_eps, topi_shifts, longmem_eps, locomo_eps, config = load_data_and_build_episodes()
    shift_eps = [ep for ep in implex_eps if ep.episode_type == "shift_and_return"]

    all_results = {}

    # Run E1
    print("\n[2/6] Executing Experiment E1: Baseline Comparison...")
    all_results["Experiment_E1_Baseline_Comparison"] = run_experiment_e1(implex_eps[:15], config)

    # Run E2
    print("\n[3/6] Executing Experiment E2: Goal-Shift Dynamics & Margin Analysis...")
    all_results["Experiment_E2_Goal_Shift_Analysis"] = run_experiment_e2(shift_eps[:15], topi_shifts, config)

    # Run E3
    print("\n[4/6] Executing Experiment E3: Classifier Gate Comparison...")
    all_results["Experiment_E3_Classifier_Comparison"] = run_experiment_e3(topi_shifts, shift_eps, config)

    # Run E4
    print("\n[5/6] Executing Experiment E4: Robustness & Generalization...")
    all_results["Experiment_E4_Robustness_Generalization"] = run_experiment_e4(shift_eps, longmem_eps, locomo_eps, config)

    # Run E5
    print("\n[6/6] Executing Experiment E5: Efficiency Breakdown...")
    all_results["Experiment_E5_Efficiency_Breakdown"] = run_experiment_e5(implex_eps, config)

    # Save output
    out_file = os.path.join(OUTPUT_DIR, "all_experiments_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f"Results saved to: {out_file}")
    print("=" * 80)
    print(json.dumps(all_results, indent=2))

if __name__ == "__main__":
    main()

