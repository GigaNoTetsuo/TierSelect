"""Constructs shift-and-return episodes and extracts ground-truth topic switches from TopiOCQA."""
import os
import json
import random
from typing import List, Dict, Any, Tuple
from src.config import Episode, DialogueChunk
from src.preprocessing.chunker import DialogueChunker

class EpisodeBuilder:
    """Builds controlled shift-and-return evaluation episodes and controls."""

    @staticmethod
    def build_from_implexconv(implex_path: str, max_episodes: int = 100, distractor_turn_count: int = 10) -> List[Episode]:
        """Build shift-and-return episodes from ImplexConv data."""
        if not os.path.exists(implex_path):
            print(f"[Warning] ImplexConv file not found: {implex_path}")
            return []

        with open(implex_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        episodes: List[Episode] = []
        random.seed(42)

        # Iterate over samples
        for idx in range(min(len(data), max_episodes)):
            sample_a = data[idx]
            qa_list = sample_a.get("qa", [])
            if not qa_list:
                continue

            target_qa = qa_list[0]
            query = target_qa.get("question", "")
            gold_answer = target_qa.get("answer", "")
            gold_session_ids = [str(x) for x in target_qa.get("retrieved_conv_ids", ["0"])]

            conv_a = sample_a.get("conversation", {})
            # Goal A: chunks from gold sessions
            goal_a_chunks: List[DialogueChunk] = []
            for s_id in gold_session_ids:
                if s_id in conv_a:
                    s_chunks = DialogueChunker.chunk_implexconv_session(s_id, conv_a[s_id], prefix=f"ep{idx}_goalA_")
                    goal_a_chunks.extend(s_chunks)

            if not goal_a_chunks:
                # Fallback to first available session
                first_k = list(conv_a.keys())[0]
                goal_a_chunks = DialogueChunker.chunk_implexconv_session(first_k, conv_a[first_k], prefix=f"ep{idx}_goalA_")

            # Goal B (distractor): take from an unrelated dialogue sample
            other_idx = (idx + 1) % len(data)
            conv_b = data[other_idx].get("conversation", {})
            goal_b_chunks: List[DialogueChunk] = []
            for s_id, s_text in conv_b.items():
                s_chunks = DialogueChunker.chunk_implexconv_session(s_id, s_text, prefix=f"ep{idx}_goalB_")
                goal_b_chunks.extend(s_chunks)
                if len(goal_b_chunks) >= distractor_turn_count:
                    goal_b_chunks = goal_b_chunks[:distractor_turn_count]
                    break

            gold_ids = [c.chunk_id for c in goal_a_chunks]

            # 1. Shift-and-Return Episode
            ep_sr = Episode(
                episode_id=f"implex_sr_{idx}",
                goal_a_chunks=goal_a_chunks,
                goal_b_chunks=goal_b_chunks,
                test_query=query,
                gold_answer=gold_answer,
                gold_evidence_chunk_ids=gold_ids,
                episode_type="shift_and_return",
                shift_turn_index=len(goal_a_chunks),
                metadata={"distractor_count": len(goal_b_chunks), "source": "ImplexConv"}
            )
            episodes.append(ep_sr)

            # 2. Continuation Control (No shift, Goal A only)
            ep_cont = Episode(
                episode_id=f"implex_cont_{idx}",
                goal_a_chunks=goal_a_chunks,
                goal_b_chunks=[],
                test_query=query,
                gold_answer=gold_answer,
                gold_evidence_chunk_ids=gold_ids,
                episode_type="continuation",
                shift_turn_index=None,
                metadata={"distractor_count": 0, "source": "ImplexConv"}
            )
            episodes.append(ep_cont)

            # 3. Stay Control (Query asks about Goal B)
            other_qa = data[other_idx].get("qa", [])
            if other_qa:
                stay_query = other_qa[0].get("question", "")
                stay_answer = other_qa[0].get("answer", "")
                ep_stay = Episode(
                    episode_id=f"implex_stay_{idx}",
                    goal_a_chunks=goal_a_chunks,
                    goal_b_chunks=goal_b_chunks,
                    test_query=stay_query,
                    gold_answer=stay_answer,
                    gold_evidence_chunk_ids=[c.chunk_id for c in goal_b_chunks],
                    episode_type="stay",
                    shift_turn_index=len(goal_a_chunks),
                    metadata={"distractor_count": len(goal_b_chunks), "source": "ImplexConv"}
                )
                episodes.append(ep_stay)

        return episodes

    @staticmethod
    def extract_topiocqa_shift_labels(topiocqa_path: str, max_conversations: int = 200) -> List[Dict[str, Any]]:
        """Extract turn-level topic switches and continue/shift ground truth from TopiOCQA."""
        if not os.path.exists(topiocqa_path):
            print(f"[Warning] TopiOCQA file not found: {topiocqa_path}")
            return []

        conv_turns: Dict[int, List[Dict[str, Any]]] = {}
        with open(topiocqa_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                conv_id = row.get("Conversation_no", 0)
                if conv_id not in conv_turns:
                    if len(conv_turns) >= max_conversations:
                        break
                    conv_turns[conv_id] = []
                conv_turns[conv_id].append(row)

        shift_examples: List[Dict[str, Any]] = []
        for conv_id, turns in conv_turns.items():
            turns_sorted = sorted(turns, key=lambda x: x.get("Turn_no", 0))
            for i in range(1, len(turns_sorted)):
                prev_topic = turns_sorted[i - 1].get("Topic", "")
                curr_topic = turns_sorted[i].get("Topic", "")
                is_shift = (prev_topic != curr_topic)
                
                # Context is the preceding query-answer turns
                history = [
                    f"User: {t.get('Question', '')} | Assistant: {t.get('Answer', '')}"
                    for t in turns_sorted[:i]
                ]
                current_query = turns_sorted[i].get("Question", "")
                
                shift_examples.append({
                    "conversation_id": conv_id,
                    "turn_no": turns_sorted[i].get("Turn_no", i + 1),
                    "history": history,
                    "query": current_query,
                    "prev_topic": prev_topic,
                    "curr_topic": curr_topic,
                    "ground_truth_shift": 1 if is_shift else 0
                })

        return shift_examples

