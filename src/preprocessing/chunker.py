"""Dialogue chunking engine for ImplexConv, LongMemEval, LoCoMo, and TopiOCQA.

Each chunk is a user-assistant turn pair with explicit backward and forward pointers.
"""
import re
from typing import List, Dict, Any, Optional
from src.config import DialogueChunk
from src.preprocessing.cleaner import clean_dialogue_turn, normalize_text

class DialogueChunker:
    """Parses raw dataset dialogue formats into structured turn-pair chunks."""

    @staticmethod
    def link_chunks(chunks: List[DialogueChunk]) -> List[DialogueChunk]:
        """Attach prev_chunk_id and next_chunk_id pointers."""
        for i, chunk in enumerate(chunks):
            if i > 0:
                chunk.prev_chunk_id = chunks[i - 1].chunk_id
            if i < len(chunks) - 1:
                chunk.next_chunk_id = chunks[i + 1].chunk_id
        return chunks

    @classmethod
    def chunk_implexconv_session(cls, session_id: str, session_text: str, prefix: str = "") -> List[DialogueChunk]:
        """Parse ImplexConv dialogue text into turn-pair chunks."""
        # Session text is structured as alternating Speaker1 / Assistant turns
        lines = re.split(r'\n+(?=(?:Speaker\s*\d+|Assistant|User|AI):\s*)', session_text.strip())
        turns = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            m = re.match(r'^(Speaker\s*\d+|User|Human|Assistant|AI):\s*(.*)$', line, flags=re.DOTALL | re.IGNORECASE)
            if m:
                speaker = m.group(1).lower()
                content = clean_dialogue_turn(m.group(2))
                role = "assistant" if "assistant" in speaker or "ai" in speaker else "user"
                turns.append((role, content))
            else:
                turns.append(("user", clean_dialogue_turn(line)))

        # Form user-assistant pairs
        chunks: List[DialogueChunk] = []
        i = 0
        pair_idx = 0
        while i < len(turns):
            role, text = turns[i]
            user_text = ""
            assistant_text = ""
            if role == "user":
                user_text = text
                if i + 1 < len(turns) and turns[i + 1][0] == "assistant":
                    assistant_text = turns[i + 1][1]
                    i += 2
                else:
                    assistant_text = ""
                    i += 1
            else:
                assistant_text = text
                i += 1

            combined_text = f"User: {user_text}\nAssistant: {assistant_text}".strip()
            chunk_id = f"{prefix}s{session_id}_c{pair_idx}"
            chunks.append(DialogueChunk(
                chunk_id=chunk_id,
                session_id=str(session_id),
                user_turn=user_text,
                assistant_turn=assistant_text,
                text=combined_text,
                metadata={"dataset": "ImplexConv", "session_id": session_id, "pair_idx": pair_idx}
            ))
            pair_idx += 1

        return cls.link_chunks(chunks)

    @classmethod
    def chunk_longmemeval_sessions(cls, haystack_sessions: List[List[Dict[str, Any]]], session_ids: Optional[List[str]] = None, prefix: str = "") -> List[DialogueChunk]:
        """Parse LongMemEval haystack sessions into turn-pair chunks."""
        chunks: List[DialogueChunk] = []
        for s_idx, session in enumerate(haystack_sessions):
            s_id = str(session_ids[s_idx]) if session_ids and s_idx < len(session_ids) else str(s_idx)
            i = 0
            pair_idx = 0
            while i < len(session):
                turn = session[i]
                role = turn.get("role", "user")
                content = clean_dialogue_turn(turn.get("content", ""))
                user_text = ""
                assistant_text = ""
                has_answer = turn.get("has_answer", False)

                if role == "user":
                    user_text = content
                    if i + 1 < len(session) and session[i + 1].get("role") == "assistant":
                        assistant_text = clean_dialogue_turn(session[i + 1].get("content", ""))
                        has_answer = has_answer or session[i + 1].get("has_answer", False)
                        i += 2
                    else:
                        i += 1
                else:
                    assistant_text = content
                    i += 1

                combined_text = f"User: {user_text}\nAssistant: {assistant_text}".strip()
                chunk_id = f"{prefix}s{s_id}_c{pair_idx}"
                chunks.append(DialogueChunk(
                    chunk_id=chunk_id,
                    session_id=s_id,
                    user_turn=user_text,
                    assistant_turn=assistant_text,
                    text=combined_text,
                    metadata={"dataset": "LongMemEval", "has_answer": has_answer, "session_id": s_id}
                ))
                pair_idx += 1

        return cls.link_chunks(chunks)

    @classmethod
    def chunk_locomo_conversation(cls, conversation_dict: Dict[str, Any], prefix: str = "") -> List[DialogueChunk]:
        """Parse LoCoMo conversation sessions into turn-pair chunks."""
        chunks: List[DialogueChunk] = []
        # Find all session keys like 'session_1', 'session_2', etc.
        session_keys = sorted(
            [k for k in conversation_dict.keys() if re.match(r'^session_\d+$', k)],
            key=lambda x: int(x.split('_')[1])
        )
        
        for s_key in session_keys:
            turns = conversation_dict[s_key]
            if not isinstance(turns, list):
                continue
            pair_idx = 0
            i = 0
            while i < len(turns):
                t = turns[i]
                text1 = clean_dialogue_turn(t.get("text", "") if isinstance(t, dict) else str(t))
                speaker1 = t.get("speaker", "user") if isinstance(t, dict) else "user"
                
                text2 = ""
                if i + 1 < len(turns):
                    t2 = turns[i + 1]
                    text2 = clean_dialogue_turn(t2.get("text", "") if isinstance(t2, dict) else str(t2))
                    i += 2
                else:
                    i += 1
                
                user_text = text1
                assistant_text = text2
                combined_text = f"{speaker1}: {user_text}\nResponse: {assistant_text}".strip()
                chunk_id = f"{prefix}{s_key}_c{pair_idx}"
                chunks.append(DialogueChunk(
                    chunk_id=chunk_id,
                    session_id=s_key,
                    user_turn=user_text,
                    assistant_turn=assistant_text,
                    text=combined_text,
                    metadata={"dataset": "LoCoMo", "session": s_key}
                ))
                pair_idx += 1

        return cls.link_chunks(chunks)

    @classmethod
    def chunk_topiocqa_conversation(cls, turn_rows: List[Dict[str, Any]], prefix: str = "") -> List[DialogueChunk]:
        """Parse grouped TopiOCQA turns of a conversation into turn-pair chunks."""
        chunks: List[DialogueChunk] = []
        for row in turn_rows:
            conv_no = row.get("Conversation_no", 0)
            turn_no = row.get("Turn_no", 0)
            q = clean_dialogue_turn(row.get("Question", ""))
            a = clean_dialogue_turn(row.get("Answer", ""))
            topic = row.get("Topic", "")
            topic_section = row.get("Topic_section", "")
            
            chunk_id = f"{prefix}conv{conv_no}_t{turn_no}"
            combined = f"User: {q}\nAssistant: {a}"
            chunks.append(DialogueChunk(
                chunk_id=chunk_id,
                session_id=str(conv_no),
                user_turn=q,
                assistant_turn=a,
                text=combined,
                metadata={
                    "dataset": "TopiOCQA",
                    "Conversation_no": conv_no,
                    "Turn_no": turn_no,
                    "topic": topic,
                    "topic_section": topic_section,
                    "gold_passage": row.get("Gold_passage", {})
                }
            ))
        return cls.link_chunks(chunks)

