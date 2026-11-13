"""Unit and integration tests for TierSelect framework components."""
import pytest
from src.config import DialogueChunk, TierSelectConfig
from src.preprocessing.chunker import DialogueChunker
from src.preprocessing.indexer import ChunkVectorIndex
from src.models.jev_classifier import JevClassifier
from src.models.compressor import TierBCompressor
from src.models.generator import ResponseGenerator
from src.tierselect.goal_tracker import GoalTracker
from src.tierselect.tier_allocator import TierAllocator
from src.tierselect.pipeline import TierSelectPipeline
from src.evaluation.selection_metrics import compute_evidence_recall, compute_context_precision, compute_context_f1
from src.evaluation.answer_metrics import compute_token_f1, compute_exact_match

def test_chunker_linking():
    c1 = DialogueChunk(chunk_id="c1", session_id="s1", user_turn="hi", assistant_turn="hello", text="User: hi\nAssistant: hello")
    c2 = DialogueChunk(chunk_id="c2", session_id="s1", user_turn="how are you", assistant_turn="fine", text="User: how are you\nAssistant: fine")
    linked = DialogueChunker.link_chunks([c1, c2])
    assert linked[0].next_chunk_id == "c2"
    assert linked[1].prev_chunk_id == "c1"

def test_vector_index():
    c1 = DialogueChunk(chunk_id="c1", session_id="s1", user_turn="paris vacation", assistant_turn="eiffel tower", text="User: paris vacation\nAssistant: eiffel tower")
    c2 = DialogueChunk(chunk_id="c2", session_id="s1", user_turn="cooking pasta", assistant_turn="boil water", text="User: cooking pasta\nAssistant: boil water")
    index = ChunkVectorIndex()
    index.build([c1, c2])
    res = index.search("trip to paris", tau_loose=0.01, k_max=5)
    assert len(res) > 0
    assert res[0][0].chunk_id == "c1"

def test_goal_tracker_and_shift():
    classifier = JevClassifier()
    gt = GoalTracker(classifier)
    c1 = DialogueChunk(chunk_id="c1", session_id="s1", user_turn="trip to europe", assistant_turn="ok", text="User: trip to europe")
    gt.add_chunk_to_active_segment(c1)
    
    # Continuation
    dec_cont, _ = gt.process_query("what places in europe?")
    assert dec_cont == "continue"
    
    # Shift
    dec_shift, _ = gt.process_query("fix my broken bicycle brakes completely unrelated")
    assert dec_shift == "shift"

def test_tier_b_compression():
    compressor = TierBCompressor(max_tokens=50)
    c = DialogueChunk(chunk_id="c10", session_id="s1", user_turn="prefer morning flights on 14th March", assistant_turn="noted", text="User: prefer morning flights on 14th March\nAssistant: noted")
    digest = compressor.compress([c])
    assert "[c10]" in digest
    assert len(digest.split()) <= 50

def test_tierselect_pipeline_execution():
    config = TierSelectConfig()
    c1 = DialogueChunk(chunk_id="c1", session_id="s1", user_turn="who wrote old man and the sea?", assistant_turn="Ernest Hemingway", text="User: who wrote old man and the sea?\nAssistant: Ernest Hemingway", is_active=True)
    c2 = DialogueChunk(chunk_id="c2", session_id="s1", user_turn="what is quantum physics?", assistant_turn="study of atoms", text="User: what is quantum physics?\nAssistant: study of atoms", is_active=False)
    index = ChunkVectorIndex()
    index.build([c1, c2])
    classifier = JevClassifier()
    pipeline = TierSelectPipeline(config, index, classifier)
    res = pipeline.run_query("tell me about Hemingway")
    assert "answer" in res
    assert "latencies" in res
    assert "context_tokens" in res
    assert len(res["tier_a_chunks"] + res["tier_b_chunks"]) > 0

def test_evaluation_metrics():
    rec = compute_evidence_recall(["c1", "c2"], ["c1"])
    assert rec == 1.0
    prec = compute_context_precision(["c1", "c2"], ["c1"])
    assert prec == 0.5
    f1 = compute_context_f1(prec, rec)
    assert round(f1, 2) == 0.67
    em = compute_exact_match("Ernest Hemingway", "ernest hemingway")
    assert em == 1.0
    tok_f1 = compute_token_f1("The author is Hemingway", "Hemingway author")
    assert tok_f1 > 0.0

