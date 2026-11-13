"""Tests for TypeSafe AI Jev integration."""
import pytest
from typesafe_sdk import Noul, Choice, Score, NoulAnswer, ChoiceAnswer
from src.config import DialogueChunk
from src.models.jev_classifier import JevClassifier

def test_typesafe_sdk_primitives():
    noul_q = Noul(instructions="Is this a goal shift?")
    assert noul_q.type == "noul"
    
    choice_q = Choice(
        instructions="Classify relevance",
        criteria={
            "relevant": "Strong evidence",
            "partially_relevant": "Background info",
            "irrelevant": "Not related"
        }
    )
    assert choice_q.type == "choice"
    assert "relevant" in choice_q.criteria

def test_jev_goal_shift_noul():
    jev = JevClassifier()
    c = DialogueChunk(chunk_id="c1", session_id="s1", user_turn="booking flights to Rome", assistant_turn="options listed", text="User: booking flights to Rome")
    
    # Continuation
    dec, p = jev.predict_goal_shift("how about hotels in Rome?", [c.text])
    assert dec == "continue"
    assert p < 0.50

    # Shift
    dec_shift, p_shift = jev.predict_goal_shift("explain how neural networks backpropagate", [c.text])
    assert dec_shift == "shift"
    assert p_shift >= 0.50

def test_jev_relevance_choice():
    jev = JevClassifier()
    c = DialogueChunk(chunk_id="c1", session_id="s1", user_turn="Who directed Inception?", assistant_turn="Christopher Nolan directed Inception in 2010.", text="User: Who directed Inception?\nAssistant: Christopher Nolan directed Inception in 2010.")
    
    typed, p = jev.score_relevance(c, query="director of Inception", goal_context="movies", is_active=True)
    assert typed in ["relevant", "partially_relevant"]
    assert p >= 0.40

def test_jev_agreement_noul():
    jev = JevClassifier()
    score = jev.evaluate_agreement_noul("Christopher Nolan", "Christopher Nolan", "Who directed Inception?")
    assert score == 1.0

