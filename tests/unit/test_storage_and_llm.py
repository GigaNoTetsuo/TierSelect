"""Tests for PostgreSQL GoalStore and LLM Provider Switching."""
import os
import pytest
from src.config import DialogueChunk
from src.storage.postgres_store import GoalStore
from src.models.llm_client import LLMClient, DEFAULT_MODELS, PROVIDER_URLS
from src.models.jev_classifier import JevClassifier
from src.tierselect.goal_tracker import GoalTracker


def test_goal_store_operations():
    """Test GoalStore CRUD operations in fallback mode."""
    store = GoalStore()
    store.reset()

    # 1. Create segment
    seg1 = store.create_segment(1)
    assert seg1 == "segment_1"
    assert store.get_active_segment_id() == "segment_1"

    # 2. Save active chunk
    chunk1 = DialogueChunk(
        chunk_id="chk_001",
        session_id="sess_1",
        user_turn="I need travel ideas for Europe.",
        assistant_turn="Paris and Rome are great choices.",
        text="user: I need travel ideas for Europe. assistant: Paris and Rome are great choices.",
        segment_id="segment_1",
        is_active=True
    )
    store.save_chunk(chunk1)

    active_chunks = store.get_active_chunks()
    assert len(active_chunks) == 1
    assert active_chunks[0].chunk_id == "chk_001"
    assert active_chunks[0].is_active is True

    # 3. Archive segment
    store.archive_segment("segment_1")
    archived_chunks = store.get_archived_chunks()
    assert len(archived_chunks) == 1
    assert archived_chunks[0].is_active is False
    assert len(store.get_active_chunks()) == 0

    # 4. Create new active segment
    seg2 = store.create_segment(2)
    assert seg2 == "segment_2"
    chunk2 = DialogueChunk(
        chunk_id="chk_002",
        session_id="sess_1",
        user_turn="What about vegan restaurants?",
        assistant_turn="Here are top vegan spots.",
        text="user: What about vegan restaurants? assistant: Here are top vegan spots.",
        segment_id="segment_2",
        is_active=True
    )
    store.save_chunk(chunk2)
    assert len(store.get_active_chunks()) == 1
    assert len(store.get_all_chunks()) == 2

    # 5. Reset
    store.reset()
    assert len(store.get_all_chunks()) == 0


def test_llm_client_provider_switching():
    """Test LLMClient configures correctly for OpenAI, Groq, and fallback."""
    # 1. Test OpenAI defaults
    client_openai = LLMClient(provider="openai", api_key="sk-test-dummy")
    assert client_openai.provider == "openai"
    assert client_openai.model == "gpt-4o-mini"
    assert client_openai.is_available is True

    # 2. Test Groq defaults
    client_groq = LLMClient(provider="groq", api_key="gsk-test-dummy")
    assert client_groq.provider == "groq"
    assert client_groq.model == "qwen-qwq-32b"
    assert client_groq.is_available is True

    # 3. Test explicit custom model
    client_custom = LLMClient(provider="groq", api_key="gsk-test-dummy", model="llama-3.3-70b-versatile")
    assert client_custom.model == "llama-3.3-70b-versatile"

    # 4. Test missing key gracefully marks unavailable without error
    client_no_key = LLMClient(provider="openai", api_key="")
    assert client_no_key.is_available is False
    reply = client_no_key.chat("Hello")
    assert reply == ""


def test_goal_tracker_with_store():
    """Test that GoalTracker integrates seamlessly with GoalStore."""
    store = GoalStore()
    store.reset()
    classifier = JevClassifier()
    tracker = GoalTracker(classifier=classifier, store=store)

    chunk = DialogueChunk(
        chunk_id="gt_001",
        session_id="s1",
        user_turn="Let us plan a vacation in Spain.",
        assistant_turn="Spain is wonderful.",
        text="Let us plan a vacation in Spain."
    )
    tracker.add_chunk_to_active_segment(chunk)
    assert len(tracker.active_segment_chunks) == 1
    assert len(store.get_active_chunks()) == 1

    # Shift query
    decision, p_shift = tracker.process_query("Completely new topic: how to fix a car engine?")
    assert decision == "shift"
    assert len(tracker.active_segment_chunks) == 0
    assert len(tracker.archived_segments) == 1
    assert len(store.get_archived_chunks()) == 1

