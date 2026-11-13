"""Goal Tracker managing active and archived dialogue goal segments.

Persists goal segments to PostgreSQL via GoalStore when a database
is configured, otherwise falls back to in-memory lists transparently.
"""
from typing import List, Tuple, Optional
from src.config import DialogueChunk
from src.models.jev_classifier import JevClassifier
from src.storage.postgres_store import GoalStore


class GoalTracker:
    """Tracks user conversational goals across turns, managing active and archived segments."""

    def __init__(self, classifier: JevClassifier, store: Optional[GoalStore] = None):
        self.classifier = classifier
        self.store = store or GoalStore()

        # In-memory mirror kept for fast access within a single run
        self.active_segment_chunks: List[DialogueChunk] = []
        self.archived_segments: List[List[DialogueChunk]] = []
        self.segment_counter: int = self.store.get_segment_counter() or 1

        # Hydrate from DB if persistent store has data
        if self.store.is_persistent:
            self.active_segment_chunks = self.store.get_active_chunks()
            self.archived_segments = self.store.get_archived_segment_chunks()

    def reset(self):
        """Reset state for a fresh conversation/episode."""
        self.active_segment_chunks = []
        self.archived_segments = []
        self.segment_counter = 1
        self.store.reset()

    def process_query(self, query: str) -> Tuple[str, float]:
        """Decide whether the new query continues or shifts the active goal segment.

        Returns:
            (decision_type, calibrated_shift_probability)
        """
        active_turns = [c.text for c in self.active_segment_chunks]
        decision, p_shift = self.classifier.predict_goal_shift(query, active_turns)

        if decision == "shift" and self.active_segment_chunks:
            # Archive the current active segment
            current_seg_id = f"segment_{self.segment_counter}"

            for c in self.active_segment_chunks:
                c.is_active = False
                c.segment_id = current_seg_id

            # Persist archive
            self.store.archive_segment(current_seg_id)
            for c in self.active_segment_chunks:
                self.store.save_chunk(c)

            self.archived_segments.append(list(self.active_segment_chunks))
            self.active_segment_chunks = []
            self.segment_counter += 1

            # Create new active segment in store
            self.store.create_segment(self.segment_counter)

        return decision, p_shift

    def add_chunk_to_active_segment(self, chunk: DialogueChunk):
        """Append a newly observed chunk to the current active goal segment."""
        seg_id = f"segment_{self.segment_counter}"

        # Ensure segment row exists
        if not self.store.get_active_segment_id():
            self.store.create_segment(self.segment_counter)

        chunk.is_active = True
        chunk.segment_id = seg_id
        self.active_segment_chunks.append(chunk)

        # Persist
        self.store.save_chunk(chunk)

    def get_goal_context(self) -> str:
        """Return text summary/concatenation of the current active goal segment."""
        if not self.active_segment_chunks:
            return ""
        return " ".join([c.user_turn for c in self.active_segment_chunks[-3:]])
