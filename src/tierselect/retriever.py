"""Loose Semantic Retrieval module for TierSelect."""
from typing import List, Tuple
from src.config import DialogueChunk
from src.preprocessing.indexer import ChunkVectorIndex

class LooseRetriever:
    """Retrieves permissive candidate pool with cosine similarity >= tau_loose."""

    def __init__(self, index: ChunkVectorIndex, tau_loose: float = 0.30, k_max: int = 50):
        self.index = index
        self.tau_loose = tau_loose
        self.k_max = k_max

    def retrieve(self, query: str) -> List[Tuple[DialogueChunk, float]]:
        """Retrieve candidates from active and archived segments alike."""
        return self.index.search(query, tau_loose=self.tau_loose, k_max=self.k_max)

