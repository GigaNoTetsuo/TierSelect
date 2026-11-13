"""Vector indexing and loose semantic retrieval for conversation chunks."""
import numpy as np
from typing import List, Tuple, Optional
from src.config import DialogueChunk
from src.models.embedding import TextEmbedder

class ChunkVectorIndex:
    """FAISS / Dense Inner-Product Vector Index for Dialogue Chunks."""

    def __init__(self, embedder: Optional[TextEmbedder] = None):
        self.embedder = embedder or TextEmbedder(model_name_or_type="lsa")
        self.chunks: List[DialogueChunk] = []
        self.vectors: Optional[np.ndarray] = None
        self._faiss_index = None

    def build(self, chunks: List[DialogueChunk]):
        """Index a list of dialogue chunks."""
        self.chunks = list(chunks)
        if not self.chunks:
            self.vectors = np.empty((0, 384), dtype=np.float32)
            return

        texts = [c.text for c in self.chunks]
        self.embedder.fit(texts)
        self.vectors = self.embedder.encode(texts)

        # Try building native FAISS index if available
        try:
            import faiss
            dim = self.vectors.shape[1]
            self._faiss_index = faiss.IndexFlatIP(dim)
            self._faiss_index.add(self.vectors)
        except Exception:
            self._faiss_index = None

    def add_chunk(self, chunk: DialogueChunk):
        """Add a single chunk to the index incrementally."""
        self.chunks.append(chunk)
        new_vec = self.embedder.encode(chunk.text).reshape(1, -1)
        if self.vectors is None or len(self.vectors) == 0:
            self.vectors = new_vec
        else:
            self.vectors = np.vstack([self.vectors, new_vec])
        if self._faiss_index is not None:
            self._faiss_index.add(new_vec)

    def search(self, query: str, tau_loose: float = 0.30, k_max: int = 50) -> List[Tuple[DialogueChunk, float]]:
        """Search chunks with cosine similarity >= tau_loose, capped at k_max."""
        if not self.chunks or self.vectors is None or len(self.vectors) == 0:
            return []

        q_vec = self.embedder.encode(query).reshape(1, -1)

        if self._faiss_index is not None:
            # Query FAISS
            scores, indices = self._faiss_index.search(q_vec, min(k_max, len(self.chunks)))
            scores = scores[0]
            indices = indices[0]
        else:
            # Exact normalized dot product
            scores = np.dot(self.vectors, q_vec.T).flatten()
            indices = np.argsort(-scores)[:k_max]
            scores = scores[indices]

        candidates = []
        for idx, score in zip(indices, scores):
            if score >= tau_loose:
                candidates.append((self.chunks[idx], float(score)))

        return candidates[:k_max]

