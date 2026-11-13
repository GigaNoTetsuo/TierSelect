"""Embedding engine with support for SentenceTransformers and fast LSA/TF-IDF fallback."""
import numpy as np
from typing import List, Union

class TextEmbedder:
    """Computes dense normalized vector representations of dialogue text and queries."""

    def __init__(self, model_name_or_type: str = "lsa", dim: int = 384):
        self.model_type = model_name_or_type
        self.dim = dim
        self._st_model = None
        self._sklearn_pipeline = None

        if model_name_or_type != "lsa":
            try:
                from sentence_transformers import SentenceTransformer
                self._st_model = SentenceTransformer(model_name_or_type)
                print(f"[TextEmbedder] Loaded SentenceTransformer: {model_name_or_type}")
            except Exception as e:
                print(f"[TextEmbedder] Could not load SentenceTransformer ({e}). Falling back to fast LSA embedder.")
                self.model_type = "lsa"

    def fit(self, texts: List[str]):
        """Fit vocabulary/projections if using LSA/TF-IDF."""
        if self.model_type == "lsa":
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
            from sklearn.pipeline import Pipeline
            
            clean_texts = [t if t.strip() else "empty turn" for t in texts]
            n_components = min(self.dim, max(16, len(clean_texts) // 2))
            self._sklearn_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)),
                ('svd', TruncatedSVD(n_components=n_components, random_state=42))
            ])
            self._sklearn_pipeline.fit(clean_texts)

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode text(s) into L2-normalized dense embedding vectors."""
        is_single = isinstance(texts, str)
        text_list = [texts] if is_single else texts
        clean_list = [t if t.strip() else "empty turn" for t in text_list]

        if self._st_model is not None:
            vectors = self._st_model.encode(clean_list, convert_to_numpy=True, normalize_embeddings=True)
            return vectors[0] if is_single else vectors

        if self._sklearn_pipeline is None:
            # Auto-fit on provided text batch
            self.fit(clean_list)

        dense = self._sklearn_pipeline.transform(clean_list)
        # L2-normalize
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = (dense / norms).astype(np.float32)
        return normalized[0] if is_single else normalized

