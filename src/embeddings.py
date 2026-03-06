
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

# Default model — small, fast, and effective for semantic search
DEFAULT_MODEL = "all-MiniLM-L6-v2"


class EmbeddingModel:
    """
    Wrapper around SentenceTransformer for generating text embeddings.

    Usage:
        model = EmbeddingModel()
        vectors = model.embed(["What is bookkeeping?", "Cash flow basics"])
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        """
        Initialise the embedding model.

        Args:
            model_name: HuggingFace model identifier.
        """
        print(f"🔧 Loading embedding model: {model_name}")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"✅ Embedding model loaded. Dimension: {self.embedding_dim}")

    def embed(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of text strings.

        Args:
            texts: List of strings to embed.
            batch_size: Number of texts to embed per batch.

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        if not texts:
            raise ValueError("Cannot embed an empty list of texts.")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 10,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalise for cosine similarity
        )
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.

        Args:
            query: The user's question or search string.

        Returns:
            1D numpy array of shape (embedding_dim,)
        """
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding[0]  # Return 1D array
