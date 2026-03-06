import os
import pickle
from typing import List, Tuple

import faiss
import numpy as np

from src.loader import Document
from src.embeddings import EmbeddingModel

# Default paths for persisting the index and document store
INDEX_PATH = "vector_store/faiss.index"
DOCS_PATH = "vector_store/documents.pkl"


class VectorStore:
    """
    FAISS-backed vector store for semantic document retrieval.

    Stores document embeddings and retrieves the most relevant
    chunks given a query embedding.
    """

    def __init__(self, embedding_dim: int):
        """
        Initialise an empty vector store.

        Args:
            embedding_dim: Dimensionality of embeddings (e.g. 384 for MiniLM).
        """
        self.embedding_dim = embedding_dim
        # IndexFlatIP = inner product search (cosine similarity when normalised)
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.documents: List[Document] = []
        print(f"📦 VectorStore initialised (dim={embedding_dim})")

    def add_documents(self, documents: List[Document], embeddings: np.ndarray) -> None:
        """
        Add documents and their embeddings to the index.

        Args:
            documents: List of Document objects.
            embeddings: numpy array of shape (n_docs, embedding_dim).
        """
        if len(documents) != embeddings.shape[0]:
            raise ValueError("Number of documents must match number of embeddings.")

        # FAISS requires float32
        vectors = embeddings.astype(np.float32)
        self.index.add(vectors)
        self.documents.extend(documents)
        print(f"✅ Added {len(documents)} chunks to vector store. Total: {self.index.ntotal}")

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Tuple[Document, float]]:
        """
        Search for the most similar documents to a query embedding.

        Args:
            query_embedding: 1D numpy array for the query.
            top_k: Number of results to return.

        Returns:
            List of (Document, similarity_score) tuples, sorted by relevance.
        """
        if self.index.ntotal == 0:
            raise RuntimeError("Vector store is empty. Build the index first.")

        # Reshape to (1, dim) for FAISS
        query_vector = query_embedding.astype(np.float32).reshape(1, -1)

        scores, indices = self.index.search(query_vector, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # -1 means no result found
                results.append((self.documents[idx], float(score)))

        return results

    def save(self, index_path: str = INDEX_PATH, docs_path: str = DOCS_PATH) -> None:
        """
        Persist the FAISS index and document list to disk.

        Args:
            index_path: Where to save the FAISS index file.
            docs_path: Where to save the serialised documents.
        """
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(docs_path, "wb") as f:
            pickle.dump(self.documents, f)
        print(f"💾 Vector store saved → {index_path}, {docs_path}")

    def load(self, index_path: str = INDEX_PATH, docs_path: str = DOCS_PATH) -> None:
        """
        Load a previously saved FAISS index and documents from disk.

        Args:
            index_path: Path to the saved FAISS index.
            docs_path: Path to the serialised documents.
        """
        if not os.path.exists(index_path) or not os.path.exists(docs_path):
            raise FileNotFoundError(
                f"Index files not found at {index_path} / {docs_path}. "
                "Run the build step first."
            )
        self.index = faiss.read_index(index_path)
        with open(docs_path, "rb") as f:
            self.documents = pickle.load(f)
        print(f"📂 Vector store loaded. Total chunks: {self.index.ntotal}")

    @property
    def is_empty(self) -> bool:
        return self.index.ntotal == 0


def build_vector_store(
    documents: List[Document],
    embedding_model: EmbeddingModel,
    save: bool = True
) -> VectorStore:
    """
    Build a vector store from a list of documents.

    Args:
        documents: List of Document chunks to index.
        embedding_model: The embedding model to use.
        save: Whether to persist the store to disk.

    Returns:
        A populated VectorStore instance.
    """
    print("\n🔨 Building vector store...")
    texts = [doc.content for doc in documents]
    embeddings = embedding_model.embed(texts)

    store = VectorStore(embedding_dim=embedding_model.embedding_dim)
    store.add_documents(documents, embeddings)

    if save:
        store.save()

    return store
