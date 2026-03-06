import os
import sys

# ── Fix: Add project root to Python path ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.loader import load_documents
from src.embeddings import EmbeddingModel
from src.vector_store import build_vector_store

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

DATA_DIR = "data"
CHUNK_SIZE = 500      # Characters per chunk
CHUNK_OVERLAP = 100   # Overlap between chunks


def build_index():
    """Build the FAISS vector index from the knowledge base documents."""
    print("=" * 60)
    print("  Hellobooks AI — Knowledge Base Indexer")
    print("=" * 60)

    # Validate data directory
    if not os.path.exists(DATA_DIR):
        print(f" Data directory '{DATA_DIR}' not found.")
        print("   Please create it and add .md files before running.")
        sys.exit(1)

    # Step 1: Load and chunk documents
    documents = load_documents(
        data_dir=DATA_DIR,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    # Step 2: Load embedding model
    embedding_model = EmbeddingModel()

    # Step 3: Build and save vector store
    vector_store = build_vector_store(
        documents=documents,
        embedding_model=embedding_model,
        save=True
    )

    print("\n" + "=" * 60)
    print(f"   Index built successfully!")
    print(f"   Documents indexed: {len(documents)} chunks")
    print(f"  Vector store saved to: vector_store/")
    print("=" * 60)
    print("\nNext step — start the API server:")
    print("  uvicorn src.api:app --reload --port 8000\n")


if __name__ == "__main__":
    build_index()