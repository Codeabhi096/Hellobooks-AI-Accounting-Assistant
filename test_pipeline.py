import os
import sys
import numpy as np
import pytest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.loader import chunk_text, Document, load_documents
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore


class TestChunkText:
    def test_basic_chunking(self):
        text = "A" * 1200
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=100)
        assert len(chunks) > 1

    def test_short_text_single_chunk(self):
        text = "Short text under 500 chars."
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=100)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_overlap(self):
        text = "word " * 200  # 1000 chars
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
       
        assert len(chunks) >= 10

    def test_empty_text(self):
        chunks = chunk_text("", chunk_size=500, chunk_overlap=100)
        assert chunks == []

    def test_exact_chunk_size(self):
        text = "A" * 500
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=0)
        assert len(chunks) == 1


class TestEmbeddingModel:
    @pytest.fixture(scope="class")
    def model(self):
        return EmbeddingModel()

    def test_model_loads(self, model):
        assert model.model is not None
        assert model.embedding_dim == 384  

    def test_embed_single_text(self, model):
        result = model.embed(["What is bookkeeping?"])
        assert result.shape == (1, 384)

    def test_embed_multiple_texts(self, model):
        texts = ["What is an invoice?", "Explain cash flow.", "Define assets."]
        result = model.embed(texts)
        assert result.shape == (3, 384)

    def test_embed_query(self, model):
        result = model.embed_query("What is profit?")
        assert result.shape == (384,)

    def test_embeddings_are_normalised(self, model):
        result = model.embed(["Normalisation test"])
        norm = np.linalg.norm(result[0])
        assert abs(norm - 1.0) < 1e-5  # Should be ~1.0

    def test_empty_text_raises(self, model):
        with pytest.raises(ValueError):
            model.embed_query("")

    def test_empty_list_raises(self, model):
        with pytest.raises(ValueError):
            model.embed([])


class TestVectorStore:
    @pytest.fixture(scope="class")
    def model(self):
        return EmbeddingModel()

    @pytest.fixture(scope="class")
    def store_with_docs(self, model):
        store = VectorStore(embedding_dim=384)
        docs = [
            Document(content="Bookkeeping is the recording of financial transactions.", source="bookkeeping.md", chunk_index=0),
            Document(content="An invoice is a bill sent to a customer.", source="invoices.md", chunk_index=0),
            Document(content="Cash flow is the movement of money in and out of a business.", source="cash_flow.md", chunk_index=0),
        ]
        texts = [d.content for d in docs]
        embeddings = model.embed(texts)
        store.add_documents(docs, embeddings)
        return store

    def test_store_initialises_empty(self):
        store = VectorStore(384)
        assert store.is_empty

    def test_store_has_correct_count(self, store_with_docs):
        assert store_with_docs.index.ntotal == 3

    def test_search_returns_results(self, model, store_with_docs):
        query = model.embed_query("What is bookkeeping?")
        results = store_with_docs.search(query, top_k=1)
        assert len(results) == 1
        doc, score = results[0]
        assert isinstance(doc, Document)
        assert isinstance(score, float)

    def test_search_top_k(self, model, store_with_docs):
        query = model.embed_query("financial records")
        results = store_with_docs.search(query, top_k=3)
        assert len(results) == 3

    def test_search_empty_store_raises(self, model):
        empty_store = VectorStore(384)
        query = model.embed_query("test query")
        with pytest.raises(RuntimeError):
            empty_store.search(query)

    def test_most_relevant_doc_returned(self, model, store_with_docs):
        query = model.embed_query("invoice billing customer")
        results = store_with_docs.search(query, top_k=1)
        doc, _ = results[0]
        assert "invoice" in doc.content.lower()


class TestLoadDocuments:
    def test_load_real_data(self):
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        if not os.path.exists(data_dir):
            pytest.skip("data/ directory not found")
        docs = load_documents(data_dir)
        assert len(docs) > 0
        for doc in docs:
            assert isinstance(doc, Document)
            assert doc.content
            assert doc.source.endswith(".md")

    def test_missing_directory_raises(self):
        with pytest.raises(FileNotFoundError):
            load_documents("/nonexistent/path/xyz")
