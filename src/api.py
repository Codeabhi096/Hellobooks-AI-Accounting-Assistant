

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore, INDEX_PATH, DOCS_PATH
from src.rag_pipeline import RAGPipeline


# Pydantic Models


class QuestionRequest(BaseModel):
    """Request body for the /ask endpoint."""
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        example="What is a balance sheet?"
    )

class AnswerResponse(BaseModel):
    """Response body for the /ask endpoint."""
    answer: str
    sources: list[str] = []
    scores: list[float] = []

class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""
    status: str
    model: str
    index_size: int



# App State (shared across requests)


# These will be initialised on startup
embedding_model: EmbeddingModel = None
vector_store: VectorStore = None
rag_pipeline: RAGPipeline = None


# Lifespan: Load models on startup


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load embedding model, vector store, and RAG pipeline on startup."""
    global embedding_model, vector_store, rag_pipeline

    print("\n🚀 Starting Hellobooks AI Accounting Assistant...")

    # Load embedding model
    embedding_model = EmbeddingModel()

    # Load vector store from disk
    vector_store = VectorStore(embedding_dim=embedding_model.embedding_dim)
    vector_store.load(INDEX_PATH, DOCS_PATH)

    # Initialise RAG pipeline
    rag_pipeline = RAGPipeline(
        embedding_model=embedding_model,
        vector_store=vector_store
    )

    print("✅ API ready to serve requests.\n")
    yield

    # Cleanup on shutdown (if needed)
    print("🛑 Shutting down...")



# FastAPI App


app = FastAPI(
    title="Hellobooks AI Accounting Assistant",
    description=(
        "A Retrieval-Augmented Generation (RAG) API that answers basic "
        "accounting questions using a curated knowledge base."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# Allow all origins for development (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Routes


@app.get("/", tags=["Info"])
def root():
    """Root endpoint — basic API info."""
    return {
        "name": "Hellobooks AI Accounting Assistant",
        "version": "1.0.0",
        "description": "Ask accounting questions, get simple answers.",
        "docs": "/docs",
        "ask_endpoint": "POST /ask"
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
def health_check():
    """Health check — confirms the model and index are loaded."""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised.")

    return HealthResponse(
        status="healthy",
        model=embedding_model.model_name,
        index_size=vector_store.index.ntotal
    )


@app.post("/ask", response_model=AnswerResponse, tags=["RAG"])
def ask_question(request: QuestionRequest):
    """
    Ask an accounting question and receive an AI-generated answer.

    The system retrieves relevant documents from the knowledge base
    and uses an LLM to generate a clear, contextual answer.
    """
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised.")

    try:
        result = rag_pipeline.answer_question(request.question)
        return AnswerResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
            scores=result.get("scores", [])
        )
    except EnvironmentError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing your question: {str(e)}"
        )
