

import os
from typing import List, Tuple

from openai import OpenAI
from dotenv import load_dotenv

from src.loader import Document
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore

# Load environment variables from .env file
load_dotenv()

# Number of chunks to retrieve for context
TOP_K = 3

# LLM model to use
LLM_MODEL = "openai/gpt-3.5-turbo"
LLM_MODEL = "mistralai/mistral-7b-instruct"
LLM_MODEL = "google/gemma-3-12b-it:free"


def build_prompt(context_chunks: List[str], question: str) -> str:
    """
    Build a RAG prompt by combining retrieved context with the user question.

    Args:
        context_chunks: List of relevant document excerpts.
        question: The user's original question.

    Returns:
        A formatted prompt string.
    """
    context = "\n\n---\n\n".join(context_chunks)

    prompt = f"""You are a helpful accounting assistant for Hellobooks, designed to answer basic accounting questions for small business owners.

Use ONLY the context provided below to answer the question. If the answer is not in the context, say: "I don't have enough information to answer that question. Please consult an accountant."

Keep your answer clear, simple, and practical — suitable for a small business owner with no accounting background.

Context:
{context}

Question:
{question}

Answer:"""

    return prompt


class RAGPipeline:
    """
    End-to-end RAG pipeline for answering accounting questions.

    Attributes:
        embedding_model: Model used to embed queries and documents.
        vector_store: FAISS store holding document embeddings.
        llm_client: OpenAI API client.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        llm_model: str = LLM_MODEL,
        top_k: int = TOP_K
    ):
        """
        Initialise the RAG pipeline.

        Args:
            embedding_model: Pre-loaded embedding model.
            vector_store: Pre-loaded and populated vector store.
            llm_model: OpenAI model identifier.
            top_k: Number of document chunks to retrieve.
        """
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.llm_model = llm_model
        self.top_k = top_k

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY not found. "
                "Please set it in your .env file or environment variables."
            )
        self.llm_client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
)
        
        print(f"✅ RAG Pipeline ready. LLM: {llm_model}, Top-K: {top_k}")

    def retrieve(self, query: str) -> List[Tuple[Document, float]]:
        """
        Embed the query and retrieve the most relevant document chunks.

        Args:
            query: The user's question.

        Returns:
            List of (Document, score) tuples ordered by relevance.
        """
        query_embedding = self.embedding_model.embed_query(query)
        results = self.vector_store.search(query_embedding, top_k=self.top_k)
        return results

    def generate(self, context_chunks: List[str], question: str) -> str:
        """
        Send context + question to the LLM and return the answer.

        Args:
            context_chunks: Retrieved document text chunks.
            question: The user's original question.

        Returns:
            The LLM's generated answer string.
        """
        prompt = build_prompt(context_chunks, question)

        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a knowledgeable and friendly accounting assistant "
                        "for small business owners. Be concise, accurate, and practical."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,      # Lower = more factual, less creative
            max_tokens=512,
        )

        answer = response.choices[0].message.content.strip()
        return answer

    def answer_question(self, query: str) -> dict:
        """
        Full RAG pipeline: retrieve relevant context, then generate an answer.

        Args:
            query: The user's question.

        Returns:
            Dict with keys: answer, sources, scores
        """
        if not query.strip():
            return {"answer": "Please provide a valid question.", "sources": [], "scores": []}

        print(f"\n🔍 Query: {query}")

        # Step 1: Retrieve relevant chunks
        results = self.retrieve(query)

        if not results:
            return {
                "answer": "I couldn't find relevant information. Please try rephrasing your question.",
                "sources": [],
                "scores": []
            }

        # Step 2: Extract context and metadata
        context_chunks = [doc.content for doc, _ in results]
        sources = [doc.source for doc, _ in results]
        scores = [round(score, 4) for _, score in results]

        print(f"📚 Retrieved {len(results)} chunks from: {set(sources)}")

        # Step 3: Generate answer with LLM
        answer = self.generate(context_chunks, query)

        print(f"💬 Answer generated ({len(answer)} chars)")

        return {
            "answer": answer,
            "sources": sources,
            "scores": scores
        }
