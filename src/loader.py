

import os
from typing import List
from dataclasses import dataclass


@dataclass
class Document:
    """Represents a single text chunk with its source metadata."""
    content: str
    source: str
    chunk_index: int


def load_markdown_files(data_dir: str) -> List[str]:
    """
    Load all markdown (.md) files from a directory.

    Args:
        data_dir: Path to the folder containing .md files.

    Returns:
        List of raw text strings, one per file.
    """
    documents = []
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith(".md"):
            filepath = os.path.join(data_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read().strip()
                if text:
                    documents.append((filename, text))
                    print(f"  ✅ Loaded: {filename} ({len(text)} chars)")

    if not documents:
        raise ValueError(f"No markdown files found in: {data_dir}")

    return documents


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> List[str]:
    """
    Split a long text into overlapping chunks.

    Args:
        text: The raw text to split.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of characters to overlap between chunks.

    Returns:
        List of text chunk strings.
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        # Move forward by chunk_size minus overlap
        start += chunk_size - chunk_overlap

    return chunks


def load_documents(data_dir: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[Document]:
    """
    Full pipeline: load markdown files, split into chunks, return Document objects.

    Args:
        data_dir: Path to the data directory.
        chunk_size: Characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        List of Document objects with content, source, and chunk index.
    """
    print(f"\n📂 Loading documents from: {data_dir}")
    raw_files = load_markdown_files(data_dir)

    all_documents: List[Document] = []
    for filename, text in raw_files:
        chunks = chunk_text(text, chunk_size, chunk_overlap)
        for i, chunk in enumerate(chunks):
            all_documents.append(Document(
                content=chunk,
                source=filename,
                chunk_index=i
            ))

    print(f"\n📄 Total chunks created: {len(all_documents)}")
    return all_documents
