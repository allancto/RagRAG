"""Retrieval module for RAG queries."""

from dataclasses import dataclass
from typing import List, Dict, Optional
from src.store import query as store_query
from config import TOP_K


@dataclass
class RetrievalResult:
    """Structured result from retrieval."""
    documents: List[str]
    metadatas: List[Dict]
    distances: List[float]

    @property
    def sources(self) -> List[str]:
        """Get unique source documents."""
        return list(set(m.get('source', 'unknown') for m in self.metadatas))

    def __len__(self) -> int:
        return len(self.documents)


def retrieve(query: str, top_k: int = TOP_K) -> RetrievalResult:
    """
    Retrieve relevant chunks for a query.

    Args:
        query: User query string
        top_k: Number of results to return

    Returns:
        RetrievalResult with documents, metadata, and distances
    """
    results = store_query(query, top_k=top_k)

    # ChromaDB returns nested lists (one per query)
    documents = results['documents'][0] if results['documents'] else []
    metadatas = results['metadatas'][0] if results['metadatas'] else []
    distances = results['distances'][0] if results['distances'] else []

    return RetrievalResult(
        documents=documents,
        metadatas=metadatas,
        distances=distances
    )


def format_context(results: RetrievalResult, include_sources: bool = True) -> str:
    """
    Format retrieved chunks into context string for LLM.

    Args:
        results: RetrievalResult from retrieve()
        include_sources: Whether to include source attribution

    Returns:
        Formatted context string
    """
    if not results.documents:
        return ""

    chunks = []
    for i, (doc, meta) in enumerate(zip(results.documents, results.metadatas), 1):
        if include_sources:
            source = meta.get('source', 'unknown')
            chunks.append(f"[{i}] (Source: {source})\n{doc}")
        else:
            chunks.append(doc)

    return "\n\n---\n\n".join(chunks)


def retrieve_and_format(query: str, top_k: int = TOP_K) -> tuple[str, RetrievalResult]:
    """
    Convenience function: retrieve and format in one call.

    Args:
        query: User query string
        top_k: Number of results

    Returns:
        Tuple of (formatted_context, raw_results)
    """
    results = retrieve(query, top_k=top_k)
    context = format_context(results)
    return context, results
