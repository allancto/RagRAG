"""Semantic Scholar API integration for paper discovery and metadata ingestion.

This module provides a hybrid approach to paper ingestion:
1. Use Semantic Scholar to discover papers and ingest lightweight summaries
2. Upgrade to full PDF content on-demand for papers you need in depth

See docs/Hybrid-Paper-Ingestion.md for the full documentation.
"""

import requests
import time
from typing import List, Dict, Optional
from pathlib import Path


SS_API_BASE = "https://api.semanticscholar.org/graph/v1"
DEFAULT_FIELDS = "title,abstract,tldr,year,citationCount,externalIds,authors,fieldsOfStudy"

# Rate limiting settings
MAX_RETRIES = 3
RETRY_DELAY = 2.0  # seconds
REQUEST_DELAY = 0.5  # delay between requests to avoid rate limiting


def _request_with_retry(url: str, params: dict, max_retries: int = MAX_RETRIES) -> requests.Response:
    """
    Make a GET request with retry logic for rate limiting.

    Args:
        url: Request URL
        params: Query parameters
        max_retries: Maximum number of retry attempts

    Returns:
        Response object

    Raises:
        requests.exceptions.HTTPError: If all retries fail
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params)

            if response.status_code == 429:  # Rate limited
                wait_time = RETRY_DELAY * (attempt + 1)  # Exponential backoff
                print(f"  Rate limited, waiting {wait_time}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue

            response.raise_for_status()
            return response

        except requests.exceptions.HTTPError as e:
            last_error = e
            if response.status_code != 429:
                raise

    raise last_error or requests.exceptions.HTTPError("Max retries exceeded")


def search_papers(
    query: str,
    limit: int = 20,
    min_citations: int = 0,
    fields: str = DEFAULT_FIELDS
) -> List[Dict]:
    """
    Search Semantic Scholar for papers matching a query.

    Args:
        query: Search query string
        limit: Max papers to return
        min_citations: Filter to papers with at least this many citations
        fields: Comma-separated fields to retrieve

    Returns:
        List of paper dictionaries
    """
    url = f"{SS_API_BASE}/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "fields": fields
    }

    response = _request_with_retry(url, params)
    data = response.json()

    papers = data.get("data", [])

    # Filter by citation count
    if min_citations > 0:
        papers = [p for p in papers if p.get("citationCount", 0) >= min_citations]

    return papers


def get_paper_by_arxiv(arxiv_id: str, fields: str = DEFAULT_FIELDS) -> Optional[Dict]:
    """
    Get a specific paper by its arxiv ID.

    Args:
        arxiv_id: Arxiv paper ID (e.g., "2005.11401")
        fields: Comma-separated fields to retrieve

    Returns:
        Paper dictionary or None if not found
    """
    url = f"{SS_API_BASE}/paper/arXiv:{arxiv_id}"
    params = {"fields": fields}

    response = requests.get(url, params=params)
    if response.status_code == 404:
        return None
    response.raise_for_status()

    return response.json()


def paper_to_chunk(paper: Dict) -> Dict:
    """
    Convert a Semantic Scholar paper to a chunk for ingestion.

    Creates a text representation suitable for embedding and retrieval.

    Args:
        paper: Paper dictionary from Semantic Scholar API

    Returns:
        Chunk dictionary with 'text', 'metadata', and 'id' keys
    """
    # Build text content
    parts = []

    title = paper.get("title", "Unknown Title")
    parts.append(f"Title: {title}")

    # Authors
    authors = paper.get("authors", [])
    if authors:
        author_names = ", ".join(a.get("name", "") for a in authors[:5])
        if len(authors) > 5:
            author_names += f" et al. ({len(authors)} authors)"
        parts.append(f"Authors: {author_names}")

    # Year and citations
    year = paper.get("year", "N/A")
    citations = paper.get("citationCount", 0)
    parts.append(f"Year: {year} | Citations: {citations}")

    # ArXiv ID
    arxiv_id = paper.get("externalIds", {}).get("ArXiv", "")
    if arxiv_id:
        parts.append(f"ArXiv: {arxiv_id}")

    # TLDR (AI summary)
    tldr = paper.get("tldr", {})
    if tldr and tldr.get("text"):
        parts.append(f"Summary: {tldr['text']}")

    # Abstract
    abstract = paper.get("abstract", "")
    if abstract:
        parts.append(f"Abstract: {abstract}")

    text = "\n\n".join(parts)

    # Build metadata
    paper_id = paper.get("paperId", "unknown")
    metadata = {
        "source": f"semantic_scholar:{paper_id}",
        "doc_type": "paper_summary",
        "title": title,
        "year": str(year),
        "citations": str(citations),
        "arxiv_id": arxiv_id,
        "has_full_pdf": "false"  # Flag for whether we've downloaded the full PDF
    }

    return {
        "text": text,
        "metadata": metadata,
        "id": f"ss_{paper_id}"
    }


def search_and_ingest(
    query: str,
    limit: int = 20,
    min_citations: int = 10
) -> List[Dict]:
    """
    Search for papers and convert them to ingestible chunks.

    Args:
        query: Search query
        limit: Max papers
        min_citations: Minimum citation threshold

    Returns:
        List of chunks ready for add_chunks()
    """
    papers = search_papers(query, limit=limit, min_citations=min_citations)
    chunks = [paper_to_chunk(p) for p in papers]
    return chunks


def get_arxiv_pdf_url(arxiv_id: str) -> str:
    """Get the PDF download URL for an arxiv paper."""
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"


def download_arxiv_pdf(arxiv_id: str, output_dir: str = "./corpus/papers") -> Optional[str]:
    """
    Download a PDF from arxiv.

    Args:
        arxiv_id: Arxiv paper ID
        output_dir: Directory to save PDF

    Returns:
        Path to downloaded file, or None if failed
    """
    url = get_arxiv_pdf_url(arxiv_id)
    output_path = Path(output_dir) / f"{arxiv_id.replace('/', '_')}.pdf"

    response = requests.get(url, stream=True)
    if response.status_code != 200:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return str(output_path)


# Convenience function to discover papers on common RAG topics
RAG_TOPICS = [
    "retrieval augmented generation",
    "dense passage retrieval",
    "vector database embedding",
    "chunking strategies NLP",
    "RAG evaluation metrics",
    "hybrid search retrieval",
    "reranking cross encoder"
]


def discover_rag_papers(min_citations: int = 50, papers_per_topic: int = 5) -> List[Dict]:
    """
    Discover high-quality papers across RAG-related topics.

    Args:
        min_citations: Minimum citations to include
        papers_per_topic: How many papers per topic

    Returns:
        List of unique paper chunks
    """
    seen_ids = set()
    all_chunks = []

    for topic in RAG_TOPICS:
        print(f"Searching: {topic}")
        try:
            chunks = search_and_ingest(topic, limit=papers_per_topic, min_citations=min_citations)
            for chunk in chunks:
                paper_id = chunk["id"]
                if paper_id not in seen_ids:
                    seen_ids.add(paper_id)
                    all_chunks.append(chunk)
            time.sleep(0.5)  # Rate limiting
        except Exception as e:
            print(f"  Error: {e}")

    print(f"Found {len(all_chunks)} unique papers")
    return all_chunks


# =============================================================================
# UPGRADE ON-DEMAND: Convert paper summaries to full PDF content
# =============================================================================

def upgrade_paper_to_full_pdf(
    arxiv_id: str,
    corpus_dir: str = "./corpus/papers",
    chunk_size: int = 512,
    overlap_ratio: float = 0.1
) -> Optional[List[Dict]]:
    """
    Upgrade a paper from summary-only to full PDF content.

    Downloads the PDF from arxiv, parses it, and returns chunks ready for ingestion.
    The chunks have metadata indicating they are full paper content.

    Args:
        arxiv_id: Arxiv paper ID (e.g., "2005.11401")
        corpus_dir: Directory to save the PDF
        chunk_size: Target chunk size in words
        overlap_ratio: Overlap ratio for chunking

    Returns:
        List of chunks from the full PDF, or None if download/parsing failed
    """
    # Import here to avoid circular imports
    from src.ingest import ingest_document

    print(f"Downloading PDF for arxiv:{arxiv_id}...")
    pdf_path = download_arxiv_pdf(arxiv_id, output_dir=corpus_dir)

    if not pdf_path:
        print(f"  Failed to download PDF")
        return None

    print(f"  Saved to: {pdf_path}")
    print(f"  Parsing and chunking...")

    try:
        chunks = ingest_document(pdf_path, chunk_size=chunk_size, overlap_ratio=overlap_ratio)

        # Update metadata to indicate full PDF
        for chunk in chunks:
            chunk["metadata"]["has_full_pdf"] = "true"
            chunk["metadata"]["arxiv_id"] = arxiv_id

        print(f"  Created {len(chunks)} chunks from full PDF")
        return chunks

    except Exception as e:
        print(f"  Failed to parse PDF: {e}")
        return None


def list_papers_without_pdfs(store) -> List[Dict]:
    """
    Query the vector store to find paper summaries that don't have full PDFs.

    Args:
        store: ChromaDB collection (from src.store)

    Returns:
        List of metadata dicts for papers without full PDFs
    """
    # Get all documents with doc_type=paper_summary and has_full_pdf=false
    results = store.get(
        where={
            "$and": [
                {"doc_type": {"$eq": "paper_summary"}},
                {"has_full_pdf": {"$eq": "false"}}
            ]
        },
        include=["metadatas"]
    )

    papers = []
    seen_arxiv = set()

    for metadata in results.get("metadatas", []):
        arxiv_id = metadata.get("arxiv_id", "")
        if arxiv_id and arxiv_id not in seen_arxiv:
            seen_arxiv.add(arxiv_id)
            papers.append({
                "arxiv_id": arxiv_id,
                "title": metadata.get("title", "Unknown"),
                "citations": metadata.get("citations", "0"),
                "year": metadata.get("year", "N/A")
            })

    # Sort by citations (highest first)
    papers.sort(key=lambda x: int(x.get("citations", 0)), reverse=True)
    return papers


def upgrade_top_papers(
    store,
    n: int = 5,
    min_citations: int = 100
) -> List[Dict]:
    """
    Automatically upgrade the top N most-cited papers to full PDF content.

    Args:
        store: ChromaDB collection
        n: Number of papers to upgrade
        min_citations: Only upgrade papers with at least this many citations

    Returns:
        List of all new chunks created
    """
    from src.store import add_chunks

    papers = list_papers_without_pdfs(store)
    papers = [p for p in papers if int(p.get("citations", 0)) >= min_citations]

    if not papers:
        print("No papers found meeting criteria for upgrade")
        return []

    print(f"Found {len(papers)} papers eligible for upgrade")
    print(f"Upgrading top {min(n, len(papers))} by citation count...")

    all_new_chunks = []

    for paper in papers[:n]:
        arxiv_id = paper["arxiv_id"]
        print(f"\n[{paper['citations']} citations] {paper['title']}")

        chunks = upgrade_paper_to_full_pdf(arxiv_id)
        if chunks:
            add_chunks(chunks)
            all_new_chunks.extend(chunks)
            print(f"  Added {len(chunks)} chunks to vector store")

        time.sleep(1)  # Be nice to arxiv

    print(f"\nTotal: Added {len(all_new_chunks)} new chunks from {len(all_new_chunks) > 0} PDFs")
    return all_new_chunks


def upgrade_paper_by_arxiv(arxiv_id: str) -> Optional[List[Dict]]:
    """
    Convenience function to upgrade a specific paper by arxiv ID.

    Downloads the PDF, parses it, and adds chunks to the vector store.

    Args:
        arxiv_id: Arxiv paper ID

    Returns:
        List of chunks added, or None if failed

    Example:
        >>> from src.semantic_scholar import upgrade_paper_by_arxiv
        >>> chunks = upgrade_paper_by_arxiv("2309.15217")  # RAGAS paper
    """
    from src.store import add_chunks

    chunks = upgrade_paper_to_full_pdf(arxiv_id)
    if chunks:
        add_chunks(chunks)
        print(f"Added {len(chunks)} chunks to vector store")
        return chunks
    return None
