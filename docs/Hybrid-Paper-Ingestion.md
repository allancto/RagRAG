# Hybrid Paper Ingestion Strategy

This document explains RagRAG's approach to ingesting academic papers using a hybrid strategy that balances coverage with depth.

---

## The Problem

Building a RAG corpus from academic papers presents a tradeoff:

| Approach | Pros | Cons |
|----------|------|------|
| Download all PDFs | Full content, methods, experiments | Slow, storage-heavy, noisy (references, equations) |
| Abstracts only | Fast, clean, broad coverage | Miss implementation details |

**Solution: Start lightweight, upgrade on-demand.**

---

## Hybrid Strategy Overview

```
┌─────────────────────────────────────────────────────────────┐
│  PHASE 1: DISCOVER (Semantic Scholar API)                   │
│                                                             │
│  - Search topics: "RAG evaluation", "chunking", etc.        │
│  - Retrieve: title, abstract, TLDR, citations, arxiv_id     │
│  - Ingest as lightweight "paper_summary" chunks             │
│  - metadata.has_full_pdf = "false"                          │
│                                                             │
│  Result: Broad coverage with minimal storage                │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 2: QUERY YOUR RAG                                    │
│                                                             │
│  User: "How do I evaluate a RAG system?"                    │
│  → Retrieves RAGAS paper summary chunk                      │
│  → Answer mentions RAGAS framework, cites arxiv:2309.15217  │
│                                                             │
│  The summary is often enough to answer the question!        │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 3: UPGRADE ON-DEMAND                                 │
│                                                             │
│  You decide: "I want the full RAGAS paper"                  │
│  → upgrade_paper_by_arxiv("2309.15217")                     │
│  → Downloads PDF from arxiv                                 │
│  → Parses and chunks full content                           │
│  → Adds to vector store                                     │
│  → Updates metadata.has_full_pdf = "true"                   │
│                                                             │
│  Now queries can access full methodology, experiments, etc. │
└─────────────────────────────────────────────────────────────┘
```

---

## What Semantic Scholar Provides

For each paper, the API returns:

```json
{
  "paperId": "abc123",
  "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
  "abstract": "Large pre-trained language models have been shown to...",
  "tldr": {
    "text": "RAG combines parametric and non-parametric memory for better factual generation"
  },
  "year": 2020,
  "citationCount": 2847,
  "authors": [{"name": "Patrick Lewis"}, ...],
  "externalIds": {"ArXiv": "2005.11401"},
  "fieldsOfStudy": ["Computer Science"]
}
```

This gets converted to a chunk like:

```
Title: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

Authors: Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin

Year: 2020 | Citations: 2847

ArXiv: 2005.11401

Summary: RAG combines parametric and non-parametric memory for better factual generation

Abstract: Large pre-trained language models have been shown to store factual knowledge...
```

**Metadata includes:**
- `doc_type: "paper_summary"`
- `has_full_pdf: "false"` (or `"true"` after upgrade)
- `arxiv_id` for easy PDF download later
- `citations` for prioritizing important papers

---

## Usage

### 1. Discover papers on a topic

```python
from src.semantic_scholar import search_and_ingest
from src.store import add_chunks

# Search for papers and get summary chunks
chunks = search_and_ingest("retrieval augmented generation", limit=20, min_citations=50)

# Add to vector store
add_chunks(chunks)
```

### 2. Bulk discover across RAG topics

```python
from src.semantic_scholar import discover_rag_papers
from src.store import add_chunks

# Searches multiple RAG-related topics
chunks = discover_rag_papers(min_citations=100, papers_per_topic=10)
add_chunks(chunks)
```

### 3. Upgrade a specific paper to full PDF

```python
from src.semantic_scholar import upgrade_paper_by_arxiv

# Download and ingest full PDF content
chunks = upgrade_paper_by_arxiv("2309.15217")  # RAGAS paper
```

### 4. See which papers could be upgraded

```python
from src.semantic_scholar import list_papers_without_pdfs
from src.store import get_collection

store = get_collection()
papers = list_papers_without_pdfs(store)

for p in papers[:10]:
    print(f"[{p['citations']} citations] {p['title']} (arxiv:{p['arxiv_id']})")
```

### 5. Auto-upgrade top cited papers

```python
from src.semantic_scholar import upgrade_top_papers
from src.store import get_collection

store = get_collection()

# Download and ingest the top 5 most-cited papers that don't have PDFs yet
upgrade_top_papers(store, n=5, min_citations=500)
```

---

## Benefits of This Approach

1. **Fast startup** - Ingest 50+ paper summaries in seconds
2. **Broad coverage** - Answer "what papers exist on X?" without downloading everything
3. **Smart prioritization** - Citation counts help identify important papers
4. **Storage efficient** - Only download PDFs you actually need
5. **Incremental depth** - Upgrade papers as your needs become clear
6. **Metadata tracking** - Always know which papers have full content

---

## RAG Topics Searched by Default

The `discover_rag_papers()` function searches these topics:

- retrieval augmented generation
- dense passage retrieval
- vector database embedding
- chunking strategies NLP
- RAG evaluation metrics
- hybrid search retrieval
- reranking cross encoder

You can customize by calling `search_and_ingest()` with your own queries.

---

## Rate Limiting

Semantic Scholar has rate limits. The module includes:
- Automatic retry with exponential backoff on 429 errors
- 0.5s delay between requests in bulk operations
- Max 3 retries per request

If you hit persistent rate limits, wait a few minutes and try again.

---

## File Locations

- **Paper summaries**: Stored only in ChromaDB (no files)
- **Downloaded PDFs**: `corpus/papers/{arxiv_id}.pdf`
- **Module**: `src/semantic_scholar.py`
