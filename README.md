# RagRAG

A RAG system for learning about RAG. RagRAG collects and indexes information about how to build, use, and maintain Retrieval Augmented Generation systems.

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/allancto/RagRAG.git
cd RagRAG

# 2. Install dependencies
pip install -r requirements.txt

# 3. Build the vector store
python scripts/setup.py

# 4. Set your API key (for generation)
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# 5. Query the RAG
python -c "from src.rag import query_rag; print(query_rag('How do I evaluate a RAG system?'))"
```

## Setup Details

The repository includes the corpus files (PDFs + framework docs) but **not** the vector store. You must run the setup script to build it:

```bash
python scripts/setup.py
```

This script:
1. **Ingests corpus files** - Parses the 25 PDFs and framework markdown docs, creates embeddings, stores in ChromaDB
2. **Fetches paper summaries** - Gets metadata from Semantic Scholar for additional RAG papers
3. **Fetches community content** - Downloads recent Reddit posts and StackOverflow questions about RAG

The setup takes a few minutes (mostly embedding generation). Once complete, you'll have ~800 chunks in your vector store.

**Options:**
```bash
python scripts/setup.py --skip-community  # Skip Reddit/StackOverflow (faster)
python scripts/setup.py --skip-papers     # Skip Semantic Scholar summaries
```

## Corpus Statistics

| Document Type | Chunks |
|---------------|--------|
| Paper (full PDF) | 564 |
| Community (Reddit) | 100 |
| Community (StackOverflow) | 100 |
| Framework docs | 31 |
| Paper summaries (metadata) | 14 |
| **Total** | **~809** |

**Source Files:**
- 25 academic papers (PDFs)
- 10 framework documentation files (ChromaDB + LlamaIndex)

## What's Included

### Academic Papers (25 PDFs)

**Foundational:**
- Original RAG paper (2020)
- Dense Passage Retrieval (DPR, 2020)
- REALM (2020)

**Surveys:**
- RAG Survey 2023
- RAG for AIGC Survey 2024

**Evaluation:**
- RAGAS - Automated RAG Evaluation
- RAGTruth - Hallucination Benchmark
- MedRAG Benchmark

**Advanced Techniques:**
- Self-RAG - Self-Reflective RAG
- CRAG - Corrective RAG
- RAPTOR - Recursive Abstractive Processing
- HyDE - Hypothetical Document Embeddings
- RAG-Fusion
- GNN-RAG - Graph Neural Retrieval
- MultiHop-RAG

**Practical:**
- Lost in the Middle (context window analysis)
- Toolformer
- Gorilla (API retrieval)
- CacheBlend (efficient RAG serving)

### Framework Documentation

- **ChromaDB** (9 docs): Getting started, architecture, querying, embedding functions, metadata filtering, etc.
- **LlamaIndex** (4 docs): RAG overview, concepts, starter examples, Q&A

### Community Content

Fetched fresh during setup:
- **~100 Reddit posts** from r/LocalLLaMA, r/MachineLearning, r/LangChain
- **~100 StackOverflow questions** on RAG, vector databases, LangChain, LlamaIndex, ChromaDB

## Architecture

```
corpus/
  papers/           # 25 PDF papers from arxiv
  frameworks/
    chromadb/       # ChromaDB documentation
    llamaindex/     # LlamaIndex documentation

src/
  ingest.py         # Document parsing (PDF, MD, HTML, TXT)
  embed.py          # Embeddings via sentence-transformers (all-MiniLM-L6-v2)
  store.py          # ChromaDB vector store operations
  retrieve.py       # Semantic search
  generate.py       # LLM generation (Claude)
  rag.py            # Full RAG pipeline
  semantic_scholar.py  # Paper discovery & hybrid ingestion
  community.py      # Reddit/StackOverflow fetching

scripts/
  setup.py          # Build vector store from corpus

data/
  chroma/           # Persistent vector store (created by setup.py)
```

## Hybrid Paper Ingestion

RagRAG uses a hybrid approach for academic papers:

1. **Discover** - Use Semantic Scholar API to find papers, ingest lightweight summaries (title, abstract, TLDR, citations)
2. **Query** - Summaries are often enough to answer questions
3. **Upgrade on-demand** - Download full PDFs only when needed

See [docs/Hybrid-Paper-Ingestion.md](docs/Hybrid-Paper-Ingestion.md) for details.

## Usage

### Query the RAG

```python
from src.rag import query_rag

response = query_rag("How do I evaluate a RAG system?")
print(response)
```

### Manual Ingestion

```python
from src.ingest import ingest_directory
from src.store import add_chunks

# Ingest all documents from corpus
chunks = ingest_directory('./corpus')
add_chunks(chunks)
```

### Discover Papers

```python
from src.semantic_scholar import discover_rag_papers, upgrade_paper_by_arxiv

# Find papers on RAG topics (summaries only)
chunks = discover_rag_papers(min_citations=100)

# Upgrade a specific paper to full PDF
upgrade_paper_by_arxiv("2309.15217")  # RAGAS paper
```

### Fetch Community Content

```python
from src.community import fetch_all_community_content
from src.store import add_chunks

content = fetch_all_community_content(reddit_limit=100, stackoverflow_limit=100)
add_chunks(content['reddit'])
add_chunks(content['stackoverflow'])
```

## Configuration

Set your API key in `.env`:

```
ANTHROPIC_API_KEY=your-key-here
```

Configuration options in `config.py`:
- `EMBEDDING_MODEL` - sentence-transformers model (default: all-MiniLM-L6-v2)
- `CHUNK_SIZE` - target chunk size in tokens (default: 512)
- `TOP_K` - number of chunks to retrieve (default: 5)
- `LLM_MODEL` - Claude model for generation

## Requirements

- Python 3.10+
- ChromaDB
- sentence-transformers
- PyPDF2
- BeautifulSoup4
- anthropic (for generation)
- requests (for API fetching)
