# RagRAG Implementation Plan

## Overview

Build a simple, self-contained RAG system that answers questions about building RAGs. The corpus consists of academic papers, framework documentation, and implementation guides.

**Stack:** ChromaDB + sentence-transformers + Claude API + pure Python

---

## Phase 1: Project Setup

### 1.1 Directory Structure
```
RagRAG/
├── README.md
├── requirements.txt
├── config.py                    # API keys, model settings
├── .env                         # Secrets (gitignored)
│
├── docs/                        # Project documentation
│   ├── Ideation-Q&A.md
│   └── Implementation-Plan.md
│
├── corpus/                      # Source documents
│   ├── SOURCES.md               # Tracks where everything came from
│   ├── papers/                  # Academic PDFs (Arxiv)
│   │   └── arxiv/
│   ├── frameworks/              # Framework documentation
│   │   ├── llamaindex/
│   │   ├── langchain/
│   │   └── chromadb/
│   └── community/               # Real-world discussions & issues
│       ├── stackoverflow/
│       ├── reddit/
│       └── geeksforgeeks/
│
├── src/
│   ├── ingest.py                # Document loading & chunking
│   ├── embed.py                 # Embedding generation
│   ├── store.py                 # ChromaDB operations
│   ├── retrieve.py              # Query & retrieval logic
│   ├── generate.py              # LLM response generation
│   └── rag.py                   # Main orchestrator (CLI)
│
└── data/
    └── chroma/                  # ChromaDB persistence (gitignored)
```

### 1.2 Dependencies
```
chromadb
sentence-transformers
anthropic
pypdf2                       # For PDF parsing
beautifulsoup4               # For HTML docs
python-dotenv                # For API key management
```

### 1.3 Configuration
- Store API keys in `.env` (gitignored)
- Default embedding model: `all-MiniLM-L6-v2`
- Default chunk size: 512 tokens, 10% overlap
- ChromaDB collection name: `ragrag_docs`

---

## Phase 2: Document Ingestion

### 2.1 Supported Formats
| Format | Parser | Notes |
|--------|--------|-------|
| PDF | PyPDF2 | Academic papers |
| Markdown | Native Python | Framework docs |
| HTML | BeautifulSoup | Web documentation |
| Plain text | Native Python | Misc guides |

### 2.2 Chunking Strategy
```python
def chunk_document(text, chunk_size=512, overlap=0.1):
    # 1. Split by natural boundaries first (paragraphs, sections)
    # 2. If chunk > chunk_size, split further
    # 3. Add overlap from previous chunk
    # 4. Preserve metadata (source file, section heading)
```

### 2.3 Metadata to Preserve
- `source`: Original filename
- `doc_type`: paper | framework | guide
- `section`: Heading or section title (if available)
- `chunk_index`: Position in original document

---

## Phase 3: Embedding & Storage

### 3.1 Embedding Pipeline
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_chunks(chunks: list[str]) -> list[list[float]]:
    return model.encode(chunks).tolist()
```

### 3.2 ChromaDB Storage
```python
import chromadb

client = chromadb.PersistentClient(path="./data/chroma")
collection = client.get_or_create_collection("ragrag_docs")

def store_chunks(chunks, embeddings, metadata):
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadata,
        ids=[generate_id(chunk) for chunk in chunks]
    )
```

### 3.3 Incremental Updates
- Hash each document to detect changes
- Only re-embed modified or new documents
- Store hash in metadata for comparison

---

## Phase 4: Retrieval

### 4.1 Basic Retrieval
```python
def retrieve(query: str, top_k: int = 5):
    query_embedding = model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results
```

### 4.2 Retrieval Enhancements (Future)
- [ ] Hybrid search (semantic + keyword via BM25)
- [ ] Reranking with cross-encoder
- [ ] MMR (Maximal Marginal Relevance) for diversity

---

## Phase 5: Generation

### 5.1 Prompt Template
```python
SYSTEM_PROMPT = """You are a helpful assistant that answers questions about
building RAG systems. Use the provided context to answer questions accurately.
If the context doesn't contain enough information, say so."""

def build_prompt(query: str, context_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    return f"""Context:
{context}

Question: {query}

Answer based on the context above:"""
```

### 5.2 Claude Integration
```python
import anthropic

client = anthropic.Anthropic()

def generate_response(query: str, context: list[str]) -> str:
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(query, context)}]
    )
    return response.content[0].text
```

---

## Phase 6: Main Interface

### 6.1 CLI Interface
```python
# rag.py
def ask(question: str) -> str:
    # 1. Retrieve relevant chunks
    results = retrieve(question, top_k=5)

    # 2. Extract document text
    context = results['documents'][0]

    # 3. Generate response
    answer = generate_response(question, context)

    return answer

if __name__ == "__main__":
    while True:
        q = input("\nAsk about RAGs: ")
        if q.lower() in ['quit', 'exit']:
            break
        print(ask(q))
```

### 6.2 Future Interfaces
- [ ] Streamlit web UI
- [ ] API endpoint (FastAPI)
- [ ] Integration with Claude Code

---

## Implementation Order

| Step | Task | Estimated Effort |
|------|------|------------------|
| 1 | Project setup, dependencies | 15 min |
| 2 | Basic PDF/MD ingestion | 30 min |
| 3 | Chunking logic | 30 min |
| 4 | Embedding + ChromaDB storage | 20 min |
| 5 | Retrieval function | 15 min |
| 6 | Claude generation | 15 min |
| 7 | CLI orchestrator | 15 min |
| 8 | Test with sample docs | 20 min |

**Total MVP: ~2.5 hours**

---

## Open Questions

1. **Corpus sourcing:** ✅ Decided
   - Arxiv papers (RAG theory, benchmarks)
   - Framework docs (LlamaIndex, LangChain, ChromaDB)
   - Community discussions (StackOverflow, Reddit, GeeksForGeeks)

2. **Embedding model:** Start with local `all-MiniLM-L6-v2` or go straight to OpenAI for better quality on technical content?

3. **Chunk size:** 512 tokens is a reasonable default, but academic papers might benefit from larger chunks (1024) to preserve context. Test both?

4. **Evaluation:** How do we measure if the RAG is working well?
   - Manual spot-checking?
   - Create a small test set of Q&A pairs?

---

## Next Steps

1. Confirm this plan works for you
2. Decide on initial corpus sources
3. Start Phase 1 implementation
