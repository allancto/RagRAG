"""RagRAG Configuration"""
import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Embedding Model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Local sentence-transformers model

# Chunking Settings
CHUNK_SIZE = 512  # tokens
CHUNK_OVERLAP = 0.1  # 10% overlap

# ChromaDB Settings
CHROMA_PERSIST_DIR = "./data/chroma"
CHROMA_COLLECTION_NAME = "ragrag_docs"

# LLM Settings
LLM_MODEL = "claude-opus-4-1-20250805"
LLM_MAX_TOKENS = 1024

# Retrieval Settings
TOP_K = 5  # Number of chunks to retrieve
