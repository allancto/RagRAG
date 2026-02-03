"""ChromaDB storage and retrieval operations."""

from typing import List, Dict, Optional
import chromadb
from chromadb.api.models.Collection import Collection
from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME
from src.embed import embed


class ChromaStore:
    """Wrapper around ChromaDB for storing and retrieving embeddings."""

    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR, collection_name: str = CHROMA_COLLECTION_NAME):
        """
        Initialize ChromaDB persistent client.

        Args:
            persist_dir: Directory to persist ChromaDB data
            collection_name: Name of the collection to use
        """
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection: Collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.collection_name = collection_name

    def add_chunks(self, chunks: List[Dict[str, str]]) -> None:
        """
        Add chunks with embeddings to ChromaDB.

        Args:
            chunks: List of dicts with 'text', 'metadata', and 'id' keys
        """
        if not chunks:
            return

        # Extract texts and generate embeddings
        texts = [chunk['text'] for chunk in chunks]
        embeddings = embed(texts)

        # Prepare data for ChromaDB
        ids = [chunk['id'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        documents = texts

        # Add to collection
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def query(self, query_text: str, top_k: int = 5) -> Dict:
        """
        Query for similar chunks.

        Args:
            query_text: Query string
            top_k: Number of results to return

        Returns:
            Dict with 'ids', 'documents', 'metadatas', 'distances'
        """
        # Embed query
        query_embedding = embed([query_text])[0]

        # Query collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        return results

    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection."""
        count = self.collection.count()
        return {
            'collection_name': self.collection_name,
            'total_documents': count
        }

    def delete_collection(self) -> None:
        """Delete the entire collection (useful for resets)."""
        self.client.delete_collection(name=self.collection_name)

    def upsert_chunks(self, chunks: List[Dict[str, str]]) -> None:
        """
        Update or insert chunks (idempotent).

        Args:
            chunks: List of dicts with 'text', 'metadata', and 'id' keys
        """
        if not chunks:
            return

        # Extract texts and generate embeddings
        texts = [chunk['text'] for chunk in chunks]
        embeddings = embed(texts)

        # Prepare data for ChromaDB
        ids = [chunk['id'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        documents = texts

        # Upsert to collection
        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def delete_by_source(self, source: str) -> None:
        """
        Delete all chunks from a specific source document.

        Args:
            source: Source filename to delete
        """
        # Query for all chunks from this source
        where_filter = {"source": {"$eq": source}}
        results = self.collection.get(where=where_filter, include=[])

        if results['ids']:
            self.collection.delete(ids=results['ids'])


# Global store instance
_store = None


def get_store() -> ChromaStore:
    """Get or create global ChromaStore instance (lazy initialization)."""
    global _store
    if _store is None:
        _store = ChromaStore()
    return _store


def add_chunks(chunks: List[Dict[str, str]]) -> None:
    """Convenience function to add chunks using global store."""
    store = get_store()
    store.add_chunks(chunks)


def query(query_text: str, top_k: int = 5) -> Dict:
    """Convenience function to query using global store."""
    store = get_store()
    return store.query(query_text, top_k=top_k)


def get_stats() -> Dict:
    """Convenience function to get store stats."""
    store = get_store()
    return store.get_collection_stats()
