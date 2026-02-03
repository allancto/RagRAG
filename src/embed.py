"""Embedding generation using sentence-transformers."""

from typing import List
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL


class EmbeddingModel:
    """Wrapper around SentenceTransformer for generating embeddings."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """
        Initialize the embedding model.

        Args:
            model_name: Name of sentence-transformer model to use
        """
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (each is a list of floats)
        """
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()

    def embed_single(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text string to embed

        Returns:
            Embedding vector as list of floats
        """
        embedding = self.model.encode(text, convert_to_tensor=False)
        return embedding.tolist()

    def get_embedding_dim(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        return self.model.get_sentence_embedding_dimension()


# Global embedding model instance
_embedding_model = None


def get_embedding_model() -> EmbeddingModel:
    """Get or create global embedding model instance (lazy initialization)."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = EmbeddingModel()
    return _embedding_model


def embed(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of texts.

    Convenience function using global embedding model.

    Args:
        texts: List of texts to embed

    Returns:
        List of embedding vectors
    """
    model = get_embedding_model()
    return model.embed_texts(texts)
