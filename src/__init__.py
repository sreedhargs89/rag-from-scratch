"""
RAG From Scratch - Source Modules

This package contains all the components needed to build a RAG system.
"""

from src.document_loader import DocumentLoader, Document
from src.chunking import (
    Chunk,
    TextChunker,
    FixedSizeChunker,
    SentenceChunker,
    RecursiveChunker
)
from src.embeddings import (
    Embedding,
    EmbeddingModel,
    SentenceTransformerModel,
    SimpleEmbeddingModel,
    EmbeddingStore,
    cosine_similarity
)
from src.vector_store import (
    SearchResult,
    VectorStore,
    SimpleVectorStore,
    FAISSVectorStore
)
from src.retrieval import (
    RetrievalResult,
    Retriever,
    HybridRetriever,
    QueryExpander,
    ReRanker
)
from src.rag_pipeline import RAGPipeline

__version__ = "1.0.0"

__all__ = [
    # Document Loading
    "DocumentLoader",
    "Document",
    # Chunking
    "Chunk",
    "TextChunker",
    "FixedSizeChunker",
    "SentenceChunker",
    "RecursiveChunker",
    # Embeddings
    "Embedding",
    "EmbeddingModel",
    "SentenceTransformerModel",
    "SimpleEmbeddingModel",
    "EmbeddingStore",
    "cosine_similarity",
    # Vector Store
    "SearchResult",
    "VectorStore",
    "SimpleVectorStore",
    "FAISSVectorStore",
    # Retrieval
    "RetrievalResult",
    "Retriever",
    "HybridRetriever",
    "QueryExpander",
    "ReRanker",
    # Pipeline
    "RAGPipeline",
]
