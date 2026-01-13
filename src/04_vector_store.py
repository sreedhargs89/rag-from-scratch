"""
Module 4: Vector Store

A vector store (vector database) is where we store embeddings and perform fast
similarity searches. This is the heart of the retrieval system in RAG.

Key Concepts:
- Fast similarity search (nearest neighbors)
- Indexing strategies
- Filtering and metadata
- Approximate vs exact search
- Persistence (saving/loading)

Why Vector Stores?
- Naive search: O(n) - compare query against every vector
- Indexed search: O(log n) or better - much faster
- Essential for large-scale RAG systems
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import json
from dataclasses import dataclass, asdict
import pickle


@dataclass
class SearchResult:
    """
    Represents a search result with relevance information.

    Attributes:
        text: The retrieved text
        score: Similarity score (higher = more similar)
        metadata: Additional information about the result
        vector: The embedding vector (optional)
    """
    text: str
    score: float
    metadata: Dict
    vector: Optional[np.ndarray] = None

    def __repr__(self):
        text_preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"SearchResult(score={self.score:.4f}, text='{text_preview}')"


class VectorStore:
    """
    Base class for vector storage and retrieval.

    This abstracts the storage mechanism so you can swap implementations.
    """

    def add(self, vectors: List[np.ndarray], texts: List[str], metadata: List[Dict]):
        """Add vectors to the store"""
        raise NotImplementedError

    def search(self, query_vector: np.ndarray, k: int = 5) -> List[SearchResult]:
        """Search for k most similar vectors"""
        raise NotImplementedError

    def save(self, filepath: str):
        """Persist the store to disk"""
        raise NotImplementedError

    def load(self, filepath: str):
        """Load the store from disk"""
        raise NotImplementedError

    def clear(self):
        """Clear all data"""
        raise NotImplementedError


class SimpleVectorStore(VectorStore):
    """
    Simple in-memory vector store using numpy.

    This is a "naive" implementation that compares the query against every vector.
    - Easy to understand
    - Good for learning and small datasets (< 10k vectors)
    - Not suitable for large-scale production

    Time Complexity:
    - Add: O(1)
    - Search: O(n) where n = number of vectors
    """

    def __init__(self):
        self.vectors: List[np.ndarray] = []
        self.texts: List[str] = []
        self.metadata_list: List[Dict] = []
        print("✓ SimpleVectorStore initialized (in-memory, exact search)")

    def add(
        self,
        vectors: List[np.ndarray],
        texts: List[str],
        metadata: Optional[List[Dict]] = None
    ):
        """
        Add vectors to the store.

        Args:
            vectors: List of embedding vectors
            texts: Corresponding text chunks
            metadata: Optional metadata for each item
        """
        if metadata is None:
            metadata = [{}] * len(vectors)

        if not (len(vectors) == len(texts) == len(metadata)):
            raise ValueError("vectors, texts, and metadata must have same length")

        self.vectors.extend(vectors)
        self.texts.extend(texts)
        self.metadata_list.extend(metadata)

        print(f"✓ Added {len(vectors)} vectors (total: {len(self.vectors)})")

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        filter_fn: Optional[callable] = None
    ) -> List[SearchResult]:
        """
        Search for k most similar vectors using cosine similarity.

        Args:
            query_vector: Query embedding
            k: Number of results to return
            filter_fn: Optional function to filter results (takes metadata, returns bool)

        Returns:
            List of SearchResult objects, sorted by similarity (highest first)
        """
        if len(self.vectors) == 0:
            return []

        # Calculate similarities with all vectors
        similarities = []
        for idx, vector in enumerate(self.vectors):
            # Apply filter if provided
            if filter_fn and not filter_fn(self.metadata_list[idx]):
                continue

            similarity = self._cosine_similarity(query_vector, vector)
            similarities.append((idx, similarity))

        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top k results
        results = []
        for idx, score in similarities[:k]:
            result = SearchResult(
                text=self.texts[idx],
                score=score,
                metadata=self.metadata_list[idx],
                vector=self.vectors[idx]
            )
            results.append(result)

        return results

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))

    def save(self, filepath: str):
        """Save to disk using pickle"""
        data = {
            "vectors": self.vectors,
            "texts": self.texts,
            "metadata_list": self.metadata_list
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        print(f"✓ Saved vector store to {filepath}")

    def load(self, filepath: str):
        """Load from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.vectors = data["vectors"]
        self.texts = data["texts"]
        self.metadata_list = data["metadata_list"]

        print(f"✓ Loaded vector store from {filepath} ({len(self.vectors)} vectors)")

    def clear(self):
        """Clear all data"""
        self.vectors = []
        self.texts = []
        self.metadata_list = []

    def __len__(self):
        return len(self.vectors)


class FAISSVectorStore(VectorStore):
    """
    Vector store using FAISS (Facebook AI Similarity Search).

    FAISS provides fast approximate nearest neighbor search:
    - Much faster than SimpleVectorStore for large datasets
    - Uses indexing (e.g., IVF, HNSW) for speed
    - Suitable for production use

    Install: pip install faiss-cpu (or faiss-gpu for GPU support)

    Time Complexity:
    - Add: O(log n) with index rebuilding
    - Search: O(log n) for approximate search
    """

    def __init__(self, dimension: int, index_type: str = "Flat"):
        """
        Args:
            dimension: Dimension of the vectors
            index_type: Type of FAISS index
                - "Flat": Exact search (slower but accurate)
                - "IVF": Approximate search (faster)
                - "HNSW": Hierarchical graph-based (fast and accurate)
        """
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "FAISS not installed. Install with: pip install faiss-cpu"
            )

        self.dimension = dimension
        self.index_type = index_type
        self.texts: List[str] = []
        self.metadata_list: List[Dict] = []

        # Create FAISS index
        if index_type == "Flat":
            self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine for normalized)
        elif index_type == "IVF":
            # IVF = Inverted File Index (clusters data for faster search)
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)  # 100 clusters
        elif index_type == "HNSW":
            # HNSW = Hierarchical Navigable Small World
            self.index = faiss.IndexHNSWFlat(dimension, 32)  # 32 = number of connections
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        print(f"✓ FAISSVectorStore initialized (type={index_type}, dim={dimension})")

    def add(
        self,
        vectors: List[np.ndarray],
        texts: List[str],
        metadata: Optional[List[Dict]] = None
    ):
        """Add vectors to FAISS index"""
        if metadata is None:
            metadata = [{}] * len(vectors)

        # Convert to numpy array and normalize (for cosine similarity)
        vectors_array = np.array(vectors).astype('float32')
        faiss.normalize_L2(vectors_array)  # Normalize for cosine similarity

        # Train index if needed (for IVF)
        if self.index_type == "IVF" and not self.index.is_trained:
            self.index.train(vectors_array)

        # Add to index
        self.index.add(vectors_array)

        # Store texts and metadata separately
        self.texts.extend(texts)
        self.metadata_list.extend(metadata)

        print(f"✓ Added {len(vectors)} vectors (total: {self.index.ntotal})")

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        filter_fn: Optional[callable] = None
    ) -> List[SearchResult]:
        """Search using FAISS"""
        import faiss

        if self.index.ntotal == 0:
            return []

        # Normalize query vector
        query = query_vector.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query)

        # Search
        scores, indices = self.index.search(query, min(k * 2, self.index.ntotal))

        # Convert to SearchResult objects
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue

            # Apply filter if provided
            if filter_fn and not filter_fn(self.metadata_list[idx]):
                continue

            result = SearchResult(
                text=self.texts[idx],
                score=float(score),
                metadata=self.metadata_list[idx]
            )
            results.append(result)

            if len(results) >= k:
                break

        return results

    def save(self, filepath: str):
        """Save FAISS index and metadata"""
        import faiss

        # Save FAISS index
        faiss.write_index(self.index, f"{filepath}.index")

        # Save metadata
        data = {
            "texts": self.texts,
            "metadata_list": self.metadata_list,
            "dimension": self.dimension,
            "index_type": self.index_type
        }
        with open(f"{filepath}.meta", 'wb') as f:
            pickle.dump(data, f)

        print(f"✓ Saved FAISS vector store to {filepath}")

    def load(self, filepath: str):
        """Load FAISS index and metadata"""
        import faiss

        # Load index
        self.index = faiss.read_index(f"{filepath}.index")

        # Load metadata
        with open(f"{filepath}.meta", 'rb') as f:
            data = pickle.load(f)

        self.texts = data["texts"]
        self.metadata_list = data["metadata_list"]
        self.dimension = data["dimension"]
        self.index_type = data["index_type"]

        print(f"✓ Loaded FAISS vector store from {filepath} ({self.index.ntotal} vectors)")

    def clear(self):
        """Clear the index"""
        self.index.reset()
        self.texts = []
        self.metadata_list = []

    def __len__(self):
        return self.index.ntotal


# ============================================================================
# EXAMPLES AND EXERCISES
# ============================================================================

def example_simple_store():
    """Example: Using SimpleVectorStore"""
    print("=" * 80)
    print("Example 1: Simple Vector Store")
    print("=" * 80)

    # Create some sample embeddings (random for demo)
    np.random.seed(42)
    dimension = 384

    texts = [
        "Machine learning is a subset of AI",
        "Deep learning uses neural networks",
        "Python is a programming language",
        "Neural networks mimic the human brain",
        "Data science involves statistics"
    ]

    # Create random vectors (in real use, these come from embedding model)
    vectors = [np.random.randn(dimension) for _ in texts]

    # Create store and add vectors
    store = SimpleVectorStore()
    store.add(vectors, texts)

    # Create a query (random for demo)
    query = np.random.randn(dimension)

    # Search
    print("\nSearching for top 3 similar texts...")
    results = store.search(query, k=3)

    print("\nResults:")
    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result.score:.4f}")
        print(f"   Text: {result.text}")


def example_with_metadata():
    """Example: Using metadata for filtering"""
    print("\n" + "=" * 80)
    print("Example 2: Metadata Filtering")
    print("=" * 80)

    np.random.seed(42)
    dimension = 384

    # Texts with categories
    texts = [
        "Python is great for data science",
        "JavaScript is used for web development",
        "Machine learning predicts patterns",
        "React is a JavaScript framework",
        "NumPy is a Python library"
    ]

    categories = ["python", "javascript", "ml", "javascript", "python"]

    vectors = [np.random.randn(dimension) for _ in texts]
    metadata = [{"category": cat} for cat in categories]

    # Add to store
    store = SimpleVectorStore()
    store.add(vectors, texts, metadata)

    # Search with filter
    query = np.random.randn(dimension)

    print("\n1. Search without filter:")
    results = store.search(query, k=3)
    for r in results:
        print(f"  - {r.text} (category: {r.metadata['category']})")

    print("\n2. Search only Python-related:")
    results = store.search(
        query,
        k=3,
        filter_fn=lambda meta: meta["category"] == "python"
    )
    for r in results:
        print(f"  - {r.text} (category: {r.metadata['category']})")


def example_persistence():
    """Example: Save and load vector store"""
    print("\n" + "=" * 80)
    print("Example 3: Persistence")
    print("=" * 80)

    np.random.seed(42)
    dimension = 384

    texts = ["Document 1", "Document 2", "Document 3"]
    vectors = [np.random.randn(dimension) for _ in texts]

    # Create and save
    print("\nCreating store...")
    store = SimpleVectorStore()
    store.add(vectors, texts)

    filepath = "../data/vector_store.pkl"
    store.save(filepath)

    # Load in new store
    print("\nLoading store...")
    new_store = SimpleVectorStore()
    new_store.load(filepath)

    print(f"Loaded store has {len(new_store)} vectors")


def example_faiss_comparison():
    """Example: Compare SimpleVectorStore vs FAISS"""
    print("\n" + "=" * 80)
    print("Example 4: FAISS Performance Comparison")
    print("=" * 80)

    try:
        import time

        dimension = 384
        n_vectors = 1000

        print(f"\nGenerating {n_vectors} random vectors...")
        np.random.seed(42)
        vectors = [np.random.randn(dimension) for _ in range(n_vectors)]
        texts = [f"Document {i}" for i in range(n_vectors)]
        query = np.random.randn(dimension)

        # Test SimpleVectorStore
        print("\n1. SimpleVectorStore (exact search):")
        simple_store = SimpleVectorStore()
        simple_store.add(vectors, texts)

        start = time.time()
        results = simple_store.search(query, k=5)
        elapsed = time.time() - start
        print(f"   Search time: {elapsed*1000:.2f}ms")

        # Test FAISS
        print("\n2. FAISSVectorStore (approximate search):")
        faiss_store = FAISSVectorStore(dimension, index_type="Flat")
        faiss_store.add(vectors, texts)

        start = time.time()
        results = faiss_store.search(query, k=5)
        elapsed = time.time() - start
        print(f"   Search time: {elapsed*1000:.2f}ms")

        print("\n💡 For larger datasets (10k+ vectors), FAISS is significantly faster!")

    except ImportError:
        print("\n⚠️  FAISS not installed")
        print("Install with: pip install faiss-cpu")


def exercise_1():
    """
    EXERCISE 1: Build a Mini Search Engine

    Task: Create a vector store with 10 documents and implement search
    """
    print("\n" + "=" * 80)
    print("EXERCISE 1: Mini Search Engine")
    print("=" * 80)

    # TODO: Create 10 text documents about different topics
    # Create random vectors for them
    # Add to store
    # Search for a query and print results

    print("\nYour code here!")


def exercise_2():
    """
    EXERCISE 2: Metadata Filtering

    Task: Add documents with dates, search only recent ones
    """
    print("\n" + "=" * 80)
    print("EXERCISE 2: Date-based Filtering")
    print("=" * 80)

    # TODO: Add documents with "date" metadata
    # Search with filter to only return documents from after a certain date

    print("\nYour code here!")


if __name__ == "__main__":
    print("\n🚀 RAG FROM SCRATCH - MODULE 4: VECTOR STORE\n")

    # Run examples
    example_simple_store()
    example_with_metadata()
    example_persistence()
    example_faiss_comparison()

    # Exercises
    exercise_1()
    exercise_2()

    print("\n" + "=" * 80)
    print("Key Takeaways:")
    print("=" * 80)
    print("✓ Vector stores enable fast similarity search")
    print("✓ SimpleVectorStore: Easy to understand, good for learning")
    print("✓ FAISS: Production-ready, much faster for large datasets")
    print("✓ Metadata filtering adds powerful query capabilities")
    print("✓ Cosine similarity is the standard metric for text embeddings")
    print("✓ Persistence allows saving/loading your vector database")
    print("✓ Next: We'll learn how to retrieve and build context for generation")
    print("=" * 80)
