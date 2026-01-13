"""
Module 3: Embeddings

Embeddings are the magic behind semantic search in RAG. This module teaches you
how to convert text into numerical vectors that capture meaning.

What are Embeddings?
- Numerical representations of text (vectors)
- Similar meanings → similar vectors
- Enable semantic search (not just keyword matching)
- Foundation of modern NLP

Key Concepts:
- Vector representations
- Semantic similarity
- Embedding models (local vs API)
- Dimensionality and its impact
- Batch processing for efficiency
"""

import numpy as np
from typing import List, Union, Optional
import json
from dataclasses import dataclass, asdict


@dataclass
class Embedding:
    """
    Represents an embedding vector with metadata.

    Attributes:
        vector: The numerical representation (numpy array)
        text: Original text that was embedded
        model: Name of the model used
        metadata: Additional information
    """
    vector: np.ndarray
    text: str
    model: str
    metadata: Optional[dict] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            "vector": self.vector.tolist(),
            "text": self.text,
            "model": self.model,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: dict):
        """Create from dictionary"""
        return cls(
            vector=np.array(data["vector"]),
            text=data["text"],
            model=data["model"],
            metadata=data.get("metadata", {})
        )

    @property
    def dimension(self) -> int:
        """Get the dimension of the embedding vector"""
        return len(self.vector)


class EmbeddingModel:
    """
    Base class for embedding models.

    This abstracts the embedding process so you can swap models easily.
    """

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text"""
        raise NotImplementedError

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed multiple texts (more efficient)"""
        return [self.embed(text) for text in texts]

    @property
    def dimension(self) -> int:
        """Get embedding dimension"""
        raise NotImplementedError

    @property
    def model_name(self) -> str:
        """Get model name"""
        raise NotImplementedError


class SentenceTransformerModel(EmbeddingModel):
    """
    Use Sentence Transformers for local embeddings.

    Advantages:
    - Free and runs locally
    - No API calls needed
    - Fast for moderate volumes
    - Many pre-trained models available

    Popular models:
    - 'all-MiniLM-L6-v2': Fast, 384 dimensions, good quality
    - 'all-mpnet-base-v2': Slower, 768 dimensions, best quality
    - 'paraphrase-MiniLM-L6-v2': Good for similarity tasks

    Install: pip install sentence-transformers
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )

        self._model_name = model_name
        self.model = SentenceTransformer(model_name)
        print(f"✓ Loaded model: {model_name}")
        print(f"  Dimension: {self.dimension}")

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text"""
        return self.model.encode(text, convert_to_numpy=True)

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """
        Embed multiple texts efficiently.

        Batch processing is much faster than embedding one-by-one.
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        return [emb for emb in embeddings]

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    @property
    def model_name(self) -> str:
        return self._model_name


class SimpleEmbeddingModel(EmbeddingModel):
    """
    A simple embedding model for learning purposes.

    This creates embeddings using basic word frequency (TF-IDF style).
    NOT suitable for production - use for understanding concepts only!

    How it works:
    1. Build vocabulary from all texts
    2. Each word gets a position in the vector
    3. Vector values = word frequency in the text
    """

    def __init__(self, vocabulary_size: int = 100):
        self.vocabulary_size = vocabulary_size
        self.vocabulary = {}  # word -> index
        self._dimension = vocabulary_size
        print(f"✓ SimpleEmbeddingModel initialized (dimension={vocabulary_size})")
        print("  ⚠️  For learning only - use SentenceTransformer in production!")

    def fit(self, texts: List[str]):
        """
        Build vocabulary from texts.

        In real RAG systems, you'd use pre-trained models instead.
        """
        # Count word frequencies
        word_counts = {}
        for text in texts:
            words = text.lower().split()
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1

        # Select top N words
        top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        top_words = top_words[:self.vocabulary_size]

        # Create vocabulary
        self.vocabulary = {word: idx for idx, (word, _) in enumerate(top_words)}
        print(f"✓ Vocabulary built: {len(self.vocabulary)} words")

    def embed(self, text: str) -> np.ndarray:
        """
        Create a simple embedding based on word frequencies.
        """
        if not self.vocabulary:
            raise ValueError("Model not fitted. Call fit() first.")

        vector = np.zeros(self.vocabulary_size)
        words = text.lower().split()

        for word in words:
            if word in self.vocabulary:
                idx = self.vocabulary[word]
                vector[idx] += 1

        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return "SimpleEmbedding"


class EmbeddingStore:
    """
    Store and manage embeddings.

    This handles:
    - Creating embeddings from text chunks
    - Saving/loading embeddings
    - Batch processing
    """

    def __init__(self, model: EmbeddingModel):
        self.model = model
        self.embeddings: List[Embedding] = []

    def embed_texts(
        self,
        texts: List[str],
        metadata_list: Optional[List[dict]] = None
    ) -> List[Embedding]:
        """
        Embed a list of texts and store them.

        Args:
            texts: List of text strings to embed
            metadata_list: Optional list of metadata dicts (one per text)

        Returns:
            List of Embedding objects
        """
        if metadata_list is None:
            metadata_list = [{}] * len(texts)

        print(f"\n🔄 Embedding {len(texts)} texts...")

        # Batch embed for efficiency
        vectors = self.model.embed_batch(texts)

        # Create Embedding objects
        embeddings = []
        for text, vector, metadata in zip(texts, vectors, metadata_list):
            embedding = Embedding(
                vector=vector,
                text=text,
                model=self.model.model_name,
                metadata=metadata
            )
            embeddings.append(embedding)

        self.embeddings.extend(embeddings)
        print(f"✓ Created {len(embeddings)} embeddings (dimension={self.model.dimension})")

        return embeddings

    def save(self, filepath: str):
        """Save embeddings to file"""
        data = [emb.to_dict() for emb in self.embeddings]

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✓ Saved {len(self.embeddings)} embeddings to {filepath}")

    def load(self, filepath: str):
        """Load embeddings from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.embeddings = [Embedding.from_dict(item) for item in data]
        print(f"✓ Loaded {len(self.embeddings)} embeddings from {filepath}")

    def clear(self):
        """Clear all stored embeddings"""
        self.embeddings = []


# ============================================================================
# SIMILARITY FUNCTIONS
# ============================================================================

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.

    Cosine similarity measures the angle between vectors:
    - 1.0 = identical direction (very similar)
    - 0.0 = perpendicular (unrelated)
    - -1.0 = opposite direction (very different)

    Formula: (A · B) / (||A|| * ||B||)
    """
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two vectors.

    Euclidean distance measures straight-line distance:
    - 0.0 = identical vectors
    - Higher = more different

    Note: For normalized vectors, cosine similarity often works better.
    """
    return np.linalg.norm(vec1 - vec2)


# ============================================================================
# EXAMPLES AND EXERCISES
# ============================================================================

def example_basic_embeddings():
    """Example: Create and compare embeddings"""
    print("=" * 80)
    print("Example 1: Basic Embeddings")
    print("=" * 80)

    # Create simple model for learning
    model = SimpleEmbeddingModel(vocabulary_size=20)

    # Sample texts
    texts = [
        "machine learning is a subset of AI",
        "deep learning uses neural networks",
        "AI enables computers to learn",
        "the weather is nice today"
    ]

    # Fit vocabulary
    model.fit(texts)

    # Create embeddings
    store = EmbeddingStore(model)
    embeddings = store.embed_texts(texts)

    # Show embedding details
    print(f"\nEmbedding dimension: {embeddings[0].dimension}")
    print(f"\nFirst embedding (truncated):")
    print(f"  Text: {embeddings[0].text}")
    print(f"  Vector: {embeddings[0].vector[:10]}...")


def example_semantic_similarity():
    """Example: Demonstrate semantic similarity"""
    print("\n" + "=" * 80)
    print("Example 2: Semantic Similarity")
    print("=" * 80)

    # Use simple model for illustration
    model = SimpleEmbeddingModel(vocabulary_size=50)

    texts = [
        "machine learning algorithms",
        "AI and machine learning",
        "deep neural networks",
        "pizza and pasta"
    ]

    model.fit(texts)
    store = EmbeddingStore(model)
    embeddings = store.embed_texts(texts)

    # Compare similarities
    print("\nSimilarity Matrix:")
    print("-" * 80)
    print(f"{'':30} ", end="")
    for i, text in enumerate(texts):
        print(f"Text{i+1:>6}", end="")
    print()

    for i, emb1 in enumerate(embeddings):
        print(f"{texts[i][:30]:30} ", end="")
        for j, emb2 in enumerate(embeddings):
            sim = cosine_similarity(emb1.vector, emb2.vector)
            print(f"{sim:6.2f}", end="")
        print()

    print("\n💡 Notice:")
    print("  - ML/AI texts have HIGH similarity to each other")
    print("  - Pizza text has LOW similarity to ML texts")


def example_sentence_transformers():
    """Example: Use real Sentence Transformers"""
    print("\n" + "=" * 80)
    print("Example 3: Sentence Transformers (Production-Ready)")
    print("=" * 80)

    try:
        # Use a real, pre-trained model
        model = SentenceTransformerModel('all-MiniLM-L6-v2')

        texts = [
            "The cat sat on the mat",
            "A feline rested on the rug",
            "Dogs are loyal animals",
            "Python is a programming language"
        ]

        store = EmbeddingStore(model)
        embeddings = store.embed_texts(texts)

        # Compare similarities
        print("\nSemantic Similarity with Real Embeddings:")
        print("-" * 80)

        query = embeddings[0]  # "The cat sat on the mat"
        print(f"Query: {query.text}\n")

        for emb in embeddings[1:]:
            sim = cosine_similarity(query.vector, emb.vector)
            print(f"Similarity to '{emb.text}': {sim:.4f}")

        print("\n💡 Notice:")
        print("  - 'feline on rug' is VERY similar (paraphrase)")
        print("  - 'dogs' is moderately similar (related topic)")
        print("  - 'Python programming' is less similar (different topic)")

    except ImportError:
        print("\n⚠️  sentence-transformers not installed")
        print("Install with: pip install sentence-transformers")


def exercise_1():
    """
    EXERCISE 1: Embedding Comparison

    Task: Compare embeddings of synonyms vs unrelated words
    """
    print("\n" + "=" * 80)
    print("EXERCISE 1: Compare Synonyms")
    print("=" * 80)

    # TODO: Create embeddings for:
    # - "happy" and "joyful" (synonyms)
    # - "happy" and "computer" (unrelated)
    # Compare their similarities

    print("\nYour code here!")


def exercise_2():
    """
    EXERCISE 2: Find Most Similar

    Task: Given a query, find the most similar text from a list
    """
    print("\n" + "=" * 80)
    print("EXERCISE 2: Find Most Similar Text")
    print("=" * 80)

    # TODO: Embed a list of texts
    # Embed a query
    # Find and print the most similar text

    print("\nYour code here!")


if __name__ == "__main__":
    print("\n🚀 RAG FROM SCRATCH - MODULE 3: EMBEDDINGS\n")

    # Run examples
    example_basic_embeddings()
    example_semantic_similarity()
    example_sentence_transformers()

    # Exercises
    exercise_1()
    exercise_2()

    print("\n" + "=" * 80)
    print("Key Takeaways:")
    print("=" * 80)
    print("✓ Embeddings convert text to numerical vectors")
    print("✓ Similar meanings → similar vectors (semantic similarity)")
    print("✓ Cosine similarity measures how similar two embeddings are")
    print("✓ Pre-trained models (like Sentence Transformers) work best")
    print("✓ Batch processing is much faster than one-by-one")
    print("✓ Dimension size affects quality and speed (384-1536 typical)")
    print("✓ Next: We'll learn how to store and search these embeddings efficiently")
    print("=" * 80)
