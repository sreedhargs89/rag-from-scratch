"""
Module 2: Text Chunking

Chunking is one of the most critical steps in RAG. It determines how well your system
can retrieve relevant information. This module teaches various chunking strategies
and when to use each one.

Why Chunking Matters:
- Embeddings work best on focused, coherent text
- LLMs have context length limits
- Smaller chunks = more precise retrieval
- Larger chunks = more context but less precision

Key Concepts:
- Fixed-size chunking
- Sentence-based chunking
- Recursive chunking with fallback delimiters
- Chunk overlap for context preservation
- Token counting vs character counting
"""

from typing import List, Optional, Callable
from dataclasses import dataclass
import re


@dataclass
class Chunk:
    """
    Represents a text chunk with metadata.

    Attributes:
        content: The chunked text
        metadata: Information about the chunk (source, position, etc.)
        chunk_id: Unique identifier for the chunk
    """
    content: str
    metadata: dict
    chunk_id: Optional[str] = None

    def __repr__(self):
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Chunk(id={self.chunk_id}, content='{preview}')"


class TextChunker:
    """
    Base class for text chunking strategies.

    This demonstrates the Strategy pattern - different ways to solve the same problem.
    """

    def chunk(self, text: str, metadata: Optional[dict] = None) -> List[Chunk]:
        """
        Split text into chunks.

        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to chunks

        Returns:
            List of Chunk objects
        """
        raise NotImplementedError("Subclasses must implement chunk()")


class FixedSizeChunker(TextChunker):
    """
    Split text into fixed-size chunks.

    This is the simplest strategy:
    - Easy to implement
    - Predictable chunk sizes
    - May break sentences/paragraphs awkwardly

    Use when: You need simple, fast chunking and don't care about semantic boundaries.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Characters to overlap between chunks (preserves context)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, metadata: Optional[dict] = None) -> List[Chunk]:
        metadata = metadata or {}
        chunks = []

        # Calculate step size (how much we advance for each chunk)
        step = self.chunk_size - self.chunk_overlap

        for i in range(0, len(text), step):
            chunk_text = text[i:i + self.chunk_size]

            if chunk_text.strip():  # Skip empty chunks
                chunk_metadata = {
                    **metadata,
                    "chunk_method": "fixed_size",
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "start_char": i,
                    "end_char": i + len(chunk_text)
                }

                chunks.append(Chunk(
                    content=chunk_text.strip(),
                    metadata=chunk_metadata,
                    chunk_id=f"chunk_{i}"
                ))

        return chunks


class SentenceChunker(TextChunker):
    """
    Split text into chunks based on sentence boundaries.

    This is better than fixed-size because:
    - Preserves sentence integrity
    - More natural semantic units
    - Better for comprehension

    Use when: You want semantic coherence and your text has clear sentence structure.
    """

    def __init__(self, max_sentences: int = 5, overlap_sentences: int = 1):
        """
        Args:
            max_sentences: Maximum sentences per chunk
            overlap_sentences: Sentences to overlap between chunks
        """
        self.max_sentences = max_sentences
        self.overlap_sentences = overlap_sentences

    def chunk(self, text: str, metadata: Optional[dict] = None) -> List[Chunk]:
        metadata = metadata or {}

        # Split into sentences (simple regex, can be improved)
        sentences = self._split_sentences(text)

        chunks = []
        i = 0

        while i < len(sentences):
            # Take max_sentences
            chunk_sentences = sentences[i:i + self.max_sentences]
            chunk_text = " ".join(chunk_sentences)

            if chunk_text.strip():
                chunk_metadata = {
                    **metadata,
                    "chunk_method": "sentence",
                    "max_sentences": self.max_sentences,
                    "sentence_start": i,
                    "sentence_end": i + len(chunk_sentences)
                }

                chunks.append(Chunk(
                    content=chunk_text.strip(),
                    metadata=chunk_metadata,
                    chunk_id=f"chunk_sent_{i}"
                ))

            # Move forward, accounting for overlap
            i += (self.max_sentences - self.overlap_sentences)

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """
        Simple sentence splitting.

        A more robust approach would use libraries like spaCy or NLTK.
        """
        # Split on . ! ? followed by space or end of string
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


class RecursiveChunker(TextChunker):
    """
    Split text using a hierarchy of delimiters.

    This is the most sophisticated approach:
    - Tries to split on paragraphs first
    - Falls back to sentences
    - Falls back to fixed size if needed

    Use when: You want maximum semantic coherence and have varying text structures.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        separators: Optional[List[str]] = None
    ):
        """
        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Characters to overlap
            separators: List of separators to try (in order of preference)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Default separators: try paragraphs, then sentences, then arbitrary
        self.separators = separators or [
            "\n\n",  # Paragraphs
            "\n",    # Lines
            ". ",    # Sentences
            "! ",    # Sentences
            "? ",    # Sentences
            ", ",    # Clauses
            " ",     # Words
            ""       # Characters
        ]

    def chunk(self, text: str, metadata: Optional[dict] = None) -> List[Chunk]:
        metadata = metadata or {}
        return self._chunk_recursive(text, metadata, 0)

    def _chunk_recursive(
        self,
        text: str,
        metadata: dict,
        separator_index: int
    ) -> List[Chunk]:
        """
        Recursively chunk text using progressively finer separators.
        """
        if not text.strip():
            return []

        # If text is small enough, return as single chunk
        if len(text) <= self.chunk_size:
            return [Chunk(
                content=text.strip(),
                metadata={**metadata, "chunk_method": "recursive"},
                chunk_id=f"chunk_{hash(text)}"
            )]

        # Try current separator
        if separator_index < len(self.separators):
            separator = self.separators[separator_index]
            splits = text.split(separator) if separator else list(text)

            # Merge splits into chunks
            return self._merge_splits(
                splits,
                separator,
                metadata,
                separator_index
            )
        else:
            # No more separators, force split
            return self._force_split(text, metadata)

    def _merge_splits(
        self,
        splits: List[str],
        separator: str,
        metadata: dict,
        separator_index: int
    ) -> List[Chunk]:
        """
        Merge splits into chunks, respecting size limits.
        """
        chunks = []
        current_chunk = []
        current_size = 0

        for split in splits:
            split_size = len(split) + len(separator)

            # If adding this split would exceed chunk_size
            if current_size + split_size > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = separator.join(current_chunk)
                chunks.append(Chunk(
                    content=chunk_text.strip(),
                    metadata={**metadata, "chunk_method": "recursive"},
                    chunk_id=f"chunk_{len(chunks)}"
                ))

                # Start new chunk with overlap
                current_chunk = current_chunk[-1:] if self.chunk_overlap > 0 else []
                current_size = len(current_chunk[0]) if current_chunk else 0

            current_chunk.append(split)
            current_size += split_size

        # Add remaining chunk
        if current_chunk:
            chunk_text = separator.join(current_chunk)
            chunks.append(Chunk(
                content=chunk_text.strip(),
                metadata={**metadata, "chunk_method": "recursive"},
                chunk_id=f"chunk_{len(chunks)}"
            ))

        return chunks

    def _force_split(self, text: str, metadata: dict) -> List[Chunk]:
        """
        Force split text when no good separators found.
        """
        chunks = []
        for i in range(0, len(text), self.chunk_size):
            chunk_text = text[i:i + self.chunk_size]
            chunks.append(Chunk(
                content=chunk_text.strip(),
                metadata={**metadata, "chunk_method": "recursive_forced"},
                chunk_id=f"chunk_{i}"
            ))
        return chunks


class SemanticChunker(TextChunker):
    """
    Split text based on semantic similarity (advanced).

    This uses embeddings to detect topic changes:
    - More intelligent than simple delimiters
    - Preserves topical coherence
    - More computationally expensive

    Note: Requires embeddings (covered in Module 3)
    """

    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold

    def chunk(self, text: str, metadata: Optional[dict] = None) -> List[Chunk]:
        # Placeholder - will implement after embeddings module
        raise NotImplementedError(
            "Semantic chunking requires embeddings (see Module 3)"
        )


# ============================================================================
# EXAMPLES AND EXERCISES
# ============================================================================

def example_compare_strategies():
    """Compare different chunking strategies"""
    print("=" * 80)
    print("Example: Comparing Chunking Strategies")
    print("=" * 80)

    sample_text = """
Machine learning is a subset of artificial intelligence. It enables computers to learn without explicit programming. The field has three main categories.

Supervised learning uses labeled data. The algorithm learns from examples with known outputs. Common applications include image classification and spam detection.

Unsupervised learning finds patterns in unlabeled data. Clustering and dimensionality reduction are key techniques. It's useful for customer segmentation and anomaly detection.

Reinforcement learning learns through trial and error. An agent interacts with an environment to maximize rewards. Applications include game playing and robotics.
"""

    print(f"\nOriginal text: {len(sample_text)} characters\n")

    # Strategy 1: Fixed Size
    print("\n1. FIXED SIZE CHUNKING (chunk_size=150, overlap=20)")
    print("-" * 80)
    fixed_chunker = FixedSizeChunker(chunk_size=150, chunk_overlap=20)
    fixed_chunks = fixed_chunker.chunk(sample_text)
    print(f"Number of chunks: {len(fixed_chunks)}")
    for i, chunk in enumerate(fixed_chunks[:3], 1):  # Show first 3
        print(f"\nChunk {i}: ({len(chunk.content)} chars)")
        print(f"  {chunk.content}")

    # Strategy 2: Sentence-based
    print("\n\n2. SENTENCE CHUNKING (max_sentences=3, overlap=1)")
    print("-" * 80)
    sentence_chunker = SentenceChunker(max_sentences=3, overlap_sentences=1)
    sentence_chunks = sentence_chunker.chunk(sample_text)
    print(f"Number of chunks: {len(sentence_chunks)}")
    for i, chunk in enumerate(sentence_chunks[:3], 1):
        print(f"\nChunk {i}: ({len(chunk.content)} chars)")
        print(f"  {chunk.content}")

    # Strategy 3: Recursive
    print("\n\n3. RECURSIVE CHUNKING (chunk_size=200, overlap=50)")
    print("-" * 80)
    recursive_chunker = RecursiveChunker(chunk_size=200, chunk_overlap=50)
    recursive_chunks = recursive_chunker.chunk(sample_text)
    print(f"Number of chunks: {len(recursive_chunks)}")
    for i, chunk in enumerate(recursive_chunks[:3], 1):
        print(f"\nChunk {i}: ({len(chunk.content)} chars)")
        print(f"  {chunk.content}")


def example_chunk_overlap():
    """Demonstrate the importance of chunk overlap"""
    print("\n\n" + "=" * 80)
    print("Example: Understanding Chunk Overlap")
    print("=" * 80)

    text = "The cat sat on the mat. The mat was very comfortable. The comfortable mat belonged to Jane."

    print(f"\nOriginal: {text}\n")

    # Without overlap
    print("WITHOUT OVERLAP:")
    print("-" * 80)
    no_overlap = FixedSizeChunker(chunk_size=30, chunk_overlap=0)
    chunks = no_overlap.chunk(text)
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}: {chunk.content}")

    # With overlap
    print("\n\nWITH OVERLAP (overlap=10):")
    print("-" * 80)
    with_overlap = FixedSizeChunker(chunk_size=30, chunk_overlap=10)
    chunks = with_overlap.chunk(text)
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}: {chunk.content}")

    print("\n💡 Notice how overlap preserves context between chunks!")


def exercise_1():
    """
    EXERCISE 1: Find Optimal Chunk Size

    Experiment with different chunk sizes and count how many chunks are created.
    """
    print("\n\n" + "=" * 80)
    print("EXERCISE 1: Find Optimal Chunk Size")
    print("=" * 80)

    text = "AI " * 500  # 1500 characters

    # TODO: Try chunk_size of 100, 200, 500, 1000
    # Count chunks for each size
    # Print results

    print("\nYour code here!")


def exercise_2():
    """
    EXERCISE 2: Custom Chunker

    Create a chunker that splits on markdown headers (# ## ###)
    """
    print("\n\n" + "=" * 80)
    print("EXERCISE 2: Create a Markdown Chunker")
    print("=" * 80)

    markdown_text = """# Introduction
This is the introduction section.

## Subsection A
Content for subsection A.

## Subsection B
Content for subsection B.

# Conclusion
Final thoughts here.
"""

    # TODO: Implement MarkdownChunker that splits on headers
    # Hint: Use regex to find headers

    print("\nYour code here!")


if __name__ == "__main__":
    print("\n🚀 RAG FROM SCRATCH - MODULE 2: TEXT CHUNKING\n")

    # Run examples
    example_compare_strategies()
    example_chunk_overlap()

    # Exercises
    exercise_1()
    exercise_2()

    print("\n\n" + "=" * 80)
    print("Key Takeaways:")
    print("=" * 80)
    print("✓ Chunking strategy significantly impacts retrieval quality")
    print("✓ Fixed-size is simple but may break semantic boundaries")
    print("✓ Sentence-based preserves meaning but varies in size")
    print("✓ Recursive is most flexible and sophisticated")
    print("✓ Overlap helps preserve context between chunks")
    print("✓ Optimal chunk size depends on your use case (typically 200-1000 chars)")
    print("✓ Next: We'll learn how to convert these chunks into embeddings")
    print("=" * 80)
