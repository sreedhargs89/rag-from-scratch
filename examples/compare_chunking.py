"""
Compare Chunking Strategies

This example shows how different chunking strategies affect retrieval quality.
Understanding chunking is crucial for optimal RAG performance.
"""

import sys
sys.path.append('..')

from src.02_chunking import FixedSizeChunker, SentenceChunker, RecursiveChunker
import os


def main():
    print("=" * 80)
    print("CHUNKING STRATEGY COMPARISON")
    print("=" * 80)

    # Sample text
    text = """
Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and humans through natural language. The ultimate goal of NLP is to enable computers to understand, interpret, and generate human language in a valuable way.

NLP combines computational linguistics with machine learning and deep learning models. These technologies enable computers to process human language in the form of text or voice data and understand its full meaning, including the speaker's intent and sentiment.

Key applications of NLP include machine translation, sentiment analysis, chatbots, and text summarization. Companies use NLP to analyze customer feedback, automate customer service, and extract insights from large amounts of unstructured text data.

Recent advances in transformer models like BERT and GPT have significantly improved NLP capabilities. These models can understand context and nuance in ways that were not possible with earlier approaches. They have achieved human-level performance on many NLP tasks.
    """

    print(f"\nOriginal text length: {len(text)} characters")
    print(f"Word count: {len(text.split())} words\n")

    # Strategy 1: Fixed Size
    print("\n" + "=" * 80)
    print("Strategy 1: Fixed Size Chunking")
    print("=" * 80)
    print("Pros: Simple, predictable chunk sizes")
    print("Cons: May break sentences awkwardly")

    fixed_chunker = FixedSizeChunker(chunk_size=200, chunk_overlap=30)
    fixed_chunks = fixed_chunker.chunk(text)

    print(f"\nResult: {len(fixed_chunks)} chunks")
    for i, chunk in enumerate(fixed_chunks, 1):
        print(f"\nChunk {i} ({len(chunk.content)} chars):")
        print(f"  {chunk.content[:100]}...")

    # Strategy 2: Sentence-based
    print("\n\n" + "=" * 80)
    print("Strategy 2: Sentence-Based Chunking")
    print("=" * 80)
    print("Pros: Preserves sentence boundaries, better semantic coherence")
    print("Cons: Variable chunk sizes")

    sentence_chunker = SentenceChunker(max_sentences=3, overlap_sentences=1)
    sentence_chunks = sentence_chunker.chunk(text)

    print(f"\nResult: {len(sentence_chunks)} chunks")
    for i, chunk in enumerate(sentence_chunks, 1):
        print(f"\nChunk {i} ({len(chunk.content)} chars, sentences: {chunk.metadata['sentence_end'] - chunk.metadata['sentence_start']}):")
        print(f"  {chunk.content[:100]}...")

    # Strategy 3: Recursive
    print("\n\n" + "=" * 80)
    print("Strategy 3: Recursive Chunking")
    print("=" * 80)
    print("Pros: Most flexible, tries to split on natural boundaries (paragraphs → sentences → words)")
    print("Cons: More complex implementation")

    recursive_chunker = RecursiveChunker(chunk_size=250, chunk_overlap=30)
    recursive_chunks = recursive_chunker.chunk(text)

    print(f"\nResult: {len(recursive_chunks)} chunks")
    for i, chunk in enumerate(recursive_chunks, 1):
        print(f"\nChunk {i} ({len(chunk.content)} chars):")
        print(f"  {chunk.content[:100]}...")

    # Comparison Summary
    print("\n\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    print(f"\n{'Strategy':<20} {'Chunks':<10} {'Avg Size':<12} {'Best For':<30}")
    print("-" * 80)

    avg_fixed = sum(len(c.content) for c in fixed_chunks) / len(fixed_chunks)
    avg_sentence = sum(len(c.content) for c in sentence_chunks) / len(sentence_chunks)
    avg_recursive = sum(len(c.content) for c in recursive_chunks) / len(recursive_chunks)

    print(f"{'Fixed Size':<20} {len(fixed_chunks):<10} {avg_fixed:<12.0f} {'Quick, simple projects':<30}")
    print(f"{'Sentence-based':<20} {len(sentence_chunks):<10} {avg_sentence:<12.0f} {'Q&A, conversations':<30}")
    print(f"{'Recursive':<20} {len(recursive_chunks):<10} {avg_recursive:<12.0f} {'General purpose, best quality':<30}")

    print("\n\n💡 Key Insights:")
    print("  1. Smaller chunks = more precise retrieval, but may lose context")
    print("  2. Larger chunks = more context, but less precise")
    print("  3. Overlap helps preserve context between chunks")
    print("  4. Choose strategy based on your content type and use case")
    print("  5. For most cases, recursive chunking (250-500 chars) works well")


if __name__ == "__main__":
    main()
