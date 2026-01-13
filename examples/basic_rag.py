"""
Basic RAG Example

This script demonstrates the simplest way to use the RAG system.
Perfect for getting started quickly!
"""

import sys
sys.path.append('..')

from src.rag_pipeline import RAGPipeline
import os


def main():
    print("=" * 80)
    print("BASIC RAG EXAMPLE")
    print("=" * 80)

    # Step 1: Create sample documents
    print("\n1. Creating sample documents...")

    docs_dir = "../data/sample_docs/"
    os.makedirs(docs_dir, exist_ok=True)

    # Create a few documents about different topics
    documents = {
        "ai.txt": """
Artificial Intelligence (AI) refers to computer systems that can perform tasks
that typically require human intelligence. These tasks include visual perception,
speech recognition, decision-making, and language translation.

AI can be categorized into:
- Narrow AI: Specialized in one task (like chess or image recognition)
- General AI: Can perform any intellectual task a human can (not yet achieved)

Modern AI is powered by machine learning, where systems learn from data rather
than being explicitly programmed for every scenario.
        """,

        "python.txt": """
Python is a high-level, interpreted programming language known for its clear
syntax and readability. Created by Guido van Rossum in 1991, it has become
one of the most popular programming languages in the world.

Python is widely used for:
- Web development (Django, Flask)
- Data science and machine learning
- Automation and scripting
- Scientific computing

Its extensive standard library and vast ecosystem of third-party packages
make it suitable for virtually any programming task.
        """,

        "climate.txt": """
Climate change refers to long-term shifts in global temperatures and weather
patterns. While climate has changed throughout Earth's history, the current
warming trend is primarily driven by human activities, particularly the
emission of greenhouse gases.

Key impacts include:
- Rising global temperatures
- Melting polar ice and rising sea levels
- More frequent extreme weather events
- Changes in precipitation patterns

Addressing climate change requires both mitigation (reducing emissions) and
adaptation (preparing for changes).
        """
    }

    for filename, content in documents.items():
        filepath = os.path.join(docs_dir, filename)
        with open(filepath, 'w') as f:
            f.write(content.strip())

    print(f"✓ Created {len(documents)} sample documents")

    # Step 2: Initialize RAG system
    print("\n2. Initializing RAG system...")

    rag = RAGPipeline(
        embedding_model_name="simple",  # Use "sentence-transformer" for production
        vector_store_type="simple",
        chunk_size=200,
        chunk_overlap=20,
        top_k=2
    )

    # Step 3: Index documents
    print("\n3. Indexing documents...")
    num_chunks = rag.index_documents(docs_dir)
    print(f"\n✅ Indexed {num_chunks} chunks")

    # Step 4: Ask questions
    print("\n" + "=" * 80)
    print("4. Asking Questions")
    print("=" * 80)

    questions = [
        "What is artificial intelligence?",
        "What is Python used for?",
        "What causes climate change?",
        "Tell me about machine learning"
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n\nQ{i}: {question}")
        print("-" * 80)

        answer, context, sources = rag.query(question, return_context=True)

        print(f"\nAnswer:\n{answer}")
        print(f"\nSources: {', '.join(sources)}")

    # Step 5: Show knowledge base stats
    print("\n\n" + "=" * 80)
    print("Knowledge Base Statistics")
    print("=" * 80)

    stats = rag.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n✅ Example complete!")


if __name__ == "__main__":
    main()
