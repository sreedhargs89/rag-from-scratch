"""
Module 6: Complete RAG Pipeline

This module brings everything together into a working RAG system.
You've learned all the components - now see how they work together!

The Complete Flow:
1. Index Phase (offline):
   - Load documents
   - Chunk text
   - Generate embeddings
   - Store in vector database

2. Query Phase (online):
   - User asks a question
   - Embed the query
   - Retrieve relevant chunks
   - Build context
   - Generate response (with LLM)

This module demonstrates both phases and provides a ready-to-use RAG system.
"""

import os
from typing import List, Optional, Dict
from pathlib import Path

# Import from previous modules
try:
    # Try relative imports (when run as module)
    from .document_loader import DocumentLoader
    from .chunking import RecursiveChunker, Chunk
    from .embeddings import SentenceTransformerModel, SimpleEmbeddingModel
    from .vector_store import SimpleVectorStore, FAISSVectorStore
    from .retrieval import Retriever
except ImportError:
    # Fall back to direct imports (when run as script)
    from document_loader import DocumentLoader
    from chunking import RecursiveChunker, Chunk
    from embeddings import SentenceTransformerModel, SimpleEmbeddingModel
    from vector_store import SimpleVectorStore, FAISSVectorStore
    from retrieval import Retriever


class RAGPipeline:
    """
    Complete RAG system that integrates all components.

    This is what you'd use in production (though you'd add error handling,
    logging, monitoring, etc.)
    """

    def __init__(
        self,
        embedding_model_name: str = "simple",  # or "sentence-transformer"
        vector_store_type: str = "simple",      # or "faiss"
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        top_k: int = 3
    ):
        """
        Initialize the RAG pipeline.

        Args:
            embedding_model_name: Which embedding model to use
            vector_store_type: Which vector store to use
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            top_k: Number of chunks to retrieve
        """
        print("🚀 Initializing RAG Pipeline...")
        print("=" * 80)

        # 1. Document Loader
        self.document_loader = DocumentLoader()
        print("✓ Document loader ready")

        # 2. Chunker
        self.chunker = RecursiveChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        print(f"✓ Chunker configured (size={chunk_size}, overlap={chunk_overlap})")

        # 3. Embedding Model
        if embedding_model_name == "simple":
            self.embedding_model = SimpleEmbeddingModel(vocabulary_size=100)
            print("✓ Using SimpleEmbeddingModel (for learning)")
        elif embedding_model_name == "sentence-transformer":
            self.embedding_model = SentenceTransformerModel('all-MiniLM-L6-v2')
            print("✓ Using SentenceTransformer (production-ready)")
        else:
            raise ValueError(f"Unknown embedding model: {embedding_model_name}")

        # 4. Vector Store
        if vector_store_type == "simple":
            self.vector_store = SimpleVectorStore()
            print("✓ Using SimpleVectorStore (in-memory)")
        elif vector_store_type == "faiss":
            dimension = self.embedding_model.dimension
            self.vector_store = FAISSVectorStore(dimension)
            print("✓ Using FAISSVectorStore (fast)")
        else:
            raise ValueError(f"Unknown vector store: {vector_store_type}")

        # 5. Retriever
        self.retriever = Retriever(
            vector_store=self.vector_store,
            embedding_model=self.embedding_model,
            top_k=top_k
        )
        print(f"✓ Retriever configured (top_k={top_k})")

        # Track indexed documents
        self.indexed_documents = []
        self.all_chunks = []

        print("=" * 80)
        print("✅ RAG Pipeline ready!\n")

    def index_documents(
        self,
        source: str,
        recursive: bool = True
    ) -> int:
        """
        Index documents from a file or directory.

        This is the "offline" phase - you do this once to build your knowledge base.

        Args:
            source: Path to file or directory
            recursive: Search subdirectories

        Returns:
            Number of chunks indexed
        """
        print(f"\n📚 INDEXING PHASE")
        print("=" * 80)

        # Step 1: Load documents
        print(f"\n1. Loading documents from: {source}")
        path = Path(source)

        if path.is_file():
            documents = [self.document_loader.load_file(source)]
        elif path.is_dir():
            documents = self.document_loader.load_directory(source, recursive=recursive)
        else:
            raise ValueError(f"Invalid source: {source}")

        if not documents:
            print("⚠️  No documents found!")
            return 0

        print(f"✓ Loaded {len(documents)} documents")

        # Step 2: Chunk documents
        print(f"\n2. Chunking documents...")
        all_chunks = []

        for doc in documents:
            chunks = self.chunker.chunk(doc.content, metadata=doc.metadata)
            all_chunks.extend(chunks)

        print(f"✓ Created {len(all_chunks)} chunks")

        # Step 3: Generate embeddings
        print(f"\n3. Generating embeddings...")

        chunk_texts = [chunk.content for chunk in all_chunks]
        chunk_metadata = [chunk.metadata for chunk in all_chunks]

        # Special handling for SimpleEmbeddingModel
        if isinstance(self.embedding_model, SimpleEmbeddingModel):
            if not self.embedding_model.vocabulary:
                print("  Building vocabulary from chunks...")
                self.embedding_model.fit(chunk_texts)

        vectors = self.embedding_model.embed_batch(chunk_texts)
        print(f"✓ Generated {len(vectors)} embeddings (dim={len(vectors[0])})")

        # Step 4: Store in vector database
        print(f"\n4. Storing in vector database...")
        self.vector_store.add(vectors, chunk_texts, chunk_metadata)
        print(f"✓ Indexed {len(vectors)} chunks")

        # Track what we've indexed
        self.indexed_documents.extend(documents)
        self.all_chunks.extend(all_chunks)

        print("\n" + "=" * 80)
        print(f"✅ INDEXING COMPLETE: {len(all_chunks)} chunks indexed")
        print("=" * 80)

        return len(all_chunks)

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        return_context: bool = False
    ) -> str:
        """
        Query the RAG system.

        This is the "online" phase - happens every time a user asks a question.

        Args:
            question: User's question
            top_k: Override default top_k
            return_context: If True, return (answer, context, sources)

        Returns:
            Generated answer (or tuple if return_context=True)
        """
        print(f"\n❓ QUERY PHASE")
        print("=" * 80)
        print(f"Question: {question}\n")

        # Step 1: Retrieve relevant chunks
        print("1. Retrieving relevant chunks...")
        results, context = self.retriever.retrieve_with_context(
            question,
            top_k=top_k
        )

        if not results:
            return "I couldn't find any relevant information to answer this question."

        print(f"✓ Retrieved {len(results)} relevant chunks")

        # Step 2: Build sources list
        sources = list(set(r.source for r in results))
        print(f"✓ From {len(sources)} sources")

        # Step 3: Generate response
        print("\n2. Generating response...")

        # Note: In production, you'd call an LLM API here (OpenAI, Anthropic, etc.)
        # For this learning example, we'll create a simple response
        answer = self._generate_simple_response(question, results)

        print("✓ Response generated")
        print("=" * 80)

        if return_context:
            return answer, context, sources
        else:
            return answer

    def _generate_simple_response(
        self,
        question: str,
        results: List
    ) -> str:
        """
        Generate a simple response without calling an LLM.

        In production, replace this with actual LLM call:

        ```python
        import anthropic
        client = anthropic.Anthropic(api_key="your-key")

        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": f"Context: {context}\\n\\nQuestion: {question}\\n\\nAnswer:"
            }]
        )
        return message.content[0].text
        ```
        """
        # For demo: just return the most relevant chunk
        most_relevant = results[0]

        response = f"""Based on the retrieved information:

{most_relevant.text}

(Note: This is a simplified demo. In production, this would be a proper LLM-generated response using the retrieved context.)

Source: {most_relevant.source}
Relevance Score: {most_relevant.score:.4f}
"""
        return response

    def save(self, directory: str):
        """Save the RAG pipeline state"""
        os.makedirs(directory, exist_ok=True)

        # Save vector store
        vector_store_path = os.path.join(directory, "vector_store")
        self.vector_store.save(vector_store_path)

        print(f"✓ Saved RAG pipeline to {directory}")

    def load(self, directory: str):
        """Load the RAG pipeline state"""
        vector_store_path = os.path.join(directory, "vector_store")
        self.vector_store.load(vector_store_path)

        print(f"✓ Loaded RAG pipeline from {directory}")

    def get_stats(self) -> Dict:
        """Get statistics about the indexed knowledge base"""
        return {
            "num_documents": len(self.indexed_documents),
            "num_chunks": len(self.all_chunks),
            "num_vectors": len(self.vector_store),
            "embedding_dimension": self.embedding_model.dimension,
            "model": self.embedding_model.model_name
        }


# ============================================================================
# EXAMPLES
# ============================================================================

def example_end_to_end():
    """Example: Complete RAG pipeline from scratch"""
    print("\n" + "=" * 80)
    print("EXAMPLE: End-to-End RAG Pipeline")
    print("=" * 80)

    # Create sample documents
    sample_dir = "../data/sample_docs/"
    os.makedirs(sample_dir, exist_ok=True)

    # Sample documents about AI
    docs = {
        "ml_basics.txt": """
Machine Learning is a subset of artificial intelligence that enables computers
to learn from data without being explicitly programmed. It uses algorithms to
identify patterns and make decisions.

There are three main types of machine learning:
1. Supervised learning - learns from labeled data
2. Unsupervised learning - finds patterns in unlabeled data
3. Reinforcement learning - learns through trial and error

Common applications include image recognition, natural language processing,
recommendation systems, and autonomous vehicles.
        """,

        "deep_learning.txt": """
Deep Learning is a specialized branch of machine learning that uses artificial
neural networks with multiple layers. These networks are inspired by the human
brain's structure.

Deep learning excels at:
- Image and video recognition
- Speech recognition and generation
- Natural language understanding
- Game playing (like AlphaGo)

Popular frameworks include TensorFlow, PyTorch, and Keras. Training deep
learning models requires large amounts of data and computational power,
often using GPUs.
        """,

        "python_ml.txt": """
Python is the most popular programming language for machine learning. It offers
powerful libraries and frameworks that make ML development easier.

Key Python libraries for ML:
- NumPy: Numerical computing
- Pandas: Data manipulation
- Scikit-learn: Traditional ML algorithms
- TensorFlow and PyTorch: Deep learning
- Matplotlib: Data visualization

Python's simple syntax and extensive ecosystem make it ideal for both beginners
and experts in machine learning.
        """
    }

    for filename, content in docs.items():
        with open(os.path.join(sample_dir, filename), 'w') as f:
            f.write(content.strip())

    print("✓ Created sample documents\n")

    # Initialize RAG pipeline
    rag = RAGPipeline(
        embedding_model_name="simple",
        vector_store_type="simple",
        chunk_size=300,
        chunk_overlap=50,
        top_k=2
    )

    # Index documents
    num_chunks = rag.index_documents(sample_dir)

    # Show stats
    stats = rag.get_stats()
    print(f"\n📊 Knowledge Base Stats:")
    print(f"  Documents: {stats['num_documents']}")
    print(f"  Chunks: {stats['num_chunks']}")
    print(f"  Embedding dimension: {stats['embedding_dimension']}")

    # Ask questions
    questions = [
        "What is machine learning?",
        "What libraries are used for ML in Python?",
        "What is deep learning good for?"
    ]

    print(f"\n\n💬 ASKING QUESTIONS")
    print("=" * 80)

    for i, question in enumerate(questions, 1):
        print(f"\n\nQuestion {i}: {question}")
        print("-" * 80)
        answer = rag.query(question)
        print(f"\nAnswer:\n{answer}")


def example_production_setup():
    """Example: Production-ready RAG with Sentence Transformers"""
    print("\n\n" + "=" * 80)
    print("EXAMPLE: Production-Ready RAG Setup")
    print("=" * 80)

    try:
        # Use real embedding model
        rag = RAGPipeline(
            embedding_model_name="sentence-transformer",
            vector_store_type="simple",  # Would use FAISS in real production
            chunk_size=500,
            chunk_overlap=100,
            top_k=3
        )

        print("\n💡 This setup uses:")
        print("  - SentenceTransformer for high-quality embeddings")
        print("  - Configurable chunk size and overlap")
        print("  - Top-k retrieval for relevant context")
        print("\nReady for production with your own documents!")

    except ImportError:
        print("\n⚠️  Install sentence-transformers for production setup")
        print("   pip install sentence-transformers")


if __name__ == "__main__":
    print("\n🚀 RAG FROM SCRATCH - MODULE 6: COMPLETE PIPELINE\n")

    # Run examples
    example_end_to_end()
    example_production_setup()

    print("\n\n" + "=" * 80)
    print("🎉 CONGRATULATIONS!")
    print("=" * 80)
    print("You've learned how to build a RAG system from scratch!")
    print()
    print("What you've learned:")
    print("  ✓ Document loading and parsing")
    print("  ✓ Text chunking strategies")
    print("  ✓ Embeddings and semantic similarity")
    print("  ✓ Vector storage and search")
    print("  ✓ Retrieval and context building")
    print("  ✓ Complete RAG pipeline integration")
    print()
    print("Next steps:")
    print("  1. Try with your own documents")
    print("  2. Experiment with different chunking strategies")
    print("  3. Use production embedding models (Sentence Transformers)")
    print("  4. Integrate with an LLM (OpenAI, Anthropic, etc.)")
    print("  5. Add evaluation metrics")
    print("  6. Deploy as an API or web app")
    print()
    print("Resources:")
    print("  - examples/basic_rag.py - Simple usage examples")
    print("  - docs/ - Advanced techniques and best practices")
    print("=" * 80)
