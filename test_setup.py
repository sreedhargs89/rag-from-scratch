#!/usr/bin/env python3
"""
Test Setup Script

Quick verification that all components are working correctly.
Run this after installation to ensure everything is set up properly.
"""

import sys
import os

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")

    try:
        print("  ✓ NumPy", end="")
        import numpy
        print(f" ({numpy.__version__})")
    except ImportError:
        print("  ✗ NumPy - Install with: pip install numpy")
        return False

    try:
        print("  ✓ scikit-learn", end="")
        import sklearn
        print(f" ({sklearn.__version__})")
    except ImportError:
        print("  ✗ scikit-learn - Install with: pip install scikit-learn")
        return False

    # Optional dependencies
    try:
        print("  ✓ sentence-transformers", end="")
        import sentence_transformers
        print(f" ({sentence_transformers.__version__})")
    except ImportError:
        print("  ⚠ sentence-transformers (optional) - Install with: pip install sentence-transformers")

    try:
        print("  ✓ FAISS", end="")
        import faiss
        print()
    except ImportError:
        print("  ⚠ FAISS (optional) - Install with: pip install faiss-cpu")

    try:
        print("  ✓ pdfplumber", end="")
        import pdfplumber
        print()
    except ImportError:
        print("  ⚠ pdfplumber (optional) - Install with: pip install pdfplumber")

    return True


def test_modules():
    """Test that our modules work"""
    print("\nTesting RAG modules...")

    # Add src to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

    try:
        print("  ✓ Document Loader", end="")
        from src.document_loader import DocumentLoader
        loader = DocumentLoader()
        print()
    except Exception as e:
        print(f"  ✗ Document Loader - {e}")
        return False

    try:
        print("  ✓ Chunking", end="")
        from src.chunking import RecursiveChunker
        chunker = RecursiveChunker()
        print()
    except Exception as e:
        print(f"  ✗ Chunking - {e}")
        return False

    try:
        print("  ✓ Embeddings", end="")
        from src.embeddings import SimpleEmbeddingModel
        model = SimpleEmbeddingModel()
        print()
    except Exception as e:
        print(f"  ✗ Embeddings - {e}")
        return False

    try:
        print("  ✓ Vector Store", end="")
        from src.vector_store import SimpleVectorStore
        store = SimpleVectorStore()
        print()
    except Exception as e:
        print(f"  ✗ Vector Store - {e}")
        return False

    try:
        print("  ✓ Retrieval", end="")
        from src.retrieval import Retriever
        print()
    except Exception as e:
        print(f"  ✗ Retrieval - {e}")
        return False

    try:
        print("  ✓ RAG Pipeline", end="")
        from src.rag_pipeline import RAGPipeline
        print()
    except Exception as e:
        print(f"  ✗ RAG Pipeline - {e}")
        return False

    return True


def test_basic_functionality():
    """Test basic RAG functionality"""
    print("\nTesting basic functionality...")

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

    try:
        from src.document_loader import DocumentLoader, Document
        from src.chunking import RecursiveChunker
        from src.embeddings import SimpleEmbeddingModel, EmbeddingStore
        from src.vector_store import SimpleVectorStore
        from src.retrieval import Retriever

        # Test document
        print("  Creating test document...", end="")
        doc = Document(
            content="Machine learning is a subset of AI. Deep learning uses neural networks.",
            metadata={"source": "test"}
        )
        print(" ✓")

        # Test chunking
        print("  Testing chunking...", end="")
        chunker = RecursiveChunker(chunk_size=50)
        chunks = chunker.chunk(doc.content)
        assert len(chunks) > 0, "No chunks created"
        print(f" ✓ ({len(chunks)} chunks)")

        # Test embeddings
        print("  Testing embeddings...", end="")
        model = SimpleEmbeddingModel(vocabulary_size=20)
        texts = [chunk.content for chunk in chunks]
        model.fit(texts)
        store = EmbeddingStore(model)
        embeddings = store.embed_texts(texts)
        assert len(embeddings) == len(chunks), "Wrong number of embeddings"
        print(f" ✓ (dim={model.dimension})")

        # Test vector store
        print("  Testing vector store...", end="")
        vector_store = SimpleVectorStore()
        vectors = [emb.vector for emb in embeddings]
        vector_store.add(vectors, texts, [{}] * len(texts))
        assert len(vector_store) == len(chunks), "Wrong number of vectors stored"
        print(f" ✓ ({len(vector_store)} vectors)")

        # Test retrieval
        print("  Testing retrieval...", end="")
        retriever = Retriever(vector_store, model, top_k=2)
        results = retriever.retrieve("What is machine learning?")
        assert len(results) > 0, "No results retrieved"
        print(f" ✓ ({len(results)} results)")

        return True

    except Exception as e:
        print(f" ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 80)
    print("RAG FROM SCRATCH - Setup Test")
    print("=" * 80)

    success = True

    # Test imports
    if not test_imports():
        success = False
        print("\n❌ Import test failed. Install missing dependencies.")

    # Test modules
    if not test_modules():
        success = False
        print("\n❌ Module test failed. Check for errors above.")

    # Test functionality
    if not test_basic_functionality():
        success = False
        print("\n❌ Functionality test failed. Check for errors above.")

    # Summary
    print("\n" + "=" * 80)
    if success:
        print("✅ All tests passed! Your setup is ready.")
        print("\nNext steps:")
        print("  1. Read GETTING_STARTED.md for the learning guide")
        print("  2. Run: python src/01_document_loader.py")
        print("  3. Or try: python examples/basic_rag.py")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        print("\nTroubleshooting:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Check Python version: python --version (need 3.8+)")
        print("  3. Read GETTING_STARTED.md for more help")
    print("=" * 80)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
