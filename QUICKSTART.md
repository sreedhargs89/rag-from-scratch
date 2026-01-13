# RAG From Scratch - Quick Start Guide

Get up and running with RAG in 5 minutes!

## Installation

```bash
# Clone or navigate to the project directory
cd rag-from-scratch

# Install dependencies
pip install -r requirements.txt
```

## Quick Start (3 Steps)

### 1. Run the Basic Example

```bash
cd examples
python basic_rag.py
```

This will:
- Create sample documents
- Build embeddings
- Index into a vector store
- Answer questions using retrieval

### 2. Try with Your Own Documents

```python
from src.rag_pipeline import RAGPipeline

# Initialize
rag = RAGPipeline(
    embedding_model_name="simple",  # Start with simple, upgrade to "sentence-transformer" later
    chunk_size=500,
    top_k=3
)

# Index your documents
rag.index_documents("path/to/your/documents/")

# Ask questions
answer = rag.query("Your question here?")
print(answer)
```

### 3. Learn the Concepts

Work through the modules in order:

```bash
# Module 1: Document Loading
python src/01_document_loader.py

# Module 2: Chunking
python src/02_chunking.py

# Module 3: Embeddings
python src/03_embeddings.py

# Module 4: Vector Store
python src/04_vector_store.py

# Module 5: Retrieval
python src/05_retrieval.py

# Module 6: Complete Pipeline
python src/06_rag_pipeline.py
```

Each module includes:
- Detailed explanations
- Working examples
- Exercises to practice

## Common Use Cases

### Use Case 1: Document Q&A

Perfect for: Company docs, manuals, knowledge bases

```python
rag = RAGPipeline(chunk_size=300, top_k=5)
rag.index_documents("docs/company_handbook/")

answer = rag.query("What is our vacation policy?")
```

### Use Case 2: Research Assistant

Perfect for: Academic papers, research notes

```python
rag = RAGPipeline(
    embedding_model_name="sentence-transformer",  # Better quality
    chunk_size=500,
    top_k=3
)
rag.index_documents("papers/")

answer = rag.query("What are the main findings about climate change?")
```

### Use Case 3: Code Documentation

Perfect for: Code repos, technical docs

```python
rag = RAGPipeline(chunk_size=200, top_k=4)
rag.index_documents("project/docs/")

answer = rag.query("How do I authenticate users?")
```

## Production Upgrade

When ready for production, upgrade to better models:

```python
rag = RAGPipeline(
    embedding_model_name="sentence-transformer",  # High-quality embeddings
    vector_store_type="faiss",                     # Fast search
    chunk_size=500,
    chunk_overlap=100,
    top_k=5
)
```

## Configuration Guide

### Chunk Size

- **Small (200-300)**: Good for precise Q&A, short answers
- **Medium (400-600)**: General purpose, balanced
- **Large (800-1000)**: Need more context, complex topics

### Chunk Overlap

- **None (0)**: Faster, but may lose context between chunks
- **Small (50-100)**: Good balance (recommended)
- **Large (200+)**: Maximum context preservation, slower

### Top-K (Number of Results)

- **Few (1-3)**: Faster, focused answers
- **Medium (3-5)**: Balanced (recommended)
- **Many (5-10)**: More context, but potentially noisy

## Troubleshooting

### Problem: Irrelevant Results

**Solutions:**
1. Increase chunk overlap
2. Try different chunking strategy
3. Use better embedding model (sentence-transformer)
4. Increase top_k to see more results

### Problem: Slow Indexing

**Solutions:**
1. Use SimpleEmbeddingModel for testing (faster but lower quality)
2. Reduce number of documents
3. Batch process large document sets

### Problem: Slow Search

**Solutions:**
1. Use FAISS instead of SimpleVectorStore
2. Reduce top_k
3. Use GPU if available (faiss-gpu)

### Problem: Poor Answer Quality

**Solutions:**
1. Improve chunking (try recursive chunker)
2. Adjust chunk size for your content
3. Increase chunk overlap
4. Use production embedding model
5. Increase top_k for more context

## Next Steps

1. **Read the README** - Understand RAG concepts deeply
2. **Work through modules** - Build knowledge systematically
3. **Experiment** - Try different configurations with your data
4. **Integrate LLM** - Connect to OpenAI/Anthropic for generation
5. **Add evaluation** - Measure retrieval quality
6. **Deploy** - Build API or web interface

## Learning Path

```
Beginner → Modules 1-3 (Document loading, Chunking, Embeddings)
           ↓
Intermediate → Modules 4-5 (Vector stores, Retrieval)
           ↓
Advanced → Module 6 + Examples (Complete pipeline)
           ↓
Production → Integrate LLM, Deploy, Optimize
```

## Resources

- **Documentation**: See `docs/` folder for deep dives
- **Examples**: See `examples/` folder for use cases
- **Help**: Open an issue or discussion on GitHub

## Quick Reference

```python
# Complete minimal example
from src.rag_pipeline import RAGPipeline

# Setup
rag = RAGPipeline()

# Index
rag.index_documents("my_docs/")

# Query
answer = rag.query("My question?")
print(answer)

# Save for later use
rag.save("my_rag_db")

# Load previously indexed data
rag.load("my_rag_db")
```

That's it! You're ready to build RAG systems. Happy learning! 🚀
