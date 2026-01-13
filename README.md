# RAG (Retrieval Augmented Generation) From Scratch

A comprehensive learning project to understand RAG concepts by building everything from the ground up.

## What is RAG?

RAG (Retrieval Augmented Generation) is a technique that enhances Large Language Models (LLMs) by retrieving relevant information from a knowledge base before generating responses. Instead of relying solely on the model's training data, RAG allows the model to access external, up-to-date information.

### Why RAG?

1. **Fresh Information**: Access to current data beyond the model's training cutoff
2. **Domain-Specific Knowledge**: Include specialized information not in general training
3. **Verifiable Responses**: Ground answers in retrievable sources
4. **Reduced Hallucinations**: Provide factual context for generation
5. **Cost-Effective**: No need to retrain models with new data

## RAG Pipeline Components

```
┌─────────────────────────────────────────────────────────────┐
│                     RAG PIPELINE                             │
└─────────────────────────────────────────────────────────────┘

1. INDEXING PHASE (Offline - Done Once)
   ┌──────────────┐
   │   Documents  │
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │ Load & Parse │  ← Read files (PDF, TXT, MD, etc.)
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │   Chunking   │  ← Split into manageable pieces
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │  Embeddings  │  ← Convert text to vectors
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │Vector Store  │  ← Store for fast retrieval
   └──────────────┘

2. RETRIEVAL PHASE (Online - Per Query)
   ┌──────────────┐
   │  User Query  │
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │ Embed Query  │  ← Convert to same vector space
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │   Search     │  ← Find similar chunks (cosine similarity)
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │  Rerank      │  ← Optional: Re-score results
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │Build Context │  ← Combine retrieved chunks
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │   Generate   │  ← LLM creates response
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │   Response   │
   └──────────────┘
```

## Key Concepts Explained

### 1. Document Loading
- Reading various file formats (PDF, TXT, DOCX, MD)
- Extracting clean text
- Preserving structure and metadata

### 2. Chunking
Breaking documents into smaller pieces:

**Why chunk?**
- Embeddings work better on focused content
- LLMs have context length limits
- Improves retrieval precision

**Strategies:**
- **Fixed-size**: Simple chunks (e.g., 500 tokens)
- **Sentence-based**: Split on sentence boundaries
- **Semantic**: Split on topic changes
- **Recursive**: Split with fallback delimiters
- **Overlap**: Include context from adjacent chunks

### 3. Embeddings
Converting text to numerical vectors that capture semantic meaning.

**How it works:**
- Text → Embedding Model → Vector (e.g., 384 or 1536 dimensions)
- Similar meanings = similar vectors
- Enables semantic search (not just keyword matching)

**Models:**
- Sentence Transformers (local, free)
- OpenAI Ada-002 (API)
- Cohere Embed (API)

### 4. Vector Storage
Storing and indexing embeddings for fast retrieval.

**Options:**
- **Simple**: In-memory with NumPy/lists
- **Libraries**: FAISS, Annoy (approximate nearest neighbors)
- **Databases**: Pinecone, Weaviate, Chroma, Qdrant

**Search Methods:**
- Cosine Similarity: Most common
- Euclidean Distance: Alternative metric
- Dot Product: Fast but requires normalized vectors

### 5. Retrieval
Finding the most relevant chunks for a query.

**Process:**
1. Embed the user's query
2. Compare with stored embeddings
3. Return top-k most similar chunks
4. Optional: Apply filters, reranking

### 6. Context Building
Preparing retrieved information for the LLM.

**Considerations:**
- Chunk ordering (by relevance or document order)
- Deduplication
- Token budget management
- Source attribution

### 7. Generation
Using an LLM to generate the final response.

**Prompt Engineering:**
```
Context: [Retrieved chunks]

Question: [User query]

Instructions: Answer based on the context above. If the context doesn't contain
enough information, say so.
```

## Project Structure

```
rag-from-scratch/
├── README.md                    # This file
├── docs/
│   ├── 01-chunking.md          # Detailed chunking guide
│   ├── 02-embeddings.md        # Embeddings deep dive
│   ├── 03-vector-search.md     # Vector search explained
│   └── 04-advanced-rag.md      # Advanced techniques
├── src/
│   ├── 01_document_loader.py   # Load various file types
│   ├── 02_chunking.py          # Text splitting strategies
│   ├── 03_embeddings.py        # Generate embeddings
│   ├── 04_vector_store.py      # Store and search vectors
│   ├── 05_retrieval.py         # Retrieve relevant chunks
│   ├── 06_rag_pipeline.py      # Complete RAG system
│   └── utils.py                # Helper functions
├── examples/
│   ├── basic_rag.py            # Simple end-to-end example
│   ├── compare_chunking.py     # Compare chunking strategies
│   └── evaluate_retrieval.py   # Measure retrieval quality
└── data/
    ├── sample_docs/            # Example documents
    └── vector_db/              # Stored vectors
```

## Getting Started

### Prerequisites

```bash
pip install sentence-transformers numpy scikit-learn pdfplumber python-docx
```

### Quick Start

1. **Prepare Documents**: Place your documents in `data/sample_docs/`

2. **Index Documents**:
```python
from src.rag_pipeline import RAGPipeline

# Initialize RAG system
rag = RAGPipeline()

# Load and index documents
rag.index_documents("data/sample_docs/")
```

3. **Query**:
```python
# Ask a question
response = rag.query("What is machine learning?")
print(response)
```

## Learning Path

Follow these modules in order:

1. **Module 1**: Document Loading (`src/01_document_loader.py`)
   - Load different file types
   - Extract clean text

2. **Module 2**: Chunking (`src/02_chunking.py`)
   - Implement various chunking strategies
   - Compare effectiveness

3. **Module 3**: Embeddings (`src/03_embeddings.py`)
   - Generate embeddings
   - Understand vector representations

4. **Module 4**: Vector Storage (`src/04_vector_store.py`)
   - Build a simple vector database
   - Implement similarity search

5. **Module 5**: Retrieval (`src/05_retrieval.py`)
   - Query the vector store
   - Rank and filter results

6. **Module 6**: Complete Pipeline (`src/06_rag_pipeline.py`)
   - Integrate all components
   - Build end-to-end RAG system

## Advanced Topics

- **Hybrid Search**: Combine keyword and semantic search
- **Reranking**: Improve retrieval with cross-encoders
- **Query Expansion**: Enhance queries before retrieval
- **Metadata Filtering**: Add structured filters
- **Evaluation**: Measure retrieval quality (MRR, NDCG)

## Resources

- [Sentence Transformers Documentation](https://www.sbert.net/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Chunking Strategies Research](https://arxiv.org/abs/2307.03172)

## Next Steps

Start with `src/01_document_loader.py` and work through each module. Each file contains detailed explanations and exercises.
