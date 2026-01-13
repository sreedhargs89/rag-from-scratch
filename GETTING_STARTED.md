# Getting Started with RAG

Welcome! This guide will walk you through learning RAG (Retrieval Augmented Generation) from the ground up.

## What You'll Learn

By the end of this tutorial, you'll understand:

1. How RAG works end-to-end
2. Every component in detail (loading, chunking, embeddings, vector search, retrieval)
3. How to build your own RAG system
4. Best practices and optimization techniques
5. When and why to use different strategies

## Learning Approach

This project uses a **learn-by-building** approach:
- Each module builds on the previous one
- Includes working code you can run and modify
- Exercises to practice concepts
- Real examples and use cases

## Prerequisites

### Knowledge
- Basic Python programming
- Understanding of: lists, dictionaries, functions, classes
- Optional but helpful: Basic linear algebra (vectors, dot products)

### Setup

1. **Install Python 3.8+**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

For a minimal setup (just to get started):
```bash
pip install numpy scikit-learn
```

For full features:
```bash
pip install numpy scikit-learn sentence-transformers faiss-cpu pdfplumber python-docx
```

## Learning Path

### Phase 1: Understand the Concepts (30 minutes)

Read the main README to understand:
- What RAG is and why it matters
- The complete pipeline
- Each component's role

### Phase 2: Hands-On Learning (3-4 hours)

Work through each module sequentially:

#### Module 1: Document Loading (30 min)
```bash
python src/01_document_loader.py
```

**You'll learn:**
- Loading different file types (TXT, PDF, DOCX, etc.)
- Extracting clean text
- Managing metadata

**Key concept**: Documents are the foundation - clean text extraction is crucial.

---

#### Module 2: Chunking (45 min)
```bash
python src/02_chunking.py
```

**You'll learn:**
- Why chunking matters
- Fixed-size vs sentence-based vs recursive chunking
- Chunk overlap and its importance
- How chunk size affects retrieval

**Key concept**: Chunking strategy dramatically impacts retrieval quality.

---

#### Module 3: Embeddings (45 min)
```bash
python src/03_embeddings.py
```

**You'll learn:**
- What embeddings are
- How they capture semantic meaning
- Using different embedding models
- Cosine similarity for comparing text

**Key concept**: Embeddings convert text to vectors, enabling semantic search.

---

#### Module 4: Vector Store (45 min)
```bash
python src/04_vector_store.py
```

**You'll learn:**
- Why we need vector databases
- Similarity search algorithms
- Metadata filtering
- Persistence (saving/loading)

**Key concept**: Vector stores enable fast similarity search at scale.

---

#### Module 5: Retrieval (45 min)
```bash
python src/05_retrieval.py
```

**You'll learn:**
- Retrieving relevant chunks
- Building context for LLMs
- Query expansion and reranking
- Hybrid search strategies

**Key concept**: Good retrieval is the key to accurate RAG responses.

---

#### Module 6: Complete Pipeline (45 min)
```bash
python src/06_rag_pipeline.py
```

**You'll learn:**
- Integrating all components
- Indexing documents (offline phase)
- Querying (online phase)
- Production considerations

**Key concept**: RAG is more than the sum of its parts - integration matters.

### Phase 3: Practice (1-2 hours)

#### Example 1: Basic RAG
```bash
python examples/basic_rag.py
```

Build a simple question-answering system.

#### Example 2: Compare Chunking
```bash
python examples/compare_chunking.py
```

See how different chunking strategies affect results.

### Phase 4: Build Your Own (Ongoing)

Now apply what you've learned:

1. **Start simple**:
```python
from src.rag_pipeline import RAGPipeline

rag = RAGPipeline()
rag.index_documents("your_documents/")
answer = rag.query("Your question?")
```

2. **Experiment**:
   - Try different chunk sizes
   - Test various embedding models
   - Adjust top_k values

3. **Iterate**:
   - Measure quality with your use case
   - Optimize based on results
   - Add advanced features as needed

## Common Questions

### Q: Which embedding model should I use?

**For learning**: Use `SimpleEmbeddingModel` (fast, easy to understand)

**For production**: Use `sentence-transformers` (high quality, still runs locally)

**For scale**: Use API-based embeddings (OpenAI, Cohere) or local sentence-transformers

### Q: What chunk size is best?

It depends on your content:
- **Q&A, FAQs**: 200-400 characters
- **General docs**: 400-600 characters
- **Technical docs**: 600-1000 characters
- **Long-form**: 800-1200 characters

**Rule of thumb**: Start with 500, adjust based on results.

### Q: How many chunks should I retrieve (top_k)?

- **Fast, focused**: 3 chunks
- **Balanced**: 5 chunks (recommended)
- **Comprehensive**: 7-10 chunks

### Q: Should I use overlap?

Yes! Overlap preserves context between chunks.
- **Typical**: 10-20% of chunk size
- **Example**: 500 char chunks → 50-100 char overlap

### Q: When should I use FAISS vs SimpleVectorStore?

- **< 10,000 chunks**: SimpleVectorStore is fine
- **10,000 - 1M chunks**: FAISS recommended
- **> 1M chunks**: FAISS with GPU or distributed vector DB

### Q: How do I connect this to an LLM?

Replace the `_generate_simple_response` method in `RAGPipeline`:

```python
def generate_with_llm(self, query, context):
    # Example with Anthropic Claude
    import anthropic

    client = anthropic.Anthropic(api_key="your-api-key")

    prompt = f"""Context:
{context}

Question: {query}

Answer the question based only on the context above. If the context doesn't contain enough information, say so."""

    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

    return message.content[0].text
```

## Troubleshooting

### Issue: ImportError when running examples

**Solution**: Install missing packages
```bash
pip install -r requirements.txt
```

### Issue: "Module not found" errors

**Solution**: Run from correct directory
```bash
# From project root
python src/01_document_loader.py

# Or from examples/
cd examples
python basic_rag.py
```

### Issue: Slow performance

**Solutions**:
1. Use SimpleEmbeddingModel for testing
2. Reduce number of documents
3. Use smaller chunk size
4. Install FAISS for faster search

### Issue: Poor retrieval quality

**Solutions**:
1. Adjust chunk size
2. Increase chunk overlap
3. Try different chunking strategy (recursive usually best)
4. Use better embedding model (sentence-transformers)
5. Increase top_k

## Next Steps

Once you've completed the learning path:

1. **Read Advanced Techniques**: `docs/advanced-rag.md`
2. **Add Evaluation**: Measure retrieval quality
3. **Integrate LLM**: Connect to OpenAI, Anthropic, or local models
4. **Optimize**: Profile and improve performance
5. **Deploy**: Build an API or web interface

## Resources

### Documentation
- `README.md` - Core concepts and architecture
- `QUICKSTART.md` - Fast setup guide
- `docs/advanced-rag.md` - Advanced techniques
- Each module's docstrings - Detailed explanations

### Code
- `src/` - All modules with examples
- `examples/` - Complete working examples

### Community
- GitHub Issues - Ask questions
- Discussions - Share your projects

## Tips for Success

1. **Don't skip modules** - Each builds on the previous
2. **Run the code** - Reading isn't enough, experiment!
3. **Do the exercises** - Practice solidifies learning
4. **Start simple** - Get basic version working first
5. **Iterate** - Gradually add complexity
6. **Measure** - Use metrics to guide improvements
7. **Share** - Teaching others reinforces your learning

## Estimated Time Investment

- **Quick overview**: 1 hour (README + one example)
- **Basic understanding**: 4-5 hours (all modules)
- **Solid grasp**: 8-10 hours (modules + exercises + experimentation)
- **Production-ready**: 20+ hours (includes optimization, evaluation, deployment)

## You're Ready!

Start with Module 1 and work your way through. Don't rush - understanding the concepts deeply is more valuable than finishing quickly.

**Have fun building! 🚀**

---

Need help? Check the troubleshooting section or open an issue on GitHub.
