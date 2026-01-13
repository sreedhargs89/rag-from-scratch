# RAG From Scratch - Cheat Sheet

Quick reference for common tasks and configurations.

## 🚀 Quick Start

```python
from src.rag_pipeline import RAGPipeline

# Initialize and use
rag = RAGPipeline()
rag.index_documents("docs/")
answer = rag.query("Your question?")
```

## 📝 Common Configurations

### Simple Setup (Learning)
```python
rag = RAGPipeline(
    embedding_model_name="simple",
    vector_store_type="simple",
    chunk_size=300,
    chunk_overlap=50,
    top_k=3
)
```

### Production Setup
```python
rag = RAGPipeline(
    embedding_model_name="sentence-transformer",
    vector_store_type="faiss",
    chunk_size=500,
    chunk_overlap=100,
    top_k=5
)
```

### For Q&A (Short Answers)
```python
rag = RAGPipeline(
    chunk_size=200,
    chunk_overlap=50,
    top_k=3
)
```

### For Technical Docs (More Context)
```python
rag = RAGPipeline(
    chunk_size=800,
    chunk_overlap=200,
    top_k=5
)
```

## 🔧 Individual Components

### Document Loading
```python
from src.document_loader import DocumentLoader

loader = DocumentLoader()

# Single file
doc = loader.load_file("document.pdf")

# Directory
docs = loader.load_directory("docs/", recursive=True)
```

### Chunking
```python
from src.chunking import FixedSizeChunker, SentenceChunker, RecursiveChunker

# Fixed size
chunker = FixedSizeChunker(chunk_size=500, chunk_overlap=50)

# Sentence-based
chunker = SentenceChunker(max_sentences=5, overlap_sentences=1)

# Recursive (recommended)
chunker = RecursiveChunker(chunk_size=500, chunk_overlap=100)

# Use
chunks = chunker.chunk(text)
```

### Embeddings
```python
from src.embeddings import SimpleEmbeddingModel, SentenceTransformerModel, EmbeddingStore

# Simple (for learning)
model = SimpleEmbeddingModel(vocabulary_size=100)
model.fit(texts)  # Must fit first!

# Production
model = SentenceTransformerModel('all-MiniLM-L6-v2')

# Store embeddings
store = EmbeddingStore(model)
embeddings = store.embed_texts(texts)
```

### Vector Store
```python
from src.vector_store import SimpleVectorStore, FAISSVectorStore

# Simple
store = SimpleVectorStore()

# FAISS (for production)
store = FAISSVectorStore(dimension=384)

# Add vectors
store.add(vectors, texts, metadata)

# Search
results = store.search(query_vector, k=5)

# With filter
results = store.search(
    query_vector,
    k=5,
    filter_fn=lambda meta: meta['category'] == 'tech'
)
```

### Retrieval
```python
from src.retrieval import Retriever

retriever = Retriever(vector_store, embedding_model, top_k=5)

# Simple retrieval
results = retriever.retrieve(query)

# With context building
results, context = retriever.retrieve_with_context(query)

# With max length
results, context = retriever.retrieve_with_context(
    query,
    max_length=2000
)
```

## 💾 Persistence

### Save RAG System
```python
rag = RAGPipeline()
rag.index_documents("docs/")
rag.save("my_rag_db")
```

### Load RAG System
```python
rag = RAGPipeline()
rag.load("my_rag_db")
answer = rag.query("Question?")
```

### Save/Load Vector Store Only
```python
# Save
vector_store.save("vectors.pkl")

# Load
vector_store.load("vectors.pkl")
```

## 🎯 Parameter Guide

### Chunk Size
| Size | Use Case | Pros | Cons |
|------|----------|------|------|
| 100-300 | Q&A, FAQs | Precise | Less context |
| 300-600 | General | Balanced | - |
| 600-1000 | Technical | More context | Less precise |
| 1000+ | Long-form | Maximum context | Slow, imprecise |

### Chunk Overlap
| Overlap | Use Case |
|---------|----------|
| 0 | Fast indexing, less storage |
| 10-20% | Balanced (recommended) |
| 30%+ | Maximum context preservation |

### Top-K (Number of Results)
| K | Use Case |
|---|----------|
| 1-3 | Fast, focused answers |
| 3-5 | Balanced (recommended) |
| 5-10 | Comprehensive context |
| 10+ | Research, exploration |

## 🔍 Similarity Scores

### Interpreting Scores
```python
score > 0.8  # Very similar (paraphrases)
score > 0.6  # Similar (related topics)
score > 0.4  # Somewhat related
score < 0.4  # Probably not relevant
```

## 📊 Common Patterns

### Filter by Date
```python
from datetime import datetime

results = retriever.retrieve(
    query,
    filter_fn=lambda meta: meta.get('date', '') > '2023-01-01'
)
```

### Filter by Category
```python
results = retriever.retrieve(
    query,
    filter_fn=lambda meta: meta.get('category') in ['tech', 'science']
)
```

### Get Source Documents
```python
results = retriever.retrieve(query)
sources = list(set(r.source for r in results))
print(f"Found info in: {sources}")
```

### Custom Context Format
```python
def custom_context(results):
    parts = []
    for i, result in enumerate(results, 1):
        parts.append(f"[{i}] {result.text}")
    return "\n\n".join(parts)
```

## 🐛 Debugging

### Check What's Indexed
```python
stats = rag.get_stats()
print(f"Documents: {stats['num_documents']}")
print(f"Chunks: {stats['num_chunks']}")
print(f"Vectors: {stats['num_vectors']}")
```

### Inspect Chunks
```python
for i, chunk in enumerate(rag.all_chunks[:5]):
    print(f"\nChunk {i}:")
    print(f"  Length: {len(chunk.content)}")
    print(f"  Preview: {chunk.content[:100]}")
    print(f"  Metadata: {chunk.metadata}")
```

### Test Retrieval
```python
results = retriever.retrieve("test query", top_k=10)
for r in results:
    print(f"Score: {r.score:.3f} | {r.text[:80]}...")
```

### Check Similarity
```python
from src.embeddings import cosine_similarity

text1 = "machine learning"
text2 = "artificial intelligence"

vec1 = model.embed(text1)
vec2 = model.embed(text2)

sim = cosine_similarity(vec1, vec2)
print(f"Similarity: {sim:.3f}")
```

## ⚡ Performance Tips

### Faster Indexing
```python
# Use SimpleEmbeddingModel for testing
rag = RAGPipeline(embedding_model_name="simple")

# Process fewer documents first
rag.index_documents("docs/", recursive=False)
```

### Faster Search
```python
# Use FAISS
rag = RAGPipeline(vector_store_type="faiss")

# Reduce top_k
rag.retriever.top_k = 3
```

### Less Memory
```python
# Smaller chunks
rag = RAGPipeline(chunk_size=200)

# Smaller vocabulary (for SimpleEmbeddingModel)
model = SimpleEmbeddingModel(vocabulary_size=50)
```

## 🔗 Integration Examples

### With OpenAI
```python
import openai

def generate_with_openai(query, context):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{
            "role": "user",
            "content": f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        }]
    )
    return response.choices[0].message.content
```

### With Anthropic Claude
```python
import anthropic

def generate_with_claude(query, context):
    client = anthropic.Anthropic(api_key="your-key")
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        }]
    )
    return message.content[0].text
```

### With Local LLM (LlamaCPP)
```python
from llama_cpp import Llama

llm = Llama(model_path="model.gguf")

def generate_with_llama(query, context):
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    output = llm(prompt, max_tokens=512)
    return output['choices'][0]['text']
```

## 📚 Module Cheat Sheet

| Module | Main Class | Key Method | Use For |
|--------|-----------|------------|---------|
| 01 | `DocumentLoader` | `load_file()` | Loading docs |
| 02 | `RecursiveChunker` | `chunk()` | Splitting text |
| 03 | `SentenceTransformerModel` | `embed()` | Creating vectors |
| 04 | `SimpleVectorStore` | `search()` | Finding similar |
| 05 | `Retriever` | `retrieve()` | Getting context |
| 06 | `RAGPipeline` | `query()` | Complete system |

## 🎓 Quick Commands

```bash
# Test setup
python test_setup.py

# Run a module
python src/01_document_loader.py

# Run example
python examples/basic_rag.py

# Install dependencies
pip install -r requirements.txt

# Install minimal
pip install numpy scikit-learn

# Install full
pip install numpy scikit-learn sentence-transformers faiss-cpu pdfplumber python-docx
```

## 🆘 Quick Fixes

| Problem | Solution |
|---------|----------|
| ImportError | `pip install -r requirements.txt` |
| Slow indexing | Use `SimpleEmbeddingModel` |
| Poor results | Adjust chunk_size, increase overlap |
| Out of memory | Reduce chunk_size, use smaller model |
| Module not found | Run from project root |

---

**Remember:** Start simple, measure, then optimize!
