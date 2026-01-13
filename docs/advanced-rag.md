# Advanced RAG Techniques

Once you've mastered the basics, these advanced techniques will take your RAG system to the next level.

## Table of Contents

1. [Hybrid Search](#hybrid-search)
2. [Reranking](#reranking)
3. [Query Transformation](#query-transformation)
4. [Hierarchical Retrieval](#hierarchical-retrieval)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Production Optimizations](#production-optimizations)

---

## 1. Hybrid Search

Combine semantic search with keyword search for better results.

### Why Hybrid?

- **Semantic search**: Finds conceptually similar content
- **Keyword search**: Finds exact matches (names, codes, specific terms)
- **Together**: Best of both worlds!

### Implementation

```python
from rank_bm25 import BM25Okapi  # pip install rank-bm25

class HybridRAG:
    def __init__(self):
        self.semantic_weight = 0.7
        self.keyword_weight = 0.3

    def search(self, query, k=5):
        # Semantic search
        semantic_results = self.vector_store.search(query, k=k*2)

        # Keyword search (BM25)
        tokenized_query = query.lower().split()
        keyword_scores = self.bm25.get_scores(tokenized_query)

        # Combine scores
        combined_results = []
        for idx, doc in enumerate(self.documents):
            semantic_score = semantic_results[idx].score if idx < len(semantic_results) else 0
            keyword_score = keyword_scores[idx]

            final_score = (self.semantic_weight * semantic_score +
                          self.keyword_weight * keyword_score)
            combined_results.append((doc, final_score))

        # Sort and return top k
        combined_results.sort(key=lambda x: x[1], reverse=True)
        return combined_results[:k]
```

### When to Use

- Technical documentation with specific terms
- Legal documents with exact phrases
- Code search
- Product catalogs with SKUs/codes

---

## 2. Reranking

Improve precision by reranking initial results with a more powerful model.

### Two-Stage Retrieval

```
Stage 1: Fast retrieval (get top 50)
    ↓
Stage 2: Accurate reranking (get top 5)
```

### Implementation with Cross-Encoder

```python
from sentence_transformers import CrossEncoder

class RerankedRAG:
    def __init__(self):
        # Fast retriever
        self.retriever = Retriever(...)

        # Accurate reranker
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def retrieve_and_rerank(self, query, k=5):
        # Stage 1: Fast retrieval (over-fetch)
        candidates = self.retriever.retrieve(query, top_k=k*10)

        # Stage 2: Accurate reranking
        query_doc_pairs = [(query, doc.text) for doc in candidates]
        rerank_scores = self.reranker.predict(query_doc_pairs)

        # Sort by reranked scores
        scored_results = list(zip(candidates, rerank_scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, score in scored_results[:k]]
```

### Benefits

- 10-30% improvement in retrieval quality
- Catches nuanced relevance
- Better than pure vector similarity

### Cost

- Slower (O(n) comparisons)
- Use only for final ranking

---

## 3. Query Transformation

Transform user queries for better retrieval.

### Techniques

#### a) Query Expansion

Add related terms:

```python
def expand_query(query):
    # Use LLM to expand
    expanded = llm.generate(f"""
    Expand this query with related terms and synonyms:
    Query: {query}
    Expanded:
    """)
    return expanded
```

#### b) Query Decomposition

Break complex queries into sub-queries:

```python
def decompose_query(query):
    # "What is AI and how does ML work?"
    # → ["What is AI?", "How does ML work?"]

    sub_queries = llm.generate(f"""
    Break this complex question into simpler sub-questions:
    Question: {query}
    Sub-questions:
    """)

    # Retrieve for each sub-query
    all_results = []
    for sub_q in sub_queries:
        results = retrieve(sub_q)
        all_results.extend(results)

    # Deduplicate and return
    return deduplicate(all_results)
```

#### c) Hypothetical Document Embeddings (HyDE)

Generate a hypothetical answer, then search for it:

```python
def hyde_search(query):
    # Generate hypothetical answer
    hyp_answer = llm.generate(f"Answer this question: {query}")

    # Search using the hypothetical answer
    results = vector_store.search(embed(hyp_answer))

    return results
```

---

## 4. Hierarchical Retrieval

Use document structure to improve retrieval.

### Concept

```
Document
  ├── Chapter 1
  │     ├── Section 1.1
  │     └── Section 1.2
  └── Chapter 2
        ├── Section 2.1
        └── Section 2.2
```

### Implementation

```python
class HierarchicalChunker:
    def chunk_with_hierarchy(self, document):
        chunks = []

        # Level 1: Chapters
        chapters = self.split_by_chapter(document)

        for chapter in chapters:
            # Level 2: Sections
            sections = self.split_by_section(chapter)

            for section in sections:
                # Level 3: Paragraphs
                paragraphs = self.split_by_paragraph(section)

                for para in paragraphs:
                    chunk = Chunk(
                        content=para,
                        metadata={
                            'chapter': chapter.title,
                            'section': section.title,
                            'hierarchy_level': 3
                        }
                    )
                    chunks.append(chunk)

        return chunks

class HierarchicalRetriever:
    def retrieve(self, query):
        # First, find relevant chapters
        relevant_chapters = self.search(query, filter="hierarchy_level:1")

        # Then, search within those chapters
        results = []
        for chapter in relevant_chapters:
            chapter_results = self.search(
                query,
                filter=f"chapter:{chapter.title}"
            )
            results.extend(chapter_results)

        return results
```

### Benefits

- Better context awareness
- Can retrieve entire sections if needed
- Maintains document structure

---

## 5. Evaluation Metrics

Measure your RAG system's performance.

### Retrieval Metrics

#### a) Mean Reciprocal Rank (MRR)

```python
def calculate_mrr(results, relevant_docs):
    """
    MRR = average of (1 / rank of first relevant doc)
    """
    reciprocal_ranks = []

    for query_results in results:
        for rank, doc in enumerate(query_results, 1):
            if doc in relevant_docs:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)

    return sum(reciprocal_ranks) / len(reciprocal_ranks)
```

#### b) NDCG (Normalized Discounted Cumulative Gain)

```python
def calculate_ndcg(results, relevance_scores, k=10):
    """
    NDCG considers both relevance and ranking position
    """
    dcg = sum(
        (2**rel - 1) / math.log2(i + 2)
        for i, rel in enumerate(relevance_scores[:k])
    )

    ideal_scores = sorted(relevance_scores, reverse=True)
    idcg = sum(
        (2**rel - 1) / math.log2(i + 2)
        for i, rel in enumerate(ideal_scores[:k])
    )

    return dcg / idcg if idcg > 0 else 0
```

#### c) Recall@K

```python
def recall_at_k(retrieved, relevant, k):
    """
    Proportion of relevant docs in top k results
    """
    retrieved_k = set(retrieved[:k])
    relevant_set = set(relevant)

    return len(retrieved_k & relevant_set) / len(relevant_set)
```

### Generation Metrics

#### a) Answer Relevance

```python
def answer_relevance(query, answer):
    """
    Use LLM to judge relevance
    """
    prompt = f"""
    Rate the relevance of this answer to the question (0-10):

    Question: {query}
    Answer: {answer}

    Rating:
    """

    rating = llm.generate(prompt)
    return int(rating)
```

#### b) Faithfulness (Groundedness)

```python
def check_faithfulness(answer, context):
    """
    Check if answer is supported by retrieved context
    """
    prompt = f"""
    Does the answer contain only information from the context?

    Context: {context}
    Answer: {answer}

    Reply: Yes/No
    Explanation:
    """

    return llm.generate(prompt)
```

---

## 6. Production Optimizations

Make your RAG system production-ready.

### Caching

```python
from functools import lru_cache
import hashlib

class CachedRAG:
    def __init__(self):
        self.embedding_cache = {}
        self.result_cache = {}

    def get_embedding(self, text):
        # Cache embeddings
        text_hash = hashlib.md5(text.encode()).hexdigest()

        if text_hash not in self.embedding_cache:
            self.embedding_cache[text_hash] = self.model.embed(text)

        return self.embedding_cache[text_hash]

    @lru_cache(maxsize=1000)
    def query(self, question):
        # Cache query results
        return self._query_impl(question)
```

### Async Processing

```python
import asyncio

class AsyncRAG:
    async def index_documents_async(self, documents):
        # Process documents in parallel
        tasks = [
            self.process_document(doc)
            for doc in documents
        ]

        results = await asyncio.gather(*tasks)
        return results

    async def process_document(self, document):
        # Chunk
        chunks = await self.chunk_async(document)

        # Embed
        embeddings = await self.embed_async(chunks)

        # Store
        await self.store_async(embeddings)
```

### Monitoring

```python
class MonitoredRAG:
    def __init__(self):
        self.metrics = {
            'queries': 0,
            'avg_latency': 0,
            'cache_hits': 0,
            'errors': 0
        }

    def query(self, question):
        start = time.time()

        try:
            result = self._query(question)

            # Update metrics
            latency = time.time() - start
            self.metrics['queries'] += 1
            self.metrics['avg_latency'] = (
                (self.metrics['avg_latency'] * (self.metrics['queries'] - 1) + latency)
                / self.metrics['queries']
            )

            return result

        except Exception as e:
            self.metrics['errors'] += 1
            raise
```

### Batch Updates

```python
class BatchRAG:
    def __init__(self):
        self.update_buffer = []
        self.buffer_size = 100

    def add_document(self, document):
        self.update_buffer.append(document)

        if len(self.update_buffer) >= self.buffer_size:
            self.flush_buffer()

    def flush_buffer(self):
        # Process all buffered documents at once
        chunks = self.chunk_all(self.update_buffer)
        embeddings = self.embed_batch(chunks)
        self.vector_store.add_batch(embeddings)

        self.update_buffer = []
```

---

## Best Practices Summary

1. **Start simple**: Basic RAG before advanced techniques
2. **Measure first**: Baseline metrics before optimization
3. **Iterate**: Improve one component at a time
4. **Test thoroughly**: Use diverse queries and edge cases
5. **Monitor production**: Track latency, quality, errors
6. **Document decisions**: Why you chose specific techniques

## Next Steps

- Implement evaluation metrics for your RAG system
- Try hybrid search on your domain
- Add reranking for better quality
- Set up monitoring and caching
- A/B test different configurations

---

## Resources

- [RAG Papers and Research](https://arxiv.org/search/?query=retrieval+augmented+generation)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [LangChain RAG Guide](https://python.langchain.com/docs/use_cases/question_answering/)
- [Anthropic RAG Guide](https://docs.anthropic.com/claude/docs/guide-to-rag)
