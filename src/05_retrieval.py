"""
Module 5: Retrieval and Context Building

This module teaches how to retrieve relevant information and prepare it for
the LLM generation step. This is where vector search meets practical RAG.

Key Concepts:
- Query processing and expansion
- Retrieval strategies
- Reranking for better results
- Context building and formatting
- Source attribution
- Hybrid search (combining keyword + semantic)

The Retrieval Pipeline:
1. Process user query
2. Search vector store
3. (Optional) Rerank results
4. Build context from top results
5. Format for LLM prompt
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from collections import defaultdict


@dataclass
class RetrievalResult:
    """
    Enhanced search result with context and sources.

    Attributes:
        text: Retrieved text chunk
        score: Relevance score
        source: Source document/file
        metadata: Additional metadata
        rank: Position in results (1 = most relevant)
    """
    text: str
    score: float
    source: str
    metadata: Dict
    rank: int = 0

    def __repr__(self):
        preview = self.text[:60] + "..." if len(self.text) > 60 else self.text
        return f"RetrievalResult(rank={self.rank}, score={self.score:.3f}, text='{preview}')"


class Retriever:
    """
    Handles the retrieval process in RAG.

    This orchestrates:
    - Vector search
    - Result ranking
    - Context building
    """

    def __init__(self, vector_store, embedding_model, top_k: int = 5):
        """
        Args:
            vector_store: VectorStore instance
            embedding_model: Model to embed queries
            top_k: Number of results to retrieve
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_fn: Optional[callable] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: User's question
            top_k: Number of results (overrides default)
            filter_fn: Optional metadata filter

        Returns:
            List of RetrievalResult objects
        """
        k = top_k or self.top_k

        # Embed the query
        query_vector = self.embedding_model.embed(query)

        # Search vector store
        search_results = self.vector_store.search(
            query_vector,
            k=k,
            filter_fn=filter_fn
        )

        # Convert to RetrievalResult objects
        results = []
        for rank, result in enumerate(search_results, 1):
            retrieval_result = RetrievalResult(
                text=result.text,
                score=result.score,
                source=result.metadata.get('source', 'Unknown'),
                metadata=result.metadata,
                rank=rank
            )
            results.append(retrieval_result)

        return results

    def build_context(
        self,
        results: List[RetrievalResult],
        max_length: Optional[int] = None,
        include_sources: bool = True
    ) -> str:
        """
        Build context string from retrieval results.

        Args:
            results: Retrieved results
            max_length: Maximum context length in characters
            include_sources: Whether to include source attribution

        Returns:
            Formatted context string
        """
        if not results:
            return ""

        context_parts = []

        for result in results:
            if include_sources:
                # Format: [Source: filename] content
                source_name = result.source.split('/')[-1]  # Just filename
                part = f"[Source: {source_name}]\n{result.text}"
            else:
                part = result.text

            # Check length limit
            if max_length:
                current_length = sum(len(p) for p in context_parts)
                if current_length + len(part) > max_length:
                    break

            context_parts.append(part)

        return "\n\n---\n\n".join(context_parts)

    def retrieve_with_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        max_length: Optional[int] = None
    ) -> Tuple[List[RetrievalResult], str]:
        """
        Convenience method: retrieve and build context in one call.

        Returns:
            Tuple of (results, context_string)
        """
        results = self.retrieve(query, top_k=top_k)
        context = self.build_context(results, max_length=max_length)
        return results, context


class HybridRetriever(Retriever):
    """
    Combines semantic search with keyword search.

    Hybrid search is often better than semantic alone:
    - Semantic: Finds conceptually similar content
    - Keyword: Finds exact matches (important for names, codes, etc.)

    This combines both using weighted scoring.
    """

    def __init__(
        self,
        vector_store,
        embedding_model,
        top_k: int = 5,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3
    ):
        super().__init__(vector_store, embedding_model, top_k)
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_fn: Optional[callable] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve using both semantic and keyword search.
        """
        k = top_k or self.top_k

        # 1. Semantic search
        semantic_results = super().retrieve(query, top_k=k * 2)

        # 2. Keyword search
        keyword_scores = self._keyword_search(query)

        # 3. Combine scores
        combined_scores = {}
        for result in semantic_results:
            text_id = id(result.text)  # Use object id as key

            # Normalize semantic score (0-1 range)
            sem_score = result.score

            # Get keyword score
            key_score = keyword_scores.get(result.text, 0.0)

            # Weighted combination
            combined = (self.semantic_weight * sem_score +
                       self.keyword_weight * key_score)

            combined_scores[text_id] = (combined, result)

        # Sort by combined score
        sorted_results = sorted(
            combined_scores.values(),
            key=lambda x: x[0],
            reverse=True
        )

        # Convert to RetrievalResult
        final_results = []
        for rank, (score, result) in enumerate(sorted_results[:k], 1):
            final_results.append(RetrievalResult(
                text=result.text,
                score=score,
                source=result.source,
                metadata=result.metadata,
                rank=rank
            ))

        return final_results

    def _keyword_search(self, query: str) -> Dict[str, float]:
        """
        Simple keyword-based scoring.

        A production system would use BM25 or Elasticsearch here.
        """
        query_terms = set(query.lower().split())
        scores = {}

        for text in self.vector_store.texts:
            text_terms = set(text.lower().split())
            # Simple overlap score
            overlap = len(query_terms & text_terms)
            if overlap > 0:
                scores[text] = overlap / len(query_terms)

        return scores


class QueryExpander:
    """
    Expands queries to improve retrieval.

    Query expansion techniques:
    - Add synonyms
    - Rephrase questions
    - Break into sub-queries

    This improves recall (finding more relevant documents).
    """

    def expand_with_synonyms(self, query: str) -> List[str]:
        """
        Expand query with synonyms.

        In production, use WordNet, ConceptNet, or LLM-based expansion.
        """
        # Simple example - in production, use proper synonym database
        synonym_map = {
            "ai": ["artificial intelligence", "machine learning"],
            "ml": ["machine learning", "ai"],
            "nn": ["neural network", "deep learning"],
        }

        queries = [query]

        for term, synonyms in synonym_map.items():
            if term in query.lower():
                for synonym in synonyms:
                    expanded = query.replace(term, synonym)
                    queries.append(expanded)

        return queries

    def break_into_subqueries(self, query: str) -> List[str]:
        """
        Break complex query into simpler sub-queries.

        Example: "What is AI and how does ML work?"
        → ["What is AI?", "How does ML work?"]
        """
        # Simple heuristic: split on "and", "or"
        parts = []
        for sep in [" and ", " or "]:
            if sep in query.lower():
                parts.extend(query.split(sep))

        return parts if parts else [query]


class ReRanker:
    """
    Reranks retrieval results for better quality.

    Why rerank?
    - Initial retrieval is fast but approximate
    - Reranking uses more expensive but accurate models
    - Improves precision of top results

    Reranking strategies:
    - Cross-encoder models (compare query-document pairs)
    - LLM-based scoring
    - Rule-based heuristics
    """

    def rerank_by_length(
        self,
        results: List[RetrievalResult],
        prefer_longer: bool = True
    ) -> List[RetrievalResult]:
        """
        Rerank based on text length.

        Useful when you want more detailed (longer) or concise (shorter) results.
        """
        scored = []
        for result in results:
            length_score = len(result.text) / 1000.0  # Normalize
            # Combine with original score
            combined = result.score + (0.1 * length_score if prefer_longer else -0.1 * length_score)
            scored.append((combined, result))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Update ranks
        reranked = []
        for rank, (score, result) in enumerate(scored, 1):
            result.rank = rank
            result.score = score
            reranked.append(result)

        return reranked

    def rerank_by_recency(
        self,
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        Rerank to prefer more recent documents.

        Requires 'date' or 'timestamp' in metadata.
        """
        scored = []
        for result in results:
            # Get date from metadata (assume ISO format)
            date_str = result.metadata.get('date', '2000-01-01')
            # Simple scoring: more recent = higher score
            date_score = float(date_str.replace('-', '')) / 10000000  # Normalize

            combined = result.score + (0.2 * date_score)
            scored.append((combined, result))

        scored.sort(key=lambda x: x[0], reverse=True)

        reranked = []
        for rank, (score, result) in enumerate(scored, 1):
            result.rank = rank
            result.score = score
            reranked.append(result)

        return reranked


# ============================================================================
# EXAMPLES AND EXERCISES
# ============================================================================

def example_basic_retrieval():
    """Example: Basic retrieval pipeline"""
    print("=" * 80)
    print("Example 1: Basic Retrieval")
    print("=" * 80)

    # Setup (simplified - normally you'd load from modules 1-4)
    from embeddings import SimpleEmbeddingModel, EmbeddingStore
    from vector_store import SimpleVectorStore

    # Sample documents
    documents = [
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "Deep learning uses neural networks with multiple layers to process data.",
        "Python is a popular programming language for data science and machine learning.",
        "Natural language processing helps computers understand human language.",
        "Reinforcement learning agents learn by interacting with environments."
    ]

    # Create embeddings
    model = SimpleEmbeddingModel(vocabulary_size=50)
    model.fit(documents)
    embedding_store = EmbeddingStore(model)
    embeddings = embedding_store.embed_texts(documents)

    # Create vector store
    vector_store = SimpleVectorStore()
    vectors = [emb.vector for emb in embeddings]
    metadata = [{"source": f"doc_{i}.txt"} for i in range(len(documents))]
    vector_store.add(vectors, documents, metadata)

    # Create retriever
    retriever = Retriever(vector_store, model, top_k=3)

    # Query
    query = "What is machine learning?"
    print(f"\nQuery: {query}")

    results = retriever.retrieve(query)

    print(f"\nTop {len(results)} Results:")
    for result in results:
        print(f"\n{result.rank}. Score: {result.score:.4f}")
        print(f"   Source: {result.source}")
        print(f"   Text: {result.text}")


def example_context_building():
    """Example: Building context for LLM"""
    print("\n" + "=" * 80)
    print("Example 2: Building Context")
    print("=" * 80)

    # Using same setup as above
    from embeddings import SimpleEmbeddingModel, EmbeddingStore
    from vector_store import SimpleVectorStore

    documents = [
        "The Eiffel Tower is in Paris, France.",
        "The Statue of Liberty is in New York, USA.",
        "The Great Wall is in China."
    ]

    model = SimpleEmbeddingModel(vocabulary_size=30)
    model.fit(documents)
    embedding_store = EmbeddingStore(model)
    embeddings = embedding_store.embed_texts(documents)

    vector_store = SimpleVectorStore()
    vectors = [emb.vector for emb in embeddings]
    metadata = [{"source": f"landmarks.txt"} for _ in documents]
    vector_store.add(vectors, documents, metadata)

    retriever = Retriever(vector_store, model, top_k=2)

    query = "Where is the Eiffel Tower?"
    results, context = retriever.retrieve_with_context(query)

    print(f"Query: {query}\n")
    print("Built Context:")
    print("-" * 80)
    print(context)
    print("-" * 80)

    print("\nThis context would be inserted into your LLM prompt:")
    print(f"""
    Context: {context}

    Question: {query}

    Answer based on the context above:
    """)


def example_hybrid_search():
    """Example: Hybrid semantic + keyword search"""
    print("\n" + "=" * 80)
    print("Example 3: Hybrid Search")
    print("=" * 80)

    print("💡 Hybrid search combines:")
    print("  - Semantic search: Finds similar meanings")
    print("  - Keyword search: Finds exact term matches")
    print("\nThis gives better results, especially for:")
    print("  - Names, codes, specific terms")
    print("  - Technical documentation")
    print("  - FAQs with specific keywords")


def exercise_1():
    """
    EXERCISE 1: Implement Query Expansion

    Task: Expand a query and retrieve for each variation
    """
    print("\n" + "=" * 80)
    print("EXERCISE 1: Query Expansion")
    print("=" * 80)

    # TODO: Use QueryExpander to expand a query
    # Retrieve results for each expanded query
    # Combine and deduplicate results

    print("\nYour code here!")


def exercise_2():
    """
    EXERCISE 2: Reranking

    Task: Retrieve results and rerank by different criteria
    """
    print("\n" + "=" * 80)
    print("EXERCISE 2: Reranking Results")
    print("=" * 80)

    # TODO: Retrieve results
    # Try different reranking strategies
    # Compare the order of results

    print("\nYour code here!")


if __name__ == "__main__":
    print("\n🚀 RAG FROM SCRATCH - MODULE 5: RETRIEVAL\n")

    # Run examples
    example_basic_retrieval()
    example_context_building()
    example_hybrid_search()

    # Exercises
    exercise_1()
    exercise_2()

    print("\n" + "=" * 80)
    print("Key Takeaways:")
    print("=" * 80)
    print("✓ Retrieval converts queries to vectors and searches for similar chunks")
    print("✓ Context building formats retrieved chunks for the LLM")
    print("✓ Hybrid search (semantic + keyword) often works best")
    print("✓ Query expansion improves recall (finds more relevant docs)")
    print("✓ Reranking improves precision (better quality in top results)")
    print("✓ Source attribution helps users verify information")
    print("✓ Next: We'll integrate everything into a complete RAG pipeline")
    print("=" * 80)
