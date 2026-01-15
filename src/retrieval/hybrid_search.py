"""
Hybrid Retrieval System
Combines Dense (embeddings) + Sparse (BM25) + Reranking
All using FREE tools
"""

from typing import List, Dict, Any
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import numpy as np
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings, EMBEDDING_CONFIG


class HybridRetriever:
    """
    Hybrid Retrieval System combining:
    1. Dense Retrieval: Semantic embeddings (FREE - sentence-transformers)
    2. Sparse Retrieval: BM25 keyword matching (FREE - rank-bm25)
    3. Reranking: CrossEncoder for final ranking (FREE - sentence-transformers)
    """
    
    def __init__(self, vector_store_path: str = None):
        print("🚀 Initializing Hybrid Retriever (100% FREE)...")
        
        # 1. Dense Retrieval: Embeddings
        print("📊 Loading embedding model (local, no API cost)...")
        self.embeddings = HuggingFaceEmbeddings(**EMBEDDING_CONFIG)
        
        # 2. Vector Store
        self.vector_store_path = vector_store_path or settings.CHROMA_PATH
        self.vector_store = None
        
        # 3. BM25 for sparse retrieval
        self.bm25 = None
        self.documents = []
        self.tokenized_corpus = []
        
        # 4. Reranker (CrossEncoder - local, free)
        if settings.USE_RERANKING:
            print("🎯 Loading reranker model (local, no API cost)...")
            self.reranker = CrossEncoder(settings.RERANKER_MODEL)
        else:
            self.reranker = None
        
        print("✅ Hybrid Retriever ready!")
    
    def load_vector_store(self):
        """Load existing vector store"""
        try:
            if not os.path.exists(self.vector_store_path):
                print(f"⚠️  Vector store not found at {self.vector_store_path}")
                print("💡 Create one by running ingestion or using create_vector_store()")
                return False
            
            self.vector_store = Chroma(
                persist_directory=self.vector_store_path,
                embedding_function=self.embeddings,
                collection_name=settings.COLLECTION_NAME
            )
            
            # Also load documents for BM25
            self._load_documents_for_bm25()
            
            print(f"✅ Loaded vector store from {self.vector_store_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading vector store: {e}")
            return False
    
    def _load_documents_for_bm25(self):
        """Load documents from vector store for BM25"""
        try:
            # Get all documents from vector store
            collection = self.vector_store._collection
            results = collection.get()
            
            if results and results['documents']:
                self.documents = [
                    Document(
                        page_content=doc,
                        metadata=meta if meta else {}
                    )
                    for doc, meta in zip(results['documents'], results['metadatas'])
                ]
                
                # Tokenize for BM25
                self.tokenized_corpus = [
                    doc.page_content.lower().split() 
                    for doc in self.documents
                ]
                
                # Initialize BM25
                if self.tokenized_corpus:
                    self.bm25 = BM25Okapi(
                        self.tokenized_corpus,
                        k1=settings.BM25_K1,
                        b=settings.BM25_B
                    )
                    print(f"✅ Loaded {len(self.documents)} documents for BM25")
        
        except Exception as e:
            print(f"⚠️  Could not load documents for BM25: {e}")
    
    def create_vector_store(self, documents: List[Document]):
        """Create new vector store from documents"""
        print(f"📚 Creating vector store with {len(documents)} documents...")
        
        # Create vector store
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.vector_store_path,
            collection_name=settings.COLLECTION_NAME
        )
        
        # Store documents for BM25
        self.documents = documents
        self.tokenized_corpus = [doc.page_content.lower().split() for doc in documents]
        
        # Initialize BM25
        self.bm25 = BM25Okapi(
            self.tokenized_corpus,
            k1=settings.BM25_K1,
            b=settings.BM25_B
        )
        
        print("✅ Vector store created!")
        return True
    
    def dense_retrieval(self, query: str, k: int = None) -> List[Document]:
        """Dense retrieval using embeddings"""
        if not self.vector_store:
            print("⚠️  Vector store not loaded. Call load_vector_store() first.")
            return []
        
        k = k or settings.TOP_K_RETRIEVAL
        
        try:
            results = self.vector_store.similarity_search(query, k=k)
            return results
        except Exception as e:
            print(f"❌ Error in dense retrieval: {e}")
            return []
    
    def sparse_retrieval(self, query: str, k: int = None) -> List[Document]:
        """Sparse retrieval using BM25"""
        if not self.bm25 or not self.documents:
            return []
        
        k = k or settings.TOP_K_RETRIEVAL
        
        try:
            tokenized_query = query.lower().split()
            scores = self.bm25.get_scores(tokenized_query)
            
            # Get top k indices
            top_k_idx = np.argsort(scores)[-k:][::-1]
            
            # Return documents with positive scores
            return [
                self.documents[idx] 
                for idx in top_k_idx 
                if scores[idx] > 0
            ]
        
        except Exception as e:
            print(f"❌ Error in sparse retrieval: {e}")
            return []
    
    def rerank_documents(
        self, 
        query: str, 
        documents: List[Document], 
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents using CrossEncoder
        Returns documents with scores
        """
        if not documents:
            return []
        
        if not self.reranker:
            # Return documents without reranking
            return [
                {
                    "document": doc,
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": 0.5  # Default score
                }
                for doc in documents[:top_k or settings.TOP_K_FINAL]
            ]
        
        top_k = top_k or settings.TOP_K_FINAL
        
        try:
            # Prepare pairs for reranking
            pairs = [[query, doc.page_content] for doc in documents]
            
            # Get reranking scores
            scores = self.reranker.predict(pairs)
            
            # Combine documents with scores
            doc_scores = [
                {
                    "document": doc,
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score)
                }
                for doc, score in zip(documents, scores)
            ]
            
            # Sort by score (descending)
            doc_scores.sort(key=lambda x: x["score"], reverse=True)
            
            return doc_scores[:top_k]
        
        except Exception as e:
            print(f"❌ Error in reranking: {e}")
            return [
                {
                    "document": doc,
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": 0.5
                }
                for doc in documents[:top_k]
            ]
    
    def hybrid_search(
        self, 
        query: str, 
        alpha: float = None
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search: combines dense + sparse + reranking
        
        Args:
            query: Search query
            alpha: Weight for dense vs sparse (None = use config weights)
        
        Returns:
            Top-k reranked documents with scores
        """
        # Use default weights if not specified
        alpha = alpha if alpha is not None else settings.DENSE_WEIGHT
        
        # Step 1: Get candidates from both retrievers
        dense_docs = self.dense_retrieval(query)
        sparse_docs = self.sparse_retrieval(query)
        
        # Step 2: Combine and deduplicate
        all_docs = []
        seen_content = set()
        
        for doc in dense_docs + sparse_docs:
            content_hash = hash(doc.page_content)
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                all_docs.append(doc)
        
        print(f"📊 Retrieved {len(dense_docs)} dense + {len(sparse_docs)} sparse = {len(all_docs)} unique docs")
        
        # Step 3: Rerank all candidates
        reranked = self.rerank_documents(query, all_docs)
        
        print(f"🎯 Reranked to top {len(reranked)} documents")
        
        return reranked
    
    def retrieve_with_metadata_filter(
        self, 
        query: str, 
        filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Retrieval with metadata filtering
        Example: filters={"source": "langchain", "doc_type": "tutorial"}
        """
        if not self.vector_store:
            return []
        
        try:
            # Dense retrieval with filter
            results = self.vector_store.similarity_search(
                query, 
                k=settings.TOP_K_RETRIEVAL,
                filter=filters
            )
            
            # Rerank
            reranked = self.rerank_documents(query, results)
            
            return reranked
        
        except Exception as e:
            print(f"❌ Error in filtered retrieval: {e}")
            return []


class QueryExpander:
    """
    Query expansion techniques:
    - HyDE (Hypothetical Document Embeddings)
    - Query Decomposition
    """
    
    def __init__(self, llm):
        self.llm = llm
    
    def hyde_expansion(self, query: str) -> str:
        """
        HyDE: Generate hypothetical document that would answer the query
        Then use that document for retrieval
        """
        prompt = f"""Given the question: "{query}"

Generate a detailed, hypothetical answer that would perfectly address this question.
This answer will be used to find similar real documents.

Write the hypothetical answer in a technical, documentation style.

Hypothetical Answer:"""
        
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            print(f"❌ Error in HyDE expansion: {e}")
            return query  # Fall back to original query
    
    def decompose_query(self, query: str) -> List[str]:
        """
        Decompose complex query into simpler sub-queries
        """
        prompt = f"""Break down this complex question into 2-3 simpler, specific sub-questions.
Each sub-question should be answerable independently.

Question: {query}

Sub-questions (one per line, without numbering):"""
        
        try:
            response = self.llm.invoke(prompt)
            sub_queries = [
                q.strip().lstrip('0123456789.-) ') 
                for q in response.content.split('\n') 
                if q.strip()
            ]
            return sub_queries[:3]  # Max 3 sub-queries
        except Exception as e:
            print(f"❌ Error in query decomposition: {e}")
            return [query]  # Fall back to original query


# Example usage and testing
if __name__ == "__main__":
    from langchain_core.documents import Document
    
    # Initialize retriever
    retriever = HybridRetriever()
    
    # Example documents
    sample_docs = [
        Document(
            page_content="LangChain is a framework for developing applications powered by language models.",
            metadata={"source": "langchain", "doc_type": "intro"}
        ),
        Document(
            page_content="LangGraph allows you to build stateful, multi-actor applications with LLMs.",
            metadata={"source": "langgraph", "doc_type": "intro"}
        ),
        Document(
            page_content="RAG combines retrieval with generation for knowledge-grounded responses.",
            metadata={"source": "general", "doc_type": "concept"}
        ),
    ]
    
    # Create vector store
    print("\n" + "="*70)
    print("Testing Hybrid Retriever")
    print("="*70 + "\n")
    
    retriever.create_vector_store(sample_docs)
    
    # Search
    query = "How to build agents with LangChain?"
    results = retriever.hybrid_search(query)
    
    print(f"\nQuery: {query}\n")
    print("Results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.3f}")
        print(f"   Content: {result['content'][:100]}...")
        print(f"   Metadata: {result['metadata']}")