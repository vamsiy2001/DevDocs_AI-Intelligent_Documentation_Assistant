"""
QUICK START - Minimal RAG System
Run this to verify your setup works!
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma  # UPDATED import
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from typing import List, Dict

# Load environment
load_dotenv()

# Verify API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_key_here":
    print("❌ ERROR: GROQ_API_KEY not set!")
    print("📝 Steps to fix:")
    print("   1. Copy .env.example to .env")
    print("   2. Get free API key from https://console.groq.com")
    print("   3. Add it to .env file")
    exit(1)


class MinimalRAG:
    """
    Minimal RAG system, fully functional!
    Perfect for testing your setup
    """
    
    def __init__(self):
        print("🚀 Initializing Minimal RAG...")
        print()
        
        # 1. Embeddings (local, free)
        print("📊 Loading embedding model (this may take a minute first time)...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # 2. LLM (Groq, free)
        print("🤖 Connecting to Groq API...")
        self.llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="llama-3.3-70b-versatile",  # UPDATED model
            temperature=0.1
        )
        
        # 3. Reranker (local, free)
        print("🎯 Loading reranker model...")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        # 4. Vector store (will be created)
        self.vector_store = None
        
        print("✅ Ready!\n")
    
    def load_sample_docs(self) -> List[Document]:
        """Load sample documentation about LangChain"""
        
        print("📚 Loading sample documentation...")
        
        docs = [
            Document(
                page_content="LangChain is a framework for developing applications powered by language models. It enables applications that are context-aware and can reason about their context to take actions.",
                metadata={"source": "langchain", "topic": "introduction", "doc_id": "1"}
            ),
            Document(
                page_content="To create a simple chain in LangChain, you combine prompts and models using the pipe operator. For example: chain = prompt | model | output_parser. This creates a sequence that flows from prompt to model to parser.",
                metadata={"source": "langchain", "topic": "chains", "doc_id": "2"}
            ),
            Document(
                page_content="LangChain agents use a language model to choose a sequence of actions to take. Unlike chains, which hardcode the sequence in code, agents use a language model as a reasoning engine to determine which actions to take and in which order.",
                metadata={"source": "langchain", "topic": "agents", "doc_id": "3"}
            ),
            Document(
                page_content="RAG (Retrieval Augmented Generation) is a technique that combines retrieval with generation. First, you retrieve relevant documents from a knowledge base, then you pass those documents to an LLM as context to generate a response.",
                metadata={"source": "langchain", "topic": "rag", "doc_id": "4"}
            ),
            Document(
                page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs. It extends LangChain Expression Language with the ability to coordinate multiple chains (or actors) across multiple steps of computation in a cyclic manner.",
                metadata={"source": "langgraph", "topic": "introduction", "doc_id": "5"}
            ),
            Document(
                page_content="Vector databases store embeddings and allow similarity search. Popular options include Chroma (local), Pinecone (cloud), Weaviate (open source), and Qdrant (cloud/local). They enable semantic search by comparing vector representations.",
                metadata={"source": "general", "topic": "vector-databases", "doc_id": "6"}
            ),
            Document(
                page_content="Embeddings are vector representations of text. They capture semantic meaning - similar texts have similar embeddings. Common models include OpenAI's text-embedding-ada-002, sentence-transformers, and Cohere embeddings.",
                metadata={"source": "general", "topic": "embeddings", "doc_id": "7"}
            ),
            Document(
                page_content="Prompt engineering is crucial for getting good results from LLMs. Key techniques include: being specific, providing examples (few-shot learning), using chain-of-thought prompting, and iterating based on outputs.",
                metadata={"source": "general", "topic": "prompts", "doc_id": "8"}
            ),
        ]
        
        print(f"✅ Loaded {len(docs)} sample documents\n")
        return docs
    
    def create_vector_store(self, documents: List[Document]):
        """Create vector store from documents"""
        print(f"💾 Creating vector store...")
        
        # Chunk documents (for longer docs)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        
        print(f"   Split into {len(chunks)} chunks")
        print(f"   Generating embeddings...")
        
        # Create vector store
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory="./minimal_chroma_db"
        )
        
        print("✅ Vector store created!\n")
    
    def retrieve_and_rerank(self, query: str, k: int = 3) -> List[Dict]:
        """Retrieve documents and rerank"""
        
        # 1. Retrieve candidates (get more than we need)
        candidates = self.vector_store.similarity_search(query, k=k*2)
        
        # 2. Rerank using CrossEncoder
        pairs = [[query, doc.page_content] for doc in candidates]
        scores = self.reranker.predict(pairs)
        
        # 3. Sort by score and return top k
        scored_docs = [
            {"document": doc, "score": float(score)}
            for doc, score in zip(candidates, scores)
        ]
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        
        return scored_docs[:k]
    
    def query(self, question: str, verbose: bool = True) -> Dict:
        """Query the RAG system"""
        
        if verbose:
            print(f"❓ Question: {question}")
            print()
        
        # 1. Retrieve relevant documents
        if verbose:
            print("🔍 Retrieving relevant documents...")
        
        retrieved = self.retrieve_and_rerank(question, k=3)
        
        if verbose:
            print(f"   Found {len(retrieved)} relevant documents")
            for i, item in enumerate(retrieved, 1):
                score = item["score"]
                topic = item["document"].metadata.get("topic", "unknown")
                print(f"   [{i}] Topic: {topic} (Relevance: {score:.2f})")
            print()
        
        # 2. Create context from retrieved docs
        context = "\n\n".join([
            f"[Document {i+1}]\n{item['document'].page_content}"
            for i, item in enumerate(retrieved)
        ])
        
        # 3. Create prompt
        prompt = f"""Use the following context to answer the question accurately and concisely. 
If the context doesn't contain enough information, say so clearly.

Context:
{context}

Question: {question}

Answer:"""
        
        # 4. Generate answer
        if verbose:
            print("✍️  Generating answer...")
            print()
        
        response = self.llm.invoke(prompt)
        
        return {
            "answer": response.content,
            "sources": [
                {
                    "content": item["document"].page_content,
                    "metadata": item["document"].metadata,
                    "score": item["score"]
                }
                for item in retrieved
            ],
            "num_sources": len(retrieved)
        }


def print_banner():
    """Print welcome banner"""
    banner = """
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║              🤖 DevDocs AI - QUICK START DEMO                  ║
║                                                                ║
║  This demonstrates a working RAG system with:                  ║
║  ✅ Dense retrieval (semantic search)                          ║
║  ✅ Reranking (improved accuracy)                              ║
║  ✅ Groq LLM (free & fast)                                     ║
║  ✅ Local vector store                                         ║
║                                                                ║
║  100% FREE - No paid APIs required!                            ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def run_demo():
    """Run the quick start demo"""
    
    print_banner()
    
    # Initialize RAG
    rag = MinimalRAG()
    
    # Load sample docs
    docs = rag.load_sample_docs()
    
    # Create vector store
    rag.create_vector_store(docs)
    
    # Test queries
    test_questions = [
        "What is LangChain?",
        "How do I create a chain in LangChain?",
        "What are agents and how do they differ from chains?",
        "Explain RAG in simple terms",
        "What is LangGraph used for?",
    ]
    
    print("="*70)
    print("🧪 TESTING RAG SYSTEM")
    print("="*70)
    print()
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*70}")
        print(f"Test {i}/{len(test_questions)}")
        print('='*70)
        
        result = rag.query(question, verbose=True)
        
        print("📝 ANSWER:")
        print("-"*70)
        print(result["answer"])
        print("-"*70)
        print(f"✅ Used {result['num_sources']} sources")
        print('='*70)
        
        if i < len(test_questions):
            input("\n⏸️  Press Enter for next question...\n")
    
    print("\n" + "="*70)
    print("✨ DEMO COMPLETE!")
    print("="*70)
    print("\n📊 Summary:")
    print(f"   ✅ Successfully answered {len(test_questions)} questions")
    print(f"   ✅ Used semantic search + reranking")
    print(f"   ✅ All tools were FREE")
    print(f"   ✅ Everything ran locally (except LLM)")
    print()


if __name__ == "__main__":
    try:
        run_demo()
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Make sure GROQ_API_KEY is set in .env")
        print("   2. Check your internet connection")
        print("   3. Verify all dependencies are installed: pip install -r requirements.txt")