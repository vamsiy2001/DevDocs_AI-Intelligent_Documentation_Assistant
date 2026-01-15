"""
Ingest Enhanced Multi-Framework Documentation
Loads comprehensive docs about LangChain, LangGraph, FastAPI, Gradio, RAG, etc.
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.enhanced_document_loader import EnhancedDocumentLoader
from src.ingestion.chunking import DocumentChunker
from src.retrieval import HybridRetriever
from src.config import settings


def main():
    """Load and ingest enhanced documentation"""
    
    print("\n" + "="*70)
    print("📚 ENHANCED DOCUMENTATION INGESTION")
    print("="*70)
    print("\nThis will load comprehensive documentation covering:")
    print("  • LangChain (overview, chains, agents, RAG)")
    print("  • LangGraph (stateful workflows, routing)")
    print("  • Vector Databases (Chroma, Pinecone, Qdrant, etc.)")
    print("  • Embeddings & Semantic Search")
    print("  • RAG Evaluation (RAGAS metrics)")
    print("  • Advanced RAG Techniques")
    print("  • FastAPI")
    print("  • Gradio")
    print("  • Prompt Engineering")
    print()
    
    # 1. Load enhanced documents
    print("📂 Loading enhanced documentation...")
    loader = EnhancedDocumentLoader()
    documents = loader.load_comprehensive_docs()
    
    print(f"\n✅ Loaded {len(documents)} documents")
    
    # Show document breakdown
    frameworks = {}
    for doc in documents:
        fw = doc.metadata.get('framework', 'unknown')
        frameworks[fw] = frameworks.get(fw, 0) + 1
    
    print("\n📊 Document breakdown by framework:")
    for fw, count in frameworks.items():
        print(f"   • {fw}: {count} documents")
    
    # 2. Chunk documents
    print(f"\n✂️  Chunking documents...")
    chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.recursive_character_split(documents)
    
    # Add source framework to chunk metadata
    print(f"✅ Created {len(chunks)} chunks")
    
    # 3. Create vector store
    print(f"\n💾 Creating vector store...")
    print(f"   This may take 2-3 minutes for embeddings...")
    
    retriever = HybridRetriever()
    success = retriever.create_vector_store(chunks)
    
    if success:
        print(f"\n{'='*70}")
        print("✅ ENHANCED DOCUMENTATION INGESTED SUCCESSFULLY!")
        print('='*70)
        
        print(f"\n📊 Summary:")
        print(f"   • Source Documents: {len(documents)}")
        print(f"   • Chunks Created: {len(chunks)}")
        print(f"   • Frameworks: {', '.join(frameworks.keys())}")
        print(f"   • Vector Store: {settings.CHROMA_PATH}")
        
        print(f"\n🎯 Your RAG system can now answer questions about:")
        print(f"   • LangChain (chains, agents, memory)")
        print(f"   • LangGraph (workflows, routing)")
        print(f"   • RAG systems (evaluation, techniques)")
        print(f"   • Vector databases (Chroma, Pinecone, etc.)")
        print(f"   • Embeddings and semantic search")
        print(f"   • FastAPI and Gradio")
        print(f"   • Prompt engineering")
        
        print()
        
        return 0
    else:
        print("\n❌ Failed to create vector store")
        return 1


if __name__ == "__main__":
    try:
        exit(main())
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)