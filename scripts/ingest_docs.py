import argparse
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.document_loader import DocumentLoader
from src.ingestion.chunking import DocumentChunker
from src.retrieval import HybridRetriever
from src.config import settings


def ingest_from_directory(directory: str, pattern: str = "**/*.md"):
    """Ingest documents from a directory"""
    print("\n" + "="*70)
    print("📚 DOCUMENT INGESTION FROM DIRECTORY")
    print("="*70)
    print(f"Source: {directory}")
    print(f"Pattern: {pattern}")
    print()
    
    # 1. Load documents
    loader = DocumentLoader()
    documents = loader.load_from_directory(directory, pattern)
    
    if not documents:
        print("❌ No documents found!")
        return False
    
    print(f"\n✅ Loaded {len(documents)} documents")
    
    # 2. Chunk documents
    print(f"\n✂️  Chunking documents...")
    chunker = DocumentChunker()
    chunks = chunker.recursive_character_split(documents)
    
    # Filter out very short chunks
    chunks = chunker.filter_by_length(chunks, min_length=50)
    
    print(f"✅ Created {len(chunks)} chunks")
    
    # 3. Create vector store
    print(f"\n💾 Creating vector store...")
    retriever = HybridRetriever()
    success = retriever.create_vector_store(chunks)
    
    if success:
        print(f"\n✅ Ingestion complete!")
        print(f"   Documents: {len(documents)}")
        print(f"   Chunks: {len(chunks)}")
        print(f"   Vector store: {settings.CHROMA_PATH}")
        return True
    else:
        print("\n❌ Failed to create vector store")
        return False


def ingest_sample_data():
    """Ingest sample documentation"""
    print("\n" + "="*70)
    print("📚 DOCUMENT INGESTION - SAMPLE DATA")
    print("="*70)
    print()
    
    # 1. Load sample documents
    loader = DocumentLoader()
    documents = loader.load_sample_docs()
    
    print(f"✅ Loaded {len(documents)} sample documents")
    
    # 2. Chunk documents
    print(f"\n✂️  Chunking documents...")
    chunker = DocumentChunker(chunk_size=400, chunk_overlap=50)
    chunks = chunker.recursive_character_split(documents)
    
    print(f"✅ Created {len(chunks)} chunks")
    
    # 3. Create vector store
    print(f"\n💾 Creating vector store...")
    retriever = HybridRetriever()
    success = retriever.create_vector_store(chunks)
    
    if success:
        print(f"\n✅ Sample data ingestion complete!")
        print(f"   You can now run the app with: python app/gradio_app.py")
        return True
    else:
        print("\n❌ Failed to create vector store")
        return False


def ingest_from_files(file_paths: list):
    """Ingest specific files"""
    print("\n" + "="*70)
    print("📚 DOCUMENT INGESTION FROM FILES")
    print("="*70)
    print(f"Files: {len(file_paths)}")
    print()
    
    # 1. Load documents
    loader = DocumentLoader()
    documents = loader.load_from_files(file_paths)
    
    if not documents:
        print("❌ No documents loaded!")
        return False
    
    print(f"\n✅ Loaded {len(documents)} documents")
    
    # 2. Chunk documents
    print(f"\n✂️  Chunking documents...")
    chunker = DocumentChunker()
    chunks = chunker.recursive_character_split(documents)
    chunks = chunker.filter_by_length(chunks, min_length=50)
    
    print(f"✅ Created {len(chunks)} chunks")
    
    # 3. Create vector store
    print(f"\n💾 Creating vector store...")
    retriever = HybridRetriever()
    success = retriever.create_vector_store(chunks)
    
    if success:
        print(f"\n✅ Ingestion complete!")
        return True
    else:
        print("\n❌ Failed to create vector store")
        return False


def main():
    """Main ingestion function"""
    parser = argparse.ArgumentParser(
        description="Ingest documents into vector store"
    )
    
    parser.add_argument(
        "--source",
        type=str,
        default="sample",
        choices=["sample", "directory", "files"],
        help="Source type: sample, directory, or files"
    )
    
    parser.add_argument(
        "--path",
        type=str,
        default="data/raw",
        help="Path to directory or file(s)"
    )
    
    parser.add_argument(
        "--pattern",
        type=str,
        default="**/*.md",
        help="File pattern for directory source (e.g., **/*.md, **/*.txt)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🚀 DevDocs AI - Document Ingestion")
    print("="*70)
    
    try:
        if args.source == "sample":
            success = ingest_sample_data()
        
        elif args.source == "directory":
            if not os.path.exists(args.path):
                print(f"❌ Directory not found: {args.path}")
                return 1
            success = ingest_from_directory(args.path, args.pattern)
        
        elif args.source == "files":
            files = args.path.split(",")
            files = [f.strip() for f in files]
            success = ingest_from_files(files)
        
        if success:
            print("\n" + "="*70)
            print("✅ INGESTION SUCCESSFUL")
            print("="*70)
            print()
            return 0
        else:
            print("\n❌ Ingestion failed")
            return 1
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())