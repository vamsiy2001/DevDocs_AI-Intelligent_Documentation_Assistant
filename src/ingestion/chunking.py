"""
Advanced Chunking Strategies
- Recursive character splitting
- Semantic chunking
- Parent-child chunking
"""

from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings


class DocumentChunker:
    """Advanced document chunking strategies"""
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        
        print(f"✂️  Document Chunker initialized")
        print(f"   Chunk size: {self.chunk_size}")
        print(f"   Overlap: {self.chunk_overlap}")
    
    def recursive_character_split(
        self, 
        documents: List[Document]
    ) -> List[Document]:
        """
        Recursive character text splitting
        Best for general text
        """
        print(f"✂️  Chunking {len(documents)} documents...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
        
        chunks = text_splitter.split_documents(documents)
        
        print(f"✅ Created {len(chunks)} chunks")
        
        return chunks
    
    def parent_child_split(
        self, 
        documents: List[Document]
    ) -> tuple[List[Document], List[Document]]:
        """
        Parent-child chunking strategy
        Creates large parent chunks and smaller child chunks
        
        Returns:
            (parent_chunks, child_chunks)
        """
        print(f"✂️  Creating parent-child chunks...")
        
        # Parent chunks (larger)
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.PARENT_CHUNK_SIZE,
            chunk_overlap=self.chunk_overlap * 2,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Child chunks (smaller)
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        parent_chunks = parent_splitter.split_documents(documents)
        
        # Create child chunks for each parent
        all_child_chunks = []
        
        for i, parent_doc in enumerate(parent_chunks):
            # Split parent into children
            child_docs = child_splitter.split_documents([parent_doc])
            
            # Add parent reference to children
            for j, child_doc in enumerate(child_docs):
                child_doc.metadata["parent_id"] = f"parent_{i}"
                child_doc.metadata["child_id"] = f"child_{i}_{j}"
                child_doc.metadata["parent_content"] = parent_doc.page_content[:500]  # Store snippet
            
            all_child_chunks.extend(child_docs)
        
        print(f"✅ Created {len(parent_chunks)} parent chunks")
        print(f"✅ Created {len(all_child_chunks)} child chunks")
        
        return parent_chunks, all_child_chunks
    
    def semantic_split(
        self, 
        documents: List[Document]
    ) -> List[Document]:
        """
        Semantic chunking based on content structure
        Tries to keep related content together
        """
        print(f"✂️  Semantic chunking {len(documents)} documents...")
        
        # Custom separators that respect document structure
        semantic_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n\n\n",  # Major sections
                "\n\n",    # Paragraphs
                "\n",      # Lines
                ". ",      # Sentences
                " "        # Words
            ],
            length_function=len
        )
        
        chunks = semantic_splitter.split_documents(documents)
        
        print(f"✅ Created {len(chunks)} semantic chunks")
        
        return chunks
    
    def add_metadata(
        self, 
        chunks: List[Document],
        metadata: dict = None
    ) -> List[Document]:
        """
        Add or update metadata for all chunks
        """
        if not metadata:
            return chunks
        
        for chunk in chunks:
            chunk.metadata.update(metadata)
        
        return chunks
    
    def filter_by_length(
        self, 
        chunks: List[Document],
        min_length: int = 50,
        max_length: int = None
    ) -> List[Document]:
        """
        Filter chunks by length
        Removes very short or very long chunks
        """
        max_length = max_length or self.chunk_size * 2
        
        filtered = [
            chunk for chunk in chunks
            if min_length <= len(chunk.page_content) <= max_length
        ]
        
        removed = len(chunks) - len(filtered)
        if removed > 0:
            print(f"⚠️  Filtered out {removed} chunks (too short/long)")
        
        return filtered


class CodeChunker:
    """Specialized chunker for code files"""
    
    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size
    
    def chunk_code(
        self, 
        documents: List[Document]
    ) -> List[Document]:
        """
        Chunk code files intelligently
        Tries to keep functions/classes together
        """
        print(f"💻 Chunking code documents...")
        
        # Code-specific separators
        code_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=50,
            separators=[
                "\n\nclass ",   # Class definitions
                "\n\ndef ",     # Function definitions
                "\n\n",         # Blank lines
                "\n",           # Lines
                " "             # Words
            ],
            length_function=len
        )
        
        chunks = code_splitter.split_documents(documents)
        
        print(f"✅ Created {len(chunks)} code chunks")
        
        return chunks


# Example usage
if __name__ == "__main__":
    from document_loader import DocumentLoader
    
    # Load sample docs
    loader = DocumentLoader()
    docs = loader.load_sample_docs()
    
    # Test chunking
    chunker = DocumentChunker(chunk_size=300, chunk_overlap=30)
    
    # Recursive split
    chunks = chunker.recursive_character_split(docs)
    
    print(f"\nExample chunk:")
    print(f"Content: {chunks[0].page_content[:200]}...")
    print(f"Metadata: {chunks[0].metadata}")
    
    # Parent-child split
    parents, children = chunker.parent_child_split(docs)
    
    print(f"\nParent-child example:")
    print(f"Child metadata: {children[0].metadata}")