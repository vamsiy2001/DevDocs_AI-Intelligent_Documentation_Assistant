"""
Document Loaders for Various Sources
- Local files (txt, md, pdf)
- Web pages
- GitHub repositories
"""

from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
    WebBaseLoader,
)
import os
from pathlib import Path


class DocumentLoader:
    """Load documents from various sources"""
    
    def __init__(self):
        print("📚 Document Loader initialized")
    
    def load_from_directory(
        self, 
        directory: str, 
        glob_pattern: str = "**/*.md"
    ) -> List[Document]:
        """
        Load documents from a directory
        
        Args:
            directory: Path to directory
            glob_pattern: File pattern (e.g., "**/*.md", "**/*.txt")
        
        Returns:
            List of Document objects
        """
        print(f"📂 Loading documents from {directory}")
        print(f"   Pattern: {glob_pattern}")
        
        try:
            loader = DirectoryLoader(
                directory,
                glob=glob_pattern,
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"},
                show_progress=True
            )
            
            documents = loader.load()
            
            print(f"✅ Loaded {len(documents)} documents")
            
            return documents
        
        except Exception as e:
            print(f"❌ Error loading directory: {e}")
            return []
    
    def load_from_files(self, file_paths: List[str]) -> List[Document]:
        """
        Load specific files
        
        Args:
            file_paths: List of file paths
        
        Returns:
            List of Document objects
        """
        documents = []
        
        print(f"📄 Loading {len(file_paths)} files...")
        
        for file_path in file_paths:
            try:
                if not os.path.exists(file_path):
                    print(f"⚠️  File not found: {file_path}")
                    continue
                
                # Determine loader based on extension
                ext = Path(file_path).suffix.lower()
                
                if ext in ['.txt', '.md', '.markdown']:
                    loader = TextLoader(file_path, encoding='utf-8')
                    docs = loader.load()
                    documents.extend(docs)
                    print(f"   ✅ {file_path}")
                else:
                    print(f"   ⏭️  Unsupported file type: {file_path}")
            
            except Exception as e:
                print(f"   ❌ Error loading {file_path}: {e}")
        
        print(f"✅ Loaded {len(documents)} documents total")
        
        return documents
    
    def load_from_urls(self, urls: List[str]) -> List[Document]:
        """
        Load documents from web URLs
        
        Args:
            urls: List of URLs to scrape
        
        Returns:
            List of Document objects
        """
        print(f"🌐 Loading {len(urls)} URLs...")
        
        documents = []
        
        for url in urls:
            try:
                loader = WebBaseLoader(url)
                docs = loader.load()
                documents.extend(docs)
                print(f"   ✅ {url}")
            
            except Exception as e:
                print(f"   ❌ Error loading {url}: {e}")
        
        print(f"✅ Loaded {len(documents)} documents from URLs")
        
        return documents
    
    def load_sample_docs(self) -> List[Document]:
        """
        Load sample documentation for testing
        """
        print("📚 Loading sample documentation...")
        
        sample_docs = [
            Document(
                page_content="""# LangChain Introduction

LangChain is a framework for developing applications powered by language models. 
It enables applications that are:
- Context-aware: connect a language model to sources of context
- Reason: rely on a language model to reason about how to answer based on provided context

The main value props of LangChain are:
1. Components: abstractions for working with language models
2. Off-the-shelf chains: built-in assemblages of components""",
                metadata={"source": "langchain", "topic": "introduction", "type": "documentation"}
            ),
            Document(
                page_content="""# Creating Chains in LangChain

Chains allow you to combine multiple components together to create a single, coherent application.

Example:
```python
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
model = ChatOpenAI()

chain = prompt | model
result = chain.invoke({"topic": "programming"})
```

This uses LCEL (LangChain Expression Language) to chain components.""",
                metadata={"source": "langchain", "topic": "chains", "type": "tutorial"}
            ),
            Document(
                page_content="""# LangChain Agents

Agents use an LLM to determine which actions to take and in what order.
Unlike chains, where the sequence is hardcoded, agents use a language model as a reasoning engine.

Key components:
- Agent: The core decision-making unit
- Tools: Functions the agent can use
- Executor: Runs the agent loop

Example tools: web search, calculator, database queries, API calls""",
                metadata={"source": "langchain", "topic": "agents", "type": "documentation"}
            ),
            Document(
                page_content="""# Retrieval Augmented Generation (RAG)

RAG is a technique for augmenting LLM knowledge with additional data.

Steps:
1. Load: Get your data (documents, web pages, etc.)
2. Split: Break documents into chunks
3. Store: Embed and store chunks in a vector database
4. Retrieve: Find relevant chunks for a query
5. Generate: Pass chunks to LLM as context

This grounds responses in your specific data.""",
                metadata={"source": "langchain", "topic": "rag", "type": "concept"}
            ),
            Document(
                page_content="""# Vector Databases

Vector databases store embeddings and enable similarity search.

Popular options:
- Chroma: Local, easy to use, good for development
- Pinecone: Cloud-based, scalable
- Weaviate: Open-source, feature-rich
- Qdrant: Fast, Rust-based
- FAISS: Facebook's library, efficient

They enable semantic search by comparing vector representations.""",
                metadata={"source": "general", "topic": "vector-databases", "type": "overview"}
            ),
        ]
        
        print(f"✅ Loaded {len(sample_docs)} sample documents")
        
        return sample_docs


# Example usage
if __name__ == "__main__":
    loader = DocumentLoader()
    
    # Test loading sample docs
    docs = loader.load_sample_docs()
    
    print(f"\nExample document:")
    print(f"Content: {docs[0].page_content[:200]}...")
    print(f"Metadata: {docs[0].metadata}")