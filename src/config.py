"""
Configuration Management for DevDocs AI
All settings using FREE tools
"""

from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings using free tier services"""
    
 
    # LLM Configuration (Groq - FREE)
    GROQ_API_KEY: str
    LLM_MODEL: str = "llama-3.3-70b-versatile"
    LLM_TEMPERATURE: float = 0.1
    MAX_TOKENS: int = 1024
 
    # Embeddings Configuration (Local - FREE)
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    EMBEDDING_DEVICE: str = "cpu"  # Use "cuda" if GPU available
    
    # Vector Store Configuration
    VECTOR_DB: str = "chroma"  # or "qdrant"
    CHROMA_PATH: str = "./data/chroma_db"
    COLLECTION_NAME: str = "devdocs"
 
    # Reranker Configuration (Local - FREE)
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    USE_RERANKING: bool = True
    
 
    # Retrieval Configuration
    TOP_K_RETRIEVAL: int = 20
    TOP_K_FINAL: int = 5       # Final results after reranking
    
    # BM25 Parameters
    BM25_K1: float = 1.5
    BM25_B: float = 0.75
    
    # Hybrid Search Weights
    DENSE_WEIGHT: float = 0.5   # Weight for dense retrieval
    SPARSE_WEIGHT: float = 0.5  # Weight for sparse (BM25) retrieval
    
    # Chunking Configuration
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    PARENT_CHUNK_SIZE: int = 2048  # For parent-child strategy
    
 
    # Web Search (Tavily - 1000 free searches/month)
    TAVILY_API_KEY: Optional[str] = None
    
 
    # Monitoring (LangSmith - 5K traces free/month)
    LANGSMITH_API_KEY: Optional[str] = None
    LANGCHAIN_TRACING_V2: bool = True
    LANGCHAIN_PROJECT: str = "devdocs-ai"
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"
    
 
    # Cache Configuration
    ENABLE_CACHE: bool = True
    CACHE_DIR: str = "./data/cache"
    CACHE_TTL: int = 3600  # 1 hour in seconds
    
 
    # Evaluation Configuration
    EVAL_DATASET_PATH: str = "./data/evaluation/test_dataset.json"
    EVAL_RESULTS_PATH: str = "./data/evaluation/results"
    
 
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = True
    
 
    # Gradio Configuration
 
    GRADIO_SHARE: bool = False
    GRADIO_SERVER_NAME: str = "0.0.0.0"
    GRADIO_SERVER_PORT: int = 7860
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"  # Allow extra fields


# Global Settings Instance
settings = Settings()


# Model Configurations
EMBEDDING_CONFIG = {
    "model_name": settings.EMBEDDING_MODEL,
    "model_kwargs": {"device": settings.EMBEDDING_DEVICE},
    "encode_kwargs": {
        "normalize_embeddings": True
    }
}

LLM_CONFIG = {
    "model": settings.LLM_MODEL,
    "temperature": settings.LLM_TEMPERATURE,
    "max_tokens": settings.MAX_TOKENS,
}


# Validation

def validate_config():
    """Validate configuration"""
    errors = []
    
    # Check required API keys
    if not settings.GROQ_API_KEY or settings.GROQ_API_KEY == "your_groq_key_here":
        errors.append("GROQ_API_KEY is not set. Get it from https://console.groq.com")
    
    # Check directories exist
    for directory in [settings.CACHE_DIR, settings.EVAL_RESULTS_PATH]:
        os.makedirs(directory, exist_ok=True)
    
    if errors:
        print("❌ Configuration Errors:")
        for error in errors:
            print(f"   • {error}")
        return False
    
    return True


def print_config():
    """Print current configuration (for debugging)"""
    print("\n" + "="*70)
    print("⚙️  CONFIGURATION")
    print("="*70)
    print(f"\n🤖 LLM:")
    print(f"   Model: {settings.LLM_MODEL}")
    print(f"   Temperature: {settings.LLM_TEMPERATURE}")
    print(f"   Max Tokens: {settings.MAX_TOKENS}")
    
    print(f"\n📊 Embeddings:")
    print(f"   Model: {settings.EMBEDDING_MODEL}")
    print(f"   Dimension: {settings.EMBEDDING_DIMENSION}")
    print(f"   Device: {settings.EMBEDDING_DEVICE}")
    
    print(f"\n💾 Vector Store:")
    print(f"   Type: {settings.VECTOR_DB}")
    print(f"   Path: {settings.CHROMA_PATH}")
    
    print(f"\n🎯 Retrieval:")
    print(f"   Initial K: {settings.TOP_K_RETRIEVAL}")
    print(f"   Final K: {settings.TOP_K_FINAL}")
    print(f"   Reranking: {'✅ Enabled' if settings.USE_RERANKING else '❌ Disabled'}")
    
    print(f"\n📈 Monitoring:")
    if settings.LANGSMITH_API_KEY:
        print(f"   LangSmith: ✅ Enabled")
        print(f"   Project: {settings.LANGCHAIN_PROJECT}")
    else:
        print(f"   LangSmith: ⏭️  Disabled (optional)")
    
    print("\n" + "="*70 + "\n")


# Startup

if __name__ == "__main__":
    # Validate and print config
    if validate_config():
        print("✅ Configuration is valid!")
        print_config()
    else:
        print("\n❌ Please fix configuration errors above")
        exit(1)
else:
    # Auto-validate on import
    if not validate_config():
        print("⚠️  Warning: Configuration has errors. Some features may not work.")