"""
Basic Import Tests
Verify that all modules can be imported correctly
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_config_import():
    """Test config module import"""
    try:
        from src import config
        from src.config import settings
        print("✅ Config import successful")
        print(f"   LLM Model: {settings.LLM_MODEL}")
        return True
    except Exception as e:
        print(f"❌ Config import failed: {e}")
        return False


def test_retrieval_imports():
    """Test retrieval module imports"""
    try:
        from src.retrieval import HybridRetriever, QueryExpander
        print("✅ Retrieval imports successful")
        return True
    except Exception as e:
        print(f"❌ Retrieval imports failed: {e}")
        return False


def test_agent_imports():
    """Test agent module imports"""
    try:
        from src.agents import SimpleRAG, RAGAgent
        print("✅ Agent imports successful")
        return True
    except Exception as e:
        print(f"❌ Agent imports failed: {e}")
        return False


def test_evaluation_imports():
    """Test evaluation module imports"""
    try:
        from src.evaluation import RAGEvaluator, CustomMetrics
        print("✅ Evaluation imports successful")
        return True
    except Exception as e:
        print(f"❌ Evaluation imports failed: {e}")
        return False


def test_ingestion_imports():
    """Test ingestion module imports"""
    try:
        from src.ingestion.document_loader import DocumentLoader
        from src.ingestion.chunking import DocumentChunker
        print("✅ Ingestion imports successful")
        return True
    except Exception as e:
        print(f"❌ Ingestion imports failed: {e}")
        return False


def test_external_dependencies():
    """Test external package imports"""
    dependencies = [
        ("langchain", "LangChain"),
        ("langchain_groq", "LangChain Groq"),
        ("langchain_huggingface", "LangChain HuggingFace"),
        ("chromadb", "ChromaDB"),
        ("sentence_transformers", "Sentence Transformers"),
        ("rank_bm25", "Rank BM25"),
        ("ragas", "RAGAS"),
        ("gradio", "Gradio"),
    ]
    
    results = []
    
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✅ {name} installed")
            results.append(True)
        except ImportError:
            print(f"❌ {name} not installed")
            results.append(False)
    
    return all(results)


def test_api_key():
    """Test API key configuration"""
    try:
        from src.config import settings
        
        if settings.GROQ_API_KEY and settings.GROQ_API_KEY != "your_groq_key_here":
            print("✅ GROQ_API_KEY is configured")
            return True
        else:
            print("⚠️  GROQ_API_KEY not configured")
            print("   Get your key from: https://console.groq.com")
            return False
    except Exception as e:
        print(f"❌ API key test failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    
    print("\n" + "="*70)
    print("🧪 RUNNING IMPORT TESTS")
    print("="*70)
    print()
    
    tests = [
        ("Config Module", test_config_import),
        ("Retrieval Module", test_retrieval_imports),
        ("Agent Module", test_agent_imports),
        ("Evaluation Module", test_evaluation_imports),
        ("Ingestion Module", test_ingestion_imports),
        ("External Dependencies", test_external_dependencies),
        ("API Key Configuration", test_api_key),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*70}")
        print(f"Testing: {test_name}")
        print('='*70)
        
        try:
            success = test_func()
            results[test_name] = success
        except Exception as e:
            print(f"❌ Test error: {e}")
            results[test_name] = False
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print("\n" + "="*70)
    print(f"Results: {passed}/{total} tests passed")
    print("="*70)
    
    if passed == total:
        print("\n🎉 All tests passed! Your setup is ready.")
        print("\n🎯 Next steps:")
        print("   1. Run quickstart: python quickstart.py")
        print("   2. Ingest documents: python scripts/ingest_docs.py")
        print("   3. Launch app: python app/gradio_app.py")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        print("\n💡 Common fixes:")
        print("   - Missing packages: pip install -r requirements.txt")
        print("   - Missing directories: python setup_project.py")
        print("   - Missing API key: Check .env file")
        return 1


if __name__ == "__main__":
    exit(run_all_tests())