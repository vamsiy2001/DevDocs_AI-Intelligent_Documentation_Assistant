import argparse
import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval import HybridRetriever
from src.agents import SimpleRAG
from src.evaluation import RAGEvaluator, CustomMetrics
from src.config import settings


def run_evaluation(config_name: str = "default", save_report: bool = True):
    """Run complete evaluation"""
    
    print("\n" + "="*70)
    print("📊 RAG EVALUATION")
    print("="*70)
    print(f"Configuration: {config_name}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 1. Initialize RAG system
    print("🚀 Initializing RAG system...")
    retriever = HybridRetriever()
    
    if not retriever.load_vector_store():
        print("\n❌ No vector store found!")
        print("💡 Run ingestion first: python scripts/ingest_docs.py")
        return False
    
    rag = SimpleRAG(retriever)
    print("✅ RAG system loaded")
    
    # 2. Initialize evaluator
    print("\n🔍 Initializing evaluator...")
    evaluator = RAGEvaluator()
    
    # 3. Create test dataset
    print("\n📝 Creating test dataset...")
    test_cases = evaluator.create_test_dataset()
    print(f"✅ Created {len(test_cases)} test cases")
    
    # 4. Prepare evaluation dataset
    print("\n🧪 Running RAG on test cases...")
    eval_dataset = evaluator.prepare_eval_dataset(rag, test_cases)
    
    # 5. Run RAGAS evaluation
    print("\n📊 Running RAGAS evaluation...")
    print("This may take a few minutes...")
    
    try:
        results = evaluator.evaluate_rag(eval_dataset)
        
        # 6. Generate report
        print("\n📄 Generating report...")
        report = evaluator.generate_report(results)
        
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        print(report)
        
        # 7. Save results
        if save_report:
            evaluator.save_results(results, config_name)
            
            # Save report as markdown
            report_path = f"{settings.EVAL_RESULTS_PATH}/report_{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            with open(report_path, 'w') as f:
                f.write(report)
            print(f"\n💾 Report saved to {report_path}")
        
        # 8. Custom metrics (latency)
        print("\n⚡ Measuring latency...")
        questions = [tc["question"] for tc in test_cases[:5]]  # Sample 5 questions
        latency_metrics = CustomMetrics.latency_metric(rag, questions)
        
        print(f"\nLatency Metrics:")
        print(f"   Average: {latency_metrics['avg_latency_ms']:.0f}ms")
        print(f"   P95: {latency_metrics['p95_latency_ms']:.0f}ms")
        
        print("\n" + "="*70)
        print("✅ EVALUATION COMPLETE")
        print("="*70)
        
        return True
    
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Make sure RAGAS is installed: pip install ragas")
        print("   2. Check your GROQ_API_KEY is valid")
        print("   3. Ensure vector store has data")
        import traceback
        traceback.print_exc()
        return False


def compare_configurations():
    """Compare multiple RAG configurations"""
    
    print("\n" + "="*70)
    print("🔬 CONFIGURATION COMPARISON")
    print("="*70)
    print("\nThis will evaluate multiple configurations and compare them.")
    print("Note: This may take 10-15 minutes.\n")
    
    configs = [
        {"name": "baseline", "description": "Default settings"},
        {"name": "high_k", "description": "Higher retrieval K"},
        {"name": "no_rerank", "description": "Without reranking"},
    ]
    
    results_summary = []
    
    for config in configs:
        print(f"\n{'='*70}")
        print(f"Testing: {config['name']} - {config['description']}")
        print('='*70)
        
        # Here you would modify RAG settings based on config
        # For now, just run with default
        success = run_evaluation(config['name'], save_report=True)
        
        if success:
            results_summary.append({
                "name": config['name'],
                "status": "✅ Success"
            })
        else:
            results_summary.append({
                "name": config['name'],
                "status": "❌ Failed"
            })
    
    # Print summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    for result in results_summary:
        print(f"{result['status']} - {result['name']}")
    
    print("\n💡 Check the results directory for detailed reports:")
    print(f"   {settings.EVAL_RESULTS_PATH}/")


def main():
    """Main evaluation function"""
    
    parser = argparse.ArgumentParser(
        description="Run RAG evaluation with RAGAS"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Configuration name"
    )
    
    parser.add_argument(
        "--save-report",
        action="store_true",
        default=True,
        help="Save evaluation report"
    )
    
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare multiple configurations"
    )
    
    args = parser.parse_args()
    
    try:
        if args.compare:
            compare_configurations()
        else:
            success = run_evaluation(args.config, args.save_report)
            return 0 if success else 1
    
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())