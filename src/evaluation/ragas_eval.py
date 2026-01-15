"""
RAG Evaluation Framework using RAGAS
Measures: Faithfulness, Answer Relevancy, Context Precision, Context Recall
"""

from typing import List, Dict, Any
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
import json
import pandas as pd
from datetime import datetime
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings, EMBEDDING_CONFIG


class RAGEvaluator:
    """
    Comprehensive RAG evaluation using RAGAS
    All free tools!
    """
    
    def __init__(self):
        print("🔍 Initializing RAG Evaluator...")
        
        # LLM for evaluation (Groq - FREE)
        self.llm = ChatGroq(
            groq_api_key=settings.GROQ_API_KEY,
            model_name=settings.LLM_MODEL,
            temperature=0
        )
        
        # Embeddings for evaluation
        self.embeddings = HuggingFaceEmbeddings(**EMBEDDING_CONFIG)
        
        # RAGAS metrics
        self.metrics = [
            faithfulness,           # Is answer faithful to context?
            answer_relevancy,       # Is answer relevant to question?
            context_precision,      # Are retrieved contexts relevant?
            context_recall,         # Did we retrieve all relevant info?
        ]
        
        print("✅ Evaluator ready!")
    
    def create_test_dataset(self) -> List[Dict[str, Any]]:
        """
        Create a test dataset with golden QA pairs
        This is CRITICAL for evaluation
        """
        test_cases = [
            {
                "question": "What is LangChain and what is it used for?",
                "ground_truth": "LangChain is a framework for developing applications powered by language models. It's used to build context-aware applications that can reason and take actions.",
                "context_needed": True
            },
            {
                "question": "How do you create a simple chain in LangChain?",
                "ground_truth": "Create a chain by combining prompts with LLMs using the pipe operator (|) or LCEL (LangChain Expression Language). Example: chain = prompt | llm | output_parser",
                "context_needed": True
            },
            {
                "question": "What is the difference between agents and chains?",
                "ground_truth": "Chains follow a predetermined sequence of steps, while agents dynamically decide which tools to use and in what order based on the input and intermediate results.",
                "context_needed": True
            },
            {
                "question": "How to implement RAG in LangChain?",
                "ground_truth": "Implement RAG by: 1) Loading documents, 2) Splitting into chunks, 3) Creating embeddings, 4) Storing in vector DB, 5) Retrieving relevant chunks, 6) Passing to LLM with prompt.",
                "context_needed": True
            },
            {
                "question": "What vector databases work with LangChain?",
                "ground_truth": "LangChain supports many vector databases including Chroma, Pinecone, Weaviate, Qdrant, FAISS, and Milvus.",
                "context_needed": True
            },
            {
                "question": "What is LangGraph?",
                "ground_truth": "LangGraph is a library for building stateful, multi-actor applications with LLMs. It extends LangChain with graph-based workflows.",
                "context_needed": True
            },
            {
                "question": "How do embeddings work?",
                "ground_truth": "Embeddings are vector representations of text that capture semantic meaning. Similar texts have similar embeddings, enabling semantic search.",
                "context_needed": True
            },
            {
                "question": "What is prompt engineering?",
                "ground_truth": "Prompt engineering is the practice of designing effective prompts for LLMs. Key techniques include being specific, providing examples, and using chain-of-thought prompting.",
                "context_needed": True
            },
        ]
        
        return test_cases
    
    def prepare_eval_dataset(
        self, 
        rag_system, 
        test_cases: List[Dict[str, Any]]
    ) -> Dataset:
        """
        Run RAG system on test cases and prepare for RAGAS evaluation
        
        Returns Dataset with columns:
        - question
        - answer (generated)
        - contexts (retrieved)
        - ground_truth
        """
        eval_data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": []
        }
        
        print(f"🧪 Running RAG on {len(test_cases)} test cases...")
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n[{i}/{len(test_cases)}] Processing: {test_case['question'][:50]}...")
            
            try:
                # Run RAG
                result = rag_system.query(test_case["question"])
                
                # Extract contexts
                contexts = [doc["content"] for doc in result["sources"]]
                
                # Add to dataset
                eval_data["question"].append(test_case["question"])
                eval_data["answer"].append(result["answer"])
                eval_data["contexts"].append(contexts)
                eval_data["ground_truth"].append(test_case["ground_truth"])
                
                print(f"   ✅ Generated answer with {len(contexts)} sources")
            
            except Exception as e:
                print(f"   ❌ Error: {e}")
                # Add placeholder to maintain dataset integrity
                eval_data["question"].append(test_case["question"])
                eval_data["answer"].append("Error generating answer")
                eval_data["contexts"].append([])
                eval_data["ground_truth"].append(test_case["ground_truth"])
        
        # Convert to HuggingFace Dataset
        dataset = Dataset.from_dict(eval_data)
        
        print(f"\n✅ Prepared evaluation dataset with {len(dataset)} examples")
        
        return dataset
    
    def evaluate_rag(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Run RAGAS evaluation
        Returns metrics scores
        """
        print("\n🎯 Running RAGAS evaluation...")
        print("Metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall")
        
        try:
            # Run evaluation
            results = evaluate(
                dataset=dataset,
                metrics=self.metrics,
                llm=self.llm,
                embeddings=self.embeddings,
            )
            
            print("\n✅ Evaluation complete!")
            
            return results
        
        except Exception as e:
            print(f"\n❌ Error during evaluation: {e}")
            print("💡 Make sure you have the latest ragas version: pip install ragas --upgrade")
            return {}
    
    def save_results(self, results: Dict[str, Any], config_name: str = "default"):
        """Save evaluation results"""
        
        # Create results directory if it doesn't exist
        os.makedirs(settings.EVAL_RESULTS_PATH, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{settings.EVAL_RESULTS_PATH}/{config_name}_{timestamp}.json"
        
        # Convert EvaluationResult to serializable dict
        if hasattr(results, 'to_pandas'):
            # It's an EvaluationResult object - convert to dict
            df = results.to_pandas()
            results_dict = {
                'summary': {
                    'faithfulness': float(df['faithfulness'].mean()) if 'faithfulness' in df else None,
                    'answer_relevancy': float(df['answer_relevancy'].mean()) if 'answer_relevancy' in df else None,
                    'context_precision': float(df['context_precision'].mean()) if 'context_precision' in df else None,
                    'context_recall': float(df['context_recall'].mean()) if 'context_recall' in df else None,
                },
                'per_question': df.to_dict('records'),
                'metadata': {
                    'num_questions': len(df),
                    'timestamp': timestamp,
                    'config': config_name
                }
            }
        else:
            # Already a dict
            results_dict = dict(results) if hasattr(results, '__iter__') else results
        
        with open(filename, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"💾 Results saved to {filename}")
        
        # Also print summary to console
        if 'summary' in results_dict:
            print("\n" + "="*70)
            print("📊 EVALUATION SUMMARY")
            print("="*70)
            for metric, value in results_dict['summary'].items():
                if value is not None:
                    print(f"{metric:20} : {value:.2%} ({value:.3f})")
        
        return filename
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a readable evaluation report"""
        
        # Handle both dict and EvaluationResult object
        if hasattr(results, 'to_pandas'):
            # It's an EvaluationResult object
            df = results.to_pandas()
            faithfulness_score = float(df['faithfulness'].mean()) if 'faithfulness' in df.columns else 0
            answer_relevancy_score = float(df['answer_relevancy'].mean()) if 'answer_relevancy' in df.columns else 0
            context_precision_score = float(df['context_precision'].mean()) if 'context_precision' in df.columns else 0
            context_recall_score = float(df['context_recall'].mean()) if 'context_recall' in df.columns else 0
        else:
            # It's a dict
            if 'summary' in results:
                # Our saved format
                faithfulness_score = results['summary'].get('faithfulness', 0)
                answer_relevancy_score = results['summary'].get('answer_relevancy', 0)
                context_precision_score = results['summary'].get('context_precision', 0)
                context_recall_score = results['summary'].get('context_recall', 0)
            else:
                # Raw dict
                faithfulness_score = results.get('faithfulness', 0)
                answer_relevancy_score = results.get('answer_relevancy', 0)
                context_precision_score = results.get('context_precision', 0)
                context_recall_score = results.get('context_recall', 0)
        
        report = f"""
# RAG Evaluation Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Overall Scores

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| Faithfulness | {faithfulness_score:.3f} | >0.8 | {'✅' if faithfulness_score > 0.8 else '⚠️'} |
| Answer Relevancy | {answer_relevancy_score:.3f} | >0.8 | {'✅' if answer_relevancy_score > 0.8 else '⚠️'} |
| Context Precision | {context_precision_score:.3f} | >0.7 | {'✅' if context_precision_score > 0.7 else '⚠️'} |
| Context Recall | {context_recall_score:.3f} | >0.7 | {'✅' if context_recall_score > 0.7 else '⚠️'} |

## Metric Definitions

**Faithfulness** (Target: >0.8)
- Measures if the answer is factually consistent with the retrieved context
- Low scores indicate hallucination

**Answer Relevancy** (Target: >0.8)
- Measures if the answer actually addresses the question
- Low scores mean the answer is off-topic

**Context Precision** (Target: >0.7)
- Measures if retrieved chunks are relevant
- Low scores mean noisy retrieval

**Context Recall** (Target: >0.7)
- Measures if all necessary information was retrieved
- Low scores mean incomplete retrieval

## Recommendations
"""
        
        # Add recommendations based on scores
        if faithfulness_score < 0.8:
            report += "\n⚠️ **Improve Faithfulness**: Add explicit grounding instructions, use citations, implement fact-checking"
        
        if answer_relevancy_score < 0.8:
            report += "\n⚠️ **Improve Answer Relevancy**: Refine prompts to stay focused on the question, add relevance checks"
        
        if context_precision_score < 0.7:
            report += "\n⚠️ **Improve Context Precision**: Enhance reranking, adjust chunk size, improve metadata filtering"
        
        if context_recall_score < 0.7:
            report += "\n⚠️ **Improve Context Recall**: Increase retrieval k, use query expansion, try hybrid search"
        
        if all([
            faithfulness_score >= 0.8,
            answer_relevancy_score >= 0.8,
            context_precision_score >= 0.7,
            context_recall_score >= 0.7
        ]):
            report += "\n✅ **Great Job!** All metrics are above target. Consider fine-tuning for specific use cases."
        
        return report


class CustomMetrics:
    """Additional metrics not in RAGAS"""
    
    @staticmethod
    def latency_metric(rag_system, questions: List[str]) -> Dict[str, float]:
        """Measure average latency"""
        import time
        
        latencies = []
        for q in questions:
            start = time.time()
            try:
                rag_system.query(q)
                latencies.append(time.time() - start)
            except:
                pass
        
        if not latencies:
            return {"avg_latency_ms": 0, "p95_latency_ms": 0}
        
        return {
            "avg_latency_ms": sum(latencies) / len(latencies) * 1000,
            "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] * 1000 if latencies else 0
        }
    
    @staticmethod
    def cost_metric(num_queries: int, avg_tokens: int) -> Dict[str, float]:
        """
        Estimate costs (Groq is free, but good to track)
        """
        return {
            "total_queries": num_queries,
            "avg_tokens_per_query": avg_tokens,
            "total_tokens": num_queries * avg_tokens,
            "estimated_cost_usd": 0.0  # Groq is free!
        }


# Example Usage
if __name__ == "__main__":
    print("Evaluation module loaded")
    print("Use this from main application or scripts/run_evaluation.py")