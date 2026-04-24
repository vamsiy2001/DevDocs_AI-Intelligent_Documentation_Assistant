---
title: DevDocs AI
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
suggested_hardware: cpu-basic
pinned: false
---

<h1 align="center">DevDocs AI</h1>

<p align="center">
  A production-grade RAG system with hybrid search, cross-encoder reranking, and agentic routing — built entirely with free tools.
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/vamsiyvk/devdocsAI">
    <img src="https://img.shields.io/badge/Live%20Demo-%F0%9F%A4%97%20HuggingFace-yellow" alt="Live Demo" />
  </a>
  <img src="https://img.shields.io/badge/Python-3.10-blue" alt="Python 3.10" />
  <img src="https://img.shields.io/badge/LLM-Groq%20%7C%20Llama%203.3%2070B-orange" alt="Groq" />
  <img src="https://img.shields.io/badge/Vector%20DB-ChromaDB-green" alt="ChromaDB" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" alt="MIT License" />
</p>

---

## Overview

DevDocs AI answers questions about developer tools, frameworks, and libraries using a multi-stage retrieval pipeline. Instead of relying on a single embedding lookup, it combines dense semantic search, sparse BM25 keyword matching, and a cross-encoder reranker to surface the most relevant context before passing it to the LLM.

The entire stack runs on free-tier services — no paid APIs beyond optional monitoring.

---

## How It Works

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                    Query Router                      │
│  (LangGraph agent — routes to retrieve / search /   │
│   direct answer based on query classification)      │
└──────────────────────┬──────────────────────────────┘
                       │
           ┌───────────▼───────────┐
           │    Hybrid Retrieval   │
           │                       │
           │  Dense  ──┐           │
           │  (SBERT)  ├──► Merge ──► CrossEncoder ──► Top-K
           │  Sparse ──┘           │    Reranker
           │  (BM25)               │
           └───────────────────────┘
                       │
           ┌───────────▼───────────┐
           │     LLM Generation    │
           │  Groq / Llama 3.3 70B │
           └───────────────────────┘
                       │
              Answer + Sources + Scores
```

---

## Stack

| Layer | Tool | Notes |
|---|---|---|
| LLM | Groq — Llama 3.3 70B | Free tier, ~300 tok/s |
| Embeddings | `all-MiniLM-L6-v2` | Local, no API cost |
| Vector DB | ChromaDB | Local persistent store |
| Sparse search | BM25 (`rank-bm25`) | In-memory, no server |
| Reranker | `ms-marco-MiniLM-L-6-v2` | CrossEncoder, local |
| Orchestration | LangChain + LangGraph | Agent routing |
| UI | Gradio 5 | Deployed on HF Spaces |
| Deployment | Docker on HF Spaces | CPU-basic tier |

---

## Project Structure

```
devdocs-ai/
├── app.py                  # HF Spaces entry point
├── Dockerfile              # Container build (CPU-optimised)
├── requirements.txt
├── app/
│   └── gradio_app.py       # Gradio interface
└── src/
    ├── config.py           # Pydantic settings
    ├── agents/
    │   └── langgraph_agent.py   # SimpleRAG + RAGAgent
    ├── retrieval/
    │   └── hybrid_search.py     # Dense + BM25 + reranking
    ├── ingestion/
    │   ├── document_loader.py
    │   └── chunking.py
    └── evaluation/
        └── ragas_eval.py        # RAGAS metrics
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- A free [Groq API key](https://console.groq.com)

### Run Locally

```bash
git clone https://github.com/vamsiy2001/DevDocs_AI-Intelligent_Documentation_Assistant.git
cd DevDocs_AI-Intelligent_Documentation_Assistant

python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Add your key
echo "GROQ_API_KEY=your_key_here" > .env

python app.py
# Open http://localhost:7860
```

### Run with Docker

```bash
docker build -t devdocs-ai .
docker run -p 7860:7860 -e GROQ_API_KEY=your_key_here devdocs-ai
```

---

## Usage

```python
from src.retrieval import HybridRetriever
from src.agents import SimpleRAG

retriever = HybridRetriever()
retriever.load_vector_store()   # or create_vector_store(docs)

rag = SimpleRAG(retriever)
result = rag.query("How do I use LangChain agents?")

print(result["answer"])
# Sources with relevance scores available in result["sources"]
```

---

## Evaluation

Evaluated on 30 test cases using the [RAGAS](https://github.com/explodinggradients/ragas) framework:

| Metric | Score | Target |
|---|---|---|
| Context Precision | 85% | > 70% |
| Answer Faithfulness | 92% | > 80% |
| Context Recall | 88% | > 70% |
| Answer Relevancy | 90% | > 80% |
| P95 Latency | 1.8s | < 2s |

```bash
# Run evaluation
python scripts/run_evaluation.py
```

---

## Deployment

The live demo runs on HuggingFace Spaces (CPU basic, free tier).

The Docker image installs `torch` CPU-only (~200 MB) before other packages so that `sentence-transformers` does not pull the 2.6 GB CUDA wheel — keeping the build within memory limits.

To deploy your own copy:

```bash
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
git push space main
```

Set `GROQ_API_KEY` in **Space Settings → Variables and secrets**.

---

## License

MIT — see [LICENSE](LICENSE).
