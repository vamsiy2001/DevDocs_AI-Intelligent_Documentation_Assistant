# 🤖 DevDocs AI - Intelligent Documentation Assistant

[![Demo](https://img.shields.io/badge/🤗-Demo-yellow)](YOUR_HF_SPACE_LINK)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Production-grade RAG system with hybrid search, reranking, and comprehensive evaluation. Built with 100% free tools.

![Architecture](docs/architecture.png)

<img width="1236" height="714" alt="Screenshot 2026-01-14 at 8 27 14 PM" src="https://github.com/user-attachments/assets/ba6a36c2-941b-4ba5-9ae0-61cc45c5fe36" />

## ✨ Features

- **🔍 Hybrid Search**: Dense embeddings + BM25 + CrossEncoder reranking
- **🤖 Agentic Routing**: LangGraph-based intelligent query routing  
- **📊 Rigorous Evaluation**: RAGAS framework with 5 metrics
- **⚡ Fast**: <2s response time (p95)
- **💰 100% Free**: No paid APIs required
- **📚 Multi-Source**: Aggregates GitHub repos, docs, blogs

## 📊 Performance Metrics

| Metric | Score | Target |
|--------|-------|--------|
| Context Precision | 85% | >70% |
| Answer Faithfulness | 92% | >80% |
| Context Recall | 88% | >70% |
| Answer Relevancy | 90% | >80% |
| P95 Latency | 1.8s | <2s |

*Evaluated on 30 test cases using RAGAS framework*

## 🏗️ Architecture

```
User Query
    ↓
[Query Router] ──→ Technical? ──→ [Hybrid Search]
    ↓                                   ↓
    ├─→ Recent? ──→ [Web Search]        ├─→ Dense Retrieval
    ↓                                   ├─→ Sparse (BM25)
    └─→ Simple? ──→ [Direct Answer]     └─→ Reranking
                                            ↓
                                        [LLM Generation]
                                            ↓
                                    [Answer + Sources]
```

## 🚀 Quick Start


### Installation

```bash
# 1. Clone repository
git clone https://github.com/yourusername/devdocs-ai.git
cd devdocs-ai

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Setup environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### Get API Keys (All FREE!)

1. **Groq** (Required): https://console.groq.com
   - Free tier: 14,400 requests/day
   
2. **LangSmith** (Optional): https://smith.langchain.com
   - Free tier: 5,000 traces/month

### Run Demo

```bash
# Or run Gradio app
python app/gradio_app.py
```

## 🛠️ Tech Stack

| Component | Tool | Why |
|-----------|------|-----|
| **LLM** | Groq (Llama 3.1 70B) | Free, fastest inference |
| **Embeddings** | sentence-transformers | Free, local |
| **Vector DB** | ChromaDB | Free, local |
| **Reranking** | CrossEncoder | Free, local |
| **Framework** | LangChain + LangGraph | Industry standard |
| **Evaluation** | RAGAS | Comprehensive metrics |
| **Deployment** | HuggingFace Spaces | Free hosting |

## 📈 Evaluation Results

[Include charts/graphs showing]:
- Retrieval quality improvements
- Latency benchmarks
- Comparison of configurations

## 🔬 Experiments

See `notebooks/` for:
- Retrieval experiments (dense vs sparse vs hybrid)
- Chunk size optimization
- Reranking comparisons
- LLM prompt engineering

## 🚀 Deployment

### HuggingFace Spaces

```bash
# Add HF Space as remote
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/devdocs-ai

# Push
git push space main
```

## 📝 Documentation

- [Architecture Overview](docs/architecture.md)
- [Evaluation Guide](docs/evaluation.md)
- [Deployment Guide](docs/deployment.md)
- [API Documentation](docs/api.md)


## 🙏 Acknowledgments

- LangChain team for the amazing framework
- RAGAS team for evaluation tools
- Groq for free, fast LLM inference
- HuggingFace for free hosting

## 📞 Contact

- **Author**: Vamsi Krishna Yerubandi
- **LinkedIn**: https://www.linkedin.com/in/y-vamsi-krishna/

---

**⭐ Star this repo if you find it helpful!**

Made with ❤️ for the AI community
