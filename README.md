# PinnacleRAG-DS

> Free hybrid RAG webapp — dense+BM25, rerank, Groq+citations, eval dashboard, health PDF pack. Local embeds. No paid APIs.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Groq Powered](https://img.shields.io/badge/LLM-Groq%20Free%20Tier-orange.svg)](https://groq.com)

**Disclaimer**: This application is for educational purposes only. It is **not** medical advice. Always consult a healthcare professional for interpreting medical lab results.

## 🏗️ Architecture

PinnacleRAG-DS is a single web application designed to beat scattered CLI scripts and fragmented tools:
- **Webapp only**: The product is the browser interface (Vanilla JS/HTML SPA). No Docker needed.
- **Hybrid RAG**: Dense vector search (ChromaDB) + BM25 sparse search fusion.
- **Reranker**: Cross-Encoder (sentence-transformers).
- **Strict Grounded Generation**: Groq LLM with forced `[n]` citations.
- **Health Pack**: Specialized pipeline for parsing lab PDFs and extracting structured JSON alongside standard chat indexing.
- **Agent Loop**: Simple 2-step agent to handle weak retrieval.
- **Budget Guard**: API hard stop on max LLM calls per session.

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- A free [Groq API key](https://console.groq.com) (or run in Mock mode)

### Setup

```bash
# Clone the repo
git clone https://github.com/NivedhN160/PinnacleRAG-DS.git
cd PinnacleRAG-DS

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### Run the Webapp

Start the FastAPI backend:
```bash
uvicorn app.main:app --reload --port 8000
```

Start the frontend (using any simple web server, e.g., Python's built-in http server):
```bash
cd web
python -m http.server 3000
```

Open your browser and navigate to `http://localhost:3000`.

## 📊 Features

- **Home**: Upload documents to `data/raw/` and rebuild index.
- **Chat**: Simple or Agent mode. Ask questions and see in-line citations mapped to source documents.
- **Eval Dashboard**: See golden-set metrics directly on screen (Faithfulness, Relevancy, Context Precision/Recall).
- **Health Pack**: Upload a lab report PDF to instantly parse metrics into structured JSON and index for chat.

## 🔧 Configuration

All parameters are centralized in `config/settings.py`:
- LLM model selection (`llama-3.3-70b-versatile`)
- Embedding model (`all-MiniLM-L6-v2`)
- Budget guard limits (`MAX_LLM_CALLS`, `MAX_TOKENS`)
- Mocking (`MOCK_IF_NO_KEY`)

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.
