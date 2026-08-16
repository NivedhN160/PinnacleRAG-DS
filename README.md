# PinnacleRAG-DS

> Next.js UI, hybrid RAG (dense+BM25), domains (Trading / Security / SEO), PDF+text upload, local embeddings, Groq only. No paid APIs.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Groq Powered](https://img.shields.io/badge/LLM-Groq%20Free%20Tier-orange.svg)](https://groq.com)

**Disclaimer**: This application is for educational purposes only. It is **not** financial or security advice. Always consult a professional for interpreting such results.

## 🏗️ Architecture

PinnacleRAG-DS is a full web application designed for domain-specific RAG:
- **Next.js UI**: A modern, interactive frontend.
- **Domains**: Built-in support for Trading, Security, SEO, and General domains.
- **Hybrid RAG**: Dense vector search (ChromaDB) + BM25 sparse search fusion.
- **Reranker**: Cross-Encoder (sentence-transformers).
- **Strict Grounded Generation**: Groq LLM with forced `[n]` citations.
- **Agent Loop**: Simple 2-step agent to handle weak retrieval.
- **Budget Guard**: API hard stop on max LLM calls per session.

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- A free [Groq API key](https://console.groq.com) (or run in Mock mode)

### Setup

```bash
# Clone the repo
git clone https://github.com/NivedhN160/PinnacleRAG-DS.git
cd PinnacleRAG-DS

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install backend dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend && npm install
cd ..

# Configure API key
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### Run the Webapp

Start both backend and frontend concurrently from the root directory:
```bash
npm start
```

Open your browser and navigate to `http://localhost:3000`.

## 📂 Domain Folders

Documents are isolated by domain for accurate retrieval. Place raw files here:
- `data/raw/trading/`
- `data/raw/security/`
- `data/raw/seo/`
- `data/raw/general/`

## 🔌 API Endpoints

- `/api/query/` - Query the RAG pipeline (requires `domain`)
- `/api/ingest/` - Rebuild index from `data/raw/`
- `/api/ingest/upload` - Upload PDF/TXT/MD files for a domain
- `/api/ingest/text` - Ingest raw text snippets
- `/api/eval/` - Run Ragas evaluation
- `/api/health/` - Check system health

## 🔧 Configuration

All parameters are centralized in `config/settings.py`:
- LLM model selection (`llama-3.3-70b-versatile`)
- Embedding model (`all-MiniLM-L6-v2`)
- Budget guard limits (`MAX_LLM_CALLS`, `MAX_TOKENS`)
- Mocking (`MOCK_IF_NO_KEY`)

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.
