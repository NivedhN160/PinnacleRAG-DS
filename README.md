# PinnacleRAG-DS

> A production-grade Retrieval-Augmented Generation system with a full Data Science pipeline — powered entirely by Groq free-tier API and open-source components.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Groq Powered](https://img.shields.io/badge/LLM-Groq%20Free%20Tier-orange.svg)](https://groq.com)

## 🏗️ Architecture

PinnacleRAG-DS applies classic Data Science workflows to RAG:

```
Documents → Load → Clean → Chunk → Embed → Index
                                                 ↓
Query → Hybrid Retrieval (Dense + BM25) → Rerank (Cross-Encoder) → Generate (Groq) → Answer + Citations
                                                                                           ↓
                                                                              Evaluate (RAGAS Metrics)
```

### Key Accuracy Levers
- **Hybrid Retrieval**: Dense vector search + BM25 sparse search fusion
- **Cross-Encoder Reranking**: Free sentence-transformers model for precision
- **Structure-Aware Chunking**: Recursive + heading-aware splitting
- **Strict Grounded Generation**: Groq LLM with citation-enforcing prompts
- **RAGAS Evaluation**: Faithfulness, relevancy, context precision/recall

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- A free [Groq API key](https://console.groq.com)

### Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/PinnacleRAG-DS.git
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

### Usage

#### 1. Ingest Documents
Place your documents (PDF, TXT, MD, DOCX) in `data/raw/`, then run:
```bash
python scripts/ingest.py
```

#### 2. Query
Interactive mode:
```bash
python scripts/query.py
```

Single question:
```bash
python scripts/query.py --question "What is the main topic of the documents?"
```

#### 3. Evaluate
Create a golden Q&A set in `data/golden/golden_set.json` and run:
```bash
python scripts/evaluate.py
```

#### 4. Batch Query
```bash
python scripts/batch_query.py --input questions.txt --output results.json
```

#### 5. API Server (Optional)
```bash
uvicorn app.main:app --reload
```

## 📊 Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Faithfulness | Answer is grounded in retrieved context | ≥ 0.85 |
| Answer Relevancy | Answer addresses the question | ≥ 0.80 |
| Context Precision | Retrieved docs are relevant | ≥ 0.75 |
| Context Recall | All needed info is retrieved | ≥ 0.75 |

## 📁 Project Structure

```
PinnacleRAG-DS/
├── config/settings.py          # All tunable parameters
├── src/
│   ├── ingestion/              # Load → Clean → Chunk
│   ├── embeddings/             # Local free embedding model
│   ├── retrieval/              # Hybrid search + reranker
│   ├── generation/             # Groq LLM + grounded prompts
│   ├── evaluation/             # RAGAS-style metrics
│   ├── pipeline/               # End-to-end orchestration
│   └── utils/                  # Logging, helpers
├── scripts/                    # CLI entry points
├── notebooks/                  # DS exploration & ablation
├── tests/                      # Unit & integration tests
└── app/                        # FastAPI interface
```

## 🔧 Configuration

All parameters are centralized in `config/settings.py`:
- LLM model selection (`llama-3.3-70b-versatile`)
- Embedding model (`all-MiniLM-L6-v2`)
- Chunk size / overlap
- Retrieval top-k and hybrid weights
- Evaluation thresholds

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

---

*Built with the philosophy: free, modular, measurable, and competitive with production systems.*
