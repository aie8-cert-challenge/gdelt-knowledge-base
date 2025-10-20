# GDELT Knowledge Graph RAG Assistant

> **Certification Challenge Project** - AI Engineering Bootcamp Cohort 8
>
> An intelligent question-answering system for GDELT (Global Database of Events, Language, and Tone) documentation, powered by Retrieval-Augmented Generation.

---

**Documentation**:
- 📋 [Full Deliverables](./docs/deliverables.md) - Complete answers to all certification requirements
- 📚 [Task Rubric](./docs/certification-challenge-task-list.md) - 100-point scoring breakdown
- 🏗️ [Architecture Diagram](./docs/cert-challenge.excalidraw) - System design (Excalidraw format)

**Datasets**:
- 📄 [Source Documents](https://huggingface.co/datasets/dwb2023/gdelt-rag-sources) - 38 PDF pages on GDELT KGs
- ⭐ [Golden Testset](https://huggingface.co/datasets/dwb2023/gdelt-rag-golden-testset) - 12 RAGAS evaluation QA pairs

---

## Table of Contents

- [Quick Start](#quick-start)
- [Documentation Guide](#documentation-guide)
- [Project Overview](#project-overview)
- [Technology Stack](#technology-stack)
- [Evaluation Results](#evaluation-results)
- [Running Evaluations](#running-evaluations)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

---

## Documentation Guide

This project has comprehensive documentation organized across multiple files:

**Core Documentation**:
- **[README.md](README.md)** (this file) - Project overview, quick start, installation
- **[CLAUDE.md](CLAUDE.md)** - Complete technical reference for AI assistants and developers
- **[docs/deliverables.md](docs/deliverables.md)** - Certification challenge answers (1,152 lines)
- **[docs/architecture.md](docs/architecture.md)** - System design patterns and architectural decisions

**Directory-Specific Guides**:
- **[scripts/README.md](scripts/README.md)** - All evaluation and utility scripts (5 scripts documented)
- **[src/README.md](src/README.md)** - Factory pattern guide, module reference, adding retrievers
- **[data/README.md](data/README.md)** - Data flow, manifest schema, file formats, lineage

**Quick Navigation**:
- 🔍 Want to understand the codebase? → Start with [CLAUDE.md](CLAUDE.md)
- 🚀 Want to run evaluations? → See [scripts/README.md](scripts/README.md)
- 🛠️ Want to add a retriever? → See [src/README.md](src/README.md#quick-start-adding-a-new-retriever)
- 📊 Want to understand data flow? → See [data/README.md](data/README.md#data-flow)
- ✅ Want to validate setup? → Run `make validate` (must pass 100%)

---

## Quick Start

### Prerequisites

- Python 3.11+
- `uv` package manager ([installation guide](https://github.com/astral-sh/uv))
- OpenAI API key (required)
- Cohere API key (optional, for reranking)

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/cert-challenge.git
cd cert-challenge

# Create virtual environment
uv venv --python 3.11
source .venv/bin/activate  # Linux/WSL/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
uv pip install -e .
```

### Set Environment Variables

```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your API keys:
# OPENAI_API_KEY=your_key_here
# COHERE_API_KEY=your_key_here  # Optional
```

### Run the Application

```bash
# Option 1: LangGraph Studio UI (Interactive, Recommended)
uv run langgraph dev --allow-blocking
# Access at: http://localhost:2024
# Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024

# Option 2: Command-line evaluation (self-contained reference)
python scripts/single_file.py

# Option 3: Modular evaluation (uses src/ modules)
python scripts/run_eval_harness.py
# Or use make command:
make eval

# Option 4: Quick validation
make validate
```

---

## Project Overview

### Problem Statement

Researchers and analysts working with GDELT struggle to quickly find answers to complex questions about knowledge graph construction, data formats, and analytical techniques without manually searching through dense technical documentation.

### Solution

An AI-powered Q&A system that provides instant, citation-backed answers from GDELT research papers using:
- Retrieval-Augmented Generation (RAG)
- Multiple retrieval strategies (naive, BM25, reranking, ensemble)
- RAGAS evaluation framework
- Production-grade technology stack

### Target Audience

- GDELT researchers implementing analysis pipelines
- Data scientists building geopolitical risk assessment systems
- Analysts investigating cross-lingual event tracking
- Teams adopting GDELT for the first time

---

## Technology Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| **LLM** | OpenAI GPT-4.1-mini | Cost-effective reasoning for RAG |
| **Embeddings** | text-embedding-3-small | Strong semantic similarity at 1536 dims |
| **Vector DB** | Qdrant | Production-grade, LangChain integration |
| **Orchestration** | LangChain + LangGraph | Battle-tested RAG abstractions |
| **Monitoring** | LangSmith | End-to-end LLM observability |
| **Evaluation** | RAGAS 0.2.10 | Purpose-built RAG metrics |
| **UI** | Streamlit | Rapid prototyping, chat interface |
| **Data** | HuggingFace Datasets | Versioned, reproducible datasets |

**Advanced Retrieval Techniques**:
- Dense vector search (baseline)
- BM25 sparse keyword matching
- Cohere rerank-v3.5 (contextual compression)
- Ensemble hybrid search (dense + sparse)

---

## Evaluation Results

### Task 5: Baseline RAG Performance

*Results from `notebooks/task5_baseline_evaluation_don.py` (Naive retriever baseline)*

| Metric | Score |
|--------|-------|
| Faithfulness | 94.48% |
| Answer Relevancy | 86.79% |
| Context Precision | 81.10% |
| Context Recall | 98.33% |
| **Average** | **90.18%** |

**Key Finding**: Strong baseline with excellent Context Recall (98.33%), but Context Precision (81.10%) is the bottleneck. Dense vector search alone retrieves relevant documents but struggles with ranking quality.

### Task 7: Comparative Retrieval Performance

*Results from `notebooks/task5_baseline_evaluation_don.py` (all 4 retrievers)*

| Retriever | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Average |
|-----------|--------------|------------------|-------------------|----------------|---------|
| **Cohere Rerank** | **96.50%** | **94.51%** | **98.61%** | **96.25%** | **96.47%** |
| Ensemble | 97.58% | 96.11% | 83.79% | 98.33% | 93.96% |
| BM25 | 95.62% | 95.34% | 87.26% | 98.33% | 94.14% |
| Naive (baseline) | 93.49% | 94.65% | 84.11% | 94.17% | 91.60% |

**Key Findings**:
- **Winner**: Cohere Rerank at **96.47%** (+5.3% over baseline)
- **Dramatic Context Precision improvement**: +17.2% (81.10% → 98.61%)
- **BM25**: +4.4% Context Recall improvement via keyword matching
- **Ensemble**: Best Faithfulness (97.58%) and Answer Relevancy (96.11%)

**Recommendation**: Deploy Cohere Rerank for production. The $20/month cost is negligible compared to the quality improvement (saves ~490 errors per 10K queries).

---

## Running Evaluations

### Step 1: Baseline Evaluation (Task 5)

```bash
# Launch Jupyter
jupyter notebook

# Open and run notebooks/task5_baseline_evaluation.ipynb
# This evaluates the naive retriever with RAGAS metrics
```

**Output**: `data/processed/baseline_ragas_results.csv`

### Step 2: Comparative Evaluation (Task 7)

```bash
# Open and run notebooks/task7_comparative_evaluation.ipynb
# This evaluates all 4 retrievers and generates comparison table
```

**Output**: `data/processed/comparative_ragas_results.csv`

### Evaluation Metrics Explained

- **Faithfulness**: Are answers grounded in retrieved context? (no hallucinations)
- **Answer Relevancy**: Does the answer address the user's question?
- **Context Precision**: Are relevant contexts ranked higher than irrelevant ones?
- **Context Recall**: Is all necessary information from ground truth retrieved?

---

## Project Structure

```
cert-challenge/
├── app/                          # Main application code
│   ├── baseline_rag.py          # Naive RAG implementation (Task 4)
│   ├── retriever_registry.py    # All retrieval strategies (Task 6)
│   └── streamlit_ui.py          # Interactive UI (demo)
├── notebooks/                    # Jupyter notebooks for evaluation
│   ├── task5_baseline_evaluation.ipynb      # RAGAS baseline metrics
│   ├── task7_comparative_evaluation.ipynb   # Retriever comparison
│   └── pdf_ingestion_pipeline*.ipynb        # Data processing
├── scripts/                      # Utility and reference scripts
│   ├── upload_to_hf.py          # Dataset uploader
│   ├── session08-ragas-rag-evals.py         # RAGAS patterns
│   └── session09-adv-retrieval*.py          # Advanced retrieval patterns
├── data/
│   ├── raw/                     # Source PDFs
│   ├── interim/                 # Intermediate processing (manifest, HF datasets)
│   └── processed/               # Evaluation results (CSV tables)
├── docs/
│   ├── deliverables.md          # Complete certification answers
│   └── certification-challenge-task-list.md # Rubric
├── pyproject.toml               # Dependencies (uv/pip)
├── .env.example                 # Environment template
└── README.md                    # This file
```

---

## Key Implementation Patterns

### Loading Documents from HuggingFace

```python
from datasets import load_dataset
from langchain_core.documents import Document

dataset = load_dataset("dwb2023/gdelt-rag-sources", split="train")

documents = []
for item in dataset:
    doc = Document(
        page_content=item["page_content"],
        metadata=item["metadata"]
    )
    documents.append(doc)
```

### Creating a RAG Chain

```python
from langchain_qdrant import QdrantVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from qdrant_client import QdrantClient

# Create Qdrant client and vector store
client = QdrantClient(host="localhost", port=6333)
vectorstore = QdrantVectorStore(
    client=client,
    collection_name="gdelt_rag",
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Build chain
from langchain.prompts import ChatPromptTemplate
from operator import itemgetter

prompt = ChatPromptTemplate.from_template(
    "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
)

chain = (
    {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
    | prompt
    | ChatOpenAI(model="gpt-4.1-mini")
)
```

### Running RAGAS Evaluation

```python
from ragas import evaluate
from ragas.metrics import Faithfulness, ResponseRelevancy

result = evaluate(
    dataset=ragas_dataset,
    metrics=[Faithfulness(), ResponseRelevancy()],
)

print(result.to_pandas())
```

---

## Development Notes

### Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional (for advanced features)
COHERE_API_KEY=...              # Reranking
LANGCHAIN_API_KEY=...           # LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=cert-challenge
```

### Python Package Requirements

Key dependencies (see `pyproject.toml` for complete list):
- `langchain>=0.3.19`
- `langchain-openai>=0.3.7`
- `langchain-cohere==0.4.4`
- `qdrant-client>=1.13.2`
- `ragas==0.2.10`
- `datasets>=3.2.0`
- `huggingface-hub>=0.26.0`
- `streamlit>=1.40.0`

### Common Issues

**Issue**: `AttributeError: 'str' object has no attribute 'metadata'`
- **Cause**: Chain output format mismatch
- **Solution**: Use `retriever_registry.py` which follows correct LCEL patterns

**Issue**: `COHERE_API_KEY not set`
- **Cause**: Cohere rerank retriever requires API key
- **Solution**: Set env var or skip Cohere rerank (3 other retrievers still work)

---

## References

- [GDELT Project](https://www.gdeltproject.org/)
- [LangChain Documentation](https://python.langchain.com/)
- [RAGAS Documentation](https://docs.ragas.io/)
- [HuggingFace Datasets](https://huggingface.co/docs/datasets/)
- Myers, A., et al. (2025). "Talking to GDELT Through Knowledge Graphs." arXiv:2503.07584v3

---

## License

Apache 2.0 (see LICENSE file)

## Contact

Don Brown (dwb2023) - AI Engineering Bootcamp Cohort 8
