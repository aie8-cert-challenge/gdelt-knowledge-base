# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Production-grade RAG system for GDELT documentation, built as an AI Engineering Bootcamp certification challenge. Implements 4 retrieval strategies (naive, BM25, ensemble, Cohere rerank) with RAGAS-based evaluation. Key architectural principle: **Parquet-first data flow** where working data lives in `data/processed/` and human-readable deliverables are derived artifacts.

## Quick Start

```bash
# 1. Setup
uv venv --python 3.11 && source .venv/bin/activate && uv pip install -e .

# 2. Run evaluation (loads datasets from HuggingFace, creates Qdrant collection)
make qdrant-up && make validate && make eval

# 3. Generate human-readable outputs
make deliverables
```

## End-to-End Evaluation Flow

| Step | Command | Intent | When to Run | Output Location |
|------|---------|--------|-------------|-----------------|
| **1** | `make qdrant-up` | Start vector database | Always (required infra) | Docker container (port 6333) |
| **2** | `make validate` | Verify env + modules work | Before first eval | Terminal (23/23 checks) |
| **3a** | `make ingest` | Extract PDFs → testset | **OPTIONAL** - only if recreating datasets | `data/interim/*.parquet` + `manifest.json` |
| **3b** | `make publish-interim` | Upload to HuggingFace | **OPTIONAL** - only if sharing datasets | HuggingFace Hub |
| **4** | `make eval` | Run RAG + RAGAS evaluation<br>**← Qdrant populated HERE** | Core workflow (loads from HF by default) | `data/processed/*.parquet` + `RUN_MANIFEST.json` |
| **5** | `make deliverables` | Convert Parquet → human-readable CSV | After eval (for review/submission) | `deliverables/evaluation_evidence/*.csv` |
| **6** | `make publish-processed` | Upload evaluation results | **OPTIONAL** - only if sharing benchmarks | HuggingFace Hub |

**Cost & Duration**:
- Step 3a (ingest): 5-10 min, ~$2-3 (one-time)
- Step 4 (eval): 20-30 min, ~$5-6 (repeatable)
- Step 5 (deliverables): <1 min, $0 (regenerable)

## Critical Architectural Decisions

### 1. Factory Pattern (Mandatory)

**Why**: Retrievers depend on runtime-loaded data (documents, vector stores) that don't exist at module import time.

```python
# ❌ BREAKS - documents don't exist yet
retriever = vector_store.as_retriever()

# ✅ WORKS - deferred initialization
def create_retrievers(documents, vector_store, k=5):
    return {"naive": vector_store.as_retriever(search_kwargs={"k": k}), ...}
```

**Key Factories**:
- `src/config.py::create_vector_store()` - Qdrant vector store
- `src/retrievers.py::create_retrievers()` - 4 retrieval strategies
- `src/graph.py::build_graph()` - LangGraph workflow

### 2. Parquet-First Data Architecture

**Source of truth**: `data/processed/*.parquet` (machine-optimized, ZSTD compressed)
**Derived artifacts**: `deliverables/*.csv` (human-readable, regenerable via `make deliverables`)

**Why**:
- Parquet files are immutable working data
- CSV files can be regenerated anytime without re-running expensive evaluation
- Clear separation between what's precious (Parquet) and what's disposable (CSV)

### 3. Inference/Evaluation Decoupling

**Critical fix**: Evaluation pipeline saves inference results (`*_evaluation_inputs.parquet`) **immediately after Step 3** (inference), NOT during Step 4 (RAGAS evaluation).

**Why**: If RAGAS fails mid-run, inference results are preserved. Enables future inference-only and evaluation-only modes.

See `scripts/run_eval_harness.py:196-198` for implementation.

### 4. When Qdrant Gets Populated

**Answer**: During `make eval` (Step 2 of evaluation pipeline)
**NOT during**: `make ingest` (that's only data preparation)

If you run `make ingest` to recreate datasets, you MUST run `make eval recreate=true` to populate Qdrant with fresh data.

## Key Technologies

**Core Stack**:
- LangChain 0.3.19+ / LangGraph 0.6.7 (pinned)
- RAGAS 0.2.10 (pinned - API changed in 0.3.x)
- Qdrant (Docker), OpenAI GPT-4.1-mini + text-embedding-3-small
- Cohere rerank-v3.5

**Data**:
- HuggingFace Datasets (versioned), PyMuPDF (PDF parsing)
- Parquet with ZSTD compression

**Development**:
- `uv` package manager, Python 3.11+
- LangSmith (tracing), LangGraph Studio UI (prototype UI)

## Repository Structure

```
src/                  # Core RAG framework (775 lines total)
├── config.py         # Cached singletons (@lru_cache)
├── retrievers.py     # Factory: create_retrievers()
├── graph.py          # Factory: build_graph()
├── state.py          # LangGraph TypedDict schema
└── utils/            # HF loaders, manifest generation

scripts/              # Executable workflows (~2800 lines total)
├── run_eval_harness.py        # RAGAS eval (uses src/)
├── run_app_validation.py      # 23-check validation suite
├── ingest_raw_pdfs.py         # PDF → interim datasets
├── generate_deliverables.py   # Parquet → CSV conversion
└── publish_*.py               # HuggingFace upload

data/
├── raw/              # Source PDFs (immutable)
├── interim/          # Extracted docs + testset + manifest.json
├── processed/        # Evaluation results (Parquet, source of truth)
└── deliverables/     # CSV files (derived, regenerable)
```

## Common Gotchas

### Do I need `make ingest`?
- ❌ NO if using existing HuggingFace datasets (most users)
- ✅ YES if modifying source PDFs or regenerating testset from scratch

### When does Qdrant get populated?
During `make eval` (Step 4 in flow table), NOT during `make ingest`.

### Parquet vs CSV files?
- `data/processed/*.parquet` = source of truth (never delete)
- `deliverables/*.csv` = derived (delete anytime, regenerate with `make deliverables`)

### Why does `make eval` reuse the Qdrant collection?
**Performance** - skips re-embedding 38 documents (~5 min). Use `make eval recreate=true` after ingesting new data or changing embedding models.

### What's the difference between `run_eval_harness.py` and `run_full_evaluation.py`?
- `run_eval_harness.py` - uses `src/` modules (268 lines, cleaner)
- `run_full_evaluation.py` - standalone reference (529 lines, all logic inline)
- **Same inputs, same outputs, identical results**

## Adding a New Retriever

```python
# 1. Edit src/retrievers.py
def create_retrievers(documents, vector_store, k=5):
    # ... existing retrievers ...
    your_retriever = YourRetrieverClass(vectorstore=vector_store, k=k)

    return {
        "naive": naive,
        "bm25": bm25,
        "ensemble": ensemble,
        "cohere_rerank": compression,
        "your_method": your_retriever,  # <-- Add here
    }

# 2. Validate + Evaluate
make validate  # Must pass 100% (23/23 checks)
make eval      # Automatically includes your new retriever
```

Results automatically appear in `data/processed/comparative_ragas_results.parquet`.

## Performance Characteristics

**Latency** (per query):
- BM25: 10-20ms (in-memory)
- Naive: 50-100ms (Qdrant vector search)
- Ensemble: 60-120ms (parallel naive + BM25)
- Cohere Rerank: 200-500ms (includes API call)

**Cost** (full evaluation):
- 48 RAG queries (12 questions × 4 retrievers)
- ~150 LLM calls for RAGAS metrics
- **Total**: ~$5-6, 20-30 minutes

## Environment Variables

```bash
# Required
OPENAI_API_KEY="sk-..."

# Optional
COHERE_API_KEY="..."           # For cohere_rerank retriever
HF_TOKEN="..."                 # For publishing to HuggingFace
LANGCHAIN_API_KEY="..."        # For LangSmith tracing

# Qdrant (URL-first convention)
QDRANT_URL="http://localhost:6333"  # Preferred
# OR
QDRANT_HOST="localhost" && QDRANT_PORT="6333"
```

## Best Practices

**DO**:
- ✅ Use factories (`create_retrievers`, `build_graph`) - never module-level init
- ✅ Save working data as Parquet in `data/processed/`
- ✅ Regenerate deliverables via `make deliverables` - never edit CSV files
- ✅ Run `make validate` before committing (100% pass required)
- ✅ Set `PYTHONPATH=.` when running scripts directly

**DON'T**:
- ❌ Initialize retrievers/graphs at module import time
- ❌ Write directly to `deliverables/` (always regenerate)
- ❌ Delete `data/processed/*.parquet` (source of truth)
- ❌ Upgrade RAGAS to 0.3.x without testing (breaking API changes)

## LangGraph State Management

Nodes return **partial state updates** (dicts), not complete states:

```python
from typing import List, TypedDict
from langchain_core.documents import Document

class State(TypedDict):
    question: str
    context: List[Document]
    response: str

def retrieve(state: State) -> dict:
    docs = retriever.invoke(state["question"])
    return {"context": docs}  # Partial update - LangGraph auto-merges

def generate(state: State) -> dict:
    response = llm.invoke(prompt)
    return {"response": response.content}  # Partial update
```

**State evolution**: `{question}` → `{question, context}` → `{question, context, response}`

## Documentation Map

**This file** - High-level overview and critical decisions

**Detailed docs** (avoid duplication):
- `scripts/README.md` - Detailed script documentation, command semantics
- `src/README.md` - Factory pattern guide, module reference
- `data/README.md` - Data flow, manifest schema, file formats
- `Makefile` - All command specifications (scope, duration, cost)
- `README.md` - Project overview, datasets, evaluation results

**Architecture docs** (auto-generated):
- `architecture/` - Claude Agent SDK analyzer outputs (component inventory, diagrams, data flows)
- `docs/initial-architecture.md` - Original design sketch (frozen, historical)

## Reproducibility & Provenance

**Manifests**:
- `data/interim/manifest.json` - Ingestion provenance (SHA-256 checksums, env versions)
- `data/processed/RUN_MANIFEST.json` - Evaluation provenance (links to ingestion manifest)

**Validation**:
```bash
make validate  # Includes manifest validation
# or
PYTHONPATH=. python scripts/validate_manifests.py
```

**Version Pinning** (critical for reproducibility):
```toml
# pyproject.toml
ragas = "==0.2.10"              # API changed in 0.3.x
langgraph = "==0.6.7"           # Graph compilation behavior
langchain-cohere = "==0.4.4"    # Reranker compatibility
```

## HuggingFace Datasets

**Published datasets**:
1. [dwb2023/gdelt-rag-sources-v2](https://huggingface.co/datasets/dwb2023/gdelt-rag-sources-v2) - 38 documents
2. [dwb2023/gdelt-rag-golden-testset-v2](https://huggingface.co/datasets/dwb2023/gdelt-rag-golden-testset-v2) - 12 QA pairs
3. [dwb2023/gdelt-rag-evaluation-inputs](https://huggingface.co/datasets/dwb2023/gdelt-rag-evaluation-inputs) - 60 records
4. [dwb2023/gdelt-rag-evaluation-metrics](https://huggingface.co/datasets/dwb2023/gdelt-rag-evaluation-metrics) - 60 records with RAGAS scores

**Scientific Value**: First publicly available evaluation suite for GDELT-focused RAG systems.

## Troubleshooting

**ModuleNotFoundError: No module named 'src'**
```bash
export PYTHONPATH=.  # Or use make commands (handle this automatically)
```

**Qdrant connection refused**
```bash
make qdrant-up
curl http://localhost:6333/collections  # Verify
```

**Validation failures**
```bash
uv pip install -e .  # Install dependencies
make env             # Check API keys and environment
```

**Deliverables missing**
```bash
make deliverables  # Regenerate from Parquet (always safe)
```

## Notes

- Python 3.11+ required for modern type hints
- RAGAS 0.2.10 pinned (API changed in 0.3.x)
- Parquet-first architecture: working data in `data/processed/`, deliverables derived
- Factory pattern is critical - never initialize retrievers/graphs at module level
- `ra_orchestrators/` directory contains Claude Agent SDK analyzer (optional tooling)
