# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a production-grade RAG (Retrieval-Augmented Generation) system for GDELT (Global Database of Events, Language, and Tone) documentation, built as a certification challenge project for AI Engineering Bootcamp Cohort 8. The system implements multiple retrieval strategies with comprehensive RAGAS-based evaluation.

## Python Environment

**Python Version**: 3.11+ (required for modern type hints)

**Package Manager**: `uv` for dependency management

### Setup

```bash
# Create and activate virtual environment
uv venv --python 3.11
source .venv/bin/activate        # Linux/WSL/Mac

# Install dependencies
uv pip install -e .
```

## Common Commands

### Data Preparation (One-Time Setup)

```bash
# Extract raw PDFs and generate golden testset (ONE-TIME, NOT REQUIRED)
# Scope:
#   1. Load PDFs from data/raw/ using PyMuPDF (12 pages → 38 documents)
#   2. Generate RAGAS golden testset via LLM synthesis (12 QA pairs)
#   3. Persist to data/interim/ in 3 formats (JSONL, Parquet, HFDS)
#   4. Create manifest.json with SHA-256 checksums and provenance
# Output:
#   - data/interim/sources.* (3 files: JSONL, Parquet, HFDS)
#   - data/interim/golden_testset.* (3 files: JSONL, Parquet, HFDS)
#   - data/interim/manifest.json (reproducibility metadata)
# Duration: 5-10 minutes
# Cost: ~$2-3 in OpenAI API calls
# Note: This is ONLY needed if you want to recreate datasets from scratch.
#       The eval pipeline loads pre-built datasets from HuggingFace Hub.
make ingest
# or
PYTHONPATH=. python scripts/ingest_raw_pdfs.py
```

### Development & Validation

```bash
# Validate complete application stack (MUST pass 100%)
# Scope: Checks environment, imports, factory patterns, graph compilation
# Does NOT create vector store or run evaluations
# Output: Terminal report (23/23 checks expected)
# Duration: 1-2 minutes
make validate
# or
PYTHONPATH=. python scripts/run_app_validation.py

# Quick validation test (alias for make validate)
make test
```

### Evaluation & Analysis

```bash
# Run full RAGAS evaluation (DEFAULT: Reuses existing Qdrant collection)
# Scope:
#   1. Loads 38 documents from HuggingFace
#   2. Creates OR reuses Qdrant collection "gdelt_comparative_eval"
#      (configurable via QDRANT_COLLECTION env var)
#   3. Creates 4 retrievers (naive, bm25, ensemble, cohere_rerank)
#   4. Runs 48 RAG queries (12 questions × 4 retrievers)
#   5. Evaluates with 4 RAGAS metrics
#   6. Saves results to data/processed/ (Parquet files, source of truth)
# Output:
#   - data/processed/*_evaluation_inputs.parquet (4 files)
#   - data/processed/*_evaluation_metrics.parquet (4 files)
#   - data/processed/comparative_ragas_results.parquet
#   - data/processed/RUN_MANIFEST.json
# Duration: 20-30 minutes
# Cost: ~$5-6 in OpenAI API calls
# Vector Store: REUSES existing collection if present (FASTER, but may use old embeddings)
# Note: This is the default behavior - fine for testing, but see recreate=true below
make eval
# or
PYTHONPATH=. uv run python scripts/run_eval_harness.py

# Force recreate Qdrant collection (REQUIRED after make ingest)
# Scope: Same as above BUT DELETES and recreates Qdrant collection from scratch
# Duration: 25-35 minutes (extra time for re-embedding all 38 documents)
# Cost: ~$5-6 in OpenAI API calls (same as default)
# Use When:
#   - After running 'make ingest' (to use fresh data)
#   - Documents changed, embeddings model changed
#   - Collection corrupted or using wrong data
#   - Starting evaluation from completely fresh state
# Important: If you just ran 'make ingest', you MUST use recreate=true
#            to ensure new documents/testset are used
make eval recreate=true

# Generate human-readable CSV deliverables from Parquet data
# Scope: Converts Parquet files in data/processed/ to CSV in deliverables/
# Input: data/processed/*.parquet (MUST exist - run make eval first)
# Output: deliverables/evaluation_evidence/*.csv (10 files + manifest)
# Duration: <1 minute
# Cost: FREE (no API calls, pure file conversion)
# Behavior: Silently skips missing Parquet files (no error thrown)
# Dependencies: REQUIRES make eval to have run successfully first
# Note: Deliverables are DERIVED artifacts - always regenerable from Parquet
make deliverables
# or
python scripts/generate_deliverables.py
```

### Publishing (Optional)

```bash
# Publish interim datasets to HuggingFace Hub (OPTIONAL, ONE-TIME)
# Scope: Uploads source documents and golden testset to HuggingFace Hub
# Input: data/interim/*.parquet (must exist - run make ingest first)
# Output:
#   - dwb2023/gdelt-rag-sources-v2 (38 documents)
#   - dwb2023/gdelt-rag-golden-testset-v2 (12 QA pairs)
# Duration: 1-2 minutes (depends on upload speed)
# Prerequisites: HF_TOKEN environment variable must be set
# Note: Only needed if you want to share datasets publicly or update them
make publish-interim
# or
PYTHONPATH=. python scripts/publish_interim_datasets.py

# Publish evaluation results to HuggingFace Hub (OPTIONAL, ONE-TIME)
# Scope: Uploads evaluation results (inputs + metrics) to HuggingFace Hub
# Input: data/processed/*.parquet (must exist - run make eval first)
# Output:
#   - dwb2023/gdelt-rag-evaluation-inputs (48 records: 4 retrievers × 12 questions)
#   - dwb2023/gdelt-rag-evaluation-metrics (48 records with RAGAS scores)
# Duration: 1-2 minutes (depends on upload speed)
# Prerequisites: HF_TOKEN environment variable must be set
# Note: Only needed for public benchmarking or sharing evaluation results
make publish-processed
# or
PYTHONPATH=. python scripts/publish_processed_datasets.py
```

### Infrastructure

```bash
# Start all services (Qdrant, Redis, Neo4j, MinIO, LangGraph)
# Scope: Launches ALL Docker containers
# Services: Qdrant (vector DB), Redis (cache), Neo4j (graph DB),
#           MinIO (object storage), LangGraph API (web service)
# Ports: 6333 (Qdrant), 6379 (Redis), 7474/7687 (Neo4j),
#        9000/9001 (MinIO), 8123 (LangGraph)
# Duration: 10-30 seconds startup
make docker-up

# Start only Qdrant (minimal requirement for this project)
# Scope: Launches ONLY Qdrant vector database
# Port: 6333 (HTTP), 6334 (gRPC)
# Duration: 5-10 seconds startup
# Note: This is all you need for make eval and make validate
make qdrant-up

# Stop all services
# Scope: Stops and removes all Docker containers
# Note: Volumes persist (Qdrant data retained)
make docker-down

# Check environment configuration
# Scope: Displays API key status, Python version, Docker status
# Output: Terminal report (no side effects)
make env
```

### LangGraph Server

```bash
# Local development server with hot reload
uv add langgraph-cli[inmem]
uv run langgraph dev --allow-blocking
# Access at http://localhost:2024
# Studio: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024

# Query via HTTP
curl -X POST http://localhost:8123/invoke \
  -H "Content-Type: application/json" \
  -d '{"question": "What is GDELT?"}'
```

### Architecture Analysis

```bash
# Generate comprehensive architecture documentation
python -m ra_orchestrators.architecture_orchestrator "GDELT architecture"
# Output: ra_output/architecture_{timestamp}/
```

### Jupyter

```bash
# Start Jupyter notebook
make notebook
```

## Complete Workflow Pipeline

### End-to-End Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ Phase 1: Data Preparation (ONE-TIME, Optional)                  │
└─────────────────────────────────────────────────────────────────┘
data/raw/*.pdf (12 pages)
  ↓ [make ingest] ~5-10 min, $2-3
  ├─ PyMuPDF extraction → 38 documents
  ├─ RAGAS testset generation → 12 QA pairs
  └─ Multi-format persistence (JSONL, Parquet, HFDS)
  ↓
data/interim/
  ├─ sources.* (3 formats)
  ├─ golden_testset.* (3 formats)
  └─ manifest.json (provenance)
  ↓ [make publish-interim] ~1-2 min (optional)
  ↓
HuggingFace Hub
  ├─ dwb2023/gdelt-rag-sources-v2
  └─ dwb2023/gdelt-rag-golden-testset-v2

┌─────────────────────────────────────────────────────────────────┐
│ Phase 2: Infrastructure Setup (Required)                        │
└─────────────────────────────────────────────────────────────────┘
[make qdrant-up] ~5-10 sec
  ↓
Qdrant running at http://localhost:6333

┌─────────────────────────────────────────────────────────────────┐
│ Phase 3: Validation (Recommended)                               │
└─────────────────────────────────────────────────────────────────┘
[make validate] ~1-2 min, FREE
  ↓
✅ 23/23 checks pass (environment, imports, factories)

┌─────────────────────────────────────────────────────────────────┐
│ Phase 4: Evaluation (Core Workflow)                             │
└─────────────────────────────────────────────────────────────────┘
[make eval] ~20-30 min, $5-6
  ↓
  1. Load 38 documents from HuggingFace Hub
  2. CREATE Qdrant collection "gdelt_comparative_eval" + embeddings ← QDRANT POPULATED HERE
  3. Create 4 retrievers (naive, bm25, ensemble, cohere_rerank)
  4. Run 48 RAG queries (12 questions × 4 retrievers)
  5. Evaluate with 4 RAGAS metrics (~150 LLM calls)
  6. Save results to data/processed/
  ↓
data/processed/
  ├─ *_evaluation_inputs.parquet (4 files)
  ├─ *_evaluation_metrics.parquet (4 files)
  ├─ comparative_ragas_results.parquet
  └─ RUN_MANIFEST.json

┌─────────────────────────────────────────────────────────────────┐
│ Phase 5: Deliverables (Human-Readable Outputs)                  │
└─────────────────────────────────────────────────────────────────┘
[make deliverables] <1 min, FREE
  ↓
  Parquet → CSV conversion
  ↓
deliverables/evaluation_evidence/
  ├─ *_evaluation_dataset.csv (4 files)
  ├─ *_detailed_results.csv (4 files)
  ├─ comparative_ragas_results.csv
  └─ RUN_MANIFEST.json (copied)

┌─────────────────────────────────────────────────────────────────┐
│ Phase 6: Publishing (Optional)                                  │
└─────────────────────────────────────────────────────────────────┘
[make publish-processed] ~1-2 min (optional)
  ↓
HuggingFace Hub
  ├─ dwb2023/gdelt-rag-evaluation-inputs
  └─ dwb2023/gdelt-rag-evaluation-metrics
```

### Quick Start Workflow

**For most users (evaluation only)**:
```bash
make qdrant-up      # Start vector database
make validate       # Verify environment (100% pass required)
make eval           # Run evaluation (creates Qdrant collection)
make deliverables   # Generate human-readable CSVs
```

**For dataset creators (complete pipeline)**:
```bash
# One-time setup
make ingest             # Extract PDFs → interim datasets
make publish-interim    # Upload to HuggingFace Hub

# Then follow evaluation workflow above
```

### Key Decision Points

1. **Do I need `make ingest`?**
   - ❌ NO if evaluating with existing datasets (most users)
   - ✅ YES if recreating datasets from scratch or modifying PDFs

2. **When does Qdrant get populated?**
   - Answer: During `make eval` (Step 2 of evaluation pipeline)
   - Not during `make ingest` (that's data preparation only)

3. **Do I need publishing commands?**
   - ❌ NO for local development and evaluation
   - ✅ YES if sharing datasets publicly or updating benchmarks

## Key Technologies

### Core Stack
- **LangChain 0.3.19+**: RAG framework, document loaders, retrievers
- **LangGraph 0.6.7**: Graph-based workflow orchestration (pinned)
- **RAGAS 0.2.10**: RAG evaluation metrics (pinned - API changed in 0.3.x)
- **Qdrant**: Vector database (Docker deployment)
- **OpenAI**: GPT-4.1-mini (LLM), text-embedding-3-small (embeddings)
- **Cohere**: rerank-v3.5 (neural reranking)

### Data & Publishing
- **HuggingFace Hub**: Dataset hosting and versioning
- **PyMuPDF 1.26.3+**: PDF parsing for document ingestion
- **Datasets 3.2.0+**: Dataset loading and publishing

### Development
- **Streamlit**: Prototype chat interface
- **Claude Agent SDK 0.1.1+**: Multi-agent architecture analysis

## Architecture Patterns

### 1. Factory Pattern (Critical)

**All retrievers and graphs use factory functions** instead of module-level initialization. This is critical because retrievers depend on runtime-loaded data (documents, vector stores) that don't exist at import time.

```python
# ❌ ANTI-PATTERN: Module-level initialization fails
retriever = vector_store.as_retriever()  # vector_store doesn't exist yet!

# ✅ CORRECT: Factory function
def create_retrievers(documents, vector_store, k=5):
    return {
        "naive": vector_store.as_retriever(search_kwargs={"k": k}),
        "bm25": BM25Retriever.from_documents(documents, k=k),
        "ensemble": EnsembleRetriever(retrievers=[naive, bm25], weights=[0.5, 0.5]),
        "cohere_rerank": ContextualCompressionRetriever(...)
    }
```

**Key Factory Functions**:
- `src/config.py::create_vector_store()` - Qdrant vector store
- `src/retrievers.py::create_retrievers()` - 4 retrieval strategies
- `src/graph.py::build_graph()` - LangGraph workflow for single retriever
- `src/graph.py::build_all_graphs()` - All retriever workflows

### 2. Singleton Pattern (Resource Caching)

Expensive resources (LLM, embeddings, Qdrant client) are cached with `@lru_cache`:

```python
from functools import lru_cache

@lru_cache(maxsize=1)
def get_llm():
    return ChatOpenAI(model="gpt-4.1-mini", temperature=0)

@lru_cache(maxsize=1)
def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small")
```

**Benefits**: Single instance per process, thread-safe, easy to clear for testing

### 3. Strategy Pattern (Retrieval)

Four retrieval strategies implement common interface (`invoke(query)`):

| Strategy | Type | Latency | Use Case |
|----------|------|---------|----------|
| **naive** | Dense vector search | ~50-100ms | General semantic search |
| **bm25** | Sparse keyword matching | ~10-20ms | Exact keywords, proper nouns |
| **ensemble** | Hybrid (50/50 weighted) | ~60-120ms | Balanced semantic + lexical |
| **cohere_rerank** | Two-stage neural reranking | ~200-500ms | Highest quality results |

### 4. Parquet-First Data Architecture

**Working data** (source of truth) lives in `data/processed/` as Parquet files. **Deliverables** in `deliverables/evaluation_evidence/` are derived CSV files regenerated via `make deliverables`.

**Critical architectural fix**: Inference results are saved immediately after Step 3 (not during RAGAS evaluation Step 4). This decouples inference from evaluation and prevents data loss if RAGAS fails mid-run.

## Repository Structure

```
src/                           # Core modular RAG framework
├── config.py                  # Cached singletons (LLM, embeddings, Qdrant)
├── retrievers.py              # Factory: create_retrievers()
├── graph.py                   # Factory: build_graph(), build_all_graphs()
├── state.py                   # TypedDict schema for LangGraph state
├── prompts.py                 # RAG prompt templates
└── utils/
    ├── loaders.py             # HuggingFace dataset loading
    └── manifest.py            # Reproducibility tracking (RUN_MANIFEST.json)

scripts/                       # Executable workflows
├── run_app_validation.py      # Environment and module validation (100% pass required)
├── run_eval_harness.py        # RAGAS evaluation (uses src/ modules)
├── run_full_evaluation.py     # RAGAS evaluation (standalone reference)
├── ingest_raw_pdfs.py         # PDF → interim datasets → golden testset
├── publish_interim_datasets.py     # Upload interim datasets to HuggingFace
├── publish_processed_datasets.py   # Upload evaluation results to HuggingFace
└── generate_deliverables.py  # Parquet → CSV conversion for human review

app/
└── graph_app.py               # LangGraph Server entrypoint (get_app())

data/
├── raw/                       # Source PDFs (immutable)
├── interim/                   # Extracted documents + golden testset + manifest.json
├── processed/                 # Evaluation results (Parquet, source of truth)
└── deliverables/              # Derived CSV files (regenerable via make deliverables)

architecture/                  # Auto-generated architecture docs (Claude Agent SDK)
├── 00_README.md              # System overview and lifecycle
├── docs/                     # Component inventory, data flows, API reference
└── diagrams/                 # Mermaid dependency and system diagrams

ra_orchestrators/             # Multi-agent architecture analysis framework
ra_agents/                    # Agent definitions (JSON)
ra_tools/                     # Tool integrations (MCP, Figma)
ra_output/                    # Analysis outputs (timestamped)

docs/
├── deliverables.md           # Certification submissions
├── certification-challenge-task-list.md  # 100-point grading breakdown
└── initial-architecture.md   # Original design sketch (frozen, not updated)
```

## Data Flow Patterns

### Document Ingestion Pipeline

```
data/raw/2503.07584v3.pdf (12 pages)
  ↓ [scripts/ingest_raw_pdfs.py]
  ├─ PyMuPDF extraction (page-level chunking → 38 documents)
  ├─ RAGAS synthetic testset generation (12 QA pairs)
  └─ Multi-format persistence (JSONL, Parquet, HFDS)
  ↓
data/interim/
  ├─ sources.{jsonl,parquet,hfds}
  ├─ golden_testset.{jsonl,parquet,hfds}
  └─ manifest.json (SHA-256 checksums + provenance)
  ↓ [scripts/publish_interim_datasets.py]
  ↓
HuggingFace Hub
  ├─ dwb2023/gdelt-rag-sources-v2
  └─ dwb2023/gdelt-rag-golden-testset-v2
```

### RAG Query Processing

```
User Question
  ↓ Graph.invoke({"question": "..."})
Retrieve Node
  ↓ retriever.invoke(question)
  ├─ Naive: Qdrant similarity search (k=5)
  ├─ BM25: In-memory lexical matching (k=5)
  ├─ Ensemble: 50/50 weighted merge
  └─ Cohere Rerank: Qdrant (k=20) → rerank → top 5
Retrieved Documents
  ↓ state["context"] = docs
Generate Node
  ↓ Format prompt with context
  ↓ llm.invoke(prompt)
Generated Response
  ↓ state["response"] = answer
Final State {question, context, response}
```

**Latency**: 1.5-3.5 seconds end-to-end (retrieval: 10-500ms, generation: 1-3s)

### Evaluation Pipeline

```
HuggingFace Datasets
  ├─ dwb2023/gdelt-rag-sources-v2 (38 docs)
  └─ dwb2023/gdelt-rag-golden-testset-v2 (12 questions)
  ↓ Load into memory
Build RAG Stack
  ↓ create_vector_store() → create_retrievers() → build_all_graphs()
Inference Loop (4 retrievers × 12 questions = 48 invocations)
  ↓ graph.invoke({"question": q}) for each
  ↓ Save *_evaluation_inputs.parquet immediately (fault tolerance)
RAGAS Evaluation (4 metrics × 12 questions × 4 retrievers)
  ├─ Faithfulness (answer grounded in context)
  ├─ Answer Relevancy (answer addresses question)
  ├─ Context Precision (relevant contexts ranked higher)
  └─ Context Recall (ground truth coverage)
  ↓ ~100-150 LLM calls for metric computation
  ↓ Save *_evaluation_metrics.parquet
Comparative Summary
  ├─ comparative_ragas_results.parquet
  └─ RUN_MANIFEST.json (reproducibility metadata)
  ↓ [make deliverables]
  ↓ Convert Parquet → CSV
deliverables/evaluation_evidence/
  ├─ *_evaluation_dataset.csv (human-readable inputs)
  ├─ *_detailed_results.csv (human-readable metrics)
  ├─ comparative_ragas_results.csv
  └─ RUN_MANIFEST.json (copied)
```

**Runtime**: 20-30 minutes | **Cost**: ~$5-6 in OpenAI API calls

## Environment Variables

```bash
# Required
OPENAI_API_KEY="sk-..."

# Optional (Cohere reranking)
COHERE_API_KEY="..."

# Vector store (URL-first convention)
QDRANT_URL="http://localhost:6333"
# OR
QDRANT_HOST="localhost"
QDRANT_PORT="6333"

# HuggingFace datasets
HF_TOKEN="..."              # For private datasets
HF_SOURCES_REV="abc123"     # Pin source dataset revision
HF_GOLDEN_REV="def456"      # Pin test set revision

# LangSmith (optional but recommended)
LANGSMITH_API_KEY="..."
LANGSMITH_PROJECT="certification-challenge"
LANGSMITH_TRACING="true"
```

## Adding a New Retriever

The system automatically evaluates new retrievers when added to the factory:

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

# 2. Validate (must pass 100%)
make validate

# 3. Evaluate (automatically includes your new retriever)
make eval
```

Results automatically appear in `comparative_ragas_results.csv`.

## Reproducibility & Provenance

### Manifest Generation

- **Ingestion manifest** (`data/interim/manifest.json`):
  - SHA256 checksums for all artifacts
  - Environment versions (Python, LangChain, RAGAS)
  - Model configurations (LLM, embeddings)
  - Execution ID and timestamp

- **Evaluation manifest** (`data/processed/RUN_MANIFEST.json`):
  - RAGAS version and metrics used
  - Retriever configurations (k values, weights, models)
  - Links to ingestion manifest (data provenance)
  - Aggregated evaluation results

### Version Pinning (Critical for Reproducibility)

```toml
# pyproject.toml - exact versions prevent API breakage
ragas = "==0.2.10"              # API changed in 0.3.x
langgraph = "==0.6.7"           # Graph compilation behavior
langchain-cohere = "==0.4.4"    # Reranker compatibility
cohere = "==5.12.0"             # API client
```

### Dataset Pinning (Optional)

```bash
# Pin to specific HuggingFace dataset commits
export HF_SOURCES_REV=main@abc123
export HF_GOLDEN_REV=main@def456

make eval
```

**Without pinning**: Dataset updates can change eval scores
**With pinning**: Same datasets every time → reproducible results

### Deterministic Execution

- Temperature=0 for all LLM calls
- Fixed random seed (42) for sampling
- No randomness in retrieval or evaluation

### RAGAS Testset Generation Behavior

**Important Note**: RAGAS may generate more questions than requested to ensure diversity and quality.

- **Requested size**: 10 questions (configured via `TESTSET_SIZE` parameter)
- **Actual size**: 12 questions (generated by RAGAS 0.2.10)
- **Why**: RAGAS overprovisioning algorithm ensures variety in question types (simple, reasoning, multi-context)
- **Manifests**: Interim manifest records requested size (10), RUN_MANIFEST records actual size (12)
- **SHA-256 validation**: Validates actual artifact files (12 questions), not requested size

This behavior is **expected and documented** - always verify actual file row counts when validating results.

### Manifest Validation

All manifests include SHA-256 checksums for data integrity. Validate them:

```bash
make validate  # Includes manifest validation
# or
PYTHONPATH=. uv run python scripts/validate_manifests.py
```

**Validation checks**:
- ✅ SHA-256 hashes match actual files
- ✅ Provenance chain intact (RUN_MANIFEST → interim manifest → source PDFs)
- ✅ All referenced files exist
- ✅ Configuration reflects actual runtime values (not hardcoded templates)

## HuggingFace Datasets

This project publishes **4 datasets** to HuggingFace Hub:

### Interim Datasets (Raw Data)
1. **[dwb2023/gdelt-rag-sources-v2](https://huggingface.co/datasets/dwb2023/gdelt-rag-sources-v2)** - 38 GDELT documentation pages
2. **[dwb2023/gdelt-rag-golden-testset-v2](https://huggingface.co/datasets/dwb2023/gdelt-rag-golden-testset-v2)** - 12 QA pairs

### Processed Datasets (Evaluation Results)
3. **[dwb2023/gdelt-rag-evaluation-inputs](https://huggingface.co/datasets/dwb2023/gdelt-rag-evaluation-inputs)** - 60 evaluation records (consolidated inputs)
4. **[dwb2023/gdelt-rag-evaluation-metrics](https://huggingface.co/datasets/dwb2023/gdelt-rag-evaluation-metrics)** - 60 evaluation records with RAGAS scores

**Scientific Value**: First publicly available evaluation suite for GDELT-focused RAG systems, enabling reproducible benchmarking of retrieval strategies.

## LangGraph State Management

Graph nodes return **partial state updates** that LangGraph auto-merges:

```python
from typing import List, TypedDict
from langchain_core.documents import Document

class State(TypedDict):
    """State schema for RAG graph"""
    question: str                  # User question
    context: List[Document]        # Retrieved documents
    response: str                  # Generated answer

def retrieve(state: State) -> dict:
    docs = retriever.invoke(state["question"])
    return {"context": docs}  # Partial update

def generate(state: State) -> dict:
    response = llm.invoke(prompt)
    return {"response": response.content}  # Partial update
```

**State evolution**: `{question}` → `{question, context}` → `{question, context, response}`

## Common Development Workflows

### Standard Development Cycle

```bash
# 1. Validate environment and modules (MUST pass 100%)
make validate

# 2. Run comparative evaluation (writes Parquet to data/processed/)
make eval

# 3. Generate human-readable deliverables (Parquet → CSV)
make deliverables

# 4. Publish datasets (optional, one-time)
python scripts/publish_interim_datasets.py        # Raw sources + golden testset
python scripts/publish_processed_datasets.py      # Evaluation results
```

### Quick Validation Before Deployment

```bash
# Fast check - validates environment + imports + factory patterns
make validate
# Expected: 23/23 checks PASS (100%)
```

### Iterative Retriever Development

```bash
# 1. Edit src/retrievers.py (add new retriever to factory)
# 2. Validate
make validate

# 3. Quick test on reused collection (faster)
make eval

# 4. Full test with fresh embeddings (slower, more accurate)
make eval recreate=true
```

## Best Practices

### DO:
✅ Use factories (`create_retrievers`, `build_graph`) instead of module-level instances
✅ Use `@lru_cache` for singletons (LLM, embeddings)
✅ Return partial state updates from LangGraph nodes
✅ Validate with `make validate` before committing
✅ Save working data as Parquet in `data/processed/`
✅ Generate deliverables via `make deliverables` (never write directly to `deliverables/`)
✅ Pin critical dependency versions (RAGAS, LangGraph, Cohere)
✅ Set `PYTHONPATH=.` when running scripts directly

### DON'T:
❌ Create retrievers/graphs at module import time
❌ Hardcode API keys (use environment variables)
❌ Return complete states from LangGraph nodes
❌ Skip validation before deployment
❌ Write directly to `deliverables/` (always regenerate from Parquet)
❌ Manually edit Parquet or CSV files (regenerate via scripts)
❌ Delete `manifest.json` (breaks provenance chain)
❌ Upgrade RAGAS to 0.3.x without testing (breaking API changes)

## Performance Characteristics

### Latency Breakdown
- **BM25 retrieval**: 10-20ms (in-memory)
- **Naive retrieval**: 50-100ms (Qdrant vector search)
- **Ensemble retrieval**: 60-120ms (parallel naive + BM25)
- **Cohere rerank**: 200-500ms (includes API call)
- **LLM generation**: 1,000-3,000ms (GPT-4.1-mini)

**Total end-to-end**: 1.5-3.5 seconds

### Cost Analysis (per query)
- **Embeddings**: ~$0.00001 (one query embedding)
- **LLM generation**: ~$0.001-$0.003 (500-1000 tokens)
- **Cohere rerank**: ~$0.0001 (one rerank call)
- **Total per query**: ~$0.001-$0.003 (dominated by LLM)

**Evaluation cost**: ~$5-6 per full RAGAS run (48 Q&A pairs + ~150 metric LLM calls)

### Scaling Considerations
- **Current**: 38 documents (trivial for Qdrant)
- **100 documents**: No changes needed
- **1,000 documents**: Consider BM25 disk-based index
- **10,000+ documents**: HNSW index tuning, chunking strategy review
- **100,000+ documents**: Distributed Qdrant, pre-filtering, caching layer

## Troubleshooting

### Import Errors
```bash
# Error: ModuleNotFoundError: No module named 'src'
# Fix: Set PYTHONPATH
export PYTHONPATH=.
python scripts/run_eval_harness.py

# Or use make commands (handles PYTHONPATH automatically)
make eval
```

### Qdrant Connection Refused
```bash
# Fix: Start Qdrant
docker-compose up -d qdrant
# Verify
curl http://localhost:6333/collections
```

### Validation Failures
```bash
# Fix: Ensure dependencies installed
uv pip install -e .

# Fix: Ensure Qdrant running
make qdrant-up

# Fix: Ensure API keys set
make env  # Check environment
```

### Deliverables Missing
```bash
# Deliverables are derived artifacts - regenerate them
make deliverables
```

### RAGAS Evaluation Stalls
```bash
# Results saved incrementally - just re-run
python scripts/run_eval_harness.py
```

## WSL Configuration

Git is configured for WSL compatibility:

```bash
git config --global core.autocrlf input
git config --global core.filemode false
git config --global pull.ff only
git config --global init.defaultBranch main
```

## Documentation Structure

- **`CLAUDE.md`** (this file) - Canonical developer guide
- **`README.md`** - Project overview and quick start
- **`architecture/`** - Auto-generated architecture documentation (Claude Agent SDK)
- **`docs/deliverables.md`** - Certification submissions
- **`docs/initial-architecture.md`** - Original design sketch (frozen, historical)
- **`scripts/README.md`** - Script usage guide
- **`data/README.md`** - Data flow and manifest schema
- **`src/README.md`** - Factory pattern guide

## Repository Analyzer Framework

This repository includes the **Repository Analyzer Framework** (`ra_orchestrators/`, `ra_agents/`, `ra_tools/`, `ra_output/`) - a portable multi-agent analysis toolkit.

**Usage**:
```bash
# Generate architecture documentation
python -m ra_orchestrators.architecture_orchestrator "GDELT architecture"
# Output: ra_output/architecture_{timestamp}/
```

See `ra_orchestrators/CLAUDE.md` and `ra_orchestrators/README.md` for details.

## Notes

- This is a production-grade certification challenge project
- All architecture documentation in `architecture/` is auto-generated (do not edit manually)
- `docs/initial-architecture.md` is frozen (historical reference only)
- Parquet-first architecture: working data in `data/processed/`, deliverables derived
- Factory pattern is critical - never initialize retrievers/graphs at module level
- Python 3.11+ required for modern type hints
- RAGAS 0.2.10 pinned (API changed in 0.3.x)