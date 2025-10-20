# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Certification challenge project for AI Engineering Bootcamp Cohort 8: a production-grade RAG system for GDELT (Global Database of Events, Language, and Tone) knowledge graphs with comparative evaluation of 4 retrieval strategies using RAGAS metrics.

**Project Goal**: Compare naive, BM25, ensemble, and Cohere rerank retrievers to determine optimal RAG configuration for GDELT documentation Q&A.

## ⚠️ Documentation Status Notice

**Current Implementation**: This project has evolved through multiple refactoring cycles. The canonical implementation is:
- **Core library**: `src/` modules (config, utils, retrievers, graph, state, prompts)
- **Scripts**: `scripts/single_file.py` (self-contained reference) and `scripts/run_eval_harness.py` (modular)
- **Deployment**: `app/graph_app.py` (LangGraph Platform entrypoint only)
- **UI**: LangGraph Studio (`uv run langgraph dev`)

**Documentation Drift**: Some documentation (initial-initial-architecture.m, deliverables.md) may reference prototype files that were refactored into `src/` modules. When in doubt, trust the code in `src/` and the commands in this CLAUDE.md file.

**Reference Implementation**: Use `scripts/single_file.py` as the learning reference - it shows the full evaluation pipeline in one file without abstractions.

## Essential Commands

### Environment Setup

```bash
# Create virtual environment (Python 3.11 required)
uv venv --python 3.11
source .venv/bin/activate

# Install all dependencies
uv pip install -e .

# Start Qdrant (required for vector search)
docker-compose up -d qdrant

# Verify environment
make env
```

### Common Development Tasks

```bash
# Validate src/ module implementation (must pass 100%)
make validate

# Run full RAGAS evaluation pipeline (~20-30 min, $5-6 cost)
make eval

# Run evaluation with fresh Qdrant collection
make eval recreate=true

# Start Jupyter for notebook work
make notebook

# Clean Python cache
make clean

# View all available commands
make help
```

### Running Specific Components

```bash
# Interactive query via main entry point
python main.py

# Run single-file evaluation pipeline
python scripts/single_file.py

# Validate LangGraph implementation
PYTHONPATH=. python scripts/validate_langgraph.py

# Run evaluation harness (modular version)
PYTHONPATH=. python scripts/run_eval_harness.py

# Test individual retriever
python -c "
from src.utils import load_documents_from_huggingface
from src.config import create_vector_store
from src.retrievers import create_retrievers
from src.graph import build_graph

documents = load_documents_from_huggingface()
vector_store = create_vector_store(documents)
retrievers = create_retrievers(documents, vector_store)
graph = build_graph(retrievers['naive'])
result = graph.invoke({'question': 'What is GDELT?'})
print(result['response'])
"
```

## Core Architecture

### Factory Pattern Philosophy

**Critical Pattern**: This codebase uses factory functions to avoid module-level initialization issues. Retrievers and graphs cannot be created at import time because they depend on runtime data (documents, vector stores).

### Three-Layer Design

**Layer 1: Scripts** (`scripts/`)
- `single_file.py` - Complete standalone evaluation (508 LOC, works without src/)
- `run_eval_harness.py` - Modular evaluation using src/ modules
- `run_app_validation.py` - Application validation (100% pass required before deployment)
- `ingest_raw_pdfs.py` - Extract raw PDFs → interim datasets + RAGAS testset
- `publish_interim_datasets.py` - Upload interim datasets to HuggingFace Hub

**Layer 2: Core Modules** (`src/`)
- `config.py` - Cached singletons: `get_llm()`, `get_embeddings()`, `get_qdrant()`, `create_vector_store()`
- `state.py` - TypedDict: `{question: str, context: List[Document], response: str}`
- `prompts.py` - Template: `BASELINE_PROMPT`
- `utils.py` - Loaders: `load_documents_from_huggingface()`, `load_golden_testset_from_huggingface()`
- `retrievers.py` - Factory: `create_retrievers(documents, vector_store) -> Dict[str, Retriever]`
- `graph.py` - Factory: `build_graph(retriever) -> CompiledGraph`, `build_all_graphs(retrievers) -> Dict[str, CompiledGraph]`

**Layer 3: External Services**
- OpenAI (gpt-4.1-mini + text-embedding-3-small)
- Cohere (rerank-v3.5)
- Qdrant (localhost:6333)
- HuggingFace (dataset hosting)

### LangGraph Workflow Pattern

All 4 retrievers use the same two-node graph:

```
START → retrieve (updates context) → generate (updates response) → END
```

**Key Implementation**:
```python
from langgraph.graph import START, StateGraph
from src.state import State

def retrieve(state):
    docs = retriever.invoke(state["question"])
    return {"context": docs}  # Partial state update

def generate(state):
    docs_content = "\n\n".join(d.page_content for d in state["context"])
    messages = rag_prompt.format_messages(question=state["question"], context=docs_content)
    response = llm.invoke(messages)
    return {"response": response.content}  # Partial state update

graph = StateGraph(State)
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)
compiled = graph.compile()
```

**Why This Pattern**: Node functions return partial state updates (dicts), not complete states. LangGraph automatically merges updates into the state.

### Retriever Implementations

```python
# src/retrievers.py - Factory pattern
def create_retrievers(documents, vector_store, k=5):
    """Create all 4 retriever strategies"""

    # 1. Naive: Dense vector search
    naive = vector_store.as_retriever(search_kwargs={"k": k})

    # 2. BM25: Sparse keyword matching
    bm25 = BM25Retriever.from_documents(documents, k=k)

    # 3. Ensemble: Hybrid (50% dense + 50% sparse)
    ensemble = EnsembleRetriever(
        retrievers=[naive, bm25],
        weights=[0.5, 0.5]
    )

    # 4. Cohere Rerank: Retrieve 20 → rerank to top k
    wide_retriever = vector_store.as_retriever(search_kwargs={"k": max(20, k)})
    compression = ContextualCompressionRetriever(
        base_compressor=CohereRerank(model="rerank-v3.5"),
        base_retriever=wide_retriever
    )

    return {
        "naive": naive,
        "bm25": bm25,
        "ensemble": ensemble,
        "cohere_rerank": compression,
    }
```

**Performance Results** (96.47% Cohere > 94.14% BM25 > 93.96% Ensemble > 91.60% Naive)

### Configuration Management

**Centralized in `src/config.py`**:
```python
from functools import lru_cache
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

@lru_cache(maxsize=1)
def get_llm():
    return ChatOpenAI(model="gpt-4.1-mini", temperature=0)

@lru_cache(maxsize=1)
def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small")

@lru_cache(maxsize=1)
def get_qdrant():
    return QdrantClient(host="localhost", port=6333)
```

**Why `lru_cache`**: Ensures singleton behavior - single LLM/embeddings instance across application lifecycle.

## Data Flow

### Evaluation Pipeline (scripts/single_file.py)

```
1. Load golden testset (12 QA pairs) from HuggingFace
2. Load source documents (38 docs) from HuggingFace
3. Create Qdrant vector store + embed all documents
4. Build 4 retriever strategies
5. Execute 48 RAG queries (4 retrievers × 12 questions)
6. Run RAGAS evaluation (4 metrics × 48 queries = 192 LLM calls)
7. Generate comparative summary CSV
8. Save results to deliverables/evaluation_evidence/

Duration: 20-30 minutes (dominated by RAGAS evaluation)
Cost: ~$5.65 per full run
```

### HuggingFace Datasets

**Sources**: `dwb2023/gdelt-rag-sources` (38 documents)
- GDELT GKG 2.1 architecture docs
- Knowledge graph construction guides
- Baltimore Bridge Collapse case study

**Golden Testset**: `dwb2023/gdelt-rag-golden-testset` (12 QA pairs)
- Synthetic questions generated via RAGAS
- Ground truth answers for evaluation
- Reference contexts for context recall metric

### Multi-Format Persistence

```
data/interim/
├── sources.jsonl          # Human-readable
├── sources.parquet        # Analytics-optimized
├── sources.hfds/          # HuggingFace Dataset (fast loading + versioning)
├── golden_testset.jsonl
├── golden_testset.parquet
├── golden_testset.hfds/
└── manifest.json          # Checksums, versions, provenance

deliverables/evaluation_evidence/
├── naive_evaluation_dataset.csv
├── naive_detailed_results.csv
├── bm25_evaluation_dataset.csv
├── bm25_detailed_results.csv
├── ensemble_evaluation_dataset.csv
├── ensemble_detailed_results.csv
├── cohere_rerank_evaluation_dataset.csv
├── cohere_rerank_detailed_results.csv
├── comparative_ragas_results.csv
└── RUN_MANIFEST.json      # Reproducibility metadata
```

## Adding New Retrievers

### Step-by-Step Pattern

```python
# 1. Add retriever to src/retrievers.py
def create_retrievers(documents, vector_store, k=5):
    # ... existing retrievers ...

    # Add your new retriever
    your_retriever = YourRetrieverClass(
        vectorstore=vector_store,
        k=k
    )

    return {
        "naive": naive,
        "bm25": bm25,
        "ensemble": ensemble,
        "cohere_rerank": compression,
        "your_method": your_retriever,  # <-- Add here
    }

# 2. Re-run evaluation - automatically includes new retriever
python scripts/single_file.py
```

**The system automatically**:
- Evaluates your new retriever against all 12 test questions
- Computes 4 RAGAS metrics
- Includes results in comparative_ragas_results.csv
- Calculates performance vs baseline

## RAGAS Evaluation

### Metrics (all scored 0-1)

```python
from ragas.metrics import (
    Faithfulness,           # Answer grounded in context? (detects hallucinations)
    ResponseRelevancy,      # Answer addresses question?
    ContextPrecision,       # Relevant contexts ranked higher?
    LLMContextRecall,       # Ground truth information retrieved?
)
```

### Schema Requirements (RAGAS 0.2.10)

**Critical**: RAGAS expects specific column names and types.

```python
# Required schema for evaluation
{
    "user_input": str,              # Question (NOT "question")
    "response": str,                # Answer (NOT "answer")
    "retrieved_contexts": List[str], # Retrieved doc.page_content (NOT List[Document]!)
    "reference": str                # Ground truth (NOT "ground_truth")
}
```

**Common Error**: Passing `List[Document]` instead of `List[str]` for `retrieved_contexts` causes RAGAS validation failure. Use `validate_and_normalize_ragas_schema()` in `single_file.py` to prevent this.

## Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-proj-...

# Optional but recommended
COHERE_API_KEY=...              # For rerank retriever (otherwise skipped)
LANGCHAIN_API_KEY=...           # For LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=cert-challenge
HF_TOKEN=hf_...                 # For dataset upload

# Optional for dataset versioning
HF_SOURCES_REV=abc123           # Pin dwb2023/gdelt-rag-sources revision
HF_GOLDEN_REV=abc123            # Pin dwb2023/gdelt-rag-golden-testset revision
```

## Infrastructure Services

### Required Service

```bash
# Start Qdrant (required for vector search)
docker-compose up -d qdrant

# Verify Qdrant is running
docker-compose ps qdrant
```

### Full Service Stack (optional)

```bash
# Start all services
docker-compose up -d

# Available services:
# - Qdrant (6333/6334) - Vector database
# - Redis (6379) - Caching layer
# - Neo4j (7474/7687) - Graph database
# - Phoenix (6006) - Arize observability
# - MinIO (9000/9001) - S3-compatible storage
# - Postgres (5432) - Relational database
# - Adminer (8080) - Database admin UI
```

**Note**: Only Qdrant is required for local development (running scripts directly). For LangGraph Platform deployment, see section below.

## LangGraph Platform Deployment

### Overview: Local vs Platform Deployment

**Local Development** (Scripts):
- Runs Python scripts directly (`python scripts/single_file.py`)
- Only requires Qdrant
- No API server

**LangGraph Platform** (API Server):
- Deploys graphs as REST API endpoints
- Requires: Qdrant + Redis + Postgres
- Containerized deployment with Docker

### Prerequisites

```bash
# Install LangGraph CLI (required for building)
pip install langgraph-cli

# Verify installation
langgraph --version
```

**Required Files**:
- `langgraph.json` - LangGraph configuration (defines graphs, dependencies, environment)
- `app/graph_app.py` - Graph entrypoint with `get_app()` function
- `docker-compose.yml` - Infrastructure orchestration

### Local Development (No Docker)

**Modern, fully self-contained workflow** - no Redis/Postgres/Docker required.

```bash
# Install LangGraph CLI with in-memory runtime
uv add langgraph-cli[inmem]

# Launch local server with Studio UI
uv run langgraph dev --allow-blocking
```

**What happens**:
- Installs LangGraph CLI with `inmem` runtime backend (ephemeral, no database)
- Launches local server at `http://localhost:2024`
- Opens Studio UI automatically in browser
- `--allow-blocking` enables synchronous dev runs (helpful for notebooks, VS Code)
- Supports hot-reload on graph changes

**Advantages**:
- No Docker dependencies
- Instant iteration
- Ideal for bootcamp demos, notebooks, quick RAG testing
- Same runtime semantics as production

**Access**:
- **API**: `http://localhost:2024`
- **Studio UI**: Auto-opens or visit `http://localhost:2024`
- **API Docs**: `http://localhost:2024/docs`

**Environment variables** (same as Docker):
```bash
export OPENAI_API_KEY=sk-...
export COHERE_API_KEY=...
export LANGSMITH_API_KEY=...  # Optional
```

### Building the Docker Image

```bash
# Build the LangGraph image
langgraph build -t gdelt-image

# This creates a Docker image containing:
# - Your application code (src/, app/)
# - All Python dependencies from langgraph.json
# - LangGraph runtime
```

**What happens during build**:
1. Reads `langgraph.json` for dependencies and configuration
2. Creates Wolfi-based container (lightweight, secure)
3. Installs Python 3.11 and all dependencies
4. Bundles your source code
5. Tags image as `gdelt-image`

**Common build issues**:
- `langgraph.json not found` - Must run from project root
- Missing dependencies - Ensure all packages listed in `langgraph.json`

### Starting the Platform

```bash
# Start all required services
docker-compose up -d

# Services will start in dependency order:
# 1. langgraph-redis (healthcheck: redis-cli ping)
# 2. langgraph-postgres (healthcheck: pg_isready)
# 3. qdrant (healthcheck: TCP connection)
# 4. langgraph-api (depends on above 3 services)
```

**Service Endpoints**:
- LangGraph API: `http://localhost:8123` (port 8000 inside container)
- Qdrant: `http://localhost:6333`
- Redis: `localhost:6379`
- Postgres: `localhost:5433` (5432 inside container)

### Verifying Deployment

```bash
# 1. Health check
curl http://localhost:8123/ok

# Expected response: {"ok": true}

# 2. Check service status
docker-compose ps
# All services must show "healthy" status

# 3. View logs
docker-compose logs langgraph-api --tail 50
docker-compose logs langgraph-redis
docker-compose logs langgraph-postgres
```

### Accessing the Deployment

**Studio UI** (Visual interface):
```
https://smith.langchain.com/studio/?baseUrl=http://localhost:8123
```

**Note**: The UI is hosted on LangChain's cloud domain but connects to your local backend via the `baseUrl` parameter.

**API Documentation** (Swagger):
```
http://localhost:8123/docs
```

This provides an interactive reference for all available API endpoints.

### LangGraph Configuration (`langgraph.json`)

The `langgraph.json` file defines the deployment configuration:

```json
{
  "name": "gdelt-langgraph",
  "python_version": "3.11",
  "image_distro": "wolfi",
  "dependencies": [...],  // All required packages
  "env": {
    "QDRANT_HOST": "qdrant",      // Container hostname
    "QDRANT_PORT": "6333",
    "OPENAI_API_KEY": "${OPENAI_API_KEY}",
    "COHERE_API_KEY": "${COHERE_API_KEY}"
  },
  "graphs": {
    "gdelt": "app.graph_app:get_app"  // Entrypoint function
  }
}
```

**Key Configuration Points**:
- `graphs.gdelt` - Defines graph name and Python import path
- `app.graph_app:get_app` - Must return a CompiledGraph
- `env.QDRANT_HOST` - Uses container name `qdrant` (Docker networking)
- Environment variables interpolated from `.env` file

### Graph Entrypoint

The `app/graph_app.py:get_app()` function is the deployment entrypoint:

```python
def get_app():
    """LangGraph Server entrypoint - returns a CompiledGraph"""
    docs = load_documents_from_huggingface()
    vs = create_vector_store(docs, recreate_collection=False)
    rets = create_retrievers(docs, vs, k=5)
    graphs = build_all_graphs(rets)
    return graphs["cohere_rerank"]  # Default retriever
```

**Important**:
- Must return a `CompiledGraph` (not a dict of graphs)
- Uses `recreate_collection=False` to preserve Qdrant data
- Loads documents on startup (cached after first run)

### Querying the Deployed API

**Essential Endpoints**:

```bash
# 1. Health check
curl http://localhost:8123/ok
# Response: {"ok": true}

# 2. View API documentation (open in browser)
http://localhost:8123/docs

# 3. List available assistants/graphs
curl -X POST http://localhost:8123/assistants/search \
  -H 'Content-Type: application/json' \
  -d '{"limit":10,"offset":0}'

# Response shows assistant_id from langgraph.json (e.g., "gdelt")
```

**Stateless Run** (no thread persistence):

```bash
# Streaming response
curl -X POST http://localhost:8123/runs/stream \
  -H 'Content-Type: application/json' \
  -d '{
    "assistant_id": "gdelt",
    "input": {"question": "What is GDELT?"},
    "stream_mode": "updates"
  }'

# Blocking (wait for result)
curl -X POST http://localhost:8123/runs/wait \
  -H 'Content-Type: application/json' \
  -d '{
    "assistant_id": "gdelt",
    "input": {"question": "What is GDELT?"}
  }'
```

**Stateful Run** (with thread for conversation memory):

```bash
# 1. Create a thread
THREAD_ID=$(curl -X POST http://localhost:8123/threads \
  -H 'Content-Type: application/json' \
  -d '{}' | jq -r '.thread_id')

echo "Created thread: $THREAD_ID"

# 2. Run on thread (blocking)
curl -X POST http://localhost:8123/threads/$THREAD_ID/runs/wait \
  -H 'Content-Type: application/json' \
  -d '{
    "assistant_id": "gdelt",
    "input": {"question": "What is GDELT GKG 2.1?"}
  }'

# 3. Run follow-up query on same thread (maintains context)
curl -X POST http://localhost:8123/threads/$THREAD_ID/runs/wait \
  -H 'Content-Type: application/json' \
  -d '{
    "assistant_id": "gdelt",
    "input": {"question": "Tell me more about the themes"}
  }'

# 4. Streaming on thread
curl -X POST http://localhost:8123/threads/$THREAD_ID/runs/stream \
  -H 'Content-Type: application/json' \
  -d '{
    "assistant_id": "gdelt",
    "input": {"question": "How does GDELT handle events?"},
    "stream_mode": "updates"
  }'
```

**Note**: The `assistant_id` value ("gdelt") comes from the `graphs` section in `langgraph.json`.

### Troubleshooting

**API won't start**:
```bash
# Check service dependencies
docker-compose ps

# All services must show "healthy" status
# If redis/postgres/qdrant are unhealthy, langgraph-api won't start

# View startup logs
docker-compose logs langgraph-api --tail 100
```

**Connection refused from langgraph-api to qdrant**:
```bash
# Verify network connectivity
docker-compose exec langgraph-api ping qdrant

# Should resolve to container IP
# If fails, check that all services on same network (gdelt-network)
```

**Missing environment variables**:
```bash
# Check that .env file exists and contains:
# OPENAI_API_KEY=sk-...
# COHERE_API_KEY=...

# Restart services after updating .env
docker-compose down
docker-compose up -d
```

**Graph not found error**:
```bash
# Verify langgraph.json graphs configuration
cat langgraph.json | jq '.graphs'

# Should show: {"gdelt": "app.graph_app:get_app"}

# Rebuild image if langgraph.json changed
langgraph build -t gdelt-image
docker-compose up -d --force-recreate langgraph-api
```

**404 on endpoints**:
```bash
# Common mistake: trying to use /invoke instead of /runs/stream
# ❌ Wrong: POST /invoke
# ✅ Correct: POST /runs/stream (stateless) or POST /threads/{id}/runs/wait (stateful)

# Check Swagger docs for available endpoints
# Open in browser: http://localhost:8123/docs
```

**assistant_id not found**:
```bash
# The assistant_id must match the graph name in langgraph.json
# Check your configuration:
cat langgraph.json | jq '.graphs'

# Use the key name (e.g., "gdelt") as assistant_id in API calls

# Verify assistants are available:
curl -X POST http://localhost:8123/assistants/search \
  -H 'Content-Type: application/json' \
  -d '{"limit":10}'
```

**Studio UI not connecting**:
```bash
# Ensure baseUrl parameter matches your deployment
# For Docker: https://smith.langchain.com/studio/?baseUrl=http://localhost:8123
# For local dev: http://localhost:2024

# Check CORS if UI can't reach backend
docker-compose logs langgraph-api | grep -i cors
```

### Reference Documentation

**Official Documentation**:
- [Self-Hosted Docker Deployment](https://langchain-ai.github.io/langgraphjs/how-tos/deploy-self-hosted/#using-docker-compose)
- [LangGraph Server API Reference](https://docs.langchain.com/langsmith/server-api-ref)
- [Local Server Tutorial](https://langchain-ai.github.io/langgraph/tutorials/langgraph-platform/local-server/)
- [Streaming API](https://docs.langchain.com/langgraph-platform/streaming)

**Quick Reference**:
- **Local dev**: `uv run langgraph dev --allow-blocking` → http://localhost:2024
- **Docker**: `langgraph build -t gdelt-image && docker-compose up -d` → http://localhost:8123
- **Studio UI (Docker)**: https://smith.langchain.com/studio/?baseUrl=http://localhost:8123
- **API Docs**: http://localhost:8123/docs (Docker) or http://localhost:2024/docs (local)

## Common Development Patterns

### Loading Documents from HuggingFace

```python
from src.utils import load_documents_from_huggingface

# Load latest version
documents = load_documents_from_huggingface()

# Pin to specific revision for reproducibility
documents = load_documents_from_huggingface(revision="abc123")

# Or use environment variable
# export HF_SOURCES_REV=abc123
documents = load_documents_from_huggingface()
```

### Creating Vector Store

```python
from src.config import create_vector_store

# Reuse existing collection (fast)
vector_store = create_vector_store(documents)

# Recreate collection (slow but ensures clean state)
vector_store = create_vector_store(documents, recreate_collection=True)

# Custom collection name
vector_store = create_vector_store(
    documents,
    collection_name="my_custom_collection",
    recreate_collection=True
)
```

### Querying the RAG System

```python
from src.utils import load_documents_from_huggingface
from src.config import create_vector_store
from src.retrievers import create_retrievers
from src.graph import build_all_graphs

# Setup
documents = load_documents_from_huggingface()
vector_store = create_vector_store(documents)
retrievers = create_retrievers(documents, vector_store)
graphs = build_all_graphs(retrievers)

# Query using compiled graph
result = graphs["cohere_rerank"].invoke({"question": "What is GDELT GKG 2.1?"})
print(result["response"])

# Access retriever directly
docs = retrievers["bm25"].invoke("What is GDELT?")
for doc in docs:
    print(doc.page_content)
```

### Reproducibility Manifest Pattern

```python
# scripts/generate_run_manifest.py creates RUN_MANIFEST.json
{
    "timestamp": "2025-01-17T12:00:00Z",
    "models": {
        "llm": "gpt-4.1-mini",
        "embeddings": "text-embedding-3-small",
        "reranker": "rerank-v3.5"
    },
    "retrievers": ["naive", "bm25", "ensemble", "cohere_rerank"],
    "ragas_metrics": ["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
    "dataset_checksums": {
        "sources": "sha256:...",
        "golden_testset": "sha256:..."
    },
    "package_versions": {...}
}
```

**All evaluation runs must generate manifest** for scientific reproducibility.

## Key Implementation Files

When working with the code, reference these locations:

**Core System**:
- [src/config.py:28-35](src/config.py) - `get_llm()` cached singleton
- [src/config.py:39-46](src/config.py) - `get_embeddings()` cached singleton
- [src/config.py:70-127](src/config.py) - `create_vector_store()` factory
- [src/retrievers.py:20-89](src/retrievers.py) - `create_retrievers()` factory (all 4 strategies)
- [src/graph.py:21-106](src/graph.py) - `build_graph()` factory (LangGraph compilation)
- [src/graph.py:109-141](src/graph.py) - `build_all_graphs()` convenience factory
- [src/utils.py:15-75](src/utils.py) - `load_documents_from_huggingface()`
- [src/state.py:7-10](src/state.py) - State TypedDict schema
- [src/prompts.py:4-12](src/prompts.py) - `BASELINE_PROMPT` template

**Evaluation Pipeline**:
- [scripts/single_file.py](scripts/single_file.py) - Standalone evaluation (works without src/)
- [scripts/run_eval_harness.py](scripts/run_eval_harness.py) - Modular evaluation (uses src/)
- [scripts/validate_langgraph.py](scripts/validate_langgraph.py) - Validation harness (must pass 100%)

**Utilities**:
- [Makefile](Makefile) - All make targets
- [docker-compose.yml](docker-compose.yml) - Infrastructure services

## Validation Requirements

**Before any code changes**, run validation:

```bash
make validate
```

**Expected output**: 100% pass rate across all checks:
1. Environment validation (API keys, Qdrant connectivity)
2. Module import validation (all src/ modules importable)
3. Retriever factory pattern (creates all 4 retrievers)
4. Graph compilation (all 4 graphs compile)
5. Functional testing (test queries work)

**Exit code 0 = ready for deployment**
**Exit code 1 = must fix issues before proceeding**

## Known Limitations & Future Work

**Current Limitations**:
1. No async execution for parallel retriever evaluation (4x speedup opportunity)
2. No embedding cache (repeated API calls for same documents)
3. Ensemble weights hardcoded at 50/50 (should be tunable)
4. No query expansion or HyDE (hypothetical document embeddings)

**Recommended Improvements** (post-certification):
1. Implement `CacheBackedEmbeddings` for embedding reuse
2. Async retriever evaluation with `asyncio.gather()`
3. Tune ensemble weights via grid search
4. Add parent document retrieval strategy
5. Implement semantic caching with Redis
6. Add LangSmith evaluation datasets for continuous monitoring

## Documentation

### Documentation Guide

This project has comprehensive documentation organized across multiple files. Use this guide to find what you need:

**Core Documentation**:
- **[README.md](README.md)** - Project overview, quick start, installation (380 lines)
- **[CLAUDE.md](CLAUDE.md)** (this file) - Complete technical reference for AI assistants (965+ lines)
- **[docs/deliverables.md](docs/deliverables.md)** - Certification challenge answers (1,152 lines)
- **[docs/initial-architecture.m](docs/initial-architecture.m)** - System design patterns and decisions (18KB)
- **[docs/certification-challenge-task-list.md](docs/certification-challenge-task-list.md)** - Scoring rubric

**Architecture Documentation** (auto-generated comprehensive analysis):
- **[architecture/README.md](architecture/README.md)** - Architecture overview and navigation guide (1,100 lines)
- **[architecture/docs/01_component_inventory.md](architecture/docs/01_component_inventory.md)** - Module-by-module reference
- **[architecture/diagrams/02_architecture_diagrams.md](architecture/diagrams/02_architecture_diagrams.md)** - Visual system overview
- **[architecture/docs/03_data_flows.md](architecture/docs/03_data_flows.md)** - Sequence diagrams and pipelines
- **[architecture/docs/04_api_reference.md](architecture/docs/04_api_reference.md)** - Comprehensive API documentation

**Directory-Specific Guides**:
- **[scripts/README.md](scripts/README.md)** - All evaluation and utility scripts (5 scripts documented)
- **[src/README.md](src/README.md)** - Factory pattern guide, module reference, adding retrievers
- **[data/README.md](data/README.md)** - Data flow, manifest schema, file formats, lineage

**Repository Analyzer Framework** (optional, for codebase analysis):
- **[ra_orchestrators/README.md](ra_orchestrators/README.md)** - Multi-domain agent orchestration framework
- **[ra_orchestrators/CLAUDE.md](ra_orchestrators/CLAUDE.md)** - Framework usage guide for AI assistants

**Quick Navigation**:
- 🔍 Want to understand the codebase? → Start with [architecture/README.md](architecture/README.md)
- 🚀 Want to run evaluations? → See "Common Development Tasks" (this file)
- 🛠️ Want to add a retriever? → See [src/README.md](src/README.md#quick-start-adding-a-new-retriever)
- 📊 Want to understand data flow? → See [architecture/docs/03_data_flows.md](architecture/docs/03_data_flows.md)
- ✅ Want to validate setup? → Run `make validate` (must pass 100%)
- 📐 Want architecture diagrams? → See [architecture/diagrams/02_architecture_diagrams.md](architecture/diagrams/02_architecture_diagrams.md)

### Module Inventory (src/)

**Core Modules**:
- `config.py` - Cached singletons for LLM, embeddings, Qdrant client
- `retrievers.py` - Factory for 4 retriever strategies (naive, BM25, ensemble, Cohere rerank)
- `graph.py` - LangGraph workflow builders
- `state.py` - TypedDict schema for graph state
- `utils/loaders.py` - HuggingFace dataset loaders
- `utils/manifest.py` - RUN_MANIFEST.json generation
- `prompts.py` - RAG prompt templates

### Script Inventory (scripts/)

**Evaluation Scripts**:
- `run_app_validation.py` - Validate src/ modules + environment (2 min, $0, must pass 100%)
- `run_eval_harness.py` - Modular RAGAS evaluation (20-30 min, $5-6)
- `run_full_evaluation.py` - Standalone RAGAS evaluation (20-30 min, $5-6)

**Data Pipeline Scripts**:
- `ingest_raw_pdfs.py` - Extract raw PDFs → interim datasets + RAGAS testset
- `publish_interim_datasets.py` - Upload interim datasets to HuggingFace Hub

### External Dependencies

- **OpenAI**: gpt-4.1-mini (LLM), text-embedding-3-small (embeddings)
- **Cohere**: rerank-v3.5 (contextual compression)
- **Qdrant**: Vector database (localhost:6333)
- **HuggingFace Hub**: Dataset hosting and versioning
- **RAGAS**: RAG evaluation framework (v0.2.10)
- **LangChain**: RAG abstractions and orchestration
- **LangGraph**: Stateful graph workflows

## Repository Analyzer Framework

This repository includes an optional **Repository Analyzer Framework** (`ra_orchestrators/`, `ra_agents/`, `ra_tools/`) - a portable, drop-in analysis toolkit for comprehensive codebase analysis. This framework was used to generate the comprehensive architecture documentation in `architecture/`.

### What It Does

The framework provides multi-domain orchestration with specialized agents for:
- **Architecture Analysis** - Code structure, patterns, diagrams, data flows, API documentation
- **UX/UI Design** - User research, information architecture, visual design, prototyping
- **DevOps** (Future) - Infrastructure analysis, CI/CD workflows, IaC generation
- **Testing** (Future) - Test strategy, coverage analysis, test generation

### Key Features

1. **Portable** - Drop into any repository without modification
2. **No Collisions** - `ra_` prefix avoids conflicts with existing code
3. **Timestamped Outputs** - Each run creates `ra_output/{domain}_{YYYYMMDD_HHMMSS}/`
4. **Extensibility** - Base framework supports new domains in <1 day
5. **Reusability** - Agents and tools shared across domains

### Usage

```bash
# Architecture analysis (used to generate architecture/ docs)
python -m ra_orchestrators.architecture_orchestrator

# UX design workflow
python -m ra_orchestrators.ux_orchestrator "Project Name"

# With timeout for long-running analyses
timeout 1800 python -m ra_orchestrators.architecture_orchestrator
```

### Documentation

- **[ra_orchestrators/README.md](ra_orchestrators/README.md)** - User-facing usage guide
- **[ra_orchestrators/CLAUDE.md](ra_orchestrators/CLAUDE.md)** - Complete technical reference for AI assistants
- **[ra_orchestrators/claude-agents-research.md](ra_orchestrators/claude-agents-research.md)** - Comprehensive research (832 lines)

### When to Use

- Generating comprehensive architecture documentation
- Analyzing new codebases for patterns and structure
- Creating UX design specifications from requirements
- Performing multi-domain repository analysis

**Note**: The framework is separate from the core GDELT RAG application and can be safely ignored for typical development work. It's primarily useful for documentation generation and deep codebase analysis.