# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Certification challenge project for AI Engineering Bootcamp Cohort 8: a production-grade RAG system for GDELT (Global Database of Events, Language, and Tone) knowledge graphs with comparative evaluation of 4 retrieval strategies using RAGAS metrics.

**Project Goal**: Compare naive, BM25, ensemble, and Cohere rerank retrievers to determine optimal RAG configuration for GDELT documentation Q&A.

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
- `validate_langgraph.py` - Validation script (100% pass required before deployment)
- `ingest.py` - PDF extraction + RAGAS testset generation
- `upload_to_hf.py` - HuggingFace dataset publisher

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

**Note**: Only Qdrant is required for baseline RAG. Other services support advanced features not currently used.

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

- [README.md](README.md) - Project overview and quick start (348 lines)
- [docs/deliverables.md](docs/deliverables.md) - Complete certification answers
- [architecture/README.md](architecture/README.md) - Comprehensive architecture documentation (1,073 lines)
- [architecture/docs/01_component_inventory.md](architecture/docs/01_component_inventory.md) - Module catalog (521 lines)
- [architecture/diagrams/02_architecture_diagrams.md](architecture/diagrams/02_architecture_diagrams.md) - Visual diagrams (786 lines)
- [architecture/docs/03_data_flows.md](architecture/docs/03_data_flows.md) - Data flow analysis (947 lines)
- [architecture/docs/04_api_reference.md](architecture/docs/04_api_reference.md) - API documentation (3,121 lines)

**Total architecture documentation**: 5,375 lines across 5 files.