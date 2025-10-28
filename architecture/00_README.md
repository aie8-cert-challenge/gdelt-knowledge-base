# Repository Architecture Documentation

## Overview

### About This Documentation

This documentation set provides comprehensive architectural analysis of the GDELT RAG (Retrieval-Augmented Generation) system. It is intended for:

- **New developers** seeking to understand the codebase structure and design patterns
- **Architects** evaluating design decisions and system integration points
- **API users** looking for implementation examples and usage patterns
- **DevOps engineers** preparing for deployment and configuration management

The documentation was generated through automated analysis of the source code and captures the system state as of January 19, 2025.

### About the GDELT RAG System

The GDELT RAG system is a production-grade question-answering pipeline built on top of GDELT (Global Database of Events, Language, and Tone) knowledge graphs. The system implements multiple retrieval strategies to compare effectiveness across different retrieval approaches, from basic semantic search to advanced neural reranking.

**Core Capabilities:**
- Multi-strategy document retrieval (dense, sparse, hybrid, reranked)
- LangGraph-based orchestration for composable RAG workflows
- Comprehensive RAGAS-based evaluation framework
- Production serving via LangGraph Server
- End-to-end reproducibility through manifest tracking

**Technology Stack:**
- **Frameworks**: LangChain 0.3.19+, LangGraph 0.6.7, RAGAS 0.2.10
- **Vector Store**: Qdrant (local Docker or cloud)
- **LLM**: OpenAI GPT-4o-mini (temperature=0 for determinism)
- **Embeddings**: OpenAI text-embedding-3-small (1536 dimensions)
- **Reranking**: Cohere rerank-v3.5

### Documentation Generation

This documentation was automatically generated on **January 19, 2025** using the Claude Agent SDK architecture analysis framework. The analysis covers:

- 13 Python modules in the `src/` package
- 5 executable scripts in `scripts/`
- 2 application modules in `app/`
- 28 external dependencies
- ~3,500 lines of production code

The documentation is organized into four main sections detailed below.

## Quick Start

### For New Developers

1. **Start with the Architecture Overview** (this document)
   - Understand the 5-layer architecture and key design patterns
   - Review the factory pattern usage (critical for understanding initialization)

2. **Read the Data Flow Analysis** ([03_data_flows.md](docs/03_data_flows.md))
   - Follow the document ingestion pipeline from PDF to vector store
   - Trace a RAG query from user input to generated response
   - Understand the evaluation pipeline workflow

3. **Explore the API Reference** ([04_api_reference.md](docs/04_api_reference.md))
   - Review configuration functions in `src/config.py`
   - Study the retriever factory pattern in `src/retrievers.py`
   - Examine graph construction in `src/graph.py`

4. **Run the Validation Script**:
   ```bash
   python scripts/run_app_validation.py
   ```
   This validates your environment and demonstrates correct usage patterns.

5. **Inspect the Component Inventory** ([01_component_inventory.md](docs/01_component_inventory.md))
   - Browse the detailed module-by-module reference
   - Understand dependencies and file locations

### For Architects

1. **Review the Architecture Diagrams** ([02_architecture_diagrams.md](diagrams/02_architecture_diagrams.md))
   - System architecture: 5-layer stack visualization
   - Component relationships: initialization vs runtime patterns
   - Module dependencies: import graph and coupling analysis
   - Retriever strategy pattern: factory-based creation flow

2. **Study Design Patterns** (see "Key Design Patterns" section below)
   - Factory pattern for deferred initialization
   - Singleton pattern for resource caching
   - Strategy pattern for retriever implementations

3. **Evaluate Performance Characteristics** (see Data Flows document, Performance section)
   - Retrieval latency: 10-500ms depending on strategy
   - LLM generation: 1-3 seconds per query
   - Evaluation costs: ~$5-6 per full RAGAS run

4. **Examine Reproducibility Guarantees** (see Component Inventory, Manifest section)
   - Version pinning for critical dependencies (RAGAS 0.2.10, LangGraph 0.6.7)
   - Manifest generation for data lineage
   - Deterministic execution (temperature=0, fixed random seeds)

### For API Users

1. **Start with the API Reference** ([04_api_reference.md](docs/04_api_reference.md))
   - Configuration module: `src/config.py` (LLM, embeddings, vector store)
   - Retriever factory: `src/retrievers.py` (4 retrieval strategies)
   - Graph builders: `src/graph.py` (LangGraph workflow construction)

2. **Review Usage Examples**:
   - Basic RAG query: API Reference, "Basic RAG Query" section
   - Custom retriever configuration: API Reference, "Custom Retriever Configuration"
   - RAGAS evaluation: API Reference, "Running Evaluations"

3. **Check Environment Configuration**:
   - Required: `OPENAI_API_KEY`
   - Optional: `COHERE_API_KEY` (for reranking)
   - Vector store: `QDRANT_URL` or `QDRANT_HOST`/`QDRANT_PORT`
   - See API Reference, "Environment Variables" section for full list

4. **Run Example Scripts**:
   ```bash
   # Validate environment
   python scripts/run_app_validation.py

   # Run evaluation
   python scripts/run_eval_harness.py

   # Start LangGraph Server
   langgraph dev
   ```

## Architecture at a Glance

### System Architecture

The system follows a clean 5-layer architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│ Execution Layer                                             │
│ - app/graph_app.py (LangGraph Server entrypoint)           │
│ - scripts/run_eval_harness.py (RAGAS evaluation)           │
│ - scripts/ingest_raw_pdfs.py (data ingestion)              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Orchestration Layer                                         │
│ - src/graph.py (LangGraph workflow builders)                │
│ - src/state.py (TypedDict state schema)                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Retrieval Layer                                             │
│ - src/retrievers.py (factory for 4 retrieval strategies)   │
│   • naive (dense vector search)                            │
│   • bm25 (sparse keyword matching)                          │
│   • ensemble (hybrid 50/50 weighted)                        │
│   • cohere_rerank (neural reranking)                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Data Layer                                                  │
│ - src/utils/loaders.py (HuggingFace dataset loading)        │
│ - src/utils/manifest.py (reproducibility tracking)          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Configuration Layer                                         │
│ - src/config.py (cached LLM, embeddings, Qdrant client)    │
│ - src/prompts.py (RAG prompt templates)                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ External Services                                           │
│ - OpenAI (LLM + embeddings)                                 │
│ - Qdrant (vector database)                                  │
│ - Cohere (reranking API)                                    │
│ - HuggingFace Hub (dataset hosting)                         │
└─────────────────────────────────────────────────────────────┘
```

**Design Principles:**
- **Downward dependencies**: Each layer depends only on layers below it
- **Factory pattern**: Stateful components created via factory functions
- **Lazy initialization**: Resources instantiated only when needed via `@lru_cache`
- **Environment-driven**: All configuration from environment variables

### Key Design Patterns

#### 1. Factory Pattern (Critical)

All retrievers and graphs use factory functions instead of module-level initialization:

```python
# ❌ ANTI-PATTERN: Module-level initialization fails
retriever = vector_store.as_retriever()  # vector_store doesn't exist yet!

# ✅ CORRECT: Factory function
def create_retrievers(documents, vector_store, k=5):
    return {
        "naive": vector_store.as_retriever(search_kwargs={"k": k}),
        "bm25": BM25Retriever.from_documents(documents, k=k),
        # ... other retrievers
    }
```

**Rationale**: Retrievers depend on runtime-loaded data (documents, vector stores) that don't exist at import time.

#### 2. Singleton Pattern (Resource Caching)

Expensive resources (LLM, embeddings, Qdrant client) are cached with `@lru_cache`:

```python
@lru_cache(maxsize=1)
def get_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)
```

**Benefits**:
- Single instance per process (no duplicate connections)
- Thread-safe caching
- Easy to clear for testing: `get_llm.cache_clear()`

#### 3. Strategy Pattern (Retrieval)

Four retrieval strategies implement a common interface (`invoke(query)`):

- **Naive**: Dense vector search (baseline)
- **BM25**: Sparse lexical matching
- **Ensemble**: 50/50 weighted hybrid
- **Cohere Rerank**: Two-stage retrieval + neural reranking

All strategies are created by the same factory function and return the same interface, enabling comparative evaluation.

#### 4. State Management (LangGraph)

Graph nodes return partial state updates that LangGraph auto-merges:

```python
def retrieve(state: State) -> dict:
    docs = retriever.invoke(state["question"])
    return {"context": docs}  # Partial update

def generate(state: State) -> dict:
    response = llm.invoke(prompt)
    return {"response": response.content}  # Partial update
```

**State evolution**: `{question}` → `{question, context}` → `{question, context, response}`

### Technology Stack

#### Core Framework Dependencies
- **LangChain 0.3.19+**: RAG framework, document loaders, retrievers
- **LangGraph 0.6.7**: Graph-based workflow orchestration (pinned version)
- **LangChain-OpenAI 0.3.7+**: OpenAI LLM and embeddings integration
- **LangChain-Cohere 0.4.4**: Cohere reranker integration (pinned version)
- **LangChain-Qdrant 0.2.0+**: Qdrant vector store integration

#### Evaluation & Data
- **RAGAS 0.2.10**: RAG evaluation metrics (pinned - API changed in 0.3.x)
- **HuggingFace Datasets 3.2.0+**: Dataset loading and publishing
- **PyMuPDF 1.26.3+**: PDF parsing for document ingestion

#### External Services
- **OpenAI API**: GPT-4o-mini (LLM) + text-embedding-3-small (embeddings)
- **Qdrant**: Vector database (local Docker or cloud)
- **Cohere API**: rerank-v3.5 model for neural reranking
- **HuggingFace Hub**: Dataset hosting (dwb2023/gdelt-rag-sources, dwb2023/gdelt-rag-golden-testset)

## Component Overview

### Public API (`src/` package)

The `src/` package provides the main public API for building RAG systems:

| Module | Purpose | Key Exports |
|--------|---------|-------------|
| `config.py` | Configuration & resource factories | `get_llm()`, `get_embeddings()`, `get_qdrant()`, `create_vector_store()` |
| `retrievers.py` | Retriever factory | `create_retrievers()` → dict of 4 retriever strategies |
| `graph.py` | LangGraph builders | `build_graph()`, `build_all_graphs()` |
| `state.py` | State schema | `State` TypedDict (question, context, response) |
| `prompts.py` | Prompt templates | `BASELINE_PROMPT` (RAG prompt) |
| `utils/loaders.py` | Document loading | `load_documents_from_huggingface()`, `load_golden_testset_from_huggingface()` |
| `utils/manifest.py` | Reproducibility | `generate_run_manifest()` |

**Example Usage** (from `src/__init__.py`):
```python
from src.utils import load_documents_from_huggingface
from src.config import create_vector_store, get_llm
from src.retrievers import create_retrievers
from src.graph import build_all_graphs

# Load data
documents = load_documents_from_huggingface()
vector_store = create_vector_store(documents, recreate_collection=True)

# Build RAG stack
retrievers = create_retrievers(documents, vector_store)
graphs = build_all_graphs(retrievers)

# Query
result = graphs['naive'].invoke({"question": "What is GDELT?"})
```

### Application Layer (`app/`)

LangGraph Server entrypoint for production deployment:

| Module | Purpose |
|--------|---------|
| `graph_app.py` | Defines `get_app()` function that returns compiled graph for LangGraph Server |

The `get_app()` function loads documents, creates vector store, builds retrievers and graphs, then returns the best-performing retriever (cohere_rerank) for serving.

**Deployment**:
```bash
langgraph dev  # Local development server
langgraph up   # Production deployment
```

### Scripts (`scripts/`)

Executable workflows for data processing, evaluation, and validation:

| Script | Purpose | Runtime |
|--------|---------|---------|
| `ingest_raw_pdfs.py` | PDF → documents → RAGAS testset → HuggingFace | 2-5 min |
| `publish_interim_datasets.py` | Upload datasets to HuggingFace Hub with READMEs | 1-2 min |
| `run_eval_harness.py` | RAGAS evaluation (using src/ modules) | 20-30 min |
| `run_full_evaluation.py` | RAGAS evaluation (inline code, original) | 20-30 min |
| `run_app_validation.py` | Validate environment and src/ modules | 1-2 min |

### Configuration

All configuration is environment-driven with sensible defaults:

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

# LangSmith (optional)
LANGSMITH_API_KEY="..."
LANGSMITH_PROJECT="certification-challenge"
LANGSMITH_TRACING="true"
```

See [04_api_reference.md](docs/04_api_reference.md) for complete environment variable reference.

## Data Flow Patterns

### Document Ingestion Pipeline

```
Raw PDFs (data/raw/)
  ↓ PyMuPDFLoader
LangChain Documents (38 docs)
  ↓ RAGAS TestsetGenerator
Golden Testset (12 questions + contexts)
  ↓ Persist in 3 formats
Interim Storage (data/interim/)
  - sources.docs.{jsonl, parquet, hfds}
  - golden_testset.{jsonl, parquet, hfds}
  - manifest.json (SHA256 checksums + metadata)
  ↓ publish_interim_datasets.py
HuggingFace Hub
  - dwb2023/gdelt-rag-sources-v2
  - dwb2023/gdelt-rag-golden-testset-v2
```

**Script**: `scripts/ingest_raw_pdfs.py`
**Runtime**: 2-5 minutes
**Cost**: ~$0.50-$1.00 (OpenAI API for testset generation)

See [03_data_flows.md](docs/03_data_flows.md) for detailed sequence diagrams.

### RAG Query Processing

```
User Question
  ↓ Graph.invoke({"question": "..."})
Retrieve Node
  ↓ retriever.invoke(question)
  ├─ Naive: Qdrant similarity search (k=5)
  ├─ BM25: In-memory lexical matching (k=5)
  ├─ Ensemble: 50/50 weighted merge (k=5+5)
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

**Latency**:
- Retrieval: 10-500ms (depends on strategy)
- Generation: 1-3 seconds (GPT-4o-mini)
- **Total**: 1.5-3.5 seconds end-to-end

### Retriever Strategies

| Strategy | Type | How It Works | Use Case | Latency |
|----------|------|--------------|----------|---------|
| **Naive** | Dense vector | OpenAI embeddings + Qdrant cosine similarity (k=5) | General semantic search | ~50-100ms |
| **BM25** | Sparse lexical | TF-IDF scoring over in-memory index (k=5) | Exact keyword matching, proper nouns | ~10-20ms |
| **Ensemble** | Hybrid | 50/50 weighted combination of naive + BM25 | Balanced semantic + lexical | ~60-120ms |
| **Cohere Rerank** | Two-stage | Dense retrieval (k=20) → neural reranking → top 5 | Highest quality (willing to pay cost) | ~200-500ms |

**Factory Function**: `src/retrievers.py::create_retrievers()`

**Evaluation Results** (from comparative analysis):
- Cohere Rerank: **Highest average score** (best overall)
- Ensemble: Strong balance between precision and recall
- Naive: Good baseline performance
- BM25: Competitive on keyword-heavy queries

See [02_architecture_diagrams.md](diagrams/02_architecture_diagrams.md) for retriever strategy pattern visualization.

### Evaluation Pipeline

```
HuggingFace Datasets
  ├─ dwb2023/gdelt-rag-sources (38 docs)
  └─ dwb2023/gdelt-rag-golden-testset (12 questions)
  ↓ Load into memory
Build RAG Stack
  ↓ create_vector_store() → create_retrievers() → build_all_graphs()
Inference Loop (4 retrievers × 12 questions = 48 invocations)
  ↓ graph.invoke({"question": q}) for each
Raw Datasets (response + retrieved_contexts)
  ↓ Save immediately (fault tolerance)
  ├─ naive_raw_dataset.parquet
  ├─ bm25_raw_dataset.parquet
  ├─ ensemble_raw_dataset.parquet
  └─ cohere_rerank_raw_dataset.parquet
  ↓ Convert to RAGAS format
RAGAS Evaluation (4 metrics × 12 questions × 4 retrievers)
  ├─ Faithfulness (answer grounded in context)
  ├─ Answer Relevancy (answer addresses question)
  ├─ Context Precision (relevant contexts ranked higher)
  └─ Context Recall (ground truth coverage)
  ↓ ~100-150 LLM calls for metric computation
Detailed Results (per-question metric scores)
  ├─ naive_detailed_results.csv
  ├─ bm25_detailed_results.csv
  ├─ ensemble_detailed_results.csv
  └─ cohere_rerank_detailed_results.csv
  ↓ Aggregate and compare
Comparative Summary
  ├─ comparative_ragas_results.csv (sorted by average)
  └─ RUN_MANIFEST.json (reproducibility metadata)
```

**Script**: `scripts/run_eval_harness.py`
**Runtime**: 20-30 minutes
**Cost**: ~$5-6 in OpenAI API calls

See [03_data_flows.md](docs/03_data_flows.md), "Evaluation Pipeline Flow" section for detailed sequence diagram.

## Key Features

### Retriever Strategies

The system implements four distinct retrieval strategies for comparative evaluation:

#### 1. Naive (Dense Vector Search)
- **Implementation**: OpenAI embeddings + Qdrant cosine similarity
- **Strengths**: Good semantic understanding, handles paraphrasing well
- **Weaknesses**: May miss exact keyword matches
- **Best for**: General semantic search, conceptual queries
- **Parameters**: k=5, distance=cosine

#### 2. BM25 (Sparse Keyword Matching)
- **Implementation**: TF-IDF based ranking over in-memory index
- **Strengths**: Fast, handles exact keywords and proper nouns well
- **Weaknesses**: No semantic understanding, vocabulary mismatch issues
- **Best for**: Keyword-heavy queries, entity names, technical terms
- **Parameters**: k=5, in-memory index

#### 3. Ensemble (Hybrid)
- **Implementation**: 50/50 weighted combination of Naive + BM25
- **Strengths**: Balances semantic and lexical retrieval
- **Weaknesses**: Twice the retrieval cost, potential redundancy
- **Best for**: Production use where both semantic and lexical matching important
- **Parameters**: weights=[0.5, 0.5], k=5 per retriever

#### 4. Cohere Rerank (Neural Reranking)
- **Implementation**: Wide retrieval (k=20) → Cohere rerank-v3.5 → top 5
- **Strengths**: Highest quality results, contextual relevance scoring
- **Weaknesses**: Higher latency (~500ms), additional API cost
- **Best for**: High-value queries where quality > cost
- **Parameters**: initial_k=20, rerank_top_n=5, model=rerank-v3.5

**Tradeoffs Summary**:
- **Latency**: BM25 (10ms) < Naive (100ms) < Ensemble (120ms) < Rerank (500ms)
- **Quality**: Rerank > Ensemble > Naive ≥ BM25 (based on RAGAS evaluation)
- **Cost**: BM25 (free) < Naive ($0.0001/query) < Ensemble ($0.0002) < Rerank ($0.0003)

### Reproducibility

The system implements comprehensive reproducibility tracking:

#### Manifest Generation
- **Ingestion manifest** (`data/interim/manifest.json`):
  - SHA256 checksums for all artifacts
  - Environment versions (Python, LangChain, RAGAS)
  - Model configurations (LLM, embeddings)
  - Schema samples from datasets
  - Execution ID and timestamp

- **Evaluation manifest** (`RUN_MANIFEST.json`):
  - RAGAS version and metrics used
  - Retriever configurations (k values, weights, models)
  - Links to ingestion manifest (data provenance)
  - Aggregated evaluation results
  - Generated by `src/utils/manifest.py::generate_run_manifest()`

#### Version Pinning
- **Exact versions** (reproducibility-critical):
  - `ragas==0.2.10` (API changed in 0.3.x)
  - `langgraph==0.6.7` (graph compilation behavior)
  - `langchain-cohere==0.4.4` (reranker compatibility)
  - `cohere==5.12.0` (API client)

- **Dataset pinning** (optional):
  - `HF_SOURCES_REV` environment variable pins source dataset
  - `HF_GOLDEN_REV` environment variable pins test set
  - Prevents score drift from dataset updates

#### Deterministic Execution
- Temperature=0 for all LLM calls
- Fixed random seed (42) for sampling
- Reproducible RAGAS testset generation
- No randomness in retrieval or evaluation

### Evaluation Framework

RAGAS (RAG Assessment) integration provides four metrics:

#### 1. Faithfulness
**Question**: Is the answer grounded in the retrieved context?
**Method**: LLM checks if answer statements are supported by context
**Range**: 0.0-1.0 (higher = better)
**Purpose**: Detect hallucination

#### 2. Answer Relevancy
**Question**: Does the answer address the question asked?
**Method**: LLM evaluates question-answer alignment
**Range**: 0.0-1.0 (higher = better)
**Purpose**: Measure response quality

#### 3. Context Precision
**Question**: Are relevant contexts ranked higher than irrelevant ones?
**Method**: Checks if ground truth contexts appear early in retrieved list
**Range**: 0.0-1.0 (higher = better)
**Purpose**: Evaluate retrieval ranking quality

#### 4. Context Recall
**Question**: Do retrieved contexts cover the ground truth?
**Method**: LLM checks if ground truth information present in retrieved contexts
**Range**: 0.0-1.0 (higher = better)
**Purpose**: Measure retrieval completeness

**RAGAS Configuration**:
- Version: 0.2.10 (pinned)
- Evaluator LLM: GPT-4o-mini (same as RAG LLM)
- Timeout: 360 seconds per evaluation
- Batch processing: ~100-150 LLM calls per full evaluation

See [04_api_reference.md](docs/04_api_reference.md), "Evaluation Harness Script" section for usage examples.

### Production Serving

LangGraph Server provides production-ready HTTP API:

#### Deployment
```bash
# Local development
langgraph dev

# Production
langgraph up --config app/langgraph.json
```

#### Features
- **Hot reloading**: Code changes reflected without restart (dev mode)
- **State management**: LangGraph handles state persistence
- **Concurrency**: Async request handling
- **Error handling**: Automatic retry and error logging
- **Tracing**: LangSmith integration for debugging

#### API Endpoints
```bash
# Invoke graph
POST http://localhost:8123/invoke
Body: {"question": "What is GDELT?"}

# Stream response
POST http://localhost:8123/stream
Body: {"question": "What is GDELT?"}

# Health check
GET http://localhost:8123/health
```

#### Configuration
- **Default retriever**: cohere_rerank (best-performing)
- **Collection reuse**: `recreate_collection=False` to avoid data loss on restart
- **Cold start**: ~5-10 seconds (document loading + vector store connection)
- **Request latency**: 1.5-3.5 seconds (retrieval + generation)

See `app/graph_app.py` for implementation details.

## Architecture Highlights

### Strengths

1. **Factory Pattern Discipline**
   - All stateful components use factory functions
   - Prevents import-time initialization failures
   - Enables flexible configuration and testing
   - Clear initialization order: data → store → retrievers → graphs

2. **Comprehensive Evaluation**
   - Four retrieval strategies compared on same metrics
   - RAGAS provides standardized, LLM-based evaluation
   - Comparative analysis identifies best performer
   - Reproducible results via manifest tracking

3. **Production-Ready Serving**
   - LangGraph Server provides HTTP API with minimal code
   - Async request handling for concurrency
   - LangSmith integration for observability
   - Environment-driven configuration

4. **Data Lineage**
   - Ingestion manifest links datasets to evaluation runs
   - SHA256 checksums prevent data corruption
   - HuggingFace Hub provides version control for datasets
   - End-to-end provenance from PDF to evaluation results

5. **Developer Experience**
   - Type hints throughout for IDE support
   - Comprehensive docstrings with examples
   - Validation script for environment checking
   - Clear error messages and troubleshooting guides

### Design Decisions

#### Why Factory Pattern?
**Decision**: Use factory functions instead of module-level initialization
**Rationale**: Retrievers depend on runtime-loaded data (documents, vector stores) that don't exist at import time. Module-level initialization would fail.
**Alternative considered**: Lazy initialization on first use
**Tradeoff**: More verbose (must call factory functions), but explicit and predictable

#### Why LangGraph?
**Decision**: Use LangGraph for workflow orchestration
**Rationale**: Provides state management, composability, and production serving (LangGraph Server) out of the box
**Alternative considered**: Custom orchestration with LangChain
**Tradeoff**: Additional dependency, but significant reduction in boilerplate

#### Why RAGAS 0.2.10?
**Decision**: Pin to RAGAS 0.2.10 instead of upgrading to 0.3.x
**Rationale**: API changed significantly in 0.3.x; 0.2.10 is stable and compatible with LangChain wrappers
**Alternative considered**: Upgrade to 0.3.x
**Tradeoff**: Miss new features in 0.3.x, but avoid breaking changes

#### Why Multiple Storage Formats?
**Decision**: Persist datasets in JSONL, Parquet, and HuggingFace formats
**Rationale**: Maximizes tool compatibility (JSONL for inspection, Parquet for analytics, HF for ecosystem integration)
**Alternative considered**: Single format (HuggingFace only)
**Tradeoff**: Storage overhead (~3× files), but worth it for flexibility

#### Why Qdrant?
**Decision**: Use Qdrant for vector storage
**Rationale**: Docker-friendly, good performance, rich filtering capabilities, mature Python client
**Alternative considered**: Chroma, Pinecone, Weaviate
**Tradeoff**: Requires separate service (Docker), but better than in-memory FAISS for production

### Performance Characteristics

#### Latency Breakdown
- **BM25 retrieval**: 10-20ms (in-memory, very fast)
- **Naive retrieval**: 50-100ms (Qdrant vector search on 38 docs)
- **Ensemble retrieval**: 60-120ms (parallel naive + BM25, then merge)
- **Cohere rerank**: 200-500ms (includes API call overhead)
- **LLM generation**: 1,000-3,000ms (GPT-4o-mini response)

**Total end-to-end**: 1.5-3.5 seconds (retrieval + generation)

#### Cost Analysis (per query)
- **Embeddings**: ~$0.00001 (one query embedding)
- **LLM generation**: ~$0.001-$0.003 (500-1000 tokens)
- **Cohere rerank**: ~$0.0001 (one rerank call)
- **Total per query**: ~$0.001-$0.003 (dominated by LLM)

**Evaluation cost**: ~$5-6 per full RAGAS run (48 Q&A pairs + ~150 metric LLM calls)

#### Bottlenecks
1. **LLM API latency** (1-3 seconds per call)
   - **Impact**: Dominates end-to-end time
   - **Mitigation**: Use faster models (gpt-3.5-turbo), streaming responses

2. **RAGAS evaluation time** (15-25 minutes for 48 examples)
   - **Impact**: Long feedback loop for retriever tuning
   - **Mitigation**: Reduce test set size for iteration, use caching

3. **Qdrant write performance** (collection recreation)
   - **Impact**: ~5-10 seconds to re-embed 38 documents
   - **Mitigation**: Use `recreate_collection=False` in production

4. **Cohere rerank latency** (~300ms overhead vs naive)
   - **Impact**: Slower queries, but highest quality results
   - **Mitigation**: Use for high-value queries only, cache results

#### Scaling Considerations
- **Current**: 38 documents, trivial for Qdrant
- **100 documents**: No changes needed
- **1,000 documents**: Consider BM25 disk-based index (current in-memory)
- **10,000+ documents**: HNSW index tuning, chunking strategy review
- **100,000+ documents**: Distributed Qdrant, pre-filtering, caching layer

## Getting Started with the Code

### Prerequisites

**Required**:
- Python 3.11+ (for modern type hints)
- Docker and Docker Compose (for Qdrant)
- OpenAI API key

**Optional**:
- Cohere API key (for reranking)
- HuggingFace token (for private datasets)
- LangSmith API key (for tracing)

**Installation**:
```bash
# Clone repository
git clone <repo-url>
cd cert-challenge

# Install dependencies (using uv)
uv sync

# Or using pip
pip install -e .

# Start Qdrant
docker-compose up -d qdrant

# Verify Qdrant
curl http://localhost:6333/collections
```

### Configuration

Create `.env` file in project root:

```bash
# Required
OPENAI_API_KEY="sk-..."

# Optional
COHERE_API_KEY="..."
HF_TOKEN="..."
LANGSMITH_API_KEY="..."

# Qdrant (defaults to localhost:6333)
QDRANT_URL="http://localhost:6333"

# HuggingFace dataset revisions (optional, for reproducibility)
HF_SOURCES_REV="abc123def456"
HF_GOLDEN_REV="def456abc123"

# LangSmith (optional)
LANGSMITH_PROJECT="certification-challenge"
LANGSMITH_TRACING="true"
```

**Validation**:
```bash
# Run validation script
python scripts/run_app_validation.py

# Expected output: All validations passed (100% pass rate)
```

### Basic Usage

**Simple RAG query**:
```python
from src.utils import load_documents_from_huggingface
from src.config import create_vector_store
from src.retrievers import create_retrievers
from src.graph import build_all_graphs

# Load data
documents = load_documents_from_huggingface()

# Build RAG stack
vector_store = create_vector_store(documents)
retrievers = create_retrievers(documents, vector_store, k=5)
graphs = build_all_graphs(retrievers)

# Query with best retriever
result = graphs['cohere_rerank'].invoke({"question": "What is GDELT?"})

# Access results
print(f"Answer: {result['response']}")
print(f"Retrieved {len(result['context'])} documents")
for i, doc in enumerate(result['context']):
    print(f"\n[{i+1}] {doc.page_content[:100]}...")
```

**Run evaluation**:
```bash
# Full RAGAS evaluation (20-30 minutes, ~$5-6)
python scripts/run_eval_harness.py

# Results saved to deliverables/evaluation_evidence/
# - comparative_ragas_results.csv (summary)
# - *_detailed_results.csv (per-question metrics)
# - RUN_MANIFEST.json (reproducibility)
```

**Start server**:
```bash
# Local development
langgraph dev

# Query via HTTP
curl -X POST http://localhost:8123/invoke \
  -H "Content-Type: application/json" \
  -d '{"question": "What is GDELT?"}'
```

See [04_api_reference.md](docs/04_api_reference.md) for comprehensive API documentation.

## Documentation Structure

### Component Inventory
**File**: [docs/01_component_inventory.md](docs/01_component_inventory.md)
**Purpose**: Detailed module-by-module reference with line numbers, function signatures, and dependencies

**Read this when**:
- Looking up specific function locations
- Understanding module dependencies
- Reviewing component metadata (versions, configurations)
- Debugging import issues

**Contents**:
- Public API modules (`src/` package)
- Application modules (`app/`)
- Scripts (`scripts/`)
- Dependencies (pyproject.toml)
- Architectural patterns (factory, singleton, state management)
- File organization and directory structure

### Architecture Diagrams
**File**: [diagrams/02_architecture_diagrams.md](diagrams/02_architecture_diagrams.md)
**Purpose**: Visual representation of system architecture, data flows, and component relationships

**Read this when**:
- Getting high-level system overview
- Understanding component interactions
- Reviewing initialization sequences
- Planning system changes or extensions

**Contents**:
- System architecture (5-layer stack)
- Component relationships (initialization vs runtime)
- Data flow architecture (PDF → evaluation)
- Class hierarchies (TypedDict, retrievers, graphs)
- Module dependencies (import graph)
- Retriever strategy pattern (factory visualization)
- LangGraph workflow architecture
- Evaluation pipeline architecture

### Data Flow Analysis
**File**: [docs/03_data_flows.md](docs/03_data_flows.md)
**Purpose**: Detailed sequence diagrams and flow descriptions for all major pipelines

**Read this when**:
- Tracing data through the system
- Understanding pipeline execution order
- Debugging data transformation issues
- Optimizing performance bottlenecks

**Contents**:
- Document ingestion flow (PDF → testset → HuggingFace)
- RAG query flow (question → retrieval → generation)
- Retriever initialization flow (factory pattern)
- Evaluation pipeline flow (RAGAS end-to-end)
- LangGraph server flow (production serving)
- Error handling and edge cases
- Performance considerations (latency, cost, bottlenecks)

### API Reference
**File**: [docs/04_api_reference.md](docs/04_api_reference.md)
**Purpose**: Comprehensive API documentation with signatures, parameters, return types, and examples

**Read this when**:
- Using the public API in your code
- Writing new scripts or applications
- Troubleshooting API usage errors
- Customizing retriever or graph configurations

**Contents**:
- Configuration module (`src/config.py`)
- Retrievers module (`src/retrievers.py`)
- Graph module (`src/graph.py`)
- State and prompts modules
- Utils module (loaders, manifest)
- Scripts (ingestion, evaluation, validation)
- Environment variables reference
- Usage patterns and best practices
- Troubleshooting guide

## Project Statistics

**Codebase Summary**:
- **Total modules**: 13 (7 in `src/`, 5 in `scripts/`, 1 in `app/`)
- **Core modules**: 6 (`config`, `retrievers`, `graph`, `state`, `prompts`, `utils`)
- **Executable scripts**: 5 (ingestion, publishing, 2× evaluation, validation)
- **Application modules**: 1 (`graph_app.py` - LangGraph Server)

**Lines of Code** (~3,500 total):
- `src/`: ~1,500 lines (public API)
- `scripts/`: ~1,800 lines (executable workflows)
- `app/`: ~20 lines (server entrypoint)
- `tests/`: ~0 lines (placeholder, uses validation script instead)

**Dependencies**:
- **Framework**: 5 (LangChain, LangGraph, LangChain-OpenAI, LangChain-Cohere, LangChain-Qdrant)
- **Vector DB**: 1 (qdrant-client)
- **Retrieval**: 1 (rank-bm25)
- **Evaluation**: 1 (RAGAS)
- **Data**: 4 (HuggingFace Hub, datasets, PyMuPDF, rapidfuzz)
- **Development**: 4 (Jupyter, Streamlit, LangGraph CLI, Claude Agent SDK)
- **Utilities**: 3 (OpenAI, Cohere, python-dotenv)
- **Total**: 20 direct dependencies

**Data Artifacts**:
- **Source documents**: 38 (from GDELT PDFs)
- **Golden testset**: 12 questions (RAGAS-generated)
- **Storage formats**: 3 per dataset (JSONL, Parquet, HuggingFace)
- **Evaluation outputs**: 4 retrievers × 3 files + 2 summary files = 14 files

**Retriever Strategies**: 4
- Naive (dense vector search)
- BM25 (sparse keyword matching)
- Ensemble (hybrid 50/50)
- Cohere Rerank (neural reranking)

**RAGAS Metrics**: 4
- Faithfulness
- Answer Relevancy
- Context Precision
- Context Recall

**External Services**: 4
- OpenAI (LLM + embeddings)
- Qdrant (vector database)
- Cohere (reranking API)
- HuggingFace Hub (dataset hosting)

## Additional Resources

### Related Documentation

**In This Repository**:
- This README (architecture overview and navigation)
- [Component Inventory](docs/01_component_inventory.md) (detailed module reference)
- [Architecture Diagrams](diagrams/02_architecture_diagrams.md) (visual system overview)
- [Data Flow Analysis](docs/03_data_flows.md) (sequence diagrams and pipelines)
- [API Reference](docs/04_api_reference.md) (comprehensive API documentation)

**Project Documentation**:
- `README.md` (project root) - Project overview and quick start
- `deliverables/` - Evaluation results and evidence
- Dataset READMEs on HuggingFace:
  - [dwb2023/gdelt-rag-sources-v2](https://huggingface.co/datasets/dwb2023/gdelt-rag-sources-v2)
  - [dwb2023/gdelt-rag-golden-testset-v2](https://huggingface.co/datasets/dwb2023/gdelt-rag-golden-testset-v2)

### External Resources

**Framework Documentation**:
- [LangChain Documentation](https://python.langchain.com/) - RAG framework
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/) - Workflow orchestration
- [RAGAS Documentation](https://docs.ragas.io/) - RAG evaluation metrics
- [Qdrant Documentation](https://qdrant.tech/documentation/) - Vector database

**API References**:
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference) - LLM and embeddings
- [Cohere Rerank API](https://docs.cohere.com/reference/rerank) - Neural reranking
- [HuggingFace Datasets](https://huggingface.co/docs/datasets/) - Dataset library

**GDELT Resources**:
- [GDELT Project](https://www.gdeltproject.org/) - Global event database
- [GDELT Documentation](https://www.gdeltproject.org/data.html) - Data documentation

**Best Practices**:
- [LangChain Best Practices](https://python.langchain.com/docs/guides/best_practices) - RAG patterns
- [LangGraph Server Deployment](https://langchain-ai.github.io/langgraph/cloud/deployment/) - Production serving
- [RAGAS Cookbook](https://docs.ragas.io/en/latest/howtos/) - Evaluation patterns

## Maintenance Notes

### When to Update This Documentation

This documentation should be updated when:

1. **Major architectural changes**:
   - New modules added to `src/`
   - New retrieval strategies implemented
   - LangGraph workflow structure changes
   - Dependency versions changed (especially pinned versions)

2. **API changes**:
   - Function signatures modified
   - New configuration options added
   - Environment variables added/removed
   - Return types changed

3. **Data flow modifications**:
   - New pipelines added (e.g., fine-tuning, distillation)
   - Storage formats changed
   - External services replaced

4. **Evaluation framework updates**:
   - RAGAS version upgraded (especially to 0.3.x)
   - New metrics added
   - Evaluation workflow changed

### How This Documentation Was Generated

This documentation was generated using the Claude Agent SDK architecture analysis framework on January 19, 2025.

**Generation Process**:
1. **Component Discovery**: Recursively scanned `src/`, `app/`, and `scripts/` directories
2. **Code Analysis**: Parsed Python files for imports, functions, classes, docstrings
3. **Dependency Mapping**: Built import graph and identified coupling
4. **Pattern Detection**: Identified factory pattern, singleton pattern, state management
5. **Diagram Generation**: Created Mermaid diagrams for architecture, data flows, class hierarchies
6. **Documentation Synthesis**: Combined analysis into structured markdown documents

**Excluded from Analysis** (as requested):
- `ra_orchestrators/` (analysis framework itself)
- `ra_agents/` (analysis framework agents)
- `ra_tools/` (analysis framework tools)
- `ra_output/` (generated outputs, except this documentation)
- `.venv/` (virtual environment)

**Tools Used**:
- Claude Agent SDK (orchestration and analysis)
- AST parsing for Python code analysis
- Grep/Glob for file discovery
- Mermaid for diagram generation

**To Regenerate**:
```bash
# Re-run architecture analysis
python -m ra_orchestrators.architecture_analysis

# Or manually update individual sections
# See ra_output/architecture_20251019_223602/docs/ for templates
```

---

**Documentation Version**: 1.0
**Generated**: January 19, 2025
**Analysis Framework**: Claude Agent SDK
**System Version**: 0.1.0
**Git Branch**: GDELT
**Git Commit**: a34bb0a (docs: update deliverables and architecture)

---

**For questions, corrections, or suggested improvements**, please:
- Open an issue in the repository
- Contact the maintainers (dwb2023)
- Review the source code directly (all documentation links include file paths and line numbers)

This is a living document - contributions welcome!
