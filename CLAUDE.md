# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Production RAG system for GDELT documentation with 4 retrieval strategies (naive, BM25, ensemble, Cohere rerank) and RAGAS evaluation. Built for AI Engineering Bootcamp Cohort 8 certification challenge.

**Core Principle**: Parquet-first data architecture where `data/processed/*.parquet` is the source of truth and `deliverables/*.csv` files are regenerable derived artifacts.

## Quick Start

```bash
# Setup
uv venv --python 3.11 && source .venv/bin/activate && uv pip install -e .

# Run full pipeline
make qdrant-up && make validate && make eval && make deliverables
```

## Common Commands

### Development
```bash
make validate          # Validate environment + modules (23 checks, must pass 100%)
make eval             # Run full evaluation: inference → RAGAS → summarize
make deliverables     # Generate human-readable CSV files from Parquet

# Three-phase evaluation (for cost control)
make inference        # Phase 1: RAG inference only (~$3-4)
make eval-metrics     # Phase 2: RAGAS scoring (~$2)
make summarize        # Phase 3: Create summary + manifest ($0)
```

### Infrastructure
```bash
make qdrant-up        # Start Qdrant vector database
make docker-up        # Start all services (Qdrant, Redis, etc.)
make notebook         # Launch Jupyter
```

### Publishing
```bash
make publish-interim     # Upload sources + testset to HuggingFace
make publish-processed   # Upload evaluation results to HuggingFace
```

### Cleaning
```bash
make clean-deliverables  # Remove CSV files (regenerable)
make clean-processed     # Remove evaluation results
make clean-all           # Reset everything
```

## Critical Architecture Patterns

### 1. Factory Pattern (Mandatory)

Retrievers depend on runtime-loaded data that doesn't exist at module import time:

```python
# ❌ BREAKS - vector store doesn't exist yet
retriever = vector_store.as_retriever()

# ✅ WORKS - deferred initialization via factory
def create_retrievers(documents, vector_store, k=5):
    return {"naive": vector_store.as_retriever(search_kwargs={"k": k}), ...}
```

**Key factories** in `src/`:
- `config.py::create_vector_store()` - Qdrant store (NOT cached, allows multiple collections)
- `config.py::get_llm()`, `get_embeddings()` - Singletons via `@lru_cache`
- `retrievers.py::create_retrievers()` - All 4 retrieval strategies
- `graph.py::build_graph()` - LangGraph workflow for single retriever
- `graph.py::build_all_graphs()` - All 4 LangGraph workflows

### 2. Parquet-First Data Flow

```
data/processed/*.parquet  →  [make deliverables]  →  deliverables/*.csv
     (source of truth)                                    (regenerable)
```

- **Parquet**: Machine-optimized, ZSTD compressed, never delete
- **CSV**: Human-readable, derived, delete/regenerate anytime
- **Never write directly to deliverables/** - always regenerate via script

### 3. Three-Phase Evaluation Pipeline

The evaluation pipeline is split into 3 independent phases for cost control and resilience:

```bash
make eval  # Runs all 3 phases sequentially:
  1. make inference     # Phase 1: RAG inference (~$3-4)
  2. make eval-metrics  # Phase 2: RAGAS scoring (~$2)
  3. make summarize     # Phase 3: Summary + manifest ($0)
```

**Critical architectural decision**: The pipeline was refactored to decouple inference from evaluation:

**Before (BROKEN)**:
```
Inference (memory only) → RAGAS (saves both inputs + metrics)
```
- If RAGAS fails mid-run, all inference results are lost
- Cannot re-run evaluation without re-running expensive inference
- Cannot run inference-only or evaluation-only modes

**After (CORRECT)**:
```
Inference (saves inputs immediately) → RAGAS (saves metrics only)
```
- `scripts/run_inference.py` saves `*_evaluation_inputs.parquet` **immediately after each retriever**
- If RAGAS fails, inference results are preserved
- Can re-run `make eval-metrics` without re-running inference
- Enables future inference-only and evaluation-only workflows

**Implementation**: Three separate scripts replace the monolithic `run_eval_harness.py`:
- `scripts/run_inference.py` - Loads data, builds graphs, runs RAG queries, saves inputs
- `scripts/run_evaluation.py` - Loads saved inputs, runs RAGAS metrics, saves metrics
- `scripts/summarize_results.py` - Aggregates results into comparative summary + manifest

### 4. LangGraph State Updates

Nodes return **partial state dicts**, not complete states:

```python
class State(TypedDict):
    question: str
    context: List[Document]
    response: str

def retrieve(state: State) -> dict:
    return {"context": retriever.invoke(state["question"])}  # Partial update

def generate(state: State) -> dict:
    return {"response": llm.invoke(messages).content}  # Partial update
```

LangGraph auto-merges: `{question}` → `{question, context}` → `{question, context, response}`

## Repository Structure

```
src/                  # Core RAG framework (775 lines)
├── config.py         # Singletons: get_llm(), get_embeddings(), create_vector_store()
├── retrievers.py     # Factory: create_retrievers() → 4 strategies
├── graph.py          # Factory: build_graph(), build_all_graphs()
├── state.py          # LangGraph State schema (TypedDict)
└── utils/            # HF loaders, manifest generation

scripts/              # Executable workflows (11 scripts, ~2800 lines)
├── run_inference.py         # Phase 1: RAG inference
├── run_evaluation.py        # Phase 2: RAGAS scoring
├── summarize_results.py     # Phase 3: Aggregate results
├── run_eval_harness.py      # Legacy: monolithic 3-phase script
├── run_app_validation.py    # Validation suite (23 checks)
├── ingest_raw_pdfs.py       # PDF → interim datasets
├── generate_deliverables.py # Parquet → CSV conversion
├── validate_manifests.py    # Manifest schema validation
└── publish_*.py             # HuggingFace uploads

data/
├── raw/                     # Source PDFs (immutable)
├── interim/                 # Extracted docs + testset + manifest.json
│   └── manifest.json        # Ingestion provenance (SHA-256, versions)
├── processed/               # Evaluation results (Parquet, source of truth)
│   ├── *_evaluation_inputs.parquet   # RAG query results (4 files)
│   ├── *_evaluation_metrics.parquet  # RAGAS scores (4 files)
│   ├── comparative_ragas_results.parquet  # Aggregated summary
│   └── RUN_MANIFEST.json    # Evaluation provenance
└── deliverables/
    └── evaluation_evidence/ # CSV files (regenerable via make deliverables)
        ├── *_evaluation_inputs.csv
        ├── *_evaluation_metrics.csv
        ├── comparative_ragas_results.csv
        └── RUN_MANIFEST.json    # Copied from processed/
```

## High-Level Architecture

**5-Layer Design**:

| Layer | Purpose | Key Modules |
|-------|---------|-------------|
| **Configuration** | External services (OpenAI, Qdrant, Cohere) | `src/config.py` |
| **Data** | Ingestion + persistence (HF datasets) | `src/utils/loaders.py`, `src/utils/manifest.py` |
| **Retrieval** | 4 strategies (naive, BM25, ensemble, rerank) | `src/retrievers.py` |
| **Orchestration** | LangGraph workflows (retrieve → generate) | `src/graph.py`, `src/state.py` |
| **Execution** | Scripts + LangGraph Server entrypoints | `scripts/`, `app/graph_app.py` |

**Design Patterns**: Factory (deferred init), Singleton (resource caching), Strategy (swappable retrievers)

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

# 2. Add to src/constants.py
RETRIEVERS = ["naive", "bm25", "ensemble", "cohere_rerank", "your_method"]

# 3. Validate + Evaluate
make validate  # Must pass 23/23 checks
make eval      # Automatically includes your new retriever
```

Results appear in `data/processed/comparative_ragas_results.parquet`.

## Common Workflows

### Standard Development Flow
```bash
make qdrant-up         # Start infrastructure
make validate          # Must pass 100% (23/23)
make eval              # Run evaluation (~$5-6, 20-30 min)
make deliverables      # Generate CSV files
```

### Cost-Conscious Evaluation
```bash
make inference         # Only if you need fresh inference (~$3-4)
make eval-metrics      # Only if you need fresh RAGAS scores (~$2)
make summarize         # Always free (local aggregation)
```

### Ingesting New Data (Rare)
```bash
# Only needed if recreating datasets from PDFs
make ingest            # PDF → interim datasets (~$2-3)
make eval recreate=true  # Force recreate Qdrant collection
```

### Publishing (One-Time)
```bash
export HF_TOKEN=hf_...
make publish-interim     # Upload sources + testset
make publish-processed   # Upload evaluation results
```

## End-to-End Workflow

### Pipeline Steps (Base Commands)

The complete evaluation pipeline in logical order:

| Step | Command | What It Does | Cost | Time | Creates |
|------|---------|--------------|------|------|---------|
| **0** | `make clean-all` | Resets workspace by removing all generated data | $0 | <1s | Clean slate |
| **1** | `make ingest` | Extracts text from PDFs and generates test questions | ~$2-3 | 5-10m | `data/interim/` datasets |
| **2** | `make publish-interim` | Uploads source docs & testset to HuggingFace | $0 | 1-2m | HF datasets (sources, testset) |
| **3** | `make validate` | Verifies environment setup (23 checks must pass) | $0 | <1s | Validation report |
| **4** | `make qdrant-up` | Starts vector database Docker container | $0 | <1s | Qdrant on port 6333 |
| **5a** | `make inference` | Runs RAG queries using all 4 retrievers | ~$3-4 | 5-10m | Inference results (Parquet) |
| **5b** | `make eval-metrics` | Scores RAG outputs with RAGAS metrics | ~$2 | 5-10m | Evaluation scores (Parquet) |
| **5c** | `make summarize` | Aggregates results across all retrievers | $0 | <1s | Comparative analysis |
| **6** | `make deliverables` | Converts Parquet to human-readable CSVs | $0 | <1s | `deliverables/` CSVs |
| **7** | `make publish-processed` | Uploads evaluation results to HuggingFace | $0 | 1-2m | HF evaluation datasets |

**Shortcut:** Steps 5a-5c can be combined with `make eval` (runs all three phases sequentially)

**Total:** ~$7-9, 30-45 minutes

### Parameter Guide

Control pipeline behavior with these parameters:

| Parameter | Commands | Purpose | Values | Example | When to Use |
|-----------|----------|---------|--------|---------|-------------|
| **VERSION** | `ingest`<br>`publish-*`<br>`inference`<br>`summarize`<br>`eval` | Dataset version tag | `v3` (default)<br>`v4`, `v5`, etc. | `make ingest VERSION=v4` | • New dataset creation<br>• Version tracking<br>• A/B testing |
| **RECREATE** | `inference`<br>`eval` | Force new Qdrant collection | `false` (default)<br>`true` | `make inference RECREATE=true` | • Collection corrupted<br>• Schema changes<br>• Fresh start needed |
| **HF_TOKEN** | `publish-*` | HuggingFace authentication | Your token | `export HF_TOKEN=hf_...` | Required for publishing |

### Common Scenarios

| Scenario | Command | Why |
|----------|---------|-----|
| **First-time setup** | `make eval RECREATE=true` | Creates new Qdrant collection and runs full pipeline |
| **Re-run with existing data** | `make eval` | Uses existing Qdrant collection, faster startup |
| **Test new version** | `make eval VERSION=v5 RECREATE=true` | Creates v5 datasets with fresh collection |
| **Fix failed RAGAS** | `make eval-metrics` | Re-runs only RAGAS scoring (inference preserved) |
| **Update CSVs only** | `make deliverables` | Regenerates CSVs from existing Parquet files |
| **Publish v4 results** | `make publish-processed VERSION=v4` | Uploads v4 evaluation to HuggingFace |

## Environment Variables

```bash
# Required
OPENAI_API_KEY="sk-..."

# Optional
COHERE_API_KEY="..."              # For cohere_rerank retriever
HF_TOKEN="..."                    # For publishing to HuggingFace
LANGCHAIN_API_KEY="..."           # For LangSmith tracing
QDRANT_URL="http://localhost:6333"  # Vector database
```

## Key Technologies

- **LangChain 0.3.19+** / **LangGraph 0.6.7** (pinned)
- **RAGAS 0.2.10** (pinned - API changed in 0.3.x, breaking)
- **Qdrant** (Docker), **OpenAI** GPT-4.1-mini + text-embedding-3-small
- **Cohere** rerank-v3.5, **HuggingFace** Datasets (versioned)
- **uv** package manager, Python 3.11+

## Performance Characteristics

**Latency per query**:
- BM25: 10-20ms (in-memory)
- Naive: 50-100ms (Qdrant search)
- Ensemble: 60-120ms (parallel)
- Cohere Rerank: 200-500ms (API call)

**Cost**:
- Full eval: 48 queries (12 questions × 4 retrievers) + ~150 RAGAS calls = **$5-6, 20-30 min**
- Inference only: ~$3-4
- RAGAS only: ~$2

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
source .venv/bin/activate  # Activate environment
make validate  # Re-run (must pass 23/23)
```

**Deliverables missing/outdated**
```bash
make deliverables  # Regenerate from Parquet (always safe)
```

**Evaluation stalls/fails**
- Results saved incrementally - just re-run `make eval`
- For phase-by-phase control: `make inference`, `make eval-metrics`, `make summarize`

## Known Issues

### Manifest Provenance Tracking
The `RUN_MANIFEST.json` contains hardcoded dataset names without version suffixes:
- Shows: `"golden_testset": "dwb2023/gdelt-rag-golden-testset"`
- Should show: `"golden_testset": "dwb2023/gdelt-rag-golden-testset-v4"`

This occurs because the three-phase pipeline is decoupled - Phase 3 (summarize) doesn't
know which HuggingFace datasets Phase 1 (inference) actually used. The evaluation
results are correct, but exact reproducibility requires manually noting the VERSION
parameter used.

**Workaround:** Always document the VERSION parameter used in your evaluation runs.

## Future Enhancements

### Immutable Data Management
- Use timestamped filenames for Parquet files (e.g., `naive_evaluation_inputs_20240302_143022.parquet`)
- Prevent accidental overwrites of evaluation results
- Enable parallel evaluation runs without conflicts

### MLOps Integration
For production use, consider integrating proper experiment tracking:
- **MLflow**: Comprehensive experiment tracking, model registry
- **Weights & Biases**: Cloud-based experiment tracking with rich visualizations
- **DVC (Data Version Control)**: Git-like versioning for data and models
- **Airflow/Prefect/Dagster**: Workflow orchestration with built-in lineage tracking

These tools solve the provenance tracking issue properly by design.

## Best Practices

**DO**:
- ✅ Use factories for retrievers/graphs - never module-level init
- ✅ Save working data as Parquet in `data/processed/`
- ✅ Regenerate deliverables via `make deliverables` - never edit CSV
- ✅ Run `make validate` before committing (100% pass required)
- ✅ Set `PYTHONPATH=.` when running scripts directly (or use make)

**DON'T**:
- ❌ Initialize retrievers/graphs at module import time
- ❌ Write directly to `deliverables/` (always regenerate)
- ❌ Delete `data/processed/*.parquet` (source of truth)
- ❌ Upgrade RAGAS to 0.3.x without testing (breaking changes)
- ❌ Skip `make validate` before deployment

## Documentation Map

- **This file** - High-level architecture and common commands
- **scripts/README.md** - Detailed script documentation (11 scripts)
- **src/README.md** - Factory pattern guide + module reference
- **data/README.md** - Data flow, manifest schema, file formats
- **Makefile** - All command specifications (scope, duration, cost)
- **README.md** - Project overview, datasets, evaluation results
- **architecture/** - Auto-generated (Claude Agent SDK analyzer)

## Reproducibility & Provenance

**Manifests track complete lineage**:
- `data/interim/manifest.json` - Ingestion provenance (SHA-256 checksums, env versions)
- `data/processed/RUN_MANIFEST.json` - Evaluation provenance (links to ingestion manifest)

**Version Pinning** (critical for reproducibility):
```toml
# pyproject.toml
ragas = "==0.2.10"              # API changed in 0.3.x
langgraph = "==0.6.7"           # Graph compilation behavior
langchain-cohere = "==0.4.4"    # Reranker compatibility
```

## HuggingFace Datasets

**Published datasets** (first public evaluation suite for GDELT RAG):
1. [dwb2023/gdelt-rag-sources-v3](https://huggingface.co/datasets/dwb2023/gdelt-rag-sources-v3) - 38 documents
2. [dwb2023/gdelt-rag-golden-testset-v3](https://huggingface.co/datasets/dwb2023/gdelt-rag-golden-testset-v3) - 12 QA pairs
3. [dwb2023/gdelt-rag-evaluation-inputs-v3](https://huggingface.co/datasets/dwb2023/gdelt-rag-evaluation-inputs-v3) - 48 records
4. [dwb2023/gdelt-rag-evaluation-metrics-v3](https://huggingface.co/datasets/dwb2023/gdelt-rag-evaluation-metrics-v3) - 48 records with RAGAS scores

Load via: `from datasets import load_dataset; ds = load_dataset("dwb2023/gdelt-rag-sources-v3")`

## Notes

- Python 3.11+ required
- RAGAS 0.2.10 pinned (breaking API changes in 0.3.x)
- Factory pattern is critical - never initialize at module level
- `ra_orchestrators/` contains Claude Agent SDK analyzer (optional tooling)
- LangGraph Studio UI available via: `uv run langgraph dev --allow-blocking`