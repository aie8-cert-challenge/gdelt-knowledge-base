# Component Inventory

## Overview

This is a production-grade RAG (Retrieval-Augmented Generation) system for querying GDELT (Global Database of Events, Language, and Tone) knowledge graphs. The codebase follows a modular architecture with clear separation between:

- **Public API** (`src/` package): Reusable, well-documented modules for RAG pipeline construction
- **Application Layer** (`app/`): LangGraph Server deployment entry points
- **Scripts** (`scripts/`): Executable workflows for data ingestion, evaluation, and validation
- **Tests** (`tests/`): Test suite structure (placeholder)

The system implements multiple retrieval strategies (naive vector search, BM25, ensemble, Cohere rerank) and uses LangGraph for composable RAG workflows. Data flows from raw PDFs through HuggingFace datasets to Qdrant vector store, with comprehensive RAGAS-based evaluation.

---

## Public API

### Core Package: `src/`

The `src/` package provides the main public API for building RAG systems. All modules follow factory pattern best practices, avoiding module-level instantiation.

**Package Metadata:**
- File: `/home/donbr/don-aie-cohort8/cert-challenge/src/__init__.py:1-42`
- Version: `0.1.0`
- Exports: `config`, `graph`, `prompts`, `retrievers`, `state`, `utils`

### Modules

#### 1. Configuration Module (`src/config.py`)

Provides cached getter functions for LLM, embeddings, and Qdrant client with environment-based configuration.

**File:** `/home/donbr/don-aie-cohort8/cert-challenge/src/config.py:1-151`

**Key Functions:**

| Function | Line | Purpose |
|----------|------|---------|
| `get_llm()` | 42-49 | Returns cached ChatOpenAI instance (temperature=0) |
| `get_embeddings()` | 52-60 | Returns cached OpenAIEmbeddings instance |
| `get_qdrant()` | 63-80 | Returns cached QdrantClient (URL-first convention) |
| `get_collection_name()` | 83-90 | Returns configured collection name |
| `create_vector_store()` | 93-150 | Factory for QdrantVectorStore with document population |

**Configuration Constants:**

| Constant | Line | Default | Purpose |
|----------|------|---------|---------|
| `QDRANT_URL` | 23 | `http://localhost:6333` | Qdrant connection URL |
| `QDRANT_HOST` | 24 | `localhost` | Fallback host |
| `QDRANT_PORT` | 25 | `6333` | Fallback port |
| `COLLECTION_NAME` | 36 | `gdelt_comparative_eval` | Vector store collection |
| `OPENAI_MODEL` | 37 | `gpt-4.1-mini` | LLM model identifier |
| `OPENAI_EMBED_MODEL` | 38 | `text-embedding-3-small` | Embedding model |

**Design Pattern:**
- Uses `@lru_cache(maxsize=1)` for singleton instances
- Reads configuration from environment variables
- Handles optional API keys gracefully

---

#### 2. Retriever Factory (`src/retrievers.py`)

Factory functions for creating retriever instances. Cannot be instantiated at module level due to document/vector store dependencies.

**File:** `/home/donbr/don-aie-cohort8/cert-challenge/src/retrievers.py:1-90`

**Key Functions:**

| Function | Line | Purpose |
|----------|------|---------|
| `create_retrievers()` | 20-89 | Factory returning dict of all 4 retriever strategies |

**Retriever Strategies:**

| Strategy | Line | Type | Description |
|----------|------|------|-------------|
| `naive` | 63 | Dense Vector | Baseline vector search (k=5) |
| `bm25` | 66 | Sparse Keyword | BM25 lexical matching (k=5) |
| `ensemble` | 69-72 | Hybrid | 50/50 weighted dense+sparse |
| `cohere_rerank` | 74-82 | Contextual Compression | Retrieves 20, reranks to top-k |

**Parameters:**
- `documents`: List[Document] - Required for BM25 initialization
- `vector_store`: QdrantVectorStore - Required for dense search
- `k`: int = 5 - Number of documents to retrieve

**Design Pattern:**
- Factory pattern prevents premature initialization
- Returns dictionary for flexible graph construction
- Explicit model versions for reproducibility (`rerank-v3.5`)

---

#### 3. LangGraph Factory (`src/graph.py`)

Factory functions for building LangGraph workflows. Follows session08 patterns for state management.

**File:** `/home/donbr/don-aie-cohort8/cert-challenge/src/graph.py:1-142`

**Key Functions:**

| Function | Line | Purpose |
|----------|------|---------|
| `build_graph()` | 21-106 | Builds single retriever graph (retrieve → generate) |
| `build_all_graphs()` | 109-141 | Builds graphs for all retrievers |

**Node Functions (Internal):**

| Node | Line | Input | Output |
|------|------|-------|--------|
| `retrieve()` | 67-78 | `state["question"]` | `{"context": List[Document]}` |
| `generate()` | 80-96 | `state["question"]`, `state["context"]` | `{"response": str}` |

**Graph Structure:**
```
START → retrieve → generate → END
```

**Design Pattern:**
- Node functions return partial state updates (dict)
- LangGraph automatically merges updates into state
- Shared prompt template from `src/prompts`
- Optional LLM and prompt_template parameters

---

#### 4. State Schema (`src/state.py`)

TypedDict defining LangGraph state structure.

**File:** `/home/donbr/don-aie-cohort8/cert-challenge/src/state.py:1-10`

**Classes:**

| Class | Line | Purpose |
|-------|------|---------|
| `State` | 7-10 | LangGraph state schema (TypedDict) |

**State Fields:**

| Field | Type | Purpose |
|-------|------|---------|
| `question` | str | User input query |
| `context` | List[Document] | Retrieved documents |
| `response` | str | Generated answer |

---

#### 5. Prompt Templates (`src/prompts.py`)

Shared prompt templates for RAG generation.

**File:** `/home/donbr/don-aie-cohort8/cert-challenge/src/prompts.py:1-12`

**Constants:**

| Constant | Line | Purpose |
|----------|------|---------|
| `BASELINE_PROMPT` | 4-12 | RAG prompt template with question/context placeholders |

**Template Structure:**
- Instructs assistant to only use provided context
- Separates question and context sections
- Used by all retriever graphs for consistency

---

#### 6. Utilities Package (`src/utils/`)

Helper functions for document loading and manifest generation.

**Package File:** `/home/donbr/don-aie-cohort8/cert-challenge/src/utils/__init__.py:1-20`

**Exports:**
- `load_documents_from_huggingface`
- `load_golden_testset_from_huggingface`
- `generate_run_manifest`

##### 6a. Document Loaders (`src/utils/loaders.py`)

**File:** `/home/donbr/don-aie-cohort8/cert-challenge/src/utils/loaders.py:1-114`

| Function | Line | Purpose |
|----------|------|---------|
| `load_documents_from_huggingface()` | 15-75 | Loads source documents from HF dataset → LangChain Documents |
| `load_golden_testset_from_huggingface()` | 78-113 | Loads golden testset from HF dataset (returns Dataset) |

**Key Features:**
- Handles nested metadata structures automatically
- Supports revision pinning for reproducibility
- Uses environment variables for defaults (`HF_SOURCES_REV`, `HF_GOLDEN_REV`)
- Converts HF datasets to LangChain Document format

##### 6b. Manifest Generator (`src/utils/manifest.py`)

**File:** `/home/donbr/don-aie-cohort8/cert-challenge/src/utils/manifest.py:1-194`

| Function | Line | Purpose |
|----------|------|---------|
| `generate_manifest()` | 21-176 | Generates RUN_MANIFEST.json for reproducibility |

**Manifest Contents:**
- Model versions (RAGAS, Python, LangChain)
- Retriever configurations (all 4 strategies)
- Evaluation settings (metrics, timeouts)
- Vector store configuration
- Optional: evaluation results summary
- Optional: data provenance (links to ingestion manifest)

**Standalone Execution:**
- Lines 179-193: Can be run directly to generate manifest template

---

## Internal Implementation

### Core Modules

#### Application Entry Point (`app/graph_app.py`)

LangGraph Server entry point for production deployment.

**File:** `/home/donbr/don-aie-cohort8/cert-challenge/app/graph_app.py:1-18`

| Function | Line | Purpose |
|----------|------|---------|
| `get_app()` | 7-17 | Returns compiled graph for LangGraph Server |

**Initialization Sequence:**
1. Load documents from HuggingFace (line 12)
2. Create vector store (line 13)
3. Create retrievers (line 14)
4. Get LLM instance (line 15)
5. Build all graphs (line 16)
6. Return default graph (`cohere_rerank`, line 17)

**Note:** Currently returns single graph; can be extended to support dynamic switching via state.

---

### Utility Modules

#### Main Entry Point (`main.py`)

Minimal entry point (not currently used in production).

**File:** `/home/donbr/don-aie-cohort8/cert-challenge/main.py:1-7`

| Function | Line | Purpose |
|----------|------|---------|
| `main()` | 1-6 | Placeholder entry point (prints "Hello from cert-challenge!") |

**Status:** Placeholder - actual functionality in `app/` and `scripts/`.

---

### Configuration

#### Project Configuration (`pyproject.toml`)

**File:** `/home/donbr/don-aie-cohort8/cert-challenge/pyproject.toml:1-29`

**Metadata:**
- Name: `certification-challenge`
- Version: `0.1.0`
- Python: `>=3.11`

**Key Dependencies:**

| Package | Version | Purpose |
|---------|---------|---------|
| `langchain` | `>=0.3.19` | Core RAG framework |
| `langgraph` | `0.6.7` | Graph-based workflows |
| `langchain-openai` | `>=0.3.7` | OpenAI integration |
| `langchain-cohere` | `0.4.4` | Cohere reranker |
| `langchain-qdrant` | `>=0.2.0` | Qdrant vector store |
| `qdrant-client` | `>=1.13.2` | Qdrant client |
| `ragas` | `0.2.10` | RAG evaluation metrics |
| `datasets` | `>=3.2.0` | HuggingFace datasets |
| `pymupdf` | `>=1.26.3` | PDF parsing |
| `rank-bm25` | `>=0.2.2` | BM25 retriever |

**Note:** Exact versions pinned for `ragas`, `langgraph`, and `langchain-cohere` for reproducibility.

---

## Entry Points

### Main Scripts

All scripts are located in `/home/donbr/don-aie-cohort8/cert-challenge/scripts/`.

#### 1. Data Ingestion Pipeline (`scripts/ingest_raw_pdfs.py`)

**File:** `/home/donbr/don-aie-cohort8/cert-challenge/scripts/ingest_raw_pdfs.py:1-302`

**Purpose:** Extract PDFs → Documents → Interim Storage → RAGAS Testset → Manifest

**Key Components:**

| Component | Line | Purpose |
|-----------|------|---------|
| `find_repo_root()` | 46-52 | Detect project root (supports scripts/notebooks) |
| `ensure_jsonable()` | 89-101 | Make metadata JSON-serializable |
| `docs_to_jsonl()` | 103-110 | Persist documents to JSONL |
| `docs_to_parquet()` | 112-116 | Persist documents to Parquet |
| `docs_to_hfds()` | 118-123 | Persist documents to HF dataset |
| `hash_file()` | 125-130 | Generate SHA256 checksums |
| `write_manifest()` | 132-133 | Write manifest JSON |
| `summarize_columns_from_jsonl()` | 135-147 | Extract schema from JSONL |
| `build_testset()` | 186-198 | Generate RAGAS golden testset |

**Workflow:**
1. Load PDFs from `data/raw/` (lines 153-158)
2. Persist to interim storage (lines 163-169)
   - JSONL: `data/interim/sources.docs.jsonl`
   - Parquet: `data/interim/sources.docs.parquet`
   - HF Dataset: `data/interim/sources.hfds/`
3. Generate RAGAS testset (lines 200-201)
4. Persist testset (lines 207-222)
   - JSONL: `data/interim/golden_testset.jsonl`
   - Parquet: `data/interim/golden_testset.parquet`
   - HF Dataset: `data/interim/golden_testset.hfds/`
5. Write manifest (lines 238-296)
   - File: `data/interim/manifest.json`
   - Contains: checksums, schema, environment, params

**Configuration:**

| Constant | Line | Default | Purpose |
|----------|------|---------|---------|
| `TESTSET_SIZE` | 80 | 10 | Number of test examples to generate |
| `MAX_DOCS` | 81 | None | Limit docs for prototyping (None = all) |
| `RANDOM_SEED` | 84 | 42 | Seed for local sampling |

**RAGAS Version Handling:**
- Lines 186-198: Uses RAGAS 0.2.x API (LangchainLLMWrapper, LangchainEmbeddingsWrapper)
- Lines 189: Note about 0.3.x deprecation
- Retry logic for transient API errors (lines 180-185)

---

#### 2. Dataset Publisher (`scripts/publish_interim_datasets.py`)

**File:** `/home/donbr/don-aie-cohort8/cert-challenge/scripts/publish_interim_datasets.py:1-300`

**Purpose:** Upload interim datasets to HuggingFace Hub with dataset cards.

**Key Components:**

| Component | Line | Purpose |
|-----------|------|---------|
| `create_sources_card()` | 34-110 | Generate README for sources dataset |
| `create_golden_testset_card()` | 113-191 | Generate README for golden testset |
| `load_manifest()` | 194-197 | Load ingestion manifest |
| `update_manifest()` | 200-223 | Add HF repo IDs to manifest |
| `main()` | 226-296 | Upload workflow |

**Configuration:**

| Constant | Line | Value | Purpose |
|----------|------|-------|---------|
| `HF_USERNAME` | 21 | `dwb2023` | HuggingFace username |
| `SOURCES_DATASET_NAME` | 22 | `dwb2023/gdelt-rag-sources-v2` | Sources repo |
| `GOLDEN_TESTSET_NAME` | 23 | `dwb2023/gdelt-rag-golden-testset-v2` | Testset repo |

**Upload Workflow:**
1. Login to HuggingFace (line 234)
2. Load datasets from disk (lines 241-243)
3. Upload sources dataset (lines 249-254)
4. Upload sources README (lines 257-264)
5. Upload golden testset (lines 268-274)
6. Upload golden testset README (lines 277-284)
7. Update manifest with repo IDs (line 290)

**Dataset Card Structure:**
- Frontmatter: license, task categories, tags, size
- Description: purpose, source, format
- Data fields documentation
- Citation information
- Creation notes

---

#### 3. Application Validation (`scripts/run_app_validation.py`)

**File:** `/home/donbr/don-aie-cohort8/cert-challenge/scripts/run_app_validation.py:1-487`

**Purpose:** Validate LangGraph implementation and demonstrate correct factory patterns.

**Validation Stages:**

| Stage | Function | Line | Purpose |
|-------|----------|------|---------|
| 1 | `check_environment()` | 74-127 | Validate API keys, Qdrant, imports |
| 2 | `test_module_imports()` | 134-174 | Test each src/ module independently |
| 3 | `demonstrate_correct_pattern()` | 181-221 | Validate factory pattern usage |
| 4 | `validate_graph_compilation()` | 228-261 | Validate LangGraph builds |
| 5 | `run_functional_tests()` | 268-304 | Test queries through graphs |
| 6 | `generate_diagnostic_report()` | 311-440 | Generate comprehensive report |

**Exit Codes:**
- 0: All validations passed
- 1: One or more validations failed

**Key Features:**
- Color-coded terminal output (lines 35-67)
- Detailed error tracing for failed imports
- Recommendations for fixing broken patterns (lines 389-436)
- Tests actual src/ modules (not inline code)

**Test Question:** "What is GDELT?" (line 277)

---

#### 4. RAGAS Evaluation Harness (`scripts/run_eval_harness.py`)

**File:** `/home/donbr/don-aie-cohort8/cert-challenge/scripts/run_eval_harness.py:1-372`

**Purpose:** Run comprehensive RAGAS evaluation using src/ modules.

**Differences from `run_full_evaluation.py`:**
- Uses src/ factory functions (not inline code)
- Otherwise identical functionality
- Serves as reference implementation

**Command-Line Arguments:**

| Argument | Line | Default | Purpose |
|----------|------|---------|---------|
| `--recreate` | 56-62 | `false` | Recreate Qdrant collection |

**Configuration:**

| Constant | Line | Value | Purpose |
|----------|------|-------|---------|
| `DATASET_SOURCES` | 73 | `dwb2023/gdelt-rag-sources` | Source dataset |
| `DATASET_GOLDEN` | 74 | `dwb2023/gdelt-rag-golden-testset` | Test dataset |
| `K` | 75 | 5 | Documents to retrieve |
| `OUT_DIR` | 76 | `deliverables/evaluation_evidence/` | Output directory |

**Workflow:**
1. Pre-flight checks (lines 88-121)
   - Validate API keys
   - Load ingestion manifest for lineage
2. Load data (lines 127-140)
   - Load source documents
   - Load golden testset
3. Build RAG stack (lines 145-164)
   - Create vector store
   - Create retrievers (4 strategies)
   - Build LangGraph workflows
4. Run inference (lines 169-203)
   - Process 12 questions × 4 retrievers
   - Save raw datasets immediately (Parquet)
5. RAGAS evaluation (lines 208-250)
   - Evaluate all retrievers
   - Save detailed results (CSV)
6. Comparative analysis (lines 255-304)
   - Calculate average scores
   - Rank retrievers
   - Calculate improvement over baseline
7. Generate manifest (lines 343-367)
   - Capture configuration
   - Link to data provenance
   - Save to `RUN_MANIFEST.json`

**Output Files:**

| File Pattern | Count | Purpose |
|--------------|-------|---------|
| `{retriever}_raw_dataset.parquet` | 4 | Raw inference outputs (6 columns) |
| `{retriever}_evaluation_dataset.csv` | 4 | RAGAS format datasets |
| `{retriever}_detailed_results.csv` | 4 | Per-question metric scores |
| `comparative_ragas_results.csv` | 1 | Summary table (sorted by average) |
| `RUN_MANIFEST.json` | 1 | Reproducibility metadata |

**Metrics:**
- Faithfulness (answer grounded in context)
- Answer Relevancy (answer addresses question)
- Context Precision (relevant contexts ranked higher)
- Context Recall (ground truth coverage)

**Estimated Runtime:** 20-30 minutes
**Estimated Cost:** $5-6 in OpenAI API calls

---

#### 5. Full Evaluation Pipeline (`scripts/run_full_evaluation.py`)

**File:** `/home/donbr/don-aie-cohort8/cert-challenge/scripts/run_full_evaluation.py:1-534`

**Purpose:** Comprehensive RAG evaluation with inline code (original implementation).

**Key Components:**

| Component | Line | Purpose |
|-----------|------|---------|
| `validate_and_normalize_ragas_schema()` | 70-129 | Ensure DataFrame matches RAGAS schema |
| Inline retriever creation | 234-259 | Create 4 retriever strategies |
| Inline graph builders | 281-325 | Build LangGraph workflows |
| Inference loop | 345-377 | Process all questions |
| RAGAS evaluation | 380-427 | Evaluate with metrics |
| Comparative analysis | 430-494 | Generate summary table |
| Manifest generation | 524-531 | Reproducibility metadata |

**Configuration:**

| Constant | Line | Value | Purpose |
|----------|------|-------|---------|
| `QDRANT_HOST` | 132 | `localhost` | Qdrant host |
| `QDRANT_PORT` | 133 | `6333` | Qdrant port |
| `COLLECTION_NAME` | 134 | `gdelt_comparative_eval` | Vector store collection |

**Differences from `run_eval_harness.py`:**
- Inline retriever/graph creation (not using src/)
- Includes schema validation helper (lines 70-129)
- Documents PDF → LangChain Document conversion (lines 182-194)
- Creates vector store from scratch (lines 199-230)

**Shared Features:**
- Same 4 retriever strategies
- Same RAGAS metrics
- Same output structure
- Same manifest generation

**Note:** This is the original implementation. `run_eval_harness.py` demonstrates how to migrate to src/ modules.

---

### Module Interfaces

#### Public Interfaces (Designed for External Use)

**Package: `src`**

Primary entry points for users building RAG systems:

```python
# Document loading
from src.utils import load_documents_from_huggingface, load_golden_testset_from_huggingface

# Vector store creation
from src.config import create_vector_store, get_llm

# Retriever creation
from src.retrievers import create_retrievers

# Graph construction
from src.graph import build_all_graphs

# Example usage (from src/__init__.py:17-37)
documents = load_documents_from_huggingface()
vector_store = create_vector_store(documents, recreate_collection=True)
retrievers = create_retrievers(documents, vector_store)
graphs = build_all_graphs(retrievers)
result = graphs['naive'].invoke({"question": "What is GDELT?"})
```

**Package: `app`**

LangGraph Server deployment interface:

```python
# Server entry point
from app.graph_app import get_app

app = get_app()  # Returns compiled graph for LangGraph Server
```

---

#### Internal Interfaces (Implementation Details)

**State Management:**

```python
from src.state import State

# State schema (used by all graphs)
{
    "question": str,        # User query
    "context": List[Document],  # Retrieved documents
    "response": str         # Generated answer
}
```

**Prompt Templates:**

```python
from src.prompts import BASELINE_PROMPT

# Shared prompt template (used by build_graph)
# Format: {question} and {context} placeholders
```

**Configuration Singletons:**

```python
from src.config import get_llm, get_embeddings, get_qdrant

# Cached instances (safe to call multiple times)
llm = get_llm()             # ChatOpenAI
embeddings = get_embeddings()  # OpenAIEmbeddings
client = get_qdrant()        # QdrantClient
```

---

## Dependencies

### Core Framework Dependencies

| Package | Version | Purpose | Source |
|---------|---------|---------|--------|
| **LangChain** | >=0.3.19 | RAG framework core | `pyproject.toml:10` |
| **LangGraph** | 0.6.7 | Graph-based workflows | `pyproject.toml:14` |
| **LangChain OpenAI** | >=0.3.7 | OpenAI LLM/embeddings | `pyproject.toml:13` |
| **LangChain Cohere** | 0.4.4 | Cohere reranker | `pyproject.toml:11` |
| **LangChain Qdrant** | >=0.2.0 | Qdrant integration | `pyproject.toml:17` |

### Vector Store Dependencies

| Package | Version | Purpose | Source |
|---------|---------|---------|--------|
| **Qdrant Client** | >=1.13.2 | Vector database client | `pyproject.toml:15` |

### Retrieval Dependencies

| Package | Version | Purpose | Source |
|---------|---------|---------|--------|
| **rank-bm25** | >=0.2.2 | BM25 sparse retriever | `pyproject.toml:16` |

### Evaluation Dependencies

| Package | Version | Purpose | Source |
|---------|---------|---------|--------|
| **RAGAS** | 0.2.10 | RAG evaluation metrics | `pyproject.toml:18` |

### Data Processing Dependencies

| Package | Version | Purpose | Source |
|---------|---------|---------|--------|
| **HuggingFace Hub** | >=0.26.0 | Dataset hosting | `pyproject.toml:22` |
| **datasets** | >=3.2.0 | Dataset loading/processing | `pyproject.toml:23` |
| **PyMuPDF** | >=1.26.3 | PDF parsing | `pyproject.toml:19` |
| **rapidfuzz** | >=3.14.1 | String matching | `pyproject.toml:20` |

### Development Dependencies

| Package | Version | Purpose | Source |
|---------|---------|---------|--------|
| **Jupyter** | >=1.1.1 | Notebook environment | `pyproject.toml:8` |
| **Streamlit** | >=1.40.0 | UI framework | `pyproject.toml:24` |
| **LangGraph CLI** | >=0.4.4 | Graph server CLI | `pyproject.toml:25` |
| **Claude Agent SDK** | >=0.1.1 | Analysis framework | `pyproject.toml:21` |

### Utility Dependencies

| Package | Version | Purpose | Source |
|---------|---------|---------|--------|
| **OpenAI** | >=2.4.0 | OpenAI API client | `pyproject.toml:26` |
| **Cohere** | 5.12.0 | Cohere API client | `pyproject.toml:12` |
| **python-dotenv** | >=1.1.1 | Environment management | `pyproject.toml:27` |

### Version Pinning Strategy

**Exact Versions (Reproducibility Critical):**
- `ragas==0.2.10` - API changed significantly between 0.2.x and 0.3.x
- `langgraph==0.6.7` - Graph compilation behavior
- `langchain-cohere==0.4.4` - Reranker model compatibility
- `cohere==5.12.0` - API client compatibility

**Minimum Versions (Latest Compatible):**
- All other packages use `>=` for security updates
- Python `>=3.11` for modern type hinting support

---

## Architectural Patterns

### Factory Pattern

All core modules use factory functions to prevent premature initialization:

**Example from `src/retrievers.py:20-89`:**
```python
def create_retrievers(documents, vector_store, k=5):
    # Creates instances only when called
    return {
        "naive": vector_store.as_retriever(...),
        "bm25": BM25Retriever.from_documents(...),
        "ensemble": EnsembleRetriever(...),
        "cohere_rerank": ContextualCompressionRetriever(...)
    }
```

**Rationale:**
- Documents and vector stores must be loaded before retriever creation
- Prevents circular dependencies
- Enables testing with mock objects
- Supports multiple configurations

### Singleton Pattern

Configuration uses LRU cache for singleton instances:

**Example from `src/config.py:42-49`:**
```python
@lru_cache(maxsize=1)
def get_llm():
    return ChatOpenAI(model=OPENAI_MODEL, temperature=0)
```

**Benefits:**
- Single LLM instance across application
- Thread-safe
- No global state
- Easy to reset for testing (`get_llm.cache_clear()`)

### State Management Pattern

LangGraph nodes return partial state updates:

**Example from `src/graph.py:67-96`:**
```python
def retrieve(state: State) -> dict:
    docs = retriever.invoke(state["question"])
    return {"context": docs}  # Partial update

def generate(state: State) -> dict:
    # Uses state["question"] and state["context"]
    return {"response": response.content}  # Partial update
```

**LangGraph Auto-Merge:**
- Nodes return dict with new/updated fields
- LangGraph merges into existing state
- No manual state copying required

---

## Data Flow

### Document Lifecycle

```
Raw PDFs (data/raw/)
  ↓ [scripts/ingest_raw_pdfs.py]
Interim Storage (data/interim/)
  - sources.docs.{jsonl,parquet,hfds}
  - golden_testset.{jsonl,parquet,hfds}
  - manifest.json
  ↓ [scripts/publish_interim_datasets.py]
HuggingFace Hub
  - dwb2023/gdelt-rag-sources-v2
  - dwb2023/gdelt-rag-golden-testset-v2
  ↓ [src/utils/loaders.py]
LangChain Documents
  ↓ [src/config.py]
Qdrant Vector Store
  ↓ [src/retrievers.py]
Retriever Instances
  ↓ [src/graph.py]
LangGraph Workflows
  ↓ [app/graph_app.py or scripts/]
Deployed Application / Evaluation
```

### Evaluation Lifecycle

```
Golden Testset (HF)
  ↓ [load_golden_testset_from_huggingface]
Pandas DataFrame
  ↓ [Graph.invoke() for each question]
Raw Datasets (response + retrieved_contexts)
  ↓ [Save immediately to Parquet]
deliverables/evaluation_evidence/{retriever}_raw_dataset.parquet
  ↓ [EvaluationDataset.from_pandas()]
RAGAS EvaluationDataset
  ↓ [evaluate() with metrics]
RAGAS Results
  ↓ [result.to_pandas()]
Detailed Results with Metric Scores
  ↓ [Save to CSV]
deliverables/evaluation_evidence/{retriever}_detailed_results.csv
  ↓ [Aggregate and compare]
Comparative Summary Table
  ↓ [Save to CSV]
deliverables/evaluation_evidence/comparative_ragas_results.csv
```

---

## File Organization

### Directory Structure

```
/home/donbr/don-aie-cohort8/cert-challenge/
├── src/                          # Public API (main package)
│   ├── __init__.py              # Package exports
│   ├── config.py                # Configuration & factories
│   ├── graph.py                 # LangGraph builders
│   ├── prompts.py               # Prompt templates
│   ├── retrievers.py            # Retriever factories
│   ├── state.py                 # State schema
│   └── utils/                   # Utility subpackage
│       ├── __init__.py          # Utility exports
│       ├── loaders.py           # Document loaders
│       └── manifest.py          # Manifest generator
├── app/                         # Application layer
│   ├── __init__.py              # Empty package marker
│   └── graph_app.py             # LangGraph Server entry point
├── scripts/                     # Executable workflows
│   ├── ingest_raw_pdfs.py       # Data ingestion pipeline
│   ├── publish_interim_datasets.py  # HF upload
│   ├── run_app_validation.py   # Validation suite
│   ├── run_eval_harness.py     # RAGAS eval (using src/)
│   └── run_full_evaluation.py  # RAGAS eval (inline code)
├── tests/                       # Test suite
│   └── __init__.py              # Test package marker
├── main.py                      # Placeholder entry point
└── pyproject.toml               # Project configuration
```

### Key File Paths

**Source Code:**
- `/home/donbr/don-aie-cohort8/cert-challenge/src/` - Main package
- `/home/donbr/don-aie-cohort8/cert-challenge/app/` - Application layer
- `/home/donbr/don-aie-cohort8/cert-challenge/scripts/` - Executable scripts

**Configuration:**
- `/home/donbr/don-aie-cohort8/cert-challenge/pyproject.toml` - Dependencies
- `/home/donbr/don-aie-cohort8/cert-challenge/.env` - Environment variables (not in repo)

**Data:**
- `/home/donbr/don-aie-cohort8/cert-challenge/data/raw/` - Source PDFs
- `/home/donbr/don-aie-cohort8/cert-challenge/data/interim/` - Processed datasets
- `/home/donbr/don-aie-cohort8/cert-challenge/deliverables/evaluation_evidence/` - Evaluation outputs

---

## Component Relationships

### Dependency Graph

```
app/graph_app.py
  → src/graph.build_all_graphs
  → src/retrievers.create_retrievers
  → src/config.create_vector_store
  → src/config.get_llm
  → src/utils.load_documents_from_huggingface

scripts/run_eval_harness.py
  → src/graph.build_all_graphs
  → src/retrievers.create_retrievers
  → src/config.create_vector_store
  → src/config.get_llm
  → src/utils.load_documents_from_huggingface
  → src/utils.load_golden_testset_from_huggingface
  → src/utils.generate_run_manifest

src/graph.py
  → src/state.State
  → src/prompts.BASELINE_PROMPT
  → src/config.get_llm

src/retrievers.py
  → langchain_community.retrievers.BM25Retriever
  → langchain.retrievers.EnsembleRetriever
  → langchain_cohere.CohereRerank
  → (no src/ dependencies)

src/config.py
  → langchain_openai.ChatOpenAI
  → langchain_openai.OpenAIEmbeddings
  → langchain_qdrant.QdrantVectorStore
  → qdrant_client.QdrantClient
  → (no src/ dependencies)
```

### Module Coupling

**Low Coupling (Good):**
- `src/config.py` - No internal dependencies
- `src/retrievers.py` - No internal dependencies
- `src/prompts.py` - No internal dependencies
- `src/state.py` - No internal dependencies

**Medium Coupling:**
- `src/graph.py` - Depends on config, state, prompts (all stable)
- `src/utils/loaders.py` - Depends on datasets, langchain (external)

**High Coupling:**
- `app/graph_app.py` - Orchestrates entire stack (expected for entry point)
- `scripts/run_eval_harness.py` - Uses all src/ modules (expected for integration script)

---

## Testing Strategy

### Current State

**Test Package:** `/home/donbr/don-aie-cohort8/cert-challenge/tests/`
- Status: Placeholder package with empty `__init__.py`
- Purpose: Reserved for future unit/integration tests

### Validation Approach

Instead of traditional unit tests, the project uses a comprehensive validation script:

**File:** `/home/donbr/don-aie-cohort8/cert-challenge/scripts/run_app_validation.py`

**Coverage:**
1. Environment validation (API keys, Qdrant connectivity)
2. Module import validation (each src/ module)
3. Factory pattern validation (correct initialization)
4. Graph compilation validation (all 4 workflows)
5. Functional validation (actual queries)
6. Configuration consistency checks

**Benefits:**
- Tests against real infrastructure (Qdrant, OpenAI)
- Validates production configuration
- Demonstrates correct usage patterns
- Provides actionable error messages

**Exit Codes:**
- 0: All validations passed (safe to deploy)
- 1: Validation failures (fix before deployment)

### Recommended Test Structure (Future)

```
tests/
├── unit/
│   ├── test_config.py           # Test configuration singletons
│   ├── test_retrievers.py       # Test retriever factories
│   ├── test_graph.py            # Test graph builders
│   └── test_utils.py            # Test utility functions
├── integration/
│   ├── test_rag_pipeline.py     # End-to-end RAG tests
│   └── test_evaluation.py       # RAGAS evaluation tests
└── fixtures/
    ├── mock_documents.py        # Sample documents
    └── mock_responses.py        # Sample LLM responses
```

---

## Summary

This GDELT RAG system demonstrates production-grade software engineering practices:

1. **Modular Architecture:** Clear separation between public API (`src/`), application layer (`app/`), and executable scripts (`scripts/`)

2. **Factory Pattern:** All components use factories to prevent premature initialization and enable flexible configuration

3. **Singleton Pattern:** Cached configuration instances prevent resource duplication

4. **State Management:** LangGraph nodes return partial updates, leveraging framework auto-merge

5. **Reproducibility:** Comprehensive manifests capture exact versions, checksums, and configuration

6. **Data Lineage:** Links from raw PDFs → interim storage → HF datasets → evaluation results

7. **Validation Over Testing:** Comprehensive validation script ensures production readiness

8. **Documentation:** Inline docstrings, type hints, and extensive comments throughout

**Total Components Analyzed:**
- 6 core modules (`src/`)
- 2 application modules (`app/`)
- 5 executable scripts (`scripts/`)
- 1 configuration file (`pyproject.toml`)
- 28 external dependencies

**Lines of Code:** ~3,500 (excluding analysis framework and generated outputs)

**Primary Contributors:** dwb2023 (based on HuggingFace username and git history)

**Project Status:** Active development, production-ready for GDELT RAG use cases
