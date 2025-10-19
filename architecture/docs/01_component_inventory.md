# Component Inventory

## Overview

This is a **GDELT RAG (Retrieval-Augmented Generation) System** built with LangChain and LangGraph. The project implements a production-grade RAG pipeline for querying knowledge graphs from the GDELT (Global Database of Events, Language, and Tone) dataset.

The codebase follows a clean **factory pattern architecture** where:
- Configuration and infrastructure are managed centrally (`src/config.py`)
- Data loading is handled by utilities (`src/utils.py`)
- Retrievers are created via factory functions (`src/retrievers.py`)
- LangGraph workflows are built dynamically (`src/graph.py`)
- State management uses TypedDict schemas (`src/state.py`)
- Prompts are centralized as constants (`src/prompts.py`)

The project includes comprehensive evaluation tooling using RAGAS metrics, data ingestion pipelines, and HuggingFace dataset management.

**Project Structure:**
- `src/` - Core RAG system library (public API)
- `scripts/` - Command-line tools and utilities (entry points)
- `main.py` - Simple hello-world entry point

---

## Public API

### Modules

#### src/ - Core RAG System
**Location:** `/home/donbr/don-aie-cohort8/cert-challenge/src/__init__.py`

The main package providing the RAG system. Exported modules:
- `config` - Configuration management
- `graph` - LangGraph workflow builders
- `prompts` - Prompt templates
- `retrievers` - Retriever factory functions
- `state` - State schema definitions
- `utils` - Document loading utilities

**Version:** 0.1.0

---

### Classes

#### State (TypedDict)
**Location:** `/home/donbr/don-aie-cohort8/cert-challenge/src/state.py:7`

```python
class State(TypedDict):
    question: str
    context: List[Document]
    response: str
```

**Purpose:** LangGraph state schema defining the data flow through RAG workflow nodes.

**Fields:**
- `question` (str) - User's input question
- `context` (List[Document]) - Retrieved documents from vector store
- `response` (str) - Generated answer from LLM

**Usage:** Used internally by all LangGraph workflows built via `build_graph()` and `build_all_graphs()`.

---

### Functions

#### Public API Functions (src/)

##### load_documents_from_huggingface()
**Location:** `/home/donbr/don-aie-cohort8/cert-challenge/src/utils.py:15`

```python
def load_documents_from_huggingface(
    dataset_name: str = "dwb2023/gdelt-rag-sources",
    split: str = "train",
    revision: str = None
) -> List[Document]
```

**Purpose:** Load documents from HuggingFace dataset and convert to LangChain Documents.

**Parameters:**
- `dataset_name` - HuggingFace dataset identifier (default: "dwb2023/gdelt-rag-sources")
- `split` - Dataset split to load (default: "train")
- `revision` - Dataset revision/commit SHA to pin (default: None, uses HF_SOURCES_REV env var)

**Returns:** List of LangChain Document objects with page_content and metadata

**Features:**
- Handles nested metadata structures automatically
- Supports revision pinning for reproducibility via parameter or `HF_SOURCES_REV` env var
- Preserves all metadata fields from source dataset

---

##### load_golden_testset_from_huggingface()
**Location:** `/home/donbr/don-aie-cohort8/cert-challenge/src/utils.py:78`

```python
def load_golden_testset_from_huggingface(
    dataset_name: str = "dwb2023/gdelt-rag-golden-testset",
    split: str = "train",
    revision: str = None
)
```

**Purpose:** Load golden testset from HuggingFace dataset for evaluation.

**Parameters:**
- `dataset_name` - HuggingFace dataset identifier (default: "dwb2023/gdelt-rag-golden-testset")
- `split` - Dataset split to load (default: "train")
- `revision` - Dataset revision/commit SHA to pin (default: None, uses HF_GOLDEN_REV env var)

**Returns:** HuggingFace Dataset object

**Features:**
- Revision pinning ensures test set consistency across runs
- Prevents score drift due to dataset updates

---

##### get_llm()
**Location:** `/home/donbr/don-aie-cohort8/cert-challenge/src/config.py:28`

```python
@lru_cache(maxsize=1)
def get_llm()
```

**Purpose:** Get cached LLM instance (ChatOpenAI with temperature=0).

**Returns:** ChatOpenAI instance configured from environment

**Configuration:**
- Model: `OPENAI_MODEL` env var (default: "gpt-4.1-mini")
- Temperature: 0 (deterministic outputs)
- Cached: Yes (singleton pattern)

---

##### get_embeddings()
**Location:** `/home/donbr/don-aie-cohort8/cert-challenge/src/config.py:39`

```python
@lru_cache(maxsize=1)
def get_embeddings()
```

**Purpose:** Get cached embeddings instance (OpenAI embeddings).

**Returns:** OpenAIEmbeddings instance

**Configuration:**
- Model: `OPENAI_EMBED_MODEL` env var (default: "text-embedding-3-small")
- Dimensions: 1536
- Cached: Yes (singleton pattern)

---

##### get_qdrant()
**Location:** `/home/donbr/don-aie-cohort8/cert-challenge/src/config.py:50`

```python
@lru_cache(maxsize=1)
def get_qdrant()
```

**Purpose:** Get cached Qdrant client instance.

**Returns:** QdrantClient connected to configured host/port

**Configuration:**
- Host: `QDRANT_HOST` env var (default: "localhost")
- Port: `QDRANT_PORT` env var (default: 6333)
- Cached: Yes (singleton pattern)

---

##### get_collection_name()
**Location:** `/home/donbr/don-aie-cohort8/cert-challenge/src/config.py:60`

```python
def get_collection_name() -> str
```

**Purpose:** Get configured Qdrant collection name.

**Returns:** Collection name string from `QDRANT_COLLECTION` env var (default: "gdelt_comparative_eval")

---

##### create_vector_store()
**Location:** `/home/donbr/don-aie-cohort8/cert-challenge/src/config.py:70`

```python
def create_vector_store(
    documents: List[Document],
    collection_name: str = None,
    recreate_collection: bool = False
) -> QdrantVectorStore
```

**Purpose:** Create and populate Qdrant vector store (factory function).

**Parameters:**
- `documents` - List of Document objects to add to vector store
- `collection_name` - Override default collection name (optional)
- `recreate_collection` - If True, delete existing collection first (default: False)

**Returns:** Populated QdrantVectorStore instance

**Features:**
- Creates Qdrant collection if it doesn't exist
- Optionally recreates collection (deletes old data)
- Automatically populates with documents
- Uses cosine distance with 1536-dimensional vectors

---

##### create_retrievers()
**Location:** `/home/donbr/don-aie-cohort8/cert-challenge/src/retrievers.py:20`

```python
def create_retrievers(
    documents: List[Document],
    vector_store: QdrantVectorStore,
    k: int = 5,
) -> Dict[str, object]
```

**Purpose:** Create all retriever instances (factory function).

**Parameters:**
- `documents` - List of Document objects (required for BM25)
- `vector_store` - Populated QdrantVectorStore instance
- `k` - Number of documents to retrieve (default: 5)

**Returns:** Dictionary mapping retriever names to retriever instances

**Retriever Strategies:**
1. **naive** - Dense vector search using embeddings
2. **bm25** - Sparse keyword matching (lexical search)
3. **ensemble** - Hybrid combination (50% dense + 50% sparse)
4. **cohere_rerank** - Contextual compression with Cohere rerank-v3.5 (retrieves 20, reranks to top k)

---

##### build_graph()
**Location:** `/home/donbr/don-aie-cohort8/cert-challenge/src/graph.py:21`

```python
def build_graph(retriever, llm=None, prompt_template: str = None)
```

**Purpose:** Build a compiled LangGraph pipeline for a single retriever.

**Parameters:**
- `retriever` - Retriever instance to use for document retrieval
- `llm` - ChatOpenAI instance (defaults to get_llm() if None)
- `prompt_template` - RAG prompt template string (defaults to BASELINE_PROMPT)

**Returns:** Compiled StateGraph that can be invoked with `{"question": "..."}`

**Graph Structure:**
- START → retrieve → generate → END
- `retrieve` node: Fetches relevant documents
- `generate` node: Produces answer based on documents

**Node Functions:**
- Return partial state updates (dict)
- LangGraph automatically merges updates into state
- Follows LangGraph best practices for state management

---

##### build_all_graphs()
**Location:** `/home/donbr/don-aie-cohort8/cert-challenge/src/graph.py:109`

```python
def build_all_graphs(retrievers: Dict[str, object], llm=None) -> Dict[str, object]
```

**Purpose:** Build compiled graphs for all retrievers (convenience function).

**Parameters:**
- `retrievers` - Dictionary of retriever instances from create_retrievers()
- `llm` - Optional ChatOpenAI instance (shared across all graphs)

**Returns:** Dictionary mapping retriever names to compiled graphs (same keys as input)

**Usage:** Creates a graph for each retriever in the dictionary, enabling easy batch evaluation.

---

## Internal Implementation

### Core Modules

#### src/config.py - Configuration Management
**Location:** `/home/donbr/don-aie-cohort8/cert-challenge/src/config.py`

**Purpose:** Centralized configuration for LLM, embeddings, and Qdrant client.

**Responsibilities:**
- Environment variable management
- Singleton pattern for expensive resources (LRU cache)
- Vector store factory functions
- Configuration constants

**Constants:**
- `QDRANT_HOST` (line 20) - Default: "localhost"
- `QDRANT_PORT` (line 21) - Default: 6333
- `COLLECTION_NAME` (line 22) - Default: "gdelt_comparative_eval"
- `OPENAI_MODEL` (line 23) - Default: "gpt-4.1-mini"
- `OPENAI_EMBED_MODEL` (line 24) - Default: "text-embedding-3-small"

**Design Pattern:** All getters use `@lru_cache(maxsize=1)` to implement singleton pattern, preventing duplicate API connections.

---

#### src/graph.py - LangGraph Workflow Factory
**Location:** `/home/donbr/don-aie-cohort8/cert-challenge/src/graph.py`

**Purpose:** Factory functions to create LangGraph workflows.

**Why Factory Pattern?**
- Graphs depend on retrievers that must be created first
- Cannot instantiate at module level (would cause import-time failures)
- Allows dynamic configuration of retrieval strategies

**Internal Node Functions:**
- `retrieve(state: State) -> dict` (line 67) - Retrieves documents for question
- `generate(state: State) -> dict` (line 80) - Generates answer from context

**Graph Construction:**
```python
graph = StateGraph(State)
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)
return graph.compile()
```

---

#### src/retrievers.py - Retriever Factory
**Location:** `/home/donbr/don-aie-cohort8/cert-challenge/src/retrievers.py`

**Purpose:** Factory functions to create retriever instances.

**Why Factory Pattern?**
- Retrievers require documents and vector stores to be created first
- Cannot instantiate at module level without triggering errors
- Enables flexible configuration of k values and retrieval strategies

**Retriever Implementations:**

1. **Naive Retriever** (line 63)
   - Type: Dense vector search
   - Implementation: `vector_store.as_retriever(search_kwargs={"k": k})`
   - Strategy: Semantic similarity using embeddings

2. **BM25 Retriever** (line 66)
   - Type: Sparse keyword matching
   - Implementation: `BM25Retriever.from_documents(documents, k=k)`
   - Strategy: Lexical matching (TF-IDF variant)

3. **Ensemble Retriever** (line 69)
   - Type: Hybrid search
   - Implementation: `EnsembleRetriever(retrievers=[naive, bm25], weights=[0.5, 0.5])`
   - Strategy: Combines dense and sparse with equal weighting

4. **Cohere Rerank Retriever** (line 76)
   - Type: Contextual compression
   - Implementation: `ContextualCompressionRetriever` with `CohereRerank(model="rerank-v3.5")`
   - Strategy: Retrieve 20 candidates, rerank to top k using Cohere's neural reranker

---

### Utility Modules

#### src/utils.py - Document Loading
**Location:** `/home/donbr/don-aie-cohort8/cert-challenge/src/utils.py`

**Purpose:** Helper functions for loading and processing documents from HuggingFace.

**Responsibilities:**
- HuggingFace dataset integration
- Document format conversion (HF Dataset → LangChain Documents)
- Metadata handling and sanitization
- Revision pinning for reproducibility

**Internal Implementation Details:**
- Handles nested metadata structures (line 64-69)
- Supports environment variable configuration for revisions (line 52, 111)
- Preserves all metadata fields from source datasets

---

#### src/state.py - State Schema
**Location:** `/home/donbr/don-aie-cohort8/cert-challenge/src/state.py`

**Purpose:** TypedDict schema for LangGraph state management.

**Design:** Simple, minimal state representation (3 fields only)
- Keeps graph logic clean and predictable
- Enables type checking in development
- Documents expected state shape for node functions

---

#### src/prompts.py - Prompt Templates
**Location:** `/home/donbr/don-aie-cohort8/cert-challenge/src/prompts.py`

**Purpose:** Centralized prompt template constants.

**Template:** `BASELINE_PROMPT` (line 4)
```
You are a helpful assistant who answers questions based on provided context.
You must only use the provided context, and cannot use your own knowledge.

### Question
{question}

### Context
{context}
```

**Design Rationale:**
- Enforces context-only answering (prevents hallucination)
- Clear structure for evaluation
- Reusable across all retrieval strategies

---

### Configuration

**Environment Variables:**
- `OPENAI_API_KEY` - Required for LLM and embeddings
- `COHERE_API_KEY` - Required for cohere_rerank retriever
- `QDRANT_HOST` - Qdrant server host (default: "localhost")
- `QDRANT_PORT` - Qdrant server port (default: "6333")
- `QDRANT_COLLECTION` - Collection name (default: "gdelt_comparative_eval")
- `OPENAI_MODEL` - LLM model (default: "gpt-4.1-mini")
- `OPENAI_EMBED_MODEL` - Embedding model (default: "text-embedding-3-small")
- `HF_SOURCES_REV` - Pin HuggingFace sources dataset revision (optional)
- `HF_GOLDEN_REV` - Pin HuggingFace golden testset revision (optional)
- `HF_HUB_DISABLE_XET` - Disable XetHub progress bars (default: "1")
- `HF_HUB_DISABLE_PROGRESS_BARS` - Disable HF progress bars (default: "1")

---

## Entry Points

### Main Scripts

#### main.py
**Location:** `/home/donbr/don-aie-cohort8/cert-challenge/main.py:5`

```python
if __name__ == "__main__":
    main()
```

**Purpose:** Simple hello-world entry point (minimal functionality).

**Output:** Prints "Hello from cert-challenge!"

---

### Utility Scripts (scripts/)

#### scripts/single_file.py - Comprehensive RAGAS Evaluation
**Location:** `/home/donbr/don-aie-cohort8/cert-challenge/scripts/single_file.py`

**Purpose:** Complete RAGAS evaluation harness in a single file (Tasks 5 & 7).

**Entry Point:** Line 6 (shebang: `#!/usr/bin/env python3`)

**What It Does:**
1. Loads 38 source documents from HuggingFace
2. Creates Qdrant vector store
3. Initializes 4 retrievers (naive, BM25, ensemble, cohere_rerank)
4. Builds LangGraph workflows for each retriever
5. Runs 12 test questions through each retriever
6. Evaluates with RAGAS metrics (faithfulness, answer_relevancy, context_precision, context_recall)
7. Generates comparative analysis table
8. Saves all results to `deliverables/evaluation_evidence/`

**Key Functions:**
- `validate_and_normalize_ragas_schema()` (line 66) - Ensures DataFrame matches RAGAS 0.2.10 schema
- `retrieve_baseline()` (line 260) - Naive dense vector search
- `retrieve_bm25()` (line 265) - BM25 sparse keyword matching
- `retrieve_reranked()` (line 270) - Cohere contextual compression
- `retrieve_ensemble()` (line 275) - Ensemble hybrid search
- `generate()` (line 281) - Shared answer generation function

**Output Files:**
- `{retriever}_raw_dataset.parquet` - Raw outputs before RAGAS evaluation
- `{retriever}_evaluation_dataset.csv` - RAGAS evaluation datasets
- `{retriever}_detailed_results.csv` - Per-question metric scores
- `comparative_ragas_results.csv` - Summary table comparing all retrievers
- `RUN_MANIFEST.json` - Reproducibility manifest

**Runtime:** 20-30 minutes
**Cost:** ~$5-6 in OpenAI API calls

---

#### scripts/run_eval_harness.py - Modular RAGAS Evaluation
**Location:** `/home/donbr/don-aie-cohort8/cert-challenge/scripts/run_eval_harness.py`

**Purpose:** Same as single_file.py but uses src/ modules instead of inline code.

**Entry Point:** Line 320 (command-line argument parsing)

**Command-Line Arguments:**
- `--recreate` - Recreate Qdrant collection (choices: "true", "false", default: "false")

**Usage:**
```bash
make eval
# or
export PYTHONPATH=.
python scripts/run_eval_harness.py --recreate false
```

**Advantages Over single_file.py:**
- Uses factory functions from src/ (no code duplication)
- Cleaner separation of concerns
- Easier to maintain and extend
- Same outputs and results

**Execution Steps:**
1. STEP 1: Loading data (sources + golden testset)
2. STEP 2: Building RAG stack (vector store + retrievers + graphs)
3. STEP 3: Running inference (12 questions × 4 retrievers)
4. STEP 4: RAGAS evaluation
5. STEP 5: Comparative analysis

---

#### scripts/validate_langgraph.py - Configuration Validation
**Location:** `/home/donbr/don-aie-cohort8/cert-challenge/scripts/validate_langgraph.py`

**Purpose:** Validate LangGraph implementation and demonstrate correct patterns.

**Entry Point:** Line 484 (`if __name__ == "__main__":`)

**What It Validates:**
1. Environment configuration (API keys, Qdrant connectivity)
2. Module imports (which src/ modules are broken)
3. Retriever factory pattern (correct initialization)
4. Graph compilation (all 4 LangGraph workflows)
5. Functional execution (test queries through each graph)
6. Configuration consistency

**Key Validation Functions:**
- `check_environment()` (line 74) - Validates API keys, Qdrant connection, critical imports
- `test_module_imports()` (line 134) - Tests importing each src/ module individually
- `demonstrate_correct_pattern()` (line 181) - Shows correct factory pattern usage
- `validate_graph_compilation()` (line 228) - Validates LangGraph compilation
- `run_functional_tests()` (line 268) - Runs test queries through all graphs
- `generate_diagnostic_report()` (line 311) - Generates final diagnostic report

**Exit Codes:**
- 0: All validations passed
- 1: One or more validations failed

**Features:**
- Color-coded terminal output
- Detailed error messages with traceback
- Recommendations for fixing issues
- Comprehensive diagnostic report

---

#### scripts/ingest.py - Data Ingestion Pipeline
**Location:** `/home/donbr/don-aie-cohort8/cert-challenge/scripts/ingest.py`

**Purpose:** Standardized RAGAS golden testset pipeline.

**Entry Point:** Jupyter notebook / script execution (cell-based)

**What It Does:**
1. Extracts PDFs to LangChain Documents
2. Sanitizes metadata for Arrow/JSON serialization
3. Generates golden testset using RAGAS TestsetGenerator
4. Persists SOURCES and GOLDEN TESTSET to `/data/interim` in JSONL, Parquet, HF-dataset
5. Writes manifest with checksums & schema for provenance

**Key Functions:**
- `ensure_jsonable()` (line 92) - Makes metadata JSON-serializable
- `docs_to_jsonl()` (line 106) - Converts docs to JSONL format
- `docs_to_parquet()` (line 115) - Converts docs to Parquet format
- `docs_to_hfds()` (line 121) - Converts docs to HuggingFace dataset
- `hash_file()` (line 128) - Computes SHA256 hash for file
- `summarize_columns_from_jsonl()` (line 138) - Extracts schema from JSONL
- `build_testset()` (line 202) - Builds testset with RAGAS (0.3.x or 0.2.x fallback)

**Configuration:**
- `TESTSET_SIZE` (line 82) - Default: 10
- `MAX_DOCS` (line 83) - Optional limit for prototyping
- `RANDOM_SEED` (line 86) - Default: 42

**Output Files:**
- `data/interim/sources.docs.jsonl`
- `data/interim/sources.docs.parquet`
- `data/interim/sources.hfds/`
- `data/interim/golden_testset.jsonl`
- `data/interim/golden_testset.parquet`
- `data/interim/golden_testset.hfds/`
- `data/interim/manifest.json`

---

#### scripts/upload_to_hf.py - HuggingFace Upload
**Location:** `/home/donbr/don-aie-cohort8/cert-challenge/scripts/upload_to_hf.py`

**Purpose:** Upload GDELT RAG datasets to HuggingFace Hub.

**Entry Point:** Line 291 (`if __name__ == "__main__":`)

**What It Does:**
1. Loads source documents and golden testset from local storage
2. Creates dataset cards with metadata
3. Uploads datasets to HuggingFace Hub
4. Updates manifest.json with dataset repo IDs

**Key Functions:**
- `create_sources_card()` (line 34) - Creates README for sources dataset
- `create_golden_testset_card()` (line 113) - Creates README for testset
- `load_manifest()` (line 194) - Loads manifest.json
- `update_manifest()` (line 200) - Updates manifest with repo IDs
- `main()` (line 219) - Main upload function

**Configuration:**
- `HF_USERNAME` (line 21) - Default: "dwb2023"
- `SOURCES_DATASET_NAME` (line 22) - Default: "dwb2023/gdelt-rag-sources"
- `GOLDEN_TESTSET_NAME` (line 23) - Default: "dwb2023/gdelt-rag-golden-testset"

**Environment Variables:**
- `HF_TOKEN` - Required for authentication

**Datasets Published:**
- https://huggingface.co/datasets/dwb2023/gdelt-rag-sources
- https://huggingface.co/datasets/dwb2023/gdelt-rag-golden-testset

---

#### scripts/generate_run_manifest.py - Manifest Generation
**Location:** `/home/donbr/don-aie-cohort8/cert-challenge/scripts/generate_run_manifest.py`

**Purpose:** Generate RUN_MANIFEST.json for reproducibility.

**Entry Point:** Line 173 (`if __name__ == "__main__":`)

**What It Captures:**
- Model versions and parameters
- Retriever configurations
- Evaluation settings
- Dependencies (Python, RAGAS, etc.)
- Results summary (if provided)

**Key Function:**
- `generate_manifest()` (line 21) - Main manifest generation function

**Manifest Structure:**
- `ragas_version` - RAGAS library version
- `python_version` - Python version
- `llm` - LLM configuration (model, temperature, provider)
- `embeddings` - Embedding configuration
- `retrievers` - List of retriever configurations
- `evaluation` - Evaluation settings (datasets, metrics, timeout)
- `vector_store` - Qdrant configuration
- `results_summary` - Evaluation results (optional)
- `generated_at` - Timestamp

**Usage:**
```python
from generate_run_manifest import generate_manifest
manifest = generate_manifest(output_path, evaluation_results, retrievers_config)
```

---

#### scripts/enrich_manifest.py - Manifest Enrichment
**Location:** `/home/donbr/don-aie-cohort8/cert-challenge/scripts/enrich_manifest.py`

**Purpose:** Enrich manifest.json with environment, artifacts, metrics, and lineage.

**Entry Point:** Line 239 (`if __name__ == "__main__":`)

**What It Adds:**
- Environment details (Python version, OS, dependency versions)
- Input metadata (source directory)
- Artifact details (bytes, rows, schemas, SHA256 hashes)
- Metrics (document counts, character statistics, avg reference contexts)
- Lineage scaffold (HuggingFace, LangSmith, Phoenix)
- Compliance scaffold (license, PII policy)
- Run details (git commit SHA, random seed)

**Key Functions:**
- `sha256()` (line 7) - Computes SHA256 hash
- `count_jsonl_rows()` (line 19) - Counts rows in JSONL file
- `hfds_rows()` (line 24) - Counts rows in HuggingFace dataset
- `parquet_rows()` (line 37) - Counts rows in Parquet file
- `pandas_schema_from_parquet()` (line 51) - Extracts schema from Parquet
- `char_stats_jsonl()` (line 61) - Computes character statistics
- `main()` (line 81) - Main enrichment function

**Usage:**
```bash
python scripts/enrich_manifest.py [path/to/manifest.json]
```

**Default Path:** `data/interim/manifest.json`

---

### Programmatic Entry Points

#### Public API Integration Pattern

**Recommended Usage Pattern:**
```python
from src.utils import load_documents_from_huggingface
from src.config import create_vector_store
from src.retrievers import create_retrievers
from src.graph import build_all_graphs

# Load data
documents = load_documents_from_huggingface()

# Create vector store
vector_store = create_vector_store(documents, recreate_collection=True)

# Create retrievers
retrievers = create_retrievers(documents, vector_store)

# Build LangGraph workflows
graphs = build_all_graphs(retrievers)

# Query the system
result = graphs['naive'].invoke({"question": "What is GDELT?"})
print(result['response'])
```

**Design Philosophy:**
- Factory pattern ensures correct initialization order
- No module-level globals or singletons (except cached config)
- Each step is explicit and configurable
- Type hints guide correct usage
- Comprehensive docstrings document patterns

---

## Dependencies

### Core RAG Stack

**LangChain Ecosystem:**
- `langchain` - Core LangChain abstractions
- `langchain-openai` - OpenAI LLM and embeddings integration
- `langchain-community` - Community retrievers (BM25)
- `langchain-cohere` - Cohere reranker integration
- `langchain-qdrant` - Qdrant vector store integration
- `langchain-core` - Core LangChain types (Document)

**LangGraph:**
- `langgraph` - State machine workflow orchestration

**Vector Database:**
- `qdrant-client` - Qdrant vector database client

**Embeddings & LLM:**
- `openai` - OpenAI API client (used via langchain-openai)

**Retrievers:**
- `cohere` - Cohere API for reranking (rerank-v3.5 model)

---

### Evaluation & Metrics

**RAGAS:**
- `ragas` - Retrieval-Augmented Generation Assessment framework
  - Version: 0.2.10 (based on imports and usage patterns)
  - Metrics: Faithfulness, ResponseRelevancy, ContextPrecision, LLMContextRecall

**Retry Logic:**
- `tenacity` - Retry logic for transient API errors

---

### Data Processing

**HuggingFace:**
- `datasets` - HuggingFace datasets library
- `huggingface-hub` - HuggingFace Hub API client

**Data Formats:**
- `pandas` - DataFrame operations
- `pyarrow` - Parquet file format support

**Document Loading:**
- `langchain-community` - DirectoryLoader, PyMuPDFLoader

---

### Infrastructure

**Python Standard Library:**
- `os`, `sys`, `pathlib` - File system operations
- `json` - JSON serialization
- `hashlib` - SHA256 hashing
- `datetime` - Timestamps
- `functools` - LRU cache decorators
- `typing` - Type hints
- `argparse` - Command-line argument parsing
- `copy` - Deep copying
- `traceback` - Error reporting

**Type Hints:**
- `typing` - Generic type hints (List, Dict, Any, Optional)
- `typing_extensions` - TypedDict (for Python 3.8+ compatibility)

---

### Development & Testing

**Environment Management:**
- Environment variables for configuration (OPENAI_API_KEY, COHERE_API_KEY, etc.)
- `.env` file support (implicit via os.getenv())

**Validation:**
- `scripts/validate_langgraph.py` - Comprehensive validation suite

---

## Architecture Patterns

### Factory Pattern

**Why?** Resolves circular dependency issues and enables dynamic configuration.

**Used In:**
- `src/config.py::create_vector_store()` - Creates and populates vector stores
- `src/retrievers.py::create_retrievers()` - Creates retriever instances
- `src/graph.py::build_graph()` - Builds LangGraph workflows
- `src/graph.py::build_all_graphs()` - Batch graph creation

**Benefits:**
- Prevents import-time errors
- Enables testing and mocking
- Supports multiple configurations
- Clear dependency injection

---

### Singleton Pattern

**Implementation:** `@lru_cache(maxsize=1)` decorator

**Used In:**
- `src/config.py::get_llm()` - Single LLM instance
- `src/config.py::get_embeddings()` - Single embeddings instance
- `src/config.py::get_qdrant()` - Single Qdrant client

**Benefits:**
- Prevents duplicate API connections
- Reduces memory footprint
- Ensures configuration consistency

---

### State Machine Pattern

**Implementation:** LangGraph StateGraph

**Components:**
- `src/state.py::State` - TypedDict schema
- `src/graph.py::build_graph()` - Graph builder
- Node functions return partial state updates
- LangGraph merges updates automatically

**Benefits:**
- Predictable state transitions
- Easy to test individual nodes
- Clear data flow
- Supports complex workflows

---

## File Organization

```
/home/donbr/don-aie-cohort8/cert-challenge/
├── main.py                           # Simple entry point
├── src/                              # Core RAG system (public API)
│   ├── __init__.py                   # Package initialization
│   ├── config.py                     # Configuration management
│   ├── graph.py                      # LangGraph workflow factory
│   ├── prompts.py                    # Prompt templates
│   ├── retrievers.py                 # Retriever factory
│   ├── state.py                      # State schema
│   └── utils.py                      # Document loading utilities
├── scripts/                          # Command-line tools
│   ├── single_file.py                # Comprehensive RAGAS evaluation (inline)
│   ├── run_eval_harness.py           # Modular RAGAS evaluation (uses src/)
│   ├── validate_langgraph.py         # Configuration validation
│   ├── ingest.py                     # Data ingestion pipeline
│   ├── upload_to_hf.py               # HuggingFace upload
│   ├── generate_run_manifest.py      # Manifest generation
│   └── enrich_manifest.py            # Manifest enrichment
├── deliverables/                     # Evaluation outputs
│   └── evaluation_evidence/
│       ├── comparative_ragas_results.csv
│       ├── {retriever}_raw_dataset.parquet
│       ├── {retriever}_evaluation_dataset.csv
│       ├── {retriever}_detailed_results.csv
│       └── RUN_MANIFEST.json
└── data/                             # Data storage
    ├── raw/                          # Raw PDFs
    ├── interim/                      # Processed datasets
    │   ├── sources.docs.jsonl
    │   ├── sources.docs.parquet
    │   ├── sources.hfds/
    │   ├── golden_testset.jsonl
    │   ├── golden_testset.parquet
    │   ├── golden_testset.hfds/
    │   └── manifest.json
    └── processed/                    # Final outputs
```

---

## Summary Statistics

**Core Library (src/):**
- 6 modules
- 1 class (State TypedDict)
- 9 public functions
- 416 total lines of code (including docstrings)

**Scripts:**
- 7 utility scripts
- 3 main entry points (single_file.py, run_eval_harness.py, validate_langgraph.py)
- 4 data pipeline scripts (ingest.py, upload_to_hf.py, generate_run_manifest.py, enrich_manifest.py)

**Retrievers:**
- 4 retrieval strategies (naive, BM25, ensemble, cohere_rerank)
- All return top-k documents (default k=5)

**Evaluation:**
- 4 RAGAS metrics (faithfulness, answer_relevancy, context_precision, context_recall)
- 12 test questions
- 38 source documents

**External Dependencies:**
- 15+ external libraries
- 2 API services (OpenAI, Cohere)
- 1 vector database (Qdrant)

---

## Notes

**Code Quality:**
- Comprehensive docstrings with examples
- Type hints throughout
- Factory pattern for clean dependency management
- Environment variable configuration
- Reproducibility via manifest files

**Testing Strategy:**
- Validation script (validate_langgraph.py) provides comprehensive checks
- Functional tests run actual queries through all graphs
- RAGAS evaluation provides quantitative metrics

**Data Lineage:**
- All datasets published to HuggingFace Hub
- SHA256 hashes tracked in manifest
- Revision pinning supported via environment variables

**Production Readiness:**
- Temperature=0 for deterministic outputs
- Retry logic for transient errors (in ingest.py)
- Proper error handling and logging
- Configurable via environment variables
- No hardcoded secrets
