# Component Inventory

**Generated:** 2025-10-18
**Project:** GDELT RAG System - Retrieval-Augmented Generation for GDELT Knowledge Graphs
**Version:** 0.1.0

## Overview

The GDELT RAG System is a production-grade Retrieval-Augmented Generation system built on LangChain and LangGraph. The codebase is organized into three primary layers:

1. **Public API** (`src/` package) - A reusable module providing configuration, retrieval strategies, and LangGraph workflow builders
2. **Entry Points** (`scripts/` and `main.py`) - Executable scripts for evaluation, data ingestion, and system validation
3. **Internal Implementation** - Helper utilities and state management for the RAG pipeline

The system implements four retrieval strategies (naive dense, BM25 sparse, ensemble hybrid, and Cohere rerank) and evaluates them using RAGAS metrics against a golden testset of GDELT-related questions.

**Key Technologies:**
- LangChain for document processing and RAG chains
- LangGraph for stateful workflow orchestration
- Qdrant for vector storage
- OpenAI for embeddings and LLM
- RAGAS for RAG evaluation
- HuggingFace Datasets for data management

---

## Public API

### Modules

The `src/` package provides the public API for building RAG systems:

| Module | Path | Purpose |
|--------|------|---------|
| `config` | `/home/donbr/don-aie-cohort8/cert-challenge/src/config.py` | Configuration management with cached LLM, embeddings, and Qdrant client |
| `utils` | `/home/donbr/don-aie-cohort8/cert-challenge/src/utils.py` | Document loading utilities from HuggingFace datasets |
| `retrievers` | `/home/donbr/don-aie-cohort8/cert-challenge/src/retrievers.py` | Retriever factory functions (naive, BM25, ensemble, rerank) |
| `graph` | `/home/donbr/don-aie-cohort8/cert-challenge/src/graph.py` | LangGraph workflow builders for RAG pipelines |
| `state` | `/home/donbr/don-aie-cohort8/cert-challenge/src/state.py` | State schema TypedDict definitions |
| `prompts` | `/home/donbr/don-aie-cohort8/cert-challenge/src/prompts.py` | Shared prompt templates |

### Classes

#### Configuration Classes

**No explicit classes defined** - The configuration module uses functional factory patterns with `@lru_cache` decorators for singleton behavior.

#### State Classes

##### `State` (TypedDict)
- **File:** `/home/donbr/don-aie-cohort8/cert-challenge/src/state.py:7`
- **Purpose:** Defines the state schema for LangGraph workflows
- **Fields:**
  - `question: str` - User input question
  - `context: List[Document]` - Retrieved context documents
  - `response: str` - Generated answer from LLM
- **Usage:** Type hint for LangGraph node functions

---

### Functions

#### Configuration Functions (`src/config.py`)

##### `get_llm()`
- **File:** `/home/donbr/don-aie-cohort8/cert-challenge/src/config.py:28`
- **Signature:** `() -> ChatOpenAI`
- **Purpose:** Returns cached ChatOpenAI instance with temperature=0
- **Caching:** `@lru_cache(maxsize=1)` - singleton pattern
- **Configuration:** Uses `OPENAI_MODEL` env var (default: "gpt-4.1-mini")
- **Public API:** Yes - intended for external use

##### `get_embeddings()`
- **File:** `/home/donbr/don-aie-cohort8/cert-challenge/src/config.py:39`
- **Signature:** `() -> OpenAIEmbeddings`
- **Purpose:** Returns cached OpenAI embeddings instance
- **Caching:** `@lru_cache(maxsize=1)` - singleton pattern
- **Configuration:** Uses `OPENAI_EMBED_MODEL` env var (default: "text-embedding-3-small")
- **Public API:** Yes - intended for external use

##### `get_qdrant()`
- **File:** `/home/donbr/don-aie-cohort8/cert-challenge/src/config.py:50`
- **Signature:** `() -> QdrantClient`
- **Purpose:** Returns cached Qdrant client instance
- **Caching:** `@lru_cache(maxsize=1)` - singleton pattern
- **Configuration:** Uses `QDRANT_HOST` and `QDRANT_PORT` env vars
- **Public API:** Yes - intended for external use

##### `get_collection_name()`
- **File:** `/home/donbr/don-aie-cohort8/cert-challenge/src/config.py:60`
- **Signature:** `() -> str`
- **Purpose:** Returns configured Qdrant collection name
- **Configuration:** Uses `QDRANT_COLLECTION` env var (default: "gdelt_comparative_eval")
- **Public API:** Yes - intended for external use

##### `create_vector_store()`
- **File:** `/home/donbr/don-aie-cohort8/cert-challenge/src/config.py:70`
- **Signature:** `(documents: List[Document], collection_name: str = None, recreate_collection: bool = False) -> QdrantVectorStore`
- **Purpose:** Factory function to create and populate Qdrant vector store
- **Features:**
  - Creates collection if it doesn't exist
  - Optionally recreates collection (deletes existing first)
  - Populates vector store with documents
  - Returns configured QdrantVectorStore instance
- **Public API:** Yes - primary factory function for vector store setup

#### Utility Functions (`src/utils.py`)

##### `load_documents_from_huggingface()`
- **File:** `/home/donbr/don-aie-cohort8/cert-challenge/src/utils.py:15`
- **Signature:** `(dataset_name: str = "dwb2023/gdelt-rag-sources", split: str = "train", revision: str = None) -> List[Document]`
- **Purpose:** Load documents from HuggingFace and convert to LangChain Documents
- **Features:**
  - Handles nested metadata structures
  - Supports revision pinning via parameter or `HF_SOURCES_REV` env var
  - Converts HF dataset format to LangChain Document objects
- **Public API:** Yes - primary data loading function

##### `load_golden_testset_from_huggingface()`
- **File:** `/home/donbr/don-aie-cohort8/cert-challenge/src/utils.py:78`
- **Signature:** `(dataset_name: str = "dwb2023/gdelt-rag-golden-testset", split: str = "train", revision: str = None)`
- **Purpose:** Load golden testset from HuggingFace for evaluation
- **Features:**
  - Returns HuggingFace Dataset object (not converted to Documents)
  - Supports revision pinning via parameter or `HF_GOLDEN_REV` env var
  - Used for RAGAS evaluation
- **Public API:** Yes - evaluation data loading function

#### Retriever Functions (`src/retrievers.py`)

##### `create_retrievers()`
- **File:** `/home/donbr/don-aie-cohort8/cert-challenge/src/retrievers.py:20`
- **Signature:** `(documents: List[Document], vector_store: QdrantVectorStore, k: int = 5) -> Dict[str, object]`
- **Purpose:** Factory function to create all retriever instances
- **Returns:** Dictionary with keys:
  - `"naive"` - Dense vector search using embeddings
  - `"bm25"` - Sparse keyword matching
  - `"ensemble"` - Hybrid (50% dense + 50% sparse)
  - `"cohere_rerank"` - Contextual compression with reranking
- **Configuration:**
  - All retrievers return up to `k` documents (default: 5)
  - Ensemble uses 50/50 weighting
  - Cohere rerank retrieves 20 docs then reranks to top k
- **Public API:** Yes - primary factory for retriever strategies

#### Graph Functions (`src/graph.py`)

##### `build_graph()`
- **File:** `/home/donbr/don-aie-cohort8/cert-challenge/src/graph.py:21`
- **Signature:** `(retriever, llm=None, prompt_template: str = None) -> CompiledGraph`
- **Purpose:** Build compiled LangGraph pipeline for a single retriever
- **Graph Structure:**
  - START → retrieve → generate → END
  - Two-node sequential pipeline
- **Internal Nodes:**
  - `retrieve(state: State) -> dict` (line 67) - Retrieves documents using retriever
  - `generate(state: State) -> dict` (line 80) - Generates answer from context using LLM
- **Public API:** Yes - builds individual RAG graph

##### `build_all_graphs()`
- **File:** `/home/donbr/don-aie-cohort8/cert-challenge/src/graph.py:109`
- **Signature:** `(retrievers: Dict[str, object], llm=None) -> Dict[str, object]`
- **Purpose:** Convenience function to build graphs for all retrievers
- **Returns:** Dictionary mapping retriever names to compiled graphs
- **Public API:** Yes - builds all RAG graphs at once

---

## Internal Implementation

### Core Modules

#### State Management (`src/state.py`)

**Purpose:** Define shared state schemas for LangGraph workflows

**Components:**
- `State` TypedDict (line 7) - Defines question, context, and response fields
- Imported by `src/graph.py` for type annotations

#### Prompt Templates (`src/prompts.py`)

**Purpose:** Centralized prompt template definitions

**Components:**
- `BASELINE_PROMPT` (line 4) - RAG prompt template with placeholders for {question} and {context}
- Used by graph builders and evaluation scripts
- Ensures consistent prompting across all retrievers

### Utility Modules

#### Data Loading (`src/utils.py`)

**Purpose:** Abstract HuggingFace dataset loading with proper error handling

**Key Features:**
- Revision pinning support for reproducibility
- Metadata structure normalization
- Conversion between HF datasets and LangChain Documents

#### Environment Configuration (`src/config.py`)

**Purpose:** Centralized configuration with sensible defaults

**Configuration Variables:**
- `QDRANT_HOST` (line 20) - default: "localhost"
- `QDRANT_PORT` (line 21) - default: 6333
- `COLLECTION_NAME` (line 22) - default: "gdelt_comparative_eval"
- `OPENAI_MODEL` (line 23) - default: "gpt-4.1-mini"
- `OPENAI_EMBED_MODEL` (line 24) - default: "text-embedding-3-small"

### Helper Functions

#### Metadata Sanitization (`scripts/ingest.py`)

##### `ensure_jsonable()`
- **File:** `/home/donbr/don-aie-cohort8/cert-challenge/scripts/ingest.py:92`
- **Signature:** `(obj: Any) -> Any`
- **Purpose:** Make metadata JSON-serializable without losing information
- **Handles:** Nested dicts, lists, Path objects, UUID, datetime

##### `docs_to_jsonl()`
- **File:** `/home/donbr/don-aie-cohort8/cert-challenge/scripts/ingest.py:106`
- **Purpose:** Persist documents to JSONL format
- **Returns:** Document count

##### `docs_to_parquet()`
- **File:** `/home/donbr/don-aie-cohort8/cert-challenge/scripts/ingest.py:115`
- **Purpose:** Persist documents to Parquet format
- **Returns:** Document count

##### `docs_to_hfds()`
- **File:** `/home/donbr/don-aie-cohort8/cert-challenge/scripts/ingest.py:121`
- **Purpose:** Persist documents to HuggingFace dataset format
- **Returns:** Document count

#### RAGAS Schema Validation (`scripts/single_file.py`)

##### `validate_and_normalize_ragas_schema()`
- **File:** `/home/donbr/don-aie-cohort8/cert-challenge/scripts/single_file.py:66`
- **Signature:** `(df: pd.DataFrame, retriever_name: str = "unknown") -> pd.DataFrame`
- **Purpose:** Ensure DataFrame matches RAGAS 0.2.10 schema requirements
- **Features:**
  - Handles different column naming conventions
  - Validates required fields (user_input, response, retrieved_contexts, reference)
  - Prevents silent breakage across RAGAS versions

#### Manifest Generation (`scripts/generate_run_manifest.py`)

##### `generate_manifest()`
- **File:** `/home/donbr/don-aie-cohort8/cert-challenge/scripts/generate_run_manifest.py:21`
- **Signature:** `(output_path: Path, evaluation_results: Optional[Dict[str, Any]] = None, retrievers_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]`
- **Purpose:** Generate reproducibility manifest with exact configuration
- **Captures:**
  - Model versions and parameters
  - Retriever configurations
  - Evaluation settings
  - Dependencies
  - Results summary

---

## Entry Points

### Main Scripts

#### Primary Evaluation Script

##### `scripts/run_eval_harness.py`
- **File:** `/home/donbr/don-aie-cohort8/cert-challenge/scripts/run_eval_harness.py`
- **Purpose:** RAGAS evaluation harness using src/ modules
- **Usage:**
  ```bash
  make eval
  # or
  PYTHONPATH=. python scripts/run_eval_harness.py --recreate=false
  ```
- **Features:**
  - Loads data using `src.utils.load_documents_from_huggingface()`
  - Creates vector store using `src.config.create_vector_store()`
  - Creates retrievers using `src.retrievers.create_retrievers()`
  - Builds graphs using `src.graph.build_all_graphs()`
  - Runs 12 questions × 4 retrievers = 48 queries
  - Evaluates with RAGAS (faithfulness, answer_relevancy, context_precision, context_recall)
  - Saves results to `deliverables/evaluation_evidence/`
- **Time:** 20-30 minutes
- **Cost:** ~$5-6 in OpenAI API calls
- **Arguments:**
  - `--recreate` (default: "false") - Recreate Qdrant collection or reuse existing

#### Legacy Standalone Script

##### `scripts/single_file.py`
- **File:** `/home/donbr/don-aie-cohort8/cert-challenge/scripts/single_file.py`
- **Purpose:** Standalone evaluation script with inline code (pre-refactor version)
- **Note:** Functionally identical to `run_eval_harness.py` but duplicates code instead of using src/ modules
- **Status:** Maintained for comparison and as reference implementation
- **Features:**
  - Lines 260-286: Inline retriever function definitions
  - Lines 290-304: Inline LangGraph construction
  - Lines 333-355: Inline inference loop with immediate persistence
  - Lines 376-403: RAGAS evaluation with immediate save

### CLI Tools

#### LangGraph Validation Tool

##### `scripts/validate_langgraph.py`
- **File:** `/home/donbr/don-aie-cohort8/cert-challenge/scripts/validate_langgraph.py`
- **Purpose:** Validate src/ module implementation and demonstrate correct patterns
- **Usage:**
  ```bash
  make validate
  # or
  PYTHONPATH=. python scripts/validate_langgraph.py
  ```
- **Validation Stages:**
  1. **Environment Validation** (line 74) - Check API keys, Qdrant connectivity, imports
  2. **Module Import Validation** (line 134) - Test importing each src/ module
  3. **Correct Pattern Demonstration** (line 181) - Demonstrate factory pattern usage
  4. **Graph Compilation Validation** (line 228) - Validate LangGraph compilation
  5. **Functional Testing** (line 268) - Run test queries through each graph
  6. **Diagnostic Report** (line 311) - Generate final report with recommendations
- **Exit Codes:**
  - 0: All validations passed
  - 1: One or more validations failed
- **Output:** Color-coded terminal output with pass/fail indicators

#### Data Ingestion Tool

##### `scripts/ingest.py`
- **File:** `/home/donbr/don-aie-cohort8/cert-challenge/scripts/ingest.py`
- **Purpose:** Standardized RAGAS golden testset pipeline
- **Features:**
  - Extract PDFs to LangChain Documents (line 157)
  - Sanitize metadata for Arrow/JSON compatibility (line 92)
  - Persist sources to JSONL, Parquet, HF-dataset (lines 166-174)
  - Generate golden testset using RAGAS (line 231)
  - Persist testset to multiple formats (lines 239-258)
  - Generate manifest with checksums & schema (line 271)
- **RAGAS Version Support:**
  - Lines 186-214: Automatic detection of RAGAS 0.3.x vs 0.2.x API
  - Lines 196-230: Retry logic with exponential backoff for API calls

#### HuggingFace Upload Tool

##### `scripts/upload_to_hf.py`
- **File:** `/home/donbr/don-aie-cohort8/cert-challenge/scripts/upload_to_hf.py`
- **Purpose:** Upload datasets to Hugging Face Hub
- **Usage:**
  ```bash
  HF_TOKEN=xxx python scripts/upload_to_hf.py
  ```
- **Features:**
  - Loads datasets from local storage (line 235)
  - Creates dataset cards with metadata (lines 34, 113)
  - Uploads to HF Hub (lines 243, 262)
  - Updates manifest with repo IDs (line 200)
- **Datasets:**
  - `dwb2023/gdelt-rag-sources` - 38 source documents
  - `dwb2023/gdelt-rag-golden-testset` - 12 QA pairs

#### Manifest Enrichment Tool

##### `scripts/enrich_manifest.py`
- **File:** `/home/donbr/don-aie-cohort8/cert-challenge/scripts/enrich_manifest.py`
- **Purpose:** Enrich manifest.json with computed metadata
- **Features:**
  - Add file hashes (line 7)
  - Compute row counts (lines 19, 37)
  - Extract schema from Parquet (line 51)
  - Compute character statistics (line 61)
  - Add environment info (line 96)
  - Add lineage scaffold (line 183)
  - Add compliance scaffold (line 194)
  - Relativize paths (line 224)

#### Manifest Generator Tool

##### `scripts/generate_run_manifest.py`
- **File:** `/home/donbr/don-aie-cohort8/cert-challenge/scripts/generate_run_manifest.py`
- **Purpose:** Generate RUN_MANIFEST.json for reproducibility
- **Usage:** Called automatically by evaluation scripts
- **Captures:**
  - Model versions (lines 40-66)
  - Retriever configurations (lines 68-104)
  - Evaluation settings (lines 106-122)
  - Vector store config (lines 124-131)
  - Results summary (lines 146-163)

#### Simple Entry Point

##### `main.py`
- **File:** `/home/donbr/don-aie-cohort8/cert-challenge/main.py:1`
- **Purpose:** Placeholder entry point for package testing
- **Function:** `main()` (line 1) - Prints "Hello from cert-challenge!"
- **Usage:** Minimal test to verify package is importable

### Build System

##### `Makefile`
- **File:** `/home/donbr/don-aie-cohort8/cert-challenge/Makefile`
- **Purpose:** Task automation and developer commands
- **Key Targets:**
  - `make validate` (line 28) - Run LangGraph validation script
  - `make eval` (line 36) - Run full RAGAS evaluation (~20-30 min)
  - `make test` (line 57) - Quick validation test
  - `make docker-up` (line 60) - Start infrastructure services
  - `make docker-down` (line 71) - Stop infrastructure services
  - `make qdrant-up` (line 76) - Start only Qdrant
  - `make env` (line 82) - Show environment configuration
  - `make clean` (line 97) - Clean Python cache
  - `make notebook` (line 108) - Start Jupyter

---

## Dependencies and Relationships

### Core Dependency Graph

```
main.py
  └─ (minimal, standalone)

scripts/run_eval_harness.py (PRIMARY ENTRY POINT)
  ├─ src.utils.load_documents_from_huggingface()
  ├─ src.utils.load_golden_testset_from_huggingface()
  ├─ src.config.create_vector_store()
  ├─ src.config.get_llm()
  ├─ src.retrievers.create_retrievers()
  └─ src.graph.build_all_graphs()

scripts/validate_langgraph.py (VALIDATION TOOL)
  ├─ src.utils.load_documents_from_huggingface()
  ├─ src.config.create_vector_store()
  ├─ src.retrievers.create_retrievers()
  └─ src.graph.build_all_graphs()

src/graph.py
  ├─ src.state.State
  ├─ src.prompts.BASELINE_PROMPT
  └─ src.config.get_llm()

src/retrievers.py
  └─ (depends on vector_store passed as argument)

src/config.py
  └─ (depends only on environment variables and langchain libraries)

src/utils.py
  └─ (depends only on datasets library and environment variables)
```

### Module Responsibilities

| Module | Primary Responsibility | Dependencies | Dependents |
|--------|----------------------|--------------|------------|
| `src/config.py` | Configuration & resource factories | Environment vars, langchain | All other src/ modules, scripts |
| `src/utils.py` | Data loading utilities | HuggingFace datasets, environment vars | Evaluation scripts |
| `src/state.py` | State schema definitions | typing_extensions | src/graph.py |
| `src/prompts.py` | Prompt templates | None | src/graph.py, evaluation scripts |
| `src/retrievers.py` | Retriever factory functions | langchain, cohere | Evaluation scripts |
| `src/graph.py` | LangGraph workflow builders | src/state, src/prompts, src/config | Evaluation scripts |

### External Dependencies

**Core Framework:**
- `langchain` - Document processing, RAG chains
- `langchain_openai` - OpenAI integrations (ChatOpenAI, OpenAIEmbeddings)
- `langchain_qdrant` - Qdrant vector store integration
- `langchain_cohere` - Cohere reranking integration
- `langchain_community` - BM25 retriever
- `langgraph` - Stateful workflow orchestration

**Vector Database:**
- `qdrant_client` - Qdrant Python client

**Evaluation:**
- `ragas` - RAG evaluation metrics

**Data Management:**
- `datasets` - HuggingFace datasets library
- `pandas` - Data manipulation
- `pyarrow` - Parquet support

**Utilities:**
- `typing_extensions` - TypedDict support
- `python-dotenv` - Environment variable management

### Inter-Module Communication Patterns

#### Factory Pattern

The codebase uses factory functions to avoid circular dependencies and enable lazy initialization:

1. **Configuration Factories** - `get_llm()`, `get_embeddings()`, `get_qdrant()` with `@lru_cache`
2. **Vector Store Factory** - `create_vector_store(documents, ...)` creates and populates Qdrant
3. **Retriever Factory** - `create_retrievers(documents, vector_store, ...)` creates all retriever instances
4. **Graph Factory** - `build_graph(retriever, ...)` and `build_all_graphs(retrievers, ...)` create compiled workflows

**Rationale:** Module-level instances would fail on import because they depend on runtime data (documents, vector stores). Factory functions defer instantiation until data is available.

#### State Management Pattern

LangGraph uses a TypedDict-based state management pattern:

1. **State Definition** - `State` TypedDict in `src/state.py` defines the schema
2. **Node Functions** - Return partial state updates as dictionaries
3. **Graph Execution** - LangGraph automatically merges updates into full state
4. **Type Safety** - TypedDict provides IDE autocomplete and type checking

**Example Flow:**
```
Initial State: {"question": "What is GDELT?"}
  ↓
retrieve node returns: {"context": [doc1, doc2, ...]}
  ↓ (merged)
Intermediate State: {"question": "What is GDELT?", "context": [doc1, doc2, ...]}
  ↓
generate node returns: {"response": "GDELT is..."}
  ↓ (merged)
Final State: {"question": "What is GDELT?", "context": [doc1, doc2, ...], "response": "GDELT is..."}
```

#### Data Flow Pattern

```
HuggingFace Datasets (dwb2023/gdelt-rag-sources)
  ↓ load_documents_from_huggingface()
LangChain Documents (List[Document])
  ↓ create_vector_store(documents)
Qdrant Vector Store (QdrantVectorStore)
  ↓ create_retrievers(documents, vector_store)
Retrievers Dict ({"naive": ..., "bm25": ..., ...})
  ↓ build_all_graphs(retrievers)
Compiled Graphs Dict ({"naive": graph, "bm25": graph, ...})
  ↓ graph.invoke({"question": "..."})
Result Dict ({"question": ..., "context": ..., "response": ...})
```

### Evaluation Architecture

The evaluation system follows a multi-stage pipeline:

1. **Data Loading** - Load source documents and golden testset from HuggingFace
2. **Vector Store Setup** - Create/populate Qdrant collection with embeddings
3. **Retriever Creation** - Instantiate all four retriever strategies
4. **Graph Compilation** - Build LangGraph workflows for each retriever
5. **Inference** - Run all test questions through all retrievers (48 total queries)
6. **RAGAS Evaluation** - Compute metrics (faithfulness, answer_relevancy, context_precision, context_recall)
7. **Results Persistence** - Save raw datasets, evaluation datasets, detailed results, comparative summary
8. **Manifest Generation** - Capture configuration for reproducibility

**Key Design Decisions:**

- **Immediate Persistence**: Each retriever's results are saved immediately after inference (line 178 in `run_eval_harness.py`), preventing data loss if RAGAS evaluation fails
- **Deep Copy Strategy**: Each retriever gets a deep copy of the golden dataset (line 161 in `run_eval_harness.py`) to avoid mutation conflicts
- **Schema Validation**: RAGAS schema is validated before evaluation (line 66 in `single_file.py`) to catch version incompatibilities early
- **Reusable Modules**: `run_eval_harness.py` uses src/ modules instead of inline code, enabling code reuse across evaluation runs

---

## Component Summary Statistics

**Total Python Files:** 13 (excluding ra_* framework)

**Module Breakdown:**
- Public API modules (src/): 6 files
- Entry point scripts (scripts/): 6 files
- Root level scripts: 1 file (main.py)

**Lines of Code (approximate):**
- src/config.py: 128 lines
- src/utils.py: 114 lines
- src/retrievers.py: 90 lines
- src/graph.py: 142 lines
- src/state.py: 10 lines
- src/prompts.py: 12 lines
- scripts/run_eval_harness.py: 323 lines
- scripts/single_file.py: 511 lines
- scripts/validate_langgraph.py: 487 lines

**Total Documented Functions:** 22
- Public API functions: 10
- Internal helper functions: 7
- Entry point functions: 5

**Total Documented Classes:** 1 (State TypedDict)

**Retriever Strategies:** 4
1. Naive (dense vector search)
2. BM25 (sparse keyword matching)
3. Ensemble (hybrid: 50% dense + 50% sparse)
4. Cohere Rerank (contextual compression)

**LangGraph Workflows:** 4 (one per retriever)

**Evaluation Metrics:** 4 (faithfulness, answer_relevancy, context_precision, context_recall)

**Test Dataset Size:** 12 QA pairs

**Source Dataset Size:** 38 documents

---

## Notes

### Architecture Patterns

1. **Factory Pattern** - Used throughout for lazy initialization of resources
2. **Singleton Pattern** - Implemented via `@lru_cache` for expensive resources (LLM, embeddings, Qdrant client)
3. **Strategy Pattern** - Multiple retriever implementations with common interface
4. **State Machine Pattern** - LangGraph workflows with explicit state transitions

### Code Quality Observations

**Strengths:**
- Clear separation of concerns (config, utils, retrievers, graph)
- Comprehensive documentation with docstrings and examples
- Type hints throughout for better IDE support
- Environment-driven configuration with sensible defaults
- Reproducibility focus (revision pinning, manifest generation)
- Immediate persistence strategy prevents data loss

**Areas for Potential Improvement:**
- No unit tests present in analyzed codebase
- Some code duplication between `run_eval_harness.py` and `single_file.py`
- Hard-coded model names in some scripts (should use config module)
- Limited error handling in some helper functions

### Reproducibility Features

The codebase implements several reproducibility mechanisms:

1. **Dataset Revision Pinning** - Support for pinning HuggingFace dataset revisions via environment variables or function parameters
2. **Run Manifests** - Automatic generation of manifests capturing exact configuration, versions, and results
3. **Deterministic LLM** - Temperature=0 for all LLM calls
4. **Checksums** - SHA256 hashes for all persisted artifacts
5. **Version Capture** - Python, package, and model versions recorded in manifests

---

**End of Component Inventory**

*This document was generated by analyzing the GDELT RAG System codebase on 2025-10-18. For the most up-to-date information, refer to the source code and inline documentation.*
