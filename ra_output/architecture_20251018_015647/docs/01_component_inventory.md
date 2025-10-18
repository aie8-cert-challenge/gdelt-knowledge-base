# Component Inventory

## Overview

This codebase implements a production-grade RAG (Retrieval-Augmented Generation) system for querying GDELT (Global Database of Events, Language, and Tone) knowledge graphs. The system is built using LangChain, LangGraph, and RAGAS evaluation frameworks.

**Architecture Pattern**: Factory-based initialization with modular components
- Core library code in `src/` using factory pattern (public API)
- Executable scripts in `scripts/` for various workflows
- Entry point at `main.py`

**Total Python modules analyzed**: 15 files (excluding ra_* framework directories)

---

## Public API

### Modules

#### src/__init__.py (Lines 1-42)
**Purpose**: Package initialization and API surface definition
- **API Type**: Public
- **Description**: Defines the public interface for the GDELT RAG System package with comprehensive docstring examples
- **Exports**: `__all__ = ["config", "graph", "prompts", "retrievers", "state", "utils"]`
- **Version**: 0.1.0
- **Usage Pattern**: Factory-based initialization (see docstring lines 17-37)

#### src/utils.py (Lines 1-114)
**Purpose**: Data loading utilities for HuggingFace datasets
- **API Type**: Public
- **Key Functions**:
  - `load_documents_from_huggingface()` - Load source documents (lines 15-75)
  - `load_golden_testset_from_huggingface()` - Load evaluation testset (lines 78-113)
- **Dependencies**: datasets, langchain_core.documents
- **Features**: Revision pinning for reproducibility, metadata handling

#### src/config.py (Lines 1-128)
**Purpose**: Configuration and resource management with caching
- **API Type**: Public
- **Key Functions**:
  - `get_llm()` - Cached LLM instance (lines 28-35)
  - `get_embeddings()` - Cached embeddings instance (lines 38-46)
  - `get_qdrant()` - Cached Qdrant client (lines 49-57)
  - `get_collection_name()` - Get collection name (lines 60-67)
  - `create_vector_store()` - Vector store factory (lines 70-127)
- **Pattern**: Singleton pattern via `@lru_cache(maxsize=1)`
- **Dependencies**: langchain_openai, langchain_qdrant, qdrant_client

#### src/retrievers.py (Lines 1-90)
**Purpose**: Retriever factory functions
- **API Type**: Public
- **Key Function**: `create_retrievers()` - Factory for all 4 retriever types (lines 20-89)
- **Retriever Types**:
  1. Naive - Dense vector search (line 63)
  2. BM25 - Sparse keyword matching (line 66)
  3. Ensemble - Hybrid dense+sparse (lines 69-72)
  4. Cohere Rerank - Contextual compression (lines 76-82)
- **Return Type**: `Dict[str, object]` mapping retriever names to instances
- **Dependencies**: langchain_community, langchain_cohere, langchain_qdrant

#### src/graph.py (Lines 1-142)
**Purpose**: LangGraph workflow factory
- **API Type**: Public
- **Key Functions**:
  - `build_graph()` - Build single retriever workflow (lines 21-106)
  - `build_all_graphs()` - Build all retriever workflows (lines 109-141)
- **Graph Structure**: START → retrieve → generate → END
- **Node Functions** (internal to build_graph):
  - `retrieve()` - Document retrieval node (lines 67-78)
  - `generate()` - Answer generation node (lines 80-96)
- **Dependencies**: langgraph, langchain_core.documents, langchain.prompts

### Classes

#### src/state.py - State (Lines 7-10)
**Purpose**: LangGraph state schema
- **API Type**: Public
- **Base Class**: `TypedDict`
- **Fields**:
  - `question: str` - User input question
  - `context: List[Document]` - Retrieved documents
  - `response: str` - Generated answer
- **Usage**: Type annotation for LangGraph state transitions

### Functions

All public functions are listed in the Modules section above. Key entry points:

1. **Data Loading**: `src.utils.load_documents_from_huggingface()` (line 15)
2. **Vector Store Setup**: `src.config.create_vector_store()` (line 70)
3. **Retriever Creation**: `src.retrievers.create_retrievers()` (line 20)
4. **Graph Building**: `src.graph.build_all_graphs()` (line 109)

---

## Internal Implementation

### Core Modules

#### src/prompts.py (Lines 1-12)
**Purpose**: Prompt templates for RAG system
- **API Type**: Internal (exported but simple constant)
- **Constants**:
  - `BASELINE_PROMPT` (lines 4-12) - RAG prompt template with question/context placeholders
- **Usage**: Shared across all retriever workflows
- **Pattern**: Simple string template, no complex logic

#### src/state.py (Lines 1-10)
**Purpose**: State schema definitions
- **API Type**: Public (but typically internal usage)
- **File Path**: /home/donbr/don-aie-cohort8/cert-challenge/src/state.py
- **See Classes section above**

### Utility Modules

#### scripts/enrich_manifest.py (Lines 1-244)
**Purpose**: Enrich manifest.json with metadata and checksums
- **API Type**: Script (not importable module)
- **Key Functions**:
  - `sha256()` - File hashing (lines 7-12)
  - `file_bytes()` - File size calculation (lines 15-16)
  - `count_jsonl_rows()` - Count JSONL records (lines 19-21)
  - `hfds_rows()` - Count HuggingFace dataset rows (lines 24-34)
  - `parquet_rows()` - Count Parquet rows (lines 37-48)
  - `pandas_schema_from_parquet()` - Extract schema (lines 51-58)
  - `char_stats_jsonl()` - Character statistics (lines 61-78)
  - `main()` - Main enrichment logic (lines 81-236)
- **Dependencies**: json, hashlib, platform, pathlib, datasets

#### scripts/generate_run_manifest.py (Lines 1-188)
**Purpose**: Generate reproducibility manifest for RAGAS evaluations
- **API Type**: Script with importable function
- **Key Function**: `generate_manifest()` (lines 21-170)
- **Captures**:
  - Model versions and parameters
  - Retriever configurations
  - Evaluation settings
  - Dependencies (RAGAS, Python versions)
- **Output**: RUN_MANIFEST.json with complete evaluation configuration

#### scripts/upload_to_hf.py (Lines 1-293)
**Purpose**: Upload datasets to HuggingFace Hub
- **API Type**: Script
- **Key Functions**:
  - `create_sources_card()` - Generate dataset card for sources (lines 34-110)
  - `create_golden_testset_card()` - Generate dataset card for testset (lines 113-191)
  - `load_manifest()` - Load manifest.json (lines 194-197)
  - `update_manifest()` - Update with HF repo IDs (lines 200-216)
  - `main()` - Main upload workflow (lines 219-292)
- **Datasets**:
  - dwb2023/gdelt-rag-sources (38 documents)
  - dwb2023/gdelt-rag-golden-testset (12 QA pairs)

### Configuration

#### Environment Variables (src/config.py lines 20-24)
- `QDRANT_HOST` - Default: "localhost"
- `QDRANT_PORT` - Default: 6333
- `QDRANT_COLLECTION` - Default: "gdelt_comparative_eval"
- `OPENAI_MODEL` - Default: "gpt-4.1-mini"
- `OPENAI_EMBED_MODEL` - Default: "text-embedding-3-small"

#### Configuration Pattern
- Uses `@lru_cache` for singleton resource management
- Environment variable defaults with `os.getenv()`
- Cached getter functions for LLM, embeddings, Qdrant client
- Factory functions for stateful resources (vector store, retrievers, graphs)

---

## Entry Points

### Main Scripts

#### main.py (Lines 1-7)
**Purpose**: Primary entry point (minimal implementation)
- **Function**: `main()` (lines 1-2) - Prints "Hello from cert-challenge!"
- **Entry Point**: `if __name__ == "__main__"` (lines 5-6)
- **Status**: Placeholder implementation
- **Path**: /home/donbr/don-aie-cohort8/cert-challenge/main.py

#### scripts/run_eval_harness.py (Lines 1-323)
**Purpose**: RAGAS evaluation harness using src/ modules
- **Entry Point**: `if __name__ == "__main__"` (line 320)
- **Command Line Args**: `--recreate` flag for Qdrant collection (lines 52-65)
- **Workflow**:
  1. Load data from HuggingFace (lines 112-120)
  2. Build RAG stack (vector store, retrievers, graphs) (lines 127-144)
  3. Run inference across all retrievers (lines 149-184)
  4. RAGAS evaluation (lines 186-232)
  5. Comparative analysis (lines 234-285)
- **Output**: CSV files in deliverables/evaluation_evidence/
- **Time**: 20-30 minutes, Cost: ~$5-6 in API calls

#### scripts/single_file.py (Lines 1-511)
**Purpose**: Comprehensive standalone RAG evaluation (legacy, inline implementation)
- **Entry Point**: Script execution (no __main__ guard)
- **Implementation**: Inline code (not using src/ modules)
- **Workflow**: Same as run_eval_harness.py but with all code inline
- **Key Functions**:
  - `validate_and_normalize_ragas_schema()` (lines 66-125)
  - Node functions: `retrieve_baseline()`, `retrieve_bm25()`, etc. (lines 260-278)
  - `generate()` (lines 281-286)
- **Output**: Same as run_eval_harness.py
- **Status**: Reference implementation, being replaced by run_eval_harness.py

#### scripts/validate_langgraph.py (Lines 1-487)
**Purpose**: Validation and diagnostic script for LangGraph configuration
- **Entry Point**: `if __name__ == "__main__"` (line 484)
- **Main Function**: `main()` (lines 447-481)
- **Validation Stages**:
  1. `check_environment()` - API keys, Qdrant, imports (lines 74-127)
  2. `test_module_imports()` - Test src/ module imports (lines 134-174)
  3. `demonstrate_correct_pattern()` - Factory pattern demo (lines 181-221)
  4. `validate_graph_compilation()` - Graph building (lines 228-261)
  5. `run_functional_tests()` - Functional testing (lines 268-304)
  6. `generate_diagnostic_report()` - Final report (lines 311-440)
- **Exit Codes**: 0 = pass, 1 = fail
- **Purpose**: Development/debugging tool

#### scripts/ingest.py (Lines 1-336)
**Purpose**: Standardized RAGAS Golden Testset Pipeline
- **Entry Point**: Notebook-style script (no __main__ guard)
- **Workflow**:
  1. Extract PDFs to LangChain Documents (lines 154-162)
  2. Persist sources to JSONL/Parquet/HF-dataset (lines 166-174)
  3. Generate golden testset with RAGAS (lines 178-232)
  4. Persist testset to multiple formats (lines 236-258)
  5. Generate manifest with checksums (lines 262-329)
- **Key Functions**:
  - `ensure_jsonable()` - Metadata sanitization (lines 92-104)
  - `docs_to_jsonl()`, `docs_to_parquet()`, `docs_to_hfds()` (lines 106-126)
  - `build_testset()` - RAGAS 0.2.x/0.3.x compatibility (lines 202-229)
- **Output**: data/interim/ with sources and golden testset
- **Pattern**: Notebook-compatible with cell markers (%%)

### Module Entry Points

No `__main__` blocks found in src/ modules - all use factory pattern for initialization.

The correct usage pattern (from src/__init__.py lines 17-37):
```python
from src.utils import load_documents_from_huggingface
from src.config import create_vector_store
from src.retrievers import create_retrievers
from src.graph import build_all_graphs

documents = load_documents_from_huggingface()
vector_store = create_vector_store(documents, recreate_collection=True)
retrievers = create_retrievers(documents, vector_store)
graphs = build_all_graphs(retrievers)
result = graphs['naive'].invoke({"question": "What is GDELT?"})
```

---

## Module Dependencies

### Dependency Graph

```
main.py (standalone, no src/ imports)

scripts/run_eval_harness.py
├── src.utils (load data)
├── src.config (vector store, LLM)
├── src.retrievers (create retrievers)
└── src.graph (build graphs)

scripts/single_file.py (no src/ imports - inline implementation)

scripts/validate_langgraph.py
├── src.utils
├── src.config
├── src.retrievers
└── src.graph

scripts/ingest.py (no src/ imports - standalone pipeline)

scripts/generate_run_manifest.py (importable by other scripts)

scripts/upload_to_hf.py (standalone)

scripts/enrich_manifest.py (standalone)
```

### Internal Module Dependencies (src/)

```
src/__init__.py (exports all modules)

src/utils
└── dependencies: datasets, langchain_core.documents

src/config
└── dependencies: langchain_openai, langchain_qdrant, qdrant_client

src/state
└── dependencies: typing_extensions, langchain_core.documents

src/prompts
└── dependencies: none (constants only)

src/retrievers
├── imports: src.config (indirectly via vector_store parameter)
└── dependencies: langchain_community, langchain_cohere, langchain_qdrant

src/graph
├── imports: src.state, src.prompts, src.config
└── dependencies: langgraph, langchain_core.documents, langchain.prompts
```

### External Dependencies

**Core LangChain Stack**:
- langchain - Core framework
- langchain_openai - OpenAI LLM and embeddings
- langchain_community - BM25 retriever
- langchain_cohere - Cohere reranking
- langchain_qdrant - Qdrant vector store integration
- langgraph - Graph-based workflows

**Vector Database**:
- qdrant_client - Qdrant vector database client

**Data & Evaluation**:
- datasets - HuggingFace datasets library
- ragas - RAG evaluation framework
- pandas - Data manipulation
- pyarrow - Parquet file support

**Utilities**:
- typing_extensions - Extended type hints (TypedDict)
- pathlib - Path manipulation
- hashlib - File hashing
- uuid - Unique identifiers

---

## Notes

### Architecture Observations

1. **Factory Pattern**: The codebase correctly uses factory functions for retrievers and graphs, avoiding module-level initialization issues (see scripts/validate_langgraph.py for pattern validation).

2. **Dual Implementation**: Two evaluation scripts exist:
   - `scripts/single_file.py` - Inline implementation (legacy, self-contained)
   - `scripts/run_eval_harness.py` - Uses src/ modules (current, modular)
   - Both produce identical results

3. **Caching Strategy**: Uses `@lru_cache(maxsize=1)` for singleton resources (LLM, embeddings, Qdrant client) to prevent resource duplication.

4. **Reproducibility**: Comprehensive manifest generation with version tracking, checksums, and configuration capture (scripts/generate_run_manifest.py).

5. **HuggingFace Integration**: Datasets published to HuggingFace Hub with full metadata cards (scripts/upload_to_hf.py).

6. **Evaluation Framework**: RAGAS-based evaluation with 4 retriever types and 4 metrics (faithfulness, answer_relevancy, context_precision, context_recall).

### Code Quality

- **Documentation**: Comprehensive docstrings with examples in all src/ modules
- **Type Hints**: TypedDict for state schema, type annotations on public functions
- **Error Handling**: Environment variable defaults, connection validation
- **Testing**: Dedicated validation script (scripts/validate_langgraph.py) with 6-stage validation

### Data Flow

```
Raw PDFs (data/raw/)
  ↓ (scripts/ingest.py)
Interim Data (data/interim/)
  - sources.docs.jsonl/parquet/hfds
  - golden_testset.jsonl/parquet/hfds
  - manifest.json
  ↓ (scripts/upload_to_hf.py)
HuggingFace Hub
  - dwb2023/gdelt-rag-sources
  - dwb2023/gdelt-rag-golden-testset
  ↓ (scripts/run_eval_harness.py)
Evaluation Results (deliverables/evaluation_evidence/)
  - comparative_ragas_results.csv
  - [retriever]_evaluation_dataset.csv
  - [retriever]_detailed_results.csv
  - RUN_MANIFEST.json
```

### Key Design Decisions

1. **Why Factory Functions?**: Retrievers and graphs depend on runtime data (documents, vector store), so they cannot be instantiated at module import time.

2. **Why TypedDict?**: LangGraph requires TypedDict for state schema to enable type checking and automatic state merging.

3. **Why RAGAS 0.2.x API?**: Explicitly following session08 patterns from AI Engineering Bootcamp with compatibility layer for 0.3.x (scripts/ingest.py lines 186-229).

4. **Why 4 Retrievers?**: Comparative evaluation of different retrieval strategies:
   - Naive (baseline dense)
   - BM25 (sparse keyword)
   - Ensemble (hybrid)
   - Cohere Rerank (advanced contextual)

5. **Why Separate Evaluation Scripts?**: Migration from inline (single_file.py) to modular (run_eval_harness.py) while maintaining backward compatibility for validation.

---

**Document Generated**: 2025-10-18
**Total Components Documented**: 15 Python files, 25+ functions, 1 class
**Lines of Code Analyzed**: ~3,500 lines (excluding ra_* framework)
