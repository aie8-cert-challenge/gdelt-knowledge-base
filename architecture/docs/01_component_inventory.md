# Component Inventory

## Overview

This codebase implements a comprehensive RAG (Retrieval-Augmented Generation) evaluation system using RAGAS metrics. The project compares multiple retrieval strategies (naive dense vector search, BM25 sparse keyword matching, Cohere reranking, and ensemble hybrid search) on a GDELT knowledge graph dataset.

**Project Structure:**
- `src/` - Core RAG system implementation (configuration, retrievers, graphs, state management)
- `scripts/` - Pipeline scripts for data ingestion, dataset generation, and upload
- `main.py` - Simple entry point placeholder

**Key Technologies:**
- LangChain for RAG pipeline orchestration
- LangGraph for stateful workflow graphs
- RAGAS for evaluation metrics
- Qdrant for vector storage
- OpenAI for embeddings and LLM inference
- Cohere for reranking

---

## Public API

### Modules

#### `src/config.py`
Configuration constants and shared model instances for the RAG system.

**Purpose:** Centralized configuration for Qdrant connection and model initialization.

**Exports:**
- `QDRANT_HOST`, `QDRANT_PORT`, `COLLECTION_NAME` - Database configuration
- `llm` - Shared ChatOpenAI instance
- `embeddings` - Shared OpenAIEmbeddings instance

#### `src/graph.py`
LangGraph-based RAG pipeline definitions for each retrieval strategy.

**Purpose:** Defines executable graphs combining retrieval and generation for evaluation.

**Exports:**
- `baseline_graph` - Naive dense vector search graph
- `bm25_graph` - BM25 sparse keyword graph
- `ensemble_graph` - Hybrid search graph
- `rerank_graph` - Cohere reranking graph
- `retrievers_config` - Dictionary mapping retriever names to graphs

#### `src/retrievers.py`
Retriever implementations for different search strategies.

**Purpose:** Initialize and configure various retrieval approaches.

**Exports:**
- `baseline_retriever` - Dense vector search (k=5)
- `bm25_retriever` - BM25 sparse keyword matching
- `compression_retriever` - Cohere rerank with contextual compression
- `ensemble_retriever` - Hybrid search (50% dense + 50% sparse)

#### `src/state.py`
TypedDict state definition for LangGraph workflows.

**Purpose:** Type-safe state management for RAG pipeline.

**Exports:**
- `State` class - TypedDict with `question`, `context`, `response` fields

#### `src/prompts.py`
Prompt templates for RAG question-answering.

**Purpose:** Centralized prompt management.

**Exports:**
- `BASELINE_PROMPT` - Template for context-grounded QA

### Classes

#### `State` (src/state.py:7)
```python
class State(TypedDict):
    question: str
    context: List[Document]
    response: str
```

**Purpose:** Type-safe state container for LangGraph RAG workflows.

**Key Fields:**
- `question` - User input query
- `context` - Retrieved documents
- `response` - Generated answer

### Functions

#### Retrieval Functions (src/graph.py)

##### `retrieve_baseline(state)` (src/graph.py:20)
```python
def retrieve_baseline(state):
    """Naive dense vector search"""
    retrieved_docs = baseline_retriever.invoke(state["question"])
    return {"context": retrieved_docs}
```

**Purpose:** Execute baseline dense vector retrieval strategy.

##### `retrieve_bm25(state)` (src/graph.py:25)
```python
def retrieve_bm25(state):
    """BM25 sparse keyword matching"""
    retrieved_docs = bm25_retriever.invoke(state["question"])
    return {"context": retrieved_docs}
```

**Purpose:** Execute BM25 lexical retrieval strategy.

##### `retrieve_reranked(state)` (src/graph.py:30)
```python
def retrieve_reranked(state):
    """Cohere contextual compression with reranking"""
    retrieved_docs = compression_retriever.invoke(state["question"])
    return {"context": retrieved_docs}
```

**Purpose:** Execute Cohere reranking retrieval strategy (retrieves 20, reranks to top 5).

##### `retrieve_ensemble(state)` (src/graph.py:35)
```python
def retrieve_ensemble(state):
    """Ensemble hybrid search (dense + sparse)"""
    retrieved_docs = ensemble_retriever.invoke(state["question"])
    return {"context": retrieved_docs}
```

**Purpose:** Execute ensemble hybrid retrieval combining dense and sparse approaches.

#### Generation Function (src/graph.py)

##### `generate(state)` (src/graph.py:41)
```python
def generate(state):
    """Generate answer from context"""
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = rag_prompt.format_messages(question=state["question"], context=docs_content)
    response = llm.invoke(messages)
    return {"response": response.content}
```

**Purpose:** Generate answers from retrieved context using LLM.

---

## Internal Implementation

### Modules

#### `src/utils.py`
Utility functions module (currently empty).

**Purpose:** Reserved for shared utility functions.

**Status:** Empty file (2 lines, only comment/whitespace)

#### `src/__init__.py`
Package initialization file.

**Status:** Empty/minimal file

### Module Configuration Details

#### `src/config.py` Implementation Details
- **Line 6-8:** Qdrant connection constants
- **Line 10:** LLM initialization with `gpt-4.1-mini` at temperature 0 for determinism
- **Line 11:** Embeddings with `text-embedding-3-small` (1536 dimensions)

#### `src/retrievers.py` Implementation Details
- **Line 17-20:** Configuration constants (duplicates config.py - potential refactor opportunity)
- **Line 23:** Qdrant client initialization
- **Line 25-27:** Collection existence check
- **Line 30-34:** Vector store creation
- **Line 37:** Baseline retriever (k=5)
- **Line 40:** BM25 retriever (k=5) - Note: `documents` variable appears undefined in this file
- **Line 43-58:** Cohere reranker configuration (retrieves k=20, reranks to top 5)

#### `src/graph.py` Implementation Details
- **Line 11-13:** Import retrieval strategies and prompts
- **Line 16-17:** Model and prompt initialization
- **Line 49-71:** Graph compilation and configuration
  - Each retrieval strategy gets its own compiled StateGraph
  - All use the same `generate` function
  - Graphs are stored in `retrievers_config` dictionary

---

## Entry Points

### Main Application Entry Point

#### `main.py`
```python
def main():
    print("Hello from cert-challenge!")

if __name__ == "__main__":
    main()
```

**Location:** `/home/donbr/don-aie-cohort8/cert-challenge/main.py:1`

**Purpose:** Placeholder entry point. Currently prints a simple message.

**Status:** Not actively used - real entry points are in `scripts/` directory.

### Script Entry Points

#### `scripts/single_file.py`
**Purpose:** Comprehensive RAG evaluation with RAGAS metrics.

**Entry Point:** Line 508 (`if __name__ == "__main__"`)

**Key Responsibilities:**
1. Load golden testset from HuggingFace (`dwb2023/gdelt-rag-golden-testset`)
2. Load source documents from HuggingFace (`dwb2023/gdelt-rag-sources`)
3. Initialize Qdrant vector store with documents
4. Create 4 retrieval strategies (naive, BM25, ensemble, Cohere rerank)
5. Generate RAG responses for all test questions × all retrievers
6. Evaluate using RAGAS metrics (faithfulness, answer_relevancy, context_precision, context_recall)
7. Generate comparative summary tables and save results
8. Create RUN_MANIFEST.json for reproducibility

**Key Classes/Functions:**
- `State` (line 254) - TypedDict for LangGraph state
- `validate_and_normalize_ragas_schema()` (line 66) - Schema validation for RAGAS 0.2.10
- `retrieve_baseline()`, `retrieve_bm25()`, `retrieve_reranked()`, `retrieve_ensemble()` (lines 260-278)
- `generate()` (line 281) - Shared answer generation function

**Configuration:**
- Qdrant: `localhost:6333`, collection `gdelt_comparative_eval`
- Models: `gpt-4.1-mini` (temperature=0), `text-embedding-3-small`
- Output: `deliverables/evaluation_evidence/`

#### `scripts/ingest.py`
**Purpose:** Standardized RAGAS golden testset pipeline - PDF extraction to dataset creation.

**Entry Point:** Jupyter-style notebook (can be run as script or in notebook environment)

**Key Responsibilities:**
1. Extract PDFs from `data/raw/` using PyMuPDFLoader
2. Sanitize metadata for Arrow/JSON compatibility
3. Generate RAGAS golden testset using TestsetGenerator
4. Persist sources and golden testset in 3 formats:
   - JSONL (`sources.docs.jsonl`, `golden_testset.jsonl`)
   - Parquet (`sources.docs.parquet`, `golden_testset.parquet`)
   - HuggingFace dataset on disk (`sources.hfds`, `golden_testset.hfds`)
5. Generate manifest.json with checksums and schema for provenance

**Key Functions:**
- `find_repo_root(start)` (line 48) - Detect repository root
- `ensure_jsonable(obj)` (line 92) - Sanitize metadata for JSON serialization
- `docs_to_jsonl()`, `docs_to_parquet()`, `docs_to_hfds()` (lines 106-126) - Multi-format persistence
- `hash_file()` (line 128) - SHA256 checksum generation
- `build_testset()` (line 202) - RAGAS testset generation with 0.3.x/0.2.x compatibility

**Configuration:**
- Paths: `data/raw/`, `data/interim/`, `data/processed/`
- Models: `gpt-4.1-mini`, `text-embedding-3-small`
- Testset size: 10 (configurable via `TESTSET_SIZE`)

#### `scripts/upload_to_hf.py`
**Purpose:** Upload GDELT RAG datasets to Hugging Face Hub.

**Entry Point:** Line 291 (`if __name__ == "__main__"`)

**Key Responsibilities:**
1. Load source documents and golden testset from local storage
2. Create dataset cards with metadata (README.md)
3. Upload datasets to Hugging Face Hub
4. Update manifest.json with dataset repo IDs

**Key Functions:**
- `create_sources_card()` (line 34) - Generate dataset card for source documents
- `create_golden_testset_card()` (line 113) - Generate dataset card for golden testset
- `load_manifest()` (line 194) - Load manifest.json
- `update_manifest()` (line 200) - Update manifest with repo IDs and upload timestamp
- `main()` (line 219) - Orchestrate upload process

**Configuration:**
- Username: `dwb2023`
- Datasets: `gdelt-rag-sources`, `gdelt-rag-golden-testset`
- Requires: `HF_TOKEN` environment variable

#### `scripts/generate_run_manifest.py`
**Purpose:** Generate RUN_MANIFEST.json for reproducibility.

**Entry Point:** Line 173 (`if __name__ == "__main__"`)

**Key Responsibilities:**
1. Capture exact configuration of RAGAS evaluation runs
2. Document model versions and parameters
3. Document retriever configurations
4. Document evaluation settings and dependencies
5. Optionally include evaluation results summary

**Key Functions:**
- `generate_manifest(output_path, evaluation_results, retrievers_config)` (line 21) - Generate complete manifest

**Manifest Structure:**
- `ragas_version`, `python_version` - Environment info
- `llm`, `embeddings` - Model configurations
- `retrievers[]` - List of retriever configurations with parameters
- `evaluation` - Golden testset info and metrics
- `vector_store` - Qdrant configuration
- `results_summary` - Optional evaluation results (if provided)

**Output:** `data/processed/RUN_MANIFEST.json`

#### `scripts/enrich_manifest.py`
**Purpose:** Enrich manifest.json with metadata, checksums, and environment details.

**Entry Point:** Line 239 (`if __name__ == "__main__"`)

**Key Responsibilities:**
1. Add environment information (Python version, OS, package versions)
2. Add artifact metadata (file sizes, SHA256 checksums, row counts)
3. Add metrics (document stats, page content statistics)
4. Add lineage scaffolding (HuggingFace, LangSmith, Phoenix)
5. Add compliance scaffolding (license, PII policy)
6. Add run details (random seed, git commit SHA)
7. Relativize paths for portability

**Key Functions:**
- `sha256(path)` (line 7) - Compute SHA256 hash
- `file_bytes(path)` (line 15) - Get file size
- `count_jsonl_rows(path)` (line 19) - Count rows in JSONL
- `hfds_rows(path)` (line 24) - Count rows in HuggingFace dataset
- `parquet_rows(path)` (line 37) - Count rows in Parquet file
- `pandas_schema_from_parquet(path)` (line 51) - Extract schema
- `char_stats_jsonl(path, field, max_scan)` (line 61) - Calculate text statistics
- `main(manifest_path)` (line 81) - Orchestrate enrichment process

**Input/Output:** `data/interim/manifest.json` (default) or path from command line

---

## Module Dependencies

### Core RAG System Dependencies

```
src/graph.py
├─→ src/prompts.py (BASELINE_PROMPT)
├─→ src/retrievers.py (baseline_retriever, bm25_retriever, compression_retriever, ensemble_retriever)
├─→ src/state.py (State)
├─→ langchain.prompts (ChatPromptTemplate)
├─→ langchain_openai (ChatOpenAI)
└─→ langgraph.graph (START, StateGraph)

src/retrievers.py
├─→ langchain.retrievers (EnsembleRetriever)
├─→ langchain.retrievers.contextual_compression (ContextualCompressionRetriever)
├─→ langchain_community.retrievers (BM25Retriever)
├─→ langchain_cohere (CohereRerank)
├─→ langchain_qdrant (QdrantVectorStore)
├─→ langchain_openai (OpenAIEmbeddings)
└─→ qdrant_client (QdrantClient)

src/config.py
├─→ langchain_openai (ChatOpenAI, OpenAIEmbeddings)
└─→ [No internal dependencies]

src/state.py
├─→ typing (List)
├─→ typing_extensions (TypedDict)
└─→ langchain_core.documents (Document)

src/prompts.py
└─→ [No dependencies - pure constants]
```

### Script Dependencies

```
scripts/single_file.py (Comprehensive evaluation script)
├─→ scripts/generate_run_manifest.py (generate_manifest)
├─→ datasets (load_dataset)
├─→ langchain (ChatPromptTemplate, Document, retrievers)
├─→ langchain_openai (ChatOpenAI, OpenAIEmbeddings)
├─→ langchain_qdrant (QdrantVectorStore)
├─→ langgraph.graph (START, StateGraph)
├─→ qdrant_client (QdrantClient)
└─→ ragas (EvaluationDataset, evaluate, metrics, wrappers)

scripts/ingest.py (Data ingestion pipeline)
├─→ langchain_core.documents (Document)
├─→ langchain_community.document_loaders (DirectoryLoader, PyMuPDFLoader)
├─→ langchain_openai (ChatOpenAI, OpenAIEmbeddings)
├─→ ragas.embeddings (LangchainEmbeddingsWrapper)
├─→ ragas.llms (LangchainLLMWrapper)
├─→ ragas.testset (TestsetGenerator)
├─→ ragas (generate - optional for 0.3.x)
├─→ datasets (Dataset)
└─→ pandas, hashlib, json

scripts/upload_to_hf.py (HuggingFace upload)
├─→ datasets (load_from_disk)
├─→ huggingface_hub (HfApi, login)
└─→ json, pathlib, datetime

scripts/generate_run_manifest.py (Manifest generation)
├─→ ragas (__version__)
└─→ json, sys, datetime, pathlib

scripts/enrich_manifest.py (Manifest enrichment)
├─→ datasets (load_from_disk)
├─→ pandas
├─→ pyarrow.parquet
└─→ json, hashlib, platform, subprocess
```

### Dependency Graph Visualization

**High-Level Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│                         Scripts Layer                        │
│  (Entry points for ingestion, evaluation, upload)            │
│                                                              │
│  single_file.py → ingest.py → upload_to_hf.py              │
│                   ↓                                          │
│            generate_run_manifest.py                         │
│            enrich_manifest.py                               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────┐
│                      Core RAG System (src/)                  │
│                                                              │
│  graph.py ←─┬─→ retrievers.py                              │
│             │   (4 retrieval strategies)                    │
│             │                                               │
│             ├─→ prompts.py                                  │
│             │   (QA templates)                              │
│             │                                               │
│             ├─→ state.py                                    │
│             │   (TypedDict state)                           │
│             │                                               │
│             └─→ config.py                                   │
│                 (Qdrant config, models)                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────┐
│                   External Dependencies                      │
│                                                              │
│  LangChain → LangGraph → Qdrant → OpenAI → Cohere          │
│  RAGAS → HuggingFace Datasets → Pandas                     │
└─────────────────────────────────────────────────────────────┘
```

### Cross-Module Import Patterns

**Pattern 1: Centralized Configuration**
- `src/config.py` exports shared model instances
- `src/retrievers.py` and `scripts/single_file.py` duplicate this configuration (anti-pattern)
- **Recommendation:** All modules should import from `src/config.py`

**Pattern 2: Graph Composition**
- `src/graph.py` orchestrates all components
- Imports retrieval strategies from `src/retrievers.py`
- Imports prompts from `src/prompts.py`
- Imports state from `src/state.py`
- Creates compiled LangGraph instances for each retrieval strategy

**Pattern 3: Script Independence**
- Each script in `scripts/` is self-contained and executable
- Scripts import from external packages (LangChain, RAGAS) directly
- `scripts/single_file.py` reimplements functionality from `src/` rather than importing
- **Recommendation:** Scripts should import from `src/` for consistency

### Key Observations

1. **Duplicate Code:** `scripts/single_file.py` (lines 239-312) duplicates functionality from `src/graph.py` and `src/retrievers.py`. This is likely intentional for a self-contained evaluation script but could lead to maintenance issues.

2. **Undefined Variable:** `src/retrievers.py` line 40 references `documents` variable which is not defined in the module scope. This would cause a runtime error unless documents are defined elsewhere.

3. **Configuration Duplication:** Both `src/config.py` and `src/retrievers.py` define `QDRANT_HOST`, `QDRANT_PORT`, and `COLLECTION_NAME`.

4. **Empty Utilities:** `src/utils.py` is currently empty but included in the project structure, suggesting planned expansion.

5. **Version Compatibility:** `scripts/ingest.py` includes fallback logic for RAGAS 0.3.x vs 0.2.x API changes (lines 188-229), showing thoughtful version management.

---

## Summary Statistics

**Total Python Modules:** 13
- Core RAG system (`src/`): 7 files (1 empty, 1 minimal)
- Pipeline scripts (`scripts/`): 5 files
- Entry point (`main.py`): 1 file

**Public API Surface:**
- Classes: 1 (`State`)
- Functions: 5 (4 retrieval + 1 generation)
- Module exports: 4 (config, graph, retrievers, prompts)

**Lines of Code (approximate):**
- `src/`: ~120 LOC (excluding empty files)
- `scripts/`: ~1,200 LOC
- **Total:** ~1,320 LOC

**Entry Points:**
- Main application: 1 (placeholder)
- Executable scripts: 5 (ingest, evaluation, upload, manifest generation, manifest enrichment)

**External Dependencies:**
- LangChain ecosystem (core, openai, qdrant, cohere, community)
- RAGAS (evaluation framework)
- Qdrant (vector database client)
- HuggingFace (datasets, hub)
- Pandas, PyArrow (data processing)
- Tenacity (retry logic)
