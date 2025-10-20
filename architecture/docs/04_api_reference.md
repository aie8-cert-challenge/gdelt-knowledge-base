# API Reference

## Overview

This document provides comprehensive API documentation for the GDELT RAG (Retrieval-Augmented Generation) system. The system is organized into modular components that work together to provide a production-grade question-answering pipeline over GDELT knowledge graphs.

**Package Structure:**
- `src/config.py` - Configuration management and resource factories
- `src/retrievers.py` - Retriever factory functions
- `src/graph.py` - LangGraph workflow builders
- `src/state.py` - State schema definitions
- `src/prompts.py` - Prompt templates
- `src/utils/` - Utility functions for data loading and manifest generation
- `scripts/` - Executable scripts for ingestion, evaluation, and validation
- `app/` - LangGraph server application

**Design Principles:**
- Factory pattern for resource creation (prevents import-time initialization)
- Cached getters for expensive resources (LLM, embeddings, Qdrant client)
- Environment-based configuration with sensible defaults
- Type hints and comprehensive docstrings

---

## Configuration Module (`src/config.py`)

### Overview

The configuration module provides cached getter functions for LLM, embeddings, and Qdrant client instances. All configuration is read from environment variables with sensible defaults. Resources are cached using `@lru_cache` to prevent duplicate initialization.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL (preferred over host/port) |
| `QDRANT_HOST` | `localhost` | Qdrant host (fallback if URL not set) |
| `QDRANT_PORT` | `6333` | Qdrant port (fallback if URL not set) |
| `QDRANT_API_KEY` | `None` | Qdrant API key (optional, omit for local Docker) |
| `OPENAI_API_KEY` | *(required)* | OpenAI API key for LLM and embeddings |
| `COHERE_API_KEY` | *(optional)* | Cohere API key for reranking |
| `HF_TOKEN` | *(optional)* | HuggingFace token for private datasets |
| `LANGSMITH_PROJECT` | `certification-challenge` | LangSmith project name |
| `LANGSMITH_TRACING` | `true` | Enable LangSmith tracing |
| `LANGSMITH_API_KEY` | *(optional)* | LangSmith API key |
| `QDRANT_COLLECTION` | `gdelt_comparative_eval` | Qdrant collection name |
| `OPENAI_MODEL` | `gpt-4.1-mini` | OpenAI chat model |
| `OPENAI_EMBED_MODEL` | `text-embedding-3-small` | OpenAI embedding model |

### Functions

#### `get_llm()`

**Source**: `src/config.py:42-49`

**Description**: Get cached LLM instance for chat completions. Uses `@lru_cache(maxsize=1)` to ensure only one instance is created per process.

**Returns**:
- `ChatOpenAI`: Configured OpenAI LLM with temperature=0 for deterministic outputs

**Example**:
```python
from src.config import get_llm

llm = get_llm()
response = llm.invoke("What is GDELT?")
print(response.content)
```

**Notes**:
- Temperature is hardcoded to 0 for reproducibility
- Model name is configured via `OPENAI_MODEL` environment variable
- Instance is cached - subsequent calls return the same object

---

#### `get_embeddings()`

**Source**: `src/config.py:52-60`

**Description**: Get cached embeddings instance for document and query vectorization.

**Returns**:
- `OpenAIEmbeddings`: Configured OpenAI embeddings model (1536 dimensions)

**Example**:
```python
from src.config import get_embeddings

embeddings = get_embeddings()
vector = embeddings.embed_query("What is GDELT?")
print(f"Embedding dimensions: {len(vector)}")  # 1536
```

**Notes**:
- Uses `text-embedding-3-small` by default (1536 dimensions)
- Cached - only one instance created per process
- Dimensions must match Qdrant collection configuration

---

#### `get_qdrant()`

**Source**: `src/config.py:63-80`

**Description**: Get cached Qdrant client instance. Uses URL-first convention - if `QDRANT_URL` is set, uses it; otherwise falls back to `QDRANT_HOST`/`QDRANT_PORT`.

**Returns**:
- `QdrantClient`: Connected Qdrant client

**Example**:
```python
from src.config import get_qdrant

client = get_qdrant()
collections = client.get_collections()
print(f"Available collections: {[c.name for c in collections.collections]}")
```

**Notes**:
- API key is only passed if `QDRANT_API_KEY` is set (prevents breaking Docker default)
- Cached for connection pooling efficiency
- Automatically handles URL vs host/port configuration

---

#### `get_collection_name()`

**Source**: `src/config.py:83-90`

**Description**: Get configured collection name for Qdrant vector store.

**Returns**:
- `str`: Collection name (default: `"gdelt_comparative_eval"`)

**Example**:
```python
from src.config import get_collection_name

collection = get_collection_name()
print(f"Using collection: {collection}")
```

---

#### `create_vector_store()`

**Source**: `src/config.py:93-150`

**Description**: Create and populate Qdrant vector store. This factory function handles collection creation, optional recreation, and document ingestion.

**Parameters**:
- `documents` (`List[Document]`): List of LangChain Document objects to add to vector store
- `collection_name` (`str`, optional): Override default collection name (defaults to `get_collection_name()`)
- `recreate_collection` (`bool`, optional): If True, delete existing collection first (default: `False`)

**Returns**:
- `QdrantVectorStore`: Populated vector store instance ready for retrieval

**Example**:
```python
from src.utils import load_documents_from_huggingface
from src.config import create_vector_store

# Load documents
documents = load_documents_from_huggingface()

# Create vector store (reuse existing collection if present)
vector_store = create_vector_store(documents)

# Force recreation (useful for development)
vector_store = create_vector_store(
    documents,
    recreate_collection=True
)

# Use custom collection name
vector_store = create_vector_store(
    documents,
    collection_name="my_custom_collection"
)
```

**Implementation Details**:
1. Checks if collection exists
2. Optionally deletes collection if `recreate_collection=True`
3. Creates collection with:
   - Vector size: 1536 (matches `text-embedding-3-small`)
   - Distance metric: COSINE
4. Populates with documents if collection is new or recreated
5. Returns connected `QdrantVectorStore` instance

**Notes**:
- Collection creation is idempotent - safe to call multiple times
- Documents are only added if collection is new or recreated
- Uses cached Qdrant client and embeddings from `get_qdrant()` and `get_embeddings()`

---

## Retrievers Module (`src/retrievers.py`)

### Overview

Provides factory functions to create retriever instances. Retrievers cannot be instantiated at module level because they require documents and vector stores that must be loaded first. This module follows the factory pattern to ensure proper initialization order.

### Functions

#### `create_retrievers()`

**Source**: `src/retrievers.py:20-89`

**Description**: Create all retriever instances for comparative evaluation. Returns a dictionary of 4 different retrieval strategies optimized for different use cases.

**Parameters**:
- `documents` (`List[Document]`): List of Document objects (required for BM25 indexing)
- `vector_store` (`QdrantVectorStore`): Populated vector store instance
- `k` (`int`, optional): Number of documents to retrieve (default: `5`)

**Returns**:
- `Dict[str, object]`: Dictionary mapping retriever names to retriever instances
  - `'naive'`: Dense vector search (baseline)
  - `'bm25'`: Sparse keyword matching
  - `'ensemble'`: Hybrid (50% dense + 50% sparse)
  - `'cohere_rerank'`: Contextual compression with reranking

**Example**:
```python
from src.utils import load_documents_from_huggingface
from src.config import create_vector_store
from src.retrievers import create_retrievers

# Setup
documents = load_documents_from_huggingface()
vector_store = create_vector_store(documents, recreate_collection=True)

# Create all retrievers
retrievers = create_retrievers(documents, vector_store, k=5)

# Use individual retrievers
naive_docs = retrievers['naive'].invoke("What is GDELT?")
bm25_docs = retrievers['bm25'].invoke("What is GDELT?")
ensemble_docs = retrievers['ensemble'].invoke("What is GDELT?")
rerank_docs = retrievers['cohere_rerank'].invoke("What is GDELT?")

print(f"Naive retrieved: {len(naive_docs)} docs")
print(f"BM25 retrieved: {len(bm25_docs)} docs")
```

**Retriever Strategies**:

1. **Naive (Dense Vector Search)**
   - Type: Semantic similarity
   - Implementation: OpenAI embeddings + cosine distance
   - Use case: General semantic search
   - Returns: Top k documents by embedding similarity

2. **BM25 (Sparse Keyword Matching)**
   - Type: Lexical matching
   - Implementation: TF-IDF based ranking
   - Use case: Exact keyword matching, proper nouns
   - Returns: Top k documents by BM25 score
   - Note: Operates on in-memory document collection

3. **Ensemble (Hybrid Search)**
   - Type: Combination of dense + sparse
   - Implementation: 50/50 weighted combination of naive + BM25
   - Use case: Balanced semantic + lexical retrieval
   - Returns: Merged results from both retrievers

4. **Cohere Rerank (Contextual Compression)**
   - Type: Two-stage retrieval with reranking
   - Implementation:
     - Stage 1: Retrieve 20 documents via dense search
     - Stage 2: Rerank using Cohere rerank-v3.5
   - Use case: Highest quality retrieval, willing to pay rerank cost
   - Returns: Top k documents after reranking
   - Requirement: `COHERE_API_KEY` must be set

**Notes**:
- All retrievers follow LangChain `Retriever` interface with `.invoke()` method
- BM25 retriever requires documents to build in-memory index
- Cohere reranker will fail if `COHERE_API_KEY` is not set
- Ensemble weights are hardcoded to [0.5, 0.5] for balanced retrieval

---

## Graph Module (`src/graph.py`)

### Overview

Provides factory functions to create compiled LangGraph workflows. Graphs cannot be instantiated at module level because they depend on retrievers that must be created first. Each graph implements a simple two-node pipeline: retrieve → generate.

### Functions

#### `build_graph()`

**Source**: `src/graph.py:21-106`

**Description**: Build a compiled LangGraph pipeline for a single retriever. Creates a two-node graph: START → retrieve → generate → END.

**Parameters**:
- `retriever` (Retriever): Retriever instance to use for document retrieval
- `llm` (`ChatOpenAI`, optional): LLM instance (defaults to `get_llm()` if None)
- `prompt_template` (`str`, optional): RAG prompt template string (defaults to `BASELINE_PROMPT`)

**Returns**:
- `CompiledGraph`: Compiled StateGraph that can be invoked with `{"question": "..."}`

**Example**:
```python
from src.utils import load_documents_from_huggingface
from src.config import create_vector_store
from src.retrievers import create_retrievers
from src.graph import build_graph

# Setup
documents = load_documents_from_huggingface()
vector_store = create_vector_store(documents)
retrievers = create_retrievers(documents, vector_store)

# Build graph for single retriever
graph = build_graph(retrievers['naive'])

# Execute query
result = graph.invoke({"question": "What is GDELT?"})
print(f"Question: {result['question']}")
print(f"Context: {len(result['context'])} documents")
print(f"Response: {result['response']}")

# Custom LLM and prompt
from src.config import get_llm
from langchain_openai import ChatOpenAI

custom_llm = ChatOpenAI(model="gpt-4", temperature=0.3)
custom_prompt = "Answer based on context:\n{context}\n\nQuestion: {question}"

custom_graph = build_graph(
    retrievers['bm25'],
    llm=custom_llm,
    prompt_template=custom_prompt
)
```

**Graph Structure**:
```
START → retrieve → generate → END
```

**Node Functions**:

1. **retrieve** (returns `{"context": List[Document]}`):
   - Input: `state["question"]`
   - Action: Invoke retriever with question
   - Output: Retrieved documents in `state["context"]`

2. **generate** (returns `{"response": str}`):
   - Input: `state["question"]`, `state["context"]`
   - Action: Format prompt and invoke LLM
   - Output: Generated answer in `state["response"]`

**State Updates**:
- Node functions return partial state updates (dict)
- LangGraph automatically merges updates into state
- Follows LangGraph best practices for state management

**Notes**:
- Default temperature is 0 for deterministic outputs
- Prompt template must include `{question}` and `{context}` placeholders
- Graph is compiled once - can be invoked multiple times

---

#### `build_all_graphs()`

**Source**: `src/graph.py:109-141`

**Description**: Build compiled graphs for all retrievers. Convenience function to create a graph for each retriever in the retrievers dictionary.

**Parameters**:
- `retrievers` (`Dict[str, object]`): Dictionary of retriever instances from `create_retrievers()`
- `llm` (`ChatOpenAI`, optional): Optional LLM instance (shared across all graphs)

**Returns**:
- `Dict[str, CompiledGraph]`: Dictionary mapping retriever names to compiled graphs (same keys as input)

**Example**:
```python
from src.utils import load_documents_from_huggingface
from src.config import create_vector_store, get_llm
from src.retrievers import create_retrievers
from src.graph import build_all_graphs

# Setup
documents = load_documents_from_huggingface()
vector_store = create_vector_store(documents)
retrievers = create_retrievers(documents, vector_store)

# Build all graphs at once
graphs = build_all_graphs(retrievers)

# All graphs ready to use
result_naive = graphs['naive'].invoke({"question": "What is GDELT?"})
result_bm25 = graphs['bm25'].invoke({"question": "What is GDELT?"})
result_ensemble = graphs['ensemble'].invoke({"question": "What is GDELT?"})
result_rerank = graphs['cohere_rerank'].invoke({"question": "What is GDELT?"})

# Shared LLM across all graphs
llm = get_llm()
graphs = build_all_graphs(retrievers, llm=llm)
```

**Notes**:
- Creates one graph per retriever using `build_graph()`
- All graphs share the same LLM instance if provided
- Output keys match input retriever keys
- Graphs are compiled and ready to invoke

---

## State Module (`src/state.py`)

### Overview

Defines the state schema for LangGraph workflows using TypedDict. The state is shared across all nodes in the graph and updated via partial dictionary returns.

### Classes

#### `State`

**Source**: `src/state.py:7-10`

**Description**: TypedDict defining the state schema for RAG workflows.

**Attributes**:
- `question` (`str`): User's input question
- `context` (`List[Document]`): Retrieved documents from retriever node
- `response` (`str`): Generated answer from generate node

**Example**:
```python
from src.state import State
from langchain_core.documents import Document

# State is updated incrementally by graph nodes
initial_state: State = {
    "question": "What is GDELT?",
    "context": [],
    "response": ""
}

# After retrieve node
state_after_retrieve: State = {
    "question": "What is GDELT?",
    "context": [
        Document(page_content="GDELT is a global database..."),
        Document(page_content="GDELT monitors news media...")
    ],
    "response": ""
}

# After generate node
final_state: State = {
    "question": "What is GDELT?",
    "context": [Document(...)],
    "response": "GDELT is a comprehensive global database..."
}
```

**Notes**:
- Used as type hint for graph builder: `StateGraph(State)`
- Node functions return partial updates: `{"context": docs}` or `{"response": answer}`
- LangGraph automatically merges partial updates into state
- All fields are required (no Optional types)

---

## Prompts Module (`src/prompts.py`)

### Overview

Defines prompt templates for RAG workflows. Currently provides a single baseline prompt optimized for grounded question-answering.

### Constants

#### `BASELINE_PROMPT`

**Source**: `src/prompts.py:4-12`

**Description**: Default RAG prompt template that enforces context-grounded answering.

**Template**:
```python
BASELINE_PROMPT = """\
You are a helpful assistant who answers questions based on provided context. You must only use the provided context, and cannot use your own knowledge.

### Question
{question}

### Context
{context}
"""
```

**Placeholders**:
- `{question}`: User's input question (injected by graph)
- `{context}`: Concatenated page_content from retrieved documents

**Usage**:
```python
from src.prompts import BASELINE_PROMPT
from langchain.prompts import ChatPromptTemplate

# Create prompt template
prompt = ChatPromptTemplate.from_template(BASELINE_PROMPT)

# Format with values
messages = prompt.format_messages(
    question="What is GDELT?",
    context="GDELT is a global database of events, language, and tone..."
)

# Use with LLM
from src.config import get_llm
llm = get_llm()
response = llm.invoke(messages)
```

**Design Rationale**:
- Enforces grounded answering (prevents hallucination)
- Clear section headers for question and context
- Compatible with all OpenAI chat models
- Used by default in `build_graph()` if no custom prompt provided

---

## Utils Module (`src/utils/`)

### Document Loader (`src/utils/loaders.py`)

#### `load_documents_from_huggingface()`

**Source**: `src/utils/loaders.py:15-75`

**Description**: Load documents from HuggingFace dataset and convert to LangChain Documents. Handles nested metadata structures and provides revision pinning for reproducibility.

**Parameters**:
- `dataset_name` (`str`, optional): HuggingFace dataset identifier (default: `"dwb2023/gdelt-rag-sources"`)
- `split` (`str`, optional): Dataset split to load (default: `"train"`)
- `revision` (`str`, optional): Dataset revision/commit SHA to pin (default: `None`, uses `HF_SOURCES_REV` env var if set)

**Returns**:
- `List[Document]`: List of LangChain Document objects with page_content and metadata

**Example**:
```python
from src.utils import load_documents_from_huggingface

# Load latest version
documents = load_documents_from_huggingface()
print(f"Loaded {len(documents)} documents")

# Pin to specific revision for reproducibility
documents = load_documents_from_huggingface(
    revision="abc123def456"
)

# Or use environment variable
import os
os.environ["HF_SOURCES_REV"] = "abc123def456"
documents = load_documents_from_huggingface()

# Load custom dataset
documents = load_documents_from_huggingface(
    dataset_name="my-org/my-dataset",
    split="validation"
)

# Inspect document structure
doc = documents[0]
print(f"Content: {doc.page_content[:100]}...")
print(f"Metadata: {doc.metadata}")
```

**Dataset Structure**:
Expected HuggingFace dataset format:
```python
{
    "page_content": "GDELT is a global database...",
    "metadata": {
        "source": "gdelt_codebook.pdf",
        "page": 1,
        "chunk_id": 0
    }
}
```

**Implementation Details**:
1. Loads dataset from HuggingFace Hub
2. Iterates over dataset items
3. Extracts `page_content` field
4. Handles nested metadata structure:
   - If `metadata` is dict, uses it directly
   - Otherwise, creates metadata from all fields except `page_content`
5. Creates LangChain Document objects
6. Returns list of documents

**Notes**:
- Revision pinning prevents dataset drift over time
- Environment variable `HF_SOURCES_REV` takes precedence over default
- Empty or missing page_content defaults to empty string
- All metadata fields are preserved

---

#### `load_golden_testset_from_huggingface()`

**Source**: `src/utils/loaders.py:78-113`

**Description**: Load golden testset from HuggingFace dataset for RAGAS evaluation.

**Parameters**:
- `dataset_name` (`str`, optional): HuggingFace dataset identifier (default: `"dwb2023/gdelt-rag-golden-testset"`)
- `split` (`str`, optional): Dataset split to load (default: `"train"`)
- `revision` (`str`, optional): Dataset revision/commit SHA to pin (default: `None`, uses `HF_GOLDEN_REV` env var if set)

**Returns**:
- `Dataset`: HuggingFace Dataset object (not LangChain Documents)

**Example**:
```python
from src.utils import load_golden_testset_from_huggingface

# Load latest version
golden_dataset = load_golden_testset_from_huggingface()
golden_df = golden_dataset.to_pandas()
print(f"Loaded {len(golden_df)} test examples")

# Inspect structure
print(golden_df.columns)
# ['user_input', 'reference_contexts', 'reference', 'synthesizer_name']

# Pin to specific revision
golden_dataset = load_golden_testset_from_huggingface(
    revision="abc123def456"
)

# Or use environment variable
import os
os.environ["HF_GOLDEN_REV"] = "abc123def456"
golden_dataset = load_golden_testset_from_huggingface()
```

**Expected Schema**:
```python
{
    "user_input": "What is GDELT?",
    "reference_contexts": ["GDELT is...", "The database monitors..."],
    "reference": "GDELT is a comprehensive global database...",
    "synthesizer_name": "gpt-4.1-mini"
}
```

**Notes**:
- Returns HuggingFace Dataset, not LangChain Documents (different from `load_documents_from_huggingface()`)
- Revision pinning ensures test set consistency across runs
- Prevents score drift due to dataset updates
- Use `.to_pandas()` for DataFrame conversion

---

### Manifest Generation (`src/utils/manifest.py`)

#### `generate_manifest()`

**Source**: `src/utils/manifest.py:21-176`

**Description**: Generate run manifest JSON for reproducibility. Captures complete evaluation configuration including model versions, retriever settings, and evaluation results.

**Parameters**:
- `output_path` (`Path`): Path to save manifest JSON
- `evaluation_results` (`Dict[str, Any]`, optional): RAGAS evaluation results by retriever
- `retrievers_config` (`Dict[str, Any]`, optional): Retriever configurations
- `data_provenance` (`Dict[str, Any]`, optional): Link to ingestion manifest

**Returns**:
- `Dict[str, Any]`: Dictionary containing the manifest (also saved to file)

**Example**:
```python
from pathlib import Path
from src.utils import generate_run_manifest
from src.retrievers import create_retrievers
from src.graph import build_all_graphs
# ... after running evaluation ...

# Generate manifest
manifest_path = Path("deliverables/evaluation_evidence/RUN_MANIFEST.json")
manifest = generate_run_manifest(
    output_path=manifest_path,
    evaluation_results=ragas_results,  # Dict[str, RagasResult]
    retrievers_config={
        "naive": {"graph": graphs["naive"], "k": 5},
        "bm25": {"graph": graphs["bm25"], "k": 5},
        "ensemble": {"graph": graphs["ensemble"], "k": 5},
        "cohere_rerank": {"graph": graphs["cohere_rerank"], "k": 5}
    },
    data_provenance={
        "ingest_manifest_id": "ragas_pipeline_abc123",
        "ingest_timestamp": "2024-01-15T10:30:00Z",
        "sources_sha256": "abcdef...",
        "golden_testset_sha256": "123456..."
    }
)

print(f"Manifest saved to: {manifest_path}")
print(f"RAGAS version: {manifest['ragas_version']}")
print(f"Retrievers: {len(manifest['retrievers'])}")
```

**Manifest Structure**:
```json
{
  "ragas_version": "0.2.10",
  "python_version": "3.11",
  "llm": {
    "model": "gpt-4.1-mini",
    "temperature": 0,
    "provider": "openai",
    "purpose": "RAG generation and RAGAS evaluation"
  },
  "embeddings": {
    "model": "text-embedding-3-small",
    "dimensions": 1536,
    "provider": "openai"
  },
  "retrievers": [
    {
      "name": "naive",
      "type": "dense_vector_search",
      "description": "Baseline dense vector search",
      "k": 5,
      "distance_metric": "cosine",
      "rerank": false
    },
    ...
  ],
  "evaluation": {
    "golden_testset": "dwb2023/gdelt-rag-golden-testset",
    "golden_testset_size": 12,
    "metrics": ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
  },
  "results_summary": {
    "naive": {
      "faithfulness": 0.95,
      "answer_relevancy": 0.87,
      "context_precision": 0.82,
      "context_recall": 0.89,
      "average": 0.8825
    },
    ...
  },
  "data_provenance": {
    "ingest_manifest_id": "ragas_pipeline_abc123",
    "sources_sha256": "abcdef..."
  },
  "generated_at": "2024-01-15T12:00:00Z",
  "generated_by": "scripts/generate_run_manifest.py"
}
```

**Implementation Details**:
1. Captures runtime environment (Python, RAGAS versions)
2. Documents model configurations (LLM, embeddings)
3. Records retriever configurations and parameters
4. Summarizes evaluation results (if provided)
5. Links to data provenance (if provided)
6. Writes JSON to output_path

**Notes**:
- Enables exact reproduction of evaluation results
- Links evaluation runs to data ingestion via manifest IDs
- All LLM calls use temperature=0 for determinism
- Manifest is both returned and saved to file

---

## Scripts

### Ingestion Script (`scripts/ingest_raw_pdfs.py`)

**Source**: `scripts/ingest_raw_pdfs.py`

**Purpose**: Extract PDFs from `data/raw/`, generate RAGAS golden testset, and persist to multiple formats (JSONL, Parquet, HuggingFace dataset).

**Usage**:
```bash
# Run ingestion pipeline
python scripts/ingest_raw_pdfs.py

# Or use Jupyter notebook interface
jupyter notebook scripts/ingest_raw_pdfs.py
```

**Configuration**:
- `OPENAI_MODEL`: LLM for testset generation (default: `gpt-4.1-mini`)
- `OPENAI_EMBED_MODEL`: Embeddings model (default: `text-embedding-3-small`)
- `TESTSET_SIZE`: Number of test examples to generate (default: `10`)
- `MAX_DOCS`: Limit source documents for prototyping (default: `None` - all docs)
- `RANDOM_SEED`: Random seed for reproducibility (default: `42`)

**Input**:
- PDFs in `data/raw/` directory

**Output**:
```
data/interim/
├── sources.docs.jsonl           # Source documents (JSONL)
├── sources.docs.parquet         # Source documents (Parquet)
├── sources.hfds/                # Source documents (HF dataset)
├── golden_testset.jsonl         # Test set (JSONL)
├── golden_testset.parquet       # Test set (Parquet)
├── golden_testset.hfds/         # Test set (HF dataset)
└── manifest.json                # Ingestion manifest with SHA256 checksums
```

**Implementation**:
1. Loads PDFs using PyMuPDFLoader
2. Generates RAGAS testset using gpt-4.1-mini
3. Persists in 3 formats (JSONL, Parquet, HF dataset)
4. Generates manifest with checksums and metadata
5. Links artifacts via manifest ID for reproducibility

**Notes**:
- Requires `OPENAI_API_KEY` to be set
- Uses tenacity retry logic for API resilience
- Compatible with both RAGAS 0.2.x and 0.3.x
- Manifest enables data lineage tracking across pipelines

---

### Evaluation Harness Script (`scripts/run_eval_harness.py`)

**Source**: `scripts/run_eval_harness.py`

**Purpose**: Run comprehensive RAGAS evaluation across all 4 retrievers (naive, BM25, ensemble, cohere_rerank) using the golden testset.

**Usage**:
```bash
# Run with existing collection (faster)
python scripts/run_eval_harness.py

# Recreate collection (fresh start)
python scripts/run_eval_harness.py --recreate true

# Or use make target
make eval
```

**Configuration**:
- `DATASET_SOURCES`: Source dataset (default: `dwb2023/gdelt-rag-sources`)
- `DATASET_GOLDEN`: Golden testset (default: `dwb2023/gdelt-rag-golden-testset`)
- `K`: Number of documents to retrieve (default: `5`)
- `RECREATE_COLLECTION`: Recreate Qdrant collection (default: `false`)

**Command Line Arguments**:
- `--recreate`: `true|false` - Recreate Qdrant collection (default: `false`)

**Environment Variables**:
- `OPENAI_API_KEY`: Required for LLM and embeddings
- `COHERE_API_KEY`: Required for cohere_rerank retriever
- `HF_SOURCES_REV`: Pin source dataset revision
- `HF_GOLDEN_REV`: Pin golden testset revision

**Output**:
```
deliverables/evaluation_evidence/
├── naive_raw_dataset.parquet              # Raw outputs (6 columns)
├── naive_evaluation_dataset.csv           # RAGAS format
├── naive_detailed_results.csv             # Per-question metrics
├── bm25_raw_dataset.parquet
├── bm25_evaluation_dataset.csv
├── bm25_detailed_results.csv
├── ensemble_raw_dataset.parquet
├── ensemble_evaluation_dataset.csv
├── ensemble_detailed_results.csv
├── cohere_rerank_raw_dataset.parquet
├── cohere_rerank_evaluation_dataset.csv
├── cohere_rerank_detailed_results.csv
├── comparative_ragas_results.csv          # Summary table
└── RUN_MANIFEST.json                      # Reproducibility manifest
```

**Execution Steps**:
1. Load source documents and golden testset
2. Create/connect to Qdrant vector store
3. Create all 4 retrievers
4. Build LangGraph workflows
5. Run inference (12 questions × 4 retrievers = 48 invocations)
6. Evaluate with RAGAS metrics:
   - Faithfulness (answer grounded in context)
   - Answer Relevancy (answer addresses question)
   - Context Precision (relevant contexts ranked higher)
   - Context Recall (ground truth coverage)
7. Generate comparative analysis
8. Save RUN_MANIFEST.json

**Time & Cost**:
- Runtime: 20-30 minutes
- API cost: ~$5-6 in OpenAI API calls
- Cohere API cost: minimal (reranking only)

**Notes**:
- Uses src/ modules (not inline code)
- Results identical to single_file.py
- Intermediate results saved immediately (fault tolerance)
- Links to ingestion manifest via data_provenance

---

### Full Evaluation Script (`scripts/run_full_evaluation.py`)

**Source**: `scripts/run_full_evaluation.py`

**Purpose**: Comprehensive single-file evaluation script (Tasks 5 & 7). Includes inline implementations for maximum portability.

**Usage**:
```bash
python scripts/run_full_evaluation.py
```

**Configuration**:
- `QDRANT_HOST`: Qdrant host (default: `localhost`)
- `QDRANT_PORT`: Qdrant port (default: `6333`)
- `COLLECTION_NAME`: Collection name (default: `gdelt_comparative_eval`)

**Environment Variables**:
- `OPENAI_API_KEY`: Required
- `COHERE_API_KEY`: Optional (cohere_rerank will fail without it)
- `HF_HUB_DISABLE_XET`: Disable progress bars (default: `1`)

**Output**: Same as `run_eval_harness.py`

**Differences from run_eval_harness.py**:
- Inline LangGraph construction (not using src/graph.py)
- Inline retriever creation (not using src/retrievers.py)
- Self-contained - no src/ dependencies
- Useful for understanding full pipeline flow

**Notes**:
- Results identical to run_eval_harness.py
- RAGAS 0.2.10 compatible
- Includes schema validation helpers
- Immediate persistence (fault tolerance)

---

### App Validation Script (`scripts/run_app_validation.py`)

**Source**: `scripts/run_app_validation.py`

**Purpose**: Validate LangGraph implementation and demonstrate correct initialization patterns.

**Usage**:
```bash
python scripts/run_app_validation.py
```

**Validation Checks**:
1. **Environment Validation**
   - OPENAI_API_KEY set
   - COHERE_API_KEY set
   - Qdrant connectivity
   - Critical package imports

2. **Module Import Validation**
   - src.config
   - src.state
   - src.prompts
   - src.retrievers
   - src.graph
   - src.utils

3. **Correct Initialization Pattern**
   - Load documents
   - Create vector store
   - Create retrievers
   - Build graphs

4. **Graph Compilation Validation**
   - All 4 graphs compile successfully

5. **Functional Testing**
   - Execute test queries
   - Validate result structure
   - Check context retrieval
   - Verify response generation

6. **Diagnostic Report**
   - Pass/fail summary
   - Critical issues
   - Recommendations

**Exit Codes**:
- `0`: All validations passed
- `1`: One or more validations failed

**Example Output**:
```
================================================================================
1. ENVIRONMENT VALIDATION
================================================================================

✓ OPENAI_API_KEY set                                        [PASS]
✓ COHERE_API_KEY set                                        [PASS]
✓ Qdrant at localhost:6333                                  [PASS]

================================================================================
5. FUNCTIONAL TESTING
================================================================================

✓ naive               functional test                       [PASS]
✓ bm25                functional test                       [PASS]
✓ ensemble            functional test                       [PASS]
✓ cohere_rerank       functional test                       [PASS]

================================================================================
6. DIAGNOSTIC REPORT
================================================================================

Overall Results:
  Total checks: 25
  Passed: 25
  Failed: 0
  Pass rate: 100.0%

✅ ALL VALIDATIONS PASSED
```

**Notes**:
- Useful for debugging src/ module issues
- Demonstrates factory pattern best practices
- Color-coded terminal output
- Detailed traceback on failures

---

## App Module (`app/graph_app.py`)

### Overview

LangGraph Server application entrypoint. Provides a deployable graph for serving via LangGraph Server.

### Functions

#### `get_app()`

**Source**: `app/graph_app.py:7-17`

**Description**: LangGraph Server entrypoint. Returns a compiled graph for serving.

**Returns**:
- `CompiledGraph`: Compiled LangGraph (default: cohere_rerank retriever)

**Example**:
```bash
# Deploy with LangGraph Server
langgraph up --config app/langgraph.json

# Or programmatic access
from app.graph_app import get_app

app = get_app()
result = app.invoke({"question": "What is GDELT?"})
print(result['response'])
```

**Implementation**:
1. Loads documents from HuggingFace
2. Creates vector store (reuses existing collection)
3. Creates all retrievers
4. Builds all graphs
5. Returns cohere_rerank graph (best performing)

**Notes**:
- Uses `recreate_collection=False` to avoid recreating on every startup
- Default retriever is cohere_rerank (can be changed)
- All src/ module factories are called in correct order
- Suitable for production deployment

---

## Configuration Reference

### Environment Variables (Comprehensive)

| Category | Variable | Default | Required | Description |
|----------|----------|---------|----------|-------------|
| **OpenAI** | `OPENAI_API_KEY` | - | Yes | OpenAI API key for LLM and embeddings |
| | `OPENAI_MODEL` | `gpt-4.1-mini` | No | Chat completion model |
| | `OPENAI_EMBED_MODEL` | `text-embedding-3-small` | No | Embedding model (1536 dims) |
| **Cohere** | `COHERE_API_KEY` | - | No | Cohere API key for reranking |
| **HuggingFace** | `HF_TOKEN` | - | No | HuggingFace token for private datasets |
| | `HF_SOURCES_REV` | - | No | Pin source dataset revision |
| | `HF_GOLDEN_REV` | - | No | Pin golden testset revision |
| | `HF_HUB_DISABLE_PROGRESS_BARS` | `1` | No | Disable progress bars |
| **Qdrant** | `QDRANT_URL` | `http://localhost:6333` | No | Qdrant server URL (preferred) |
| | `QDRANT_HOST` | `localhost` | No | Qdrant host (fallback) |
| | `QDRANT_PORT` | `6333` | No | Qdrant port (fallback) |
| | `QDRANT_API_KEY` | - | No | Qdrant API key (omit for local) |
| | `QDRANT_COLLECTION` | `gdelt_comparative_eval` | No | Collection name |
| **LangSmith** | `LANGSMITH_PROJECT` | `certification-challenge` | No | LangSmith project name |
| | `LANGSMITH_TRACING` | `true` | No | Enable tracing |
| | `LANGSMITH_API_KEY` | - | No | LangSmith API key |

### Vector Store Configuration

**Qdrant Collection Settings**:
```python
{
    "collection_name": "gdelt_comparative_eval",
    "vector_size": 1536,                    # Matches text-embedding-3-small
    "distance": Distance.COSINE,            # Cosine similarity
    "host": "localhost",
    "port": 6333
}
```

**Connection Patterns**:
```python
# URL-based (preferred)
QDRANT_URL=http://localhost:6333

# Host/port-based (fallback)
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Cloud deployment
QDRANT_URL=https://abc123.us-east-1.aws.cloud.qdrant.io
QDRANT_API_KEY=your_api_key
```

### LLM Configuration

**OpenAI Settings**:
```python
{
    "model": "gpt-4.1-mini",
    "temperature": 0,                       # Deterministic
    "timeout": 60,
    "max_retries": 6
}
```

**Embedding Settings**:
```python
{
    "model": "text-embedding-3-small",
    "dimensions": 1536,
    "timeout": 60,
    "max_retries": 6
}
```

**Cohere Rerank Settings**:
```python
{
    "model": "rerank-v3.5",
    "initial_k": 20,                        # Wide retrieval
    "top_n": 5                             # Rerank to top 5
}
```

---

## Usage Patterns and Best Practices

### Basic RAG Query

**End-to-End Example**:
```python
from src.utils import load_documents_from_huggingface
from src.config import create_vector_store, get_llm
from src.retrievers import create_retrievers
from src.graph import build_all_graphs

# 1. Load documents from HuggingFace
documents = load_documents_from_huggingface()
print(f"Loaded {len(documents)} documents")

# 2. Create vector store (reuse existing collection)
vector_store = create_vector_store(documents)

# 3. Create all retrievers
retrievers = create_retrievers(documents, vector_store, k=5)

# 4. Build LangGraph workflows
graphs = build_all_graphs(retrievers)

# 5. Query the system
question = "What is GDELT?"
result = graphs['cohere_rerank'].invoke({"question": question})

# 6. Access results
print(f"Question: {result['question']}")
print(f"Retrieved {len(result['context'])} contexts")
print(f"Answer: {result['response']}")
```

### Custom Retriever Configuration

**Different k Values**:
```python
from src.retrievers import create_retrievers

# High-recall configuration (retrieve more documents)
retrievers_high_recall = create_retrievers(
    documents,
    vector_store,
    k=10
)

# Precision-focused (retrieve fewer, high-quality documents)
retrievers_precision = create_retrievers(
    documents,
    vector_store,
    k=3
)
```

**Single Retriever**:
```python
# If you only need one retriever type
naive_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

from src.graph import build_graph
graph = build_graph(naive_retriever)
```

### Running Evaluations

**Full RAGAS Evaluation**:
```python
from pathlib import Path
from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.metrics import Faithfulness, ResponseRelevancy, ContextPrecision, LLMContextRecall
from ragas.llms import LangchainLLMWrapper
from src.config import get_llm
from src.utils import load_golden_testset_from_huggingface

# Load golden testset
golden_dataset = load_golden_testset_from_huggingface()
golden_df = golden_dataset.to_pandas()

# Run inference
results_df = golden_df.copy()
results_df['response'] = None
results_df['retrieved_contexts'] = None

for idx, row in results_df.iterrows():
    result = graphs['cohere_rerank'].invoke({"question": row['user_input']})
    results_df.at[idx, 'response'] = result['response']
    results_df.at[idx, 'retrieved_contexts'] = [d.page_content for d in result['context']]

# Create RAGAS dataset
eval_dataset = EvaluationDataset.from_pandas(results_df)

# Run evaluation
evaluator_llm = LangchainLLMWrapper(get_llm())
run_config = RunConfig(timeout=360)

ragas_result = evaluate(
    dataset=eval_dataset,
    metrics=[
        Faithfulness(),
        ResponseRelevancy(),
        ContextPrecision(),
        LLMContextRecall()
    ],
    llm=evaluator_llm,
    run_config=run_config
)

# Save results
output_dir = Path("deliverables/evaluation_evidence")
output_dir.mkdir(parents=True, exist_ok=True)

ragas_result.to_pandas().to_csv(
    output_dir / "detailed_results.csv",
    index=False
)
```

### Error Handling

**Graceful Degradation**:
```python
import os
from src.retrievers import create_retrievers

# Check for Cohere API key
if os.getenv("COHERE_API_KEY"):
    # Use all retrievers including rerank
    retrievers = create_retrievers(documents, vector_store)
    graphs = build_all_graphs(retrievers)
else:
    print("WARNING: COHERE_API_KEY not set - skipping cohere_rerank")
    # Use only retrievers that don't need Cohere
    naive_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    from langchain_community.retrievers import BM25Retriever
    bm25_retriever = BM25Retriever.from_documents(documents, k=5)

    from src.graph import build_graph
    graphs = {
        'naive': build_graph(naive_retriever),
        'bm25': build_graph(bm25_retriever)
    }
```

**Connection Retry**:
```python
from tenacity import retry, stop_after_attempt, wait_exponential
from qdrant_client import QdrantClient

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10)
)
def connect_to_qdrant():
    client = QdrantClient(host="localhost", port=6333, timeout=5)
    client.get_collections()  # Test connection
    return client

try:
    client = connect_to_qdrant()
    print("Connected to Qdrant")
except Exception as e:
    print(f"Failed to connect to Qdrant: {e}")
    print("Please start Qdrant: docker-compose up -d qdrant")
```

**API Rate Limiting**:
```python
from tenacity import retry, retry_if_exception_type, wait_exponential
from openai import RateLimitError, APITimeoutError

@retry(
    retry=retry_if_exception_type((RateLimitError, APITimeoutError)),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    stop=stop_after_attempt(5)
)
def query_with_retry(graph, question):
    return graph.invoke({"question": question})

# Usage
try:
    result = query_with_retry(graphs['naive'], "What is GDELT?")
except Exception as e:
    print(f"Failed after retries: {e}")
```

---

## Type Reference

### State Types

**State (TypedDict)**:
```python
from typing import List
from typing_extensions import TypedDict
from langchain_core.documents import Document

class State(TypedDict):
    question: str
    context: List[Document]
    response: str
```

**Usage in Type Hints**:
```python
from src.state import State

def my_node_function(state: State) -> dict:
    """Node function type hint example"""
    # Access state
    question = state["question"]

    # Return partial update
    return {"response": "Answer here"}
```

### Return Types

**Common Return Types**:
```python
from typing import Dict, List
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langgraph.graph import CompiledGraph

# Configuration functions
def get_llm() -> ChatOpenAI: ...
def get_embeddings() -> OpenAIEmbeddings: ...
def get_qdrant() -> QdrantClient: ...
def create_vector_store(...) -> QdrantVectorStore: ...

# Retriever functions
def create_retrievers(...) -> Dict[str, object]: ...
# Returns: {"naive": Retriever, "bm25": Retriever, ...}

# Graph functions
def build_graph(...) -> CompiledGraph: ...
def build_all_graphs(...) -> Dict[str, CompiledGraph]: ...
# Returns: {"naive": CompiledGraph, "bm25": CompiledGraph, ...}

# Utility functions
def load_documents_from_huggingface(...) -> List[Document]: ...
def load_golden_testset_from_huggingface(...) -> Dataset: ...
def generate_manifest(...) -> Dict[str, Any]: ...
```

---

## Examples

### End-to-End RAG Pipeline

```python
#!/usr/bin/env python3
"""Complete RAG pipeline example"""

from src.utils import load_documents_from_huggingface
from src.config import create_vector_store
from src.retrievers import create_retrievers
from src.graph import build_all_graphs

def main():
    print("1. Loading documents...")
    documents = load_documents_from_huggingface()
    print(f"   Loaded {len(documents)} documents")

    print("\n2. Creating vector store...")
    vector_store = create_vector_store(
        documents,
        recreate_collection=True
    )
    print("   Vector store ready")

    print("\n3. Creating retrievers...")
    retrievers = create_retrievers(documents, vector_store, k=5)
    print(f"   Created {len(retrievers)} retrievers")

    print("\n4. Building graphs...")
    graphs = build_all_graphs(retrievers)
    print(f"   Built {len(graphs)} graphs")

    print("\n5. Running queries...")
    questions = [
        "What is GDELT?",
        "How does GDELT monitor news?",
        "What data sources does GDELT use?"
    ]

    for question in questions:
        print(f"\n   Q: {question}")
        result = graphs['cohere_rerank'].invoke({"question": question})
        print(f"   A: {result['response'][:150]}...")
        print(f"   Retrieved: {len(result['context'])} contexts")

if __name__ == "__main__":
    main()
```

### Custom Graph Construction

```python
"""Build custom graph with modified prompt and LLM settings"""

from src.utils import load_documents_from_huggingface
from src.config import create_vector_store
from src.graph import build_graph
from langchain_openai import ChatOpenAI

# Load data
documents = load_documents_from_huggingface()
vector_store = create_vector_store(documents)

# Create custom retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 10})

# Custom LLM with different settings
custom_llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.3,  # Slight creativity
    max_tokens=500
)

# Custom prompt template
custom_prompt = """\
You are a GDELT expert assistant. Provide detailed, technical answers.

Question: {question}

Context:
{context}

Answer:
"""

# Build custom graph
custom_graph = build_graph(
    retriever=retriever,
    llm=custom_llm,
    prompt_template=custom_prompt
)

# Use it
result = custom_graph.invoke({
    "question": "Explain GDELT's event taxonomy"
})
print(result['response'])
```

### Evaluation Harness

```python
"""Minimal evaluation harness example"""

from src.utils import (
    load_documents_from_huggingface,
    load_golden_testset_from_huggingface
)
from src.config import create_vector_store, get_llm
from src.retrievers import create_retrievers
from src.graph import build_all_graphs

from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.metrics import Faithfulness, ResponseRelevancy
from ragas.llms import LangchainLLMWrapper

# Setup
documents = load_documents_from_huggingface()
vector_store = create_vector_store(documents)
retrievers = create_retrievers(documents, vector_store)
graphs = build_all_graphs(retrievers)

# Load test set
golden_dataset = load_golden_testset_from_huggingface()
golden_df = golden_dataset.to_pandas()

# Run inference for one retriever
results_df = golden_df.copy()
results_df['response'] = None
results_df['retrieved_contexts'] = None

for idx, row in results_df.iterrows():
    result = graphs['naive'].invoke({"question": row['user_input']})
    results_df.at[idx, 'response'] = result['response']
    results_df.at[idx, 'retrieved_contexts'] = [
        d.page_content for d in result['context']
    ]

# Evaluate with RAGAS
eval_dataset = EvaluationDataset.from_pandas(results_df)
evaluator_llm = LangchainLLMWrapper(get_llm())

ragas_result = evaluate(
    dataset=eval_dataset,
    metrics=[Faithfulness(), ResponseRelevancy()],
    llm=evaluator_llm,
    run_config=RunConfig(timeout=360)
)

# Show results
print(ragas_result.to_pandas())
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue: "OPENAI_API_KEY not set"

**Symptom**: Runtime error when importing config module or calling LLM

**Solution**:
```bash
# Set in environment
export OPENAI_API_KEY="sk-..."

# Or in .env file
echo "OPENAI_API_KEY=sk-..." >> .env

# Verify
python -c "import os; print('Set' if os.getenv('OPENAI_API_KEY') else 'Not set')"
```

#### Issue: "Cannot connect to Qdrant at localhost:6333"

**Symptom**: Connection refused or timeout errors

**Solution**:
```bash
# Check if Qdrant is running
docker ps | grep qdrant

# Start Qdrant
docker-compose up -d qdrant

# Verify connectivity
curl http://localhost:6333/collections

# Check logs
docker-compose logs qdrant
```

#### Issue: "Cohere rerank retriever fails"

**Symptom**: Error when using `retrievers['cohere_rerank']`

**Solution**:
```bash
# Check if API key is set
echo $COHERE_API_KEY

# Set API key
export COHERE_API_KEY="your_key_here"

# Or skip cohere_rerank
python -c "
from src.retrievers import create_retrievers
# Use only naive, bm25, ensemble (skip cohere_rerank)
"
```

#### Issue: "Import errors from src/ modules"

**Symptom**: `ModuleNotFoundError: No module named 'src'`

**Solution**:
```bash
# Ensure you're in project root
cd /path/to/cert-challenge

# Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or use absolute imports in scripts
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

#### Issue: "Vector store has wrong dimensions"

**Symptom**: Dimension mismatch error when adding documents

**Solution**:
```python
# Recreate collection with correct dimensions
from src.config import create_vector_store
from src.utils import load_documents_from_huggingface

documents = load_documents_from_huggingface()
vector_store = create_vector_store(
    documents,
    recreate_collection=True  # Force recreation
)
```

#### Issue: "RAGAS evaluation hangs or times out"

**Symptom**: Evaluation freezes or exceeds timeout

**Solution**:
```python
# Increase timeout
from ragas import RunConfig

run_config = RunConfig(
    timeout=600,  # 10 minutes instead of 6
    max_workers=2  # Reduce parallelism
)

# Or reduce test set size
golden_df = golden_df.head(5)  # Test with fewer examples first
```

#### Issue: "Rate limit errors from OpenAI"

**Symptom**: `RateLimitError: Rate limit exceeded`

**Solution**:
```python
# Add retry logic
from tenacity import retry, wait_exponential, stop_after_attempt
from openai import RateLimitError

@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(5)
)
def query_with_retry(graph, question):
    return graph.invoke({"question": question})

# Or reduce parallelism in RAGAS
run_config = RunConfig(max_workers=1)  # Sequential evaluation
```

---

## Best Practices Summary

### Configuration Management
- Use environment variables for all secrets and configuration
- Pin dataset revisions for reproducibility (`HF_SOURCES_REV`, `HF_GOLDEN_REV`)
- Use `.env` file for local development
- Never commit API keys to version control

### Resource Initialization
- Always use factory functions (never module-level initialization)
- Cache expensive resources with `@lru_cache`
- Create resources in correct order: documents → vector store → retrievers → graphs
- Use `recreate_collection=False` in production to avoid data loss

### Error Handling
- Check for required API keys before execution
- Use tenacity for retry logic on transient errors
- Provide fallback options when optional services unavailable
- Save intermediate results early (fault tolerance)

### Evaluation
- Pin dataset versions for reproducible results
- Save RUN_MANIFEST.json for each evaluation run
- Link evaluation to data ingestion via manifest IDs
- Use deterministic settings (temperature=0) for reproducibility

### Performance
- Reuse existing Qdrant collections when possible
- Use cached LLM and embedding instances
- Parallelize RAGAS evaluation with `max_workers`
- Monitor API usage and costs

### Documentation
- Use type hints for all function signatures
- Write comprehensive docstrings with examples
- Document environment variables and defaults
- Include troubleshooting guides for common issues

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2024-01-15 | Initial API reference documentation |

---

## Additional Resources

**Related Documentation**:
- System Architecture: `01_system_architecture.md`
- Data Flow: `02_data_flow.md`
- Deployment Guide: `03_deployment.md`

**External References**:
- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [RAGAS Documentation](https://docs.ragas.io/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)

**Repository**:
- GitHub: `don-aie-cohort8/cert-challenge`
- Branch: `GDELT`

---

*This API reference is automatically maintained and updated with each release. For questions or corrections, please open an issue in the repository.*
