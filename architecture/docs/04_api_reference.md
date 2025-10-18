# API Reference

## Overview

This API reference documents all public functions, classes, and modules in the GDELT RAG Comparative Evaluation project. The project implements a Retrieval-Augmented Generation (RAG) system with multiple retrieval strategies evaluated using the RAGAS framework.

**Navigation:**
- [Core Modules](#core-modules-src) - Production code in `src/`
- [Utility Scripts](#utility-scripts-scripts) - Pipeline and deployment scripts
- [Configuration Reference](#configuration-reference) - Environment variables and settings
- [Type Definitions](#type-definitions) - TypedDict schemas and data structures
- [Usage Patterns](#usage-patterns) - Common workflows and examples
- [Best Practices](#best-practices) - Guidelines and recommendations

**Key Technologies:**
- **LangChain**: Document loading, retrieval, prompts
- **LangGraph**: State graph workflow orchestration
- **RAGAS**: RAG evaluation metrics (v0.2.10)
- **Qdrant**: Vector database for embeddings
- **OpenAI**: LLM (gpt-4.1-mini) and embeddings (text-embedding-3-small)
- **Cohere**: Reranking (rerank-v3.5)

---

## Core Modules (src/)

### Module: config.py

**Source:** `/home/donbr/don-aie-cohort8/cert-challenge/src/config.py`

Central configuration module for Qdrant connection, LLM, and embeddings.

#### Constants

##### `QDRANT_HOST`
**Source:** `src/config.py:6`

**Type:** `str`

**Value:** `"localhost"`

**Description:** Qdrant vector database host address.

---

##### `QDRANT_PORT`
**Source:** `src/config.py:7`

**Type:** `int`

**Value:** `6333`

**Description:** Qdrant vector database port number.

---

##### `COLLECTION_NAME`
**Source:** `src/config.py:8`

**Type:** `str`

**Value:** `"gdelt_comparative_eval"`

**Description:** Qdrant collection name for storing document embeddings.

---

#### Global Objects

##### `llm`
**Source:** `src/config.py:10`

**Type:** `ChatOpenAI`

**Description:** OpenAI LLM instance configured for RAG generation.

**Configuration:**
- Model: `gpt-4.1-mini`
- Temperature: `0` (deterministic outputs)

**Example:**
```python
from src.config import llm

# Use for generation
response = llm.invoke("What is GDELT?")
print(response.content)
```

---

##### `embeddings`
**Source:** `src/config.py:11`

**Type:** `OpenAIEmbeddings`

**Description:** OpenAI embeddings instance for document and query vectorization.

**Configuration:**
- Model: `text-embedding-3-small`
- Dimensions: 1536

**Example:**
```python
from src.config import embeddings

# Embed a query
query_vector = embeddings.embed_query("What are GDELT themes?")
print(f"Vector dimensions: {len(query_vector)}")

# Embed documents
docs = ["Document 1", "Document 2"]
doc_vectors = embeddings.embed_documents(docs)
```

---

### Module: state.py

**Source:** `/home/donbr/don-aie-cohort8/cert-challenge/src/state.py`

Defines the state schema for LangGraph workflows.

#### Classes

##### `State`
**Source:** `src/state.py:7-10`

**Type:** `TypedDict`

**Description:** State container for LangGraph RAG workflow. Tracks question, retrieved context, and generated response across graph nodes.

**Fields:**
- `question` (str): User input question
- `context` (List[Document]): Retrieved LangChain documents
- `response` (str): Generated answer from LLM

**Example:**
```python
from src.state import State
from langchain_core.documents import Document

# Initialize state
initial_state: State = {
    "question": "What is GDELT Translingual?",
    "context": [],
    "response": ""
}

# Update state with retrieved documents
state: State = {
    "question": "What is GDELT Translingual?",
    "context": [
        Document(page_content="GDELT Translingual...", metadata={"page": 5})
    ],
    "response": "GDELT Translingual is..."
}
```

**Notes:**
- Used across all retriever graph workflows
- Immutable between nodes (LangGraph creates new state copies)
- `context` must contain LangChain `Document` objects, not raw strings

---

### Module: prompts.py

**Source:** `/home/donbr/don-aie-cohort8/cert-challenge/src/prompts.py`

Prompt templates for RAG generation.

#### Constants

##### `BASELINE_PROMPT`
**Source:** `src/prompts.py:4-12`

**Type:** `str`

**Description:** Template for RAG answer generation. Instructs LLM to answer based only on provided context without using external knowledge.

**Template Variables:**
- `{question}`: User input question
- `{context}`: Retrieved document text (concatenated)

**Template:**
```text
You are a helpful assistant who answers questions based on provided context. You must only use the provided context, and cannot use your own knowledge.

### Question
{question}

### Context
{context}
```

**Example:**
```python
from langchain.prompts import ChatPromptTemplate
from src.prompts import BASELINE_PROMPT

# Create prompt template
rag_prompt = ChatPromptTemplate.from_template(BASELINE_PROMPT)

# Format prompt
messages = rag_prompt.format_messages(
    question="What is GDELT?",
    context="GDELT is a global database of events..."
)
```

**Notes:**
- Shared across all retriever strategies
- Temperature=0 in LLM ensures deterministic responses given same context
- Context grounding prevents hallucinations

---

### Module: retrievers.py

**Source:** `/home/donbr/don-aie-cohort8/cert-challenge/src/retrievers.py`

Defines all retrieval strategies for the RAG system.

#### Configuration Constants

##### `QDRANT_HOST`, `QDRANT_PORT`, `COLLECTION_NAME`
**Source:** `src/retrievers.py:17-19`

Same as in `config.py` (duplicated for module independence).

---

##### `embeddings`
**Source:** `src/retrievers.py:20`

OpenAI embeddings instance (same configuration as `config.py`).

---

#### Global Objects

##### `qdrant_client`
**Source:** `src/retrievers.py:23`

**Type:** `QdrantClient`

**Description:** Connected Qdrant client instance.

**Example:**
```python
from src.retrievers import qdrant_client

# Check collections
collections = qdrant_client.get_collections()
print(f"Available collections: {[c.name for c in collections.collections]}")
```

---

##### `vector_store`
**Source:** `src/retrievers.py:30-34`

**Type:** `QdrantVectorStore`

**Description:** LangChain wrapper for Qdrant vector store.

**Configuration:**
- Client: Connected `qdrant_client`
- Collection: `gdelt_comparative_eval`
- Embeddings: OpenAI text-embedding-3-small

**Example:**
```python
from src.retrievers import vector_store

# Search similar documents
results = vector_store.similarity_search("GDELT themes", k=5)
for doc in results:
    print(doc.page_content[:100])
```

---

#### Retrievers

##### `baseline_retriever`
**Source:** `src/retrievers.py:37`

**Type:** `VectorStoreRetriever`

**Description:** Naive dense vector search retriever (baseline strategy).

**Configuration:**
- Strategy: Dense vector similarity search
- k: 5 documents
- Distance: Cosine similarity

**Example:**
```python
from src.retrievers import baseline_retriever

# Retrieve documents
docs = baseline_retriever.invoke("What is GDELT Translingual?")
print(f"Retrieved {len(docs)} documents")
for doc in docs:
    print(f"- {doc.metadata.get('page', 'unknown')}: {doc.page_content[:80]}...")
```

---

##### `bm25_retriever`
**Source:** `src/retrievers.py:40`

**Type:** `BM25Retriever`

**Description:** Sparse keyword matching retriever using BM25 algorithm.

**Configuration:**
- Strategy: Lexical search (term frequency)
- k: 5 documents
- Source: Initialized from `documents` list

**Example:**
```python
from src.retrievers import bm25_retriever

# Retrieve using keyword matching
docs = bm25_retriever.invoke("JSON CSV formats GDELT")
print(f"BM25 retrieved {len(docs)} documents")
```

**Notes:**
- Requires `documents` variable to be in scope (loaded from vector store)
- Good for exact keyword matches
- Complements dense retrieval in ensemble approach

---

##### `compression_retriever`
**Source:** `src/retrievers.py:54-58`

**Type:** `ContextualCompressionRetriever`

**Description:** Cohere reranking retriever with contextual compression.

**Configuration:**
- Base retriever: Dense vector search (k=20)
- Compressor: Cohere rerank-v3.5
- Final k: 5 documents (top 5 after reranking)

**Pipeline:**
1. Retrieve 20 candidates using dense search
2. Rerank using Cohere's semantic reranking
3. Return top 5 most relevant

**Example:**
```python
from src.retrievers import compression_retriever

# Retrieve with reranking
docs = compression_retriever.invoke("Explain GDELT proximity context")
print(f"Reranked top {len(docs)} documents")
```

**Notes:**
- Requires `COHERE_API_KEY` environment variable
- More expensive (20 initial retrievals + reranking)
- Best semantic relevance among all strategies

---

##### `ensemble_retriever`
**Source:** `src/retrievers.py:46-49`

**Type:** `EnsembleRetriever`

**Description:** Hybrid retriever combining dense and sparse search.

**Configuration:**
- Components: `baseline_retriever` (dense) + `bm25_retriever` (sparse)
- Weights: [0.5, 0.5] (equal weighting)
- k: 5 documents total

**Example:**
```python
from src.retrievers import ensemble_retriever

# Hybrid retrieval
docs = ensemble_retriever.invoke("GDELT themes emotions languages")
print(f"Ensemble retrieved {len(docs)} documents")
```

**Notes:**
- Balances semantic similarity (dense) with keyword matching (sparse)
- Good general-purpose retriever
- Results are merged and deduplicated

---

### Module: graph.py

**Source:** `/home/donbr/don-aie-cohort8/cert-challenge/src/graph.py`

LangGraph workflow definitions for each retrieval strategy.

#### Configuration

##### `llm`
**Source:** `src/graph.py:16`

LLM instance for generation (same as `config.py`).

---

##### `rag_prompt`
**Source:** `src/graph.py:17`

**Type:** `ChatPromptTemplate`

ChatPromptTemplate created from `BASELINE_PROMPT`.

---

#### Retrieval Functions

##### `retrieve_baseline(state)`
**Source:** `src/graph.py:20-23`

**Description:** Naive dense vector search retrieval node.

**Parameters:**
- `state` (State): LangGraph state containing `question`

**Returns:**
- `dict`: Updated state with `context` field containing retrieved documents

**Example:**
```python
from src.graph import retrieve_baseline

state = {"question": "What is GDELT?"}
result = retrieve_baseline(state)
print(f"Retrieved {len(result['context'])} documents")
```

---

##### `retrieve_bm25(state)`
**Source:** `src/graph.py:25-28`

**Description:** BM25 sparse keyword matching retrieval node.

**Parameters:**
- `state` (State): LangGraph state containing `question`

**Returns:**
- `dict`: Updated state with `context` field

**Example:**
```python
from src.graph import retrieve_bm25

state = {"question": "GDELT JSON CSV"}
result = retrieve_bm25(state)
print(f"BM25 retrieved {len(result['context'])} documents")
```

---

##### `retrieve_reranked(state)`
**Source:** `src/graph.py:30-33`

**Description:** Cohere contextual compression with reranking retrieval node.

**Parameters:**
- `state` (State): LangGraph state containing `question`

**Returns:**
- `dict`: Updated state with `context` field

**Example:**
```python
from src.graph import retrieve_reranked

state = {"question": "Explain GDELT proximity"}
result = retrieve_reranked(state)
print(f"Reranked {len(result['context'])} documents")
```

---

##### `retrieve_ensemble(state)`
**Source:** `src/graph.py:35-38`

**Description:** Ensemble hybrid search retrieval node.

**Parameters:**
- `state` (State): LangGraph state containing `question`

**Returns:**
- `dict`: Updated state with `context` field

**Example:**
```python
from src.graph import retrieve_ensemble

state = {"question": "GDELT themes and emotions"}
result = retrieve_ensemble(state)
print(f"Ensemble retrieved {len(result['context'])} documents")
```

---

#### Generation Function

##### `generate(state)`
**Source:** `src/graph.py:41-46`

**Description:** Generate answer from retrieved context using LLM.

**Parameters:**
- `state` (State): LangGraph state containing `question` and `context`

**Returns:**
- `dict`: Updated state with `response` field containing generated answer

**Implementation:**
1. Concatenates `page_content` from all context documents
2. Formats prompt with question and concatenated context
3. Invokes LLM with formatted prompt
4. Returns LLM response content

**Example:**
```python
from src.graph import generate
from langchain_core.documents import Document

state = {
    "question": "What is GDELT?",
    "context": [
        Document(page_content="GDELT is a global database...", metadata={})
    ]
}
result = generate(state)
print(f"Generated answer: {result['response']}")
```

**Notes:**
- Context documents are joined with double newlines
- Uses `BASELINE_PROMPT` template
- Temperature=0 ensures deterministic generation

---

#### Compiled Graphs

##### `baseline_graph`
**Source:** `src/graph.py:50-52`

**Type:** `CompiledGraph`

**Description:** LangGraph workflow for baseline retrieval strategy.

**Workflow:**
```
START -> retrieve_baseline -> generate -> END
```

**Example:**
```python
from src.graph import baseline_graph

# Run complete RAG workflow
result = baseline_graph.invoke({"question": "What is GDELT?"})
print(f"Question: {result['question']}")
print(f"Retrieved {len(result['context'])} documents")
print(f"Answer: {result['response']}")
```

---

##### `bm25_graph`
**Source:** `src/graph.py:54-56`

**Type:** `CompiledGraph`

**Description:** LangGraph workflow for BM25 retrieval strategy.

**Workflow:**
```
START -> retrieve_bm25 -> generate -> END
```

---

##### `ensemble_graph`
**Source:** `src/graph.py:58-60`

**Type:** `CompiledGraph`

**Description:** LangGraph workflow for ensemble retrieval strategy.

**Workflow:**
```
START -> retrieve_ensemble -> generate -> END
```

---

##### `rerank_graph`
**Source:** `src/graph.py:62-64`

**Type:** `CompiledGraph`

**Description:** LangGraph workflow for reranked retrieval strategy.

**Workflow:**
```
START -> retrieve_reranked -> generate -> END
```

---

##### `retrievers_config`
**Source:** `src/graph.py:66-71`

**Type:** `dict[str, CompiledGraph]`

**Description:** Dictionary mapping retriever names to compiled LangGraph workflows.

**Keys:**
- `"naive"`: baseline_graph
- `"bm25"`: bm25_graph
- `"ensemble"`: ensemble_graph
- `"cohere_rerank"`: rerank_graph

**Example:**
```python
from src.graph import retrievers_config

# Iterate through all retrievers
for name, graph in retrievers_config.items():
    result = graph.invoke({"question": "What is GDELT?"})
    print(f"{name}: {result['response'][:100]}...")
```

**Notes:**
- Used by evaluation scripts to test all retrieval strategies
- All graphs share same generation function
- Names match RAGAS evaluation output

---

### Module: utils.py

**Source:** `/home/donbr/don-aie-cohort8/cert-challenge/src/utils.py`

Currently empty utility module for future helper functions.

---

## Utility Scripts (scripts/)

### Script: ingest.py

**Source:** `/home/donbr/don-aie-cohort8/cert-challenge/scripts/ingest.py`

**Purpose:** Complete data ingestion pipeline - extracts PDFs, generates RAGAS golden testset, persists to multiple formats.

**Entry Point:** Can be run as Jupyter notebook or Python script

**Key Features:**
- PDF extraction to LangChain Documents
- RAGAS testset generation (RAGAS 0.2.x and 0.3.x compatible)
- Multi-format persistence (JSONL, Parquet, HuggingFace datasets)
- Manifest generation with checksums and provenance
- Retries with exponential backoff for API calls

#### Configuration Constants

##### `QDRANT_HOST`, `QDRANT_PORT`, `COLLECTION_NAME`
**Source:** `scripts/ingest.py:17-19`

Same as core modules.

---

##### `OPENAI_MODEL`
**Source:** `scripts/ingest.py:78`

**Value:** `"gpt-4.1-mini"`

LLM for RAGAS testset generation.

---

##### `OPENAI_EMBED_MODEL`
**Source:** `scripts/ingest.py:79`

**Value:** `"text-embedding-3-small"`

Embeddings for RAGAS testset generation.

---

##### `TESTSET_SIZE`
**Source:** `scripts/ingest.py:82`

**Value:** `10`

Number of synthetic test cases to generate.

---

##### `MAX_DOCS`
**Source:** `scripts/ingest.py:83`

**Value:** `None`

Optional limit for document processing (for prototyping).

---

##### `RANDOM_SEED`
**Source:** `scripts/ingest.py:86`

**Value:** `42`

Random seed for reproducibility.

---

#### Main Functions

##### `find_repo_root(start: Path) -> Path`
**Source:** `scripts/ingest.py:48-54`

**Description:** Finds repository root by looking for `pyproject.toml` or `.git` directory.

**Parameters:**
- `start` (Path): Starting directory path

**Returns:**
- `Path`: Repository root path

**Example:**
```python
from pathlib import Path
from scripts.ingest import find_repo_root

repo_root = find_repo_root(Path.cwd())
print(f"Repository root: {repo_root}")
```

---

##### `ensure_jsonable(obj: Any) -> Any`
**Source:** `scripts/ingest.py:92-104`

**Description:** Makes metadata JSON-serializable without losing information.

**Parameters:**
- `obj` (Any): Object to convert (dict, list, primitive, or custom type)

**Returns:**
- `Any`: JSON-serializable version of input

**Example:**
```python
from pathlib import Path
from datetime import datetime
from scripts.ingest import ensure_jsonable

metadata = {
    "path": Path("/data/file.pdf"),
    "created": datetime.now(),
    "page": 5,
    "nested": {"key": "value"}
}

json_safe = ensure_jsonable(metadata)
# All Path and datetime objects converted to strings
```

**Notes:**
- Preserves nested structures (dicts, lists)
- Converts Path, UUID, datetime to strings
- Prevents Arrow/Parquet serialization errors

---

##### `docs_to_jsonl(docs: Iterable[Document], path: Path) -> int`
**Source:** `scripts/ingest.py:106-113`

**Description:** Persists LangChain documents to JSONL format.

**Parameters:**
- `docs` (Iterable[Document]): Documents to persist
- `path` (Path): Output JSONL file path

**Returns:**
- `int`: Number of documents written

**Example:**
```python
from pathlib import Path
from langchain_core.documents import Document
from scripts.ingest import docs_to_jsonl

docs = [
    Document(page_content="Text 1", metadata={"page": 1}),
    Document(page_content="Text 2", metadata={"page": 2})
]

count = docs_to_jsonl(docs, Path("data/output.jsonl"))
print(f"Wrote {count} documents")
```

---

##### `docs_to_parquet(docs: Iterable[Document], path: Path) -> int`
**Source:** `scripts/ingest.py:115-119`

**Description:** Persists LangChain documents to Parquet format.

**Parameters:**
- `docs` (Iterable[Document]): Documents to persist
- `path` (Path): Output Parquet file path

**Returns:**
- `int`: Number of documents written

**Example:**
```python
from pathlib import Path
from scripts.ingest import docs_to_parquet

count = docs_to_parquet(docs, Path("data/output.parquet"))
print(f"Wrote {count} documents to Parquet")
```

**Notes:**
- Flattens metadata into columns
- Uses pandas DataFrame intermediate format

---

##### `docs_to_hfds(docs: Iterable[Document], path: Path) -> int`
**Source:** `scripts/ingest.py:121-126`

**Description:** Persists LangChain documents to HuggingFace dataset on disk.

**Parameters:**
- `docs` (Iterable[Document]): Documents to persist
- `path` (Path): Output directory path

**Returns:**
- `int`: Number of documents written

**Example:**
```python
from pathlib import Path
from scripts.ingest import docs_to_hfds

count = docs_to_hfds(docs, Path("data/output.hfds"))
print(f"Wrote {count} documents to HF dataset")

# Later: reload
from datasets import load_from_disk
dataset = load_from_disk("data/output.hfds")
```

---

##### `hash_file(path: Path, algo: str = "sha256") -> str`
**Source:** `scripts/ingest.py:128-133`

**Description:** Computes file hash for integrity verification.

**Parameters:**
- `path` (Path): File to hash
- `algo` (str): Hash algorithm (default: "sha256")

**Returns:**
- `str`: Hexadecimal hash digest

**Example:**
```python
from pathlib import Path
from scripts.ingest import hash_file

checksum = hash_file(Path("data/sources.jsonl"))
print(f"SHA256: {checksum}")
```

---

##### `write_manifest(manifest_path: Path, payload: Dict[str, Any]) -> None`
**Source:** `scripts/ingest.py:135-136`

**Description:** Writes manifest JSON file.

**Parameters:**
- `manifest_path` (Path): Output manifest file path
- `payload` (Dict[str, Any]): Manifest data

**Example:**
```python
from pathlib import Path
from scripts.ingest import write_manifest

manifest = {
    "id": "run_123",
    "generated_at": "2025-01-01T00:00:00Z",
    "paths": {...}
}

write_manifest(Path("data/manifest.json"), manifest)
```

---

##### `summarize_columns_from_jsonl(path: Path, sample_n: int = 5) -> Dict[str, Any]`
**Source:** `scripts/ingest.py:138-150`

**Description:** Extracts column schema and samples from JSONL file.

**Parameters:**
- `path` (Path): JSONL file path
- `sample_n` (int): Number of sample rows to include (default: 5)

**Returns:**
- `Dict[str, Any]`: Dictionary with `columns` (list) and `sample` (list)

**Example:**
```python
from pathlib import Path
from scripts.ingest import summarize_columns_from_jsonl

schema = summarize_columns_from_jsonl(Path("data/sources.jsonl"), sample_n=3)
print(f"Columns: {schema['columns']}")
print(f"Sample count: {len(schema['sample'])}")
```

---

##### `build_testset(docs, size: int)`
**Source:** `scripts/ingest.py:202-229`

**Description:** Generates RAGAS testset with automatic version detection (0.2.x vs 0.3.x).

**Parameters:**
- `docs` (List[Document]): Source documents for testset generation
- `size` (int): Number of test cases to generate

**Returns:**
- `TestSet`: RAGAS testset object

**Implementation:**
- Tries RAGAS 0.3.x `generate()` API first
- Falls back to RAGAS 0.2.x `TestsetGenerator` API
- Includes retry logic with exponential backoff for API errors

**Example:**
```python
from scripts.ingest import build_testset

testset = build_testset(documents, size=12)
print(f"Generated {len(testset)} test cases")

# Save to JSONL
testset.to_jsonl("data/golden_testset.jsonl")
```

**Notes:**
- Requires `OPENAI_API_KEY` environment variable
- Retries up to 3 times on rate limits
- Supports both LangChain Document input

---

#### Workflow

**Pipeline Execution:**

```python
# 1. Load PDFs
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader

loader = DirectoryLoader("data/raw", glob="*.pdf", loader_cls=PyMuPDFLoader)
docs = loader.load()

# 2. Persist sources
docs_to_jsonl(docs, Path("data/interim/sources.jsonl"))
docs_to_parquet(docs, Path("data/interim/sources.parquet"))
docs_to_hfds(docs, Path("data/interim/sources.hfds"))

# 3. Generate testset
golden_testset = build_testset(docs, size=12)

# 4. Persist testset
golden_testset.to_jsonl("data/interim/golden_testset.jsonl")
golden_testset.to_hf_dataset().save_to_disk("data/interim/golden_testset.hfds")

# 5. Generate manifest
manifest = {
    "id": "pipeline_run_123",
    "generated_at": datetime.utcnow().isoformat() + "Z",
    "paths": {...},
    "fingerprints": {...}
}
write_manifest(Path("data/interim/manifest.json"), manifest)
```

**Output Structure:**
```
data/interim/
├── sources.jsonl              # Source documents (JSONL)
├── sources.parquet            # Source documents (Parquet)
├── sources.hfds/              # Source documents (HF dataset)
├── golden_testset.jsonl       # Test cases (JSONL)
├── golden_testset.parquet     # Test cases (Parquet)
├── golden_testset.hfds/       # Test cases (HF dataset)
└── manifest.json              # Provenance and checksums
```

---

### Script: single_file.py

**Source:** `/home/donbr/don-aie-cohort8/cert-challenge/scripts/single_file.py`

**Purpose:** End-to-end RAG evaluation script with RAGAS metrics for multiple retrieval strategies.

**Entry Point:**
```bash
python scripts/single_file.py
```

**Key Features:**
- Loads golden testset from HuggingFace
- Loads source documents from HuggingFace
- Creates Qdrant vector store
- Builds all 4 retrieval strategies
- Runs RAGAS evaluation (faithfulness, answer_relevancy, context_precision, context_recall)
- Generates comparative results table
- Saves all results and manifests

#### Helper Functions

##### `validate_and_normalize_ragas_schema(df: pd.DataFrame, retriever_name: str = "unknown") -> pd.DataFrame`
**Source:** `scripts/single_file.py:66-125`

**Description:** Ensures DataFrame matches RAGAS 0.2.10 schema requirements.

**Parameters:**
- `df` (pd.DataFrame): DataFrame to validate
- `retriever_name` (str): Retriever name for logging (default: "unknown")

**Returns:**
- `pd.DataFrame`: Validated DataFrame with normalized column names

**Raises:**
- `ValueError`: If required columns are missing or have wrong types

**Column Mapping:**
```python
rename_map = {
    'question': 'user_input',
    'answer': 'response',
    'contexts': 'retrieved_contexts',
    'ground_truth': 'reference',
    'ground_truths': 'reference',
    'reference_contexts': 'reference',
}
```

**Expected Columns:**
- `user_input` (str): Question
- `response` (str): Generated answer
- `retrieved_contexts` (list[str]): Retrieved document texts
- `reference` (str): Ground truth answer

**Example:**
```python
import pandas as pd
from scripts.single_file import validate_and_normalize_ragas_schema

# DataFrame with alternate column names
df = pd.DataFrame({
    'question': ['What is GDELT?'],
    'answer': ['GDELT is...'],
    'contexts': [['Context 1', 'Context 2']],
    'ground_truth': ['Ground truth answer']
})

# Normalize
validated_df = validate_and_normalize_ragas_schema(df, 'naive')
print(validated_df.columns)
# Output: ['user_input', 'response', 'retrieved_contexts', 'reference']
```

**Notes:**
- Critical for RAGAS evaluation compatibility
- Prevents silent failures from schema mismatches
- Validates both column names and data types

---

#### Workflow

**Complete Evaluation Pipeline:**

```python
# Configuration
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "gdelt_comparative_eval"

# 1. Load golden testset
from datasets import load_dataset
golden_dataset = load_dataset("dwb2023/gdelt-rag-golden-testset", split="train")
golden_df = golden_dataset.to_pandas()

# 2. Load source documents
sources_dataset = load_dataset("dwb2023/gdelt-rag-sources", split="train")
documents = [
    Document(page_content=item["page_content"], metadata=item.get("metadata", {}))
    for item in sources_dataset
]

# 3. Create Qdrant vector store
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore

qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings
)
vector_store.add_documents(documents)

# 4. Create retrievers
baseline_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
bm25_retriever = BM25Retriever.from_documents(documents, k=5)
# ... (compression, ensemble)

# 5. Build LangGraph workflows
from langgraph.graph import START, StateGraph

baseline_graph = StateGraph(State).add_sequence([retrieve_baseline, generate])
baseline_graph.add_edge(START, "retrieve_baseline")
baseline_graph = baseline_graph.compile()
# ... (other graphs)

# 6. Run questions through all retrievers
for retriever_name, graph in retrievers_config.items():
    for idx, row in datasets[retriever_name].iterrows():
        result = graph.invoke({"question": row['user_input']})
        datasets[retriever_name].at[idx, 'response'] = result['response']
        datasets[retriever_name].at[idx, 'retrieved_contexts'] = [
            doc.page_content for doc in result['context']
        ]

# 7. Create RAGAS EvaluationDatasets
from ragas import EvaluationDataset

evaluation_datasets = {}
for retriever_name, dataset in datasets.items():
    validated = validate_and_normalize_ragas_schema(dataset, retriever_name)
    evaluation_datasets[retriever_name] = EvaluationDataset.from_pandas(validated)

# 8. Run RAGAS evaluation
from ragas import evaluate
from ragas.metrics import Faithfulness, ResponseRelevancy, ContextPrecision, LLMContextRecall

evaluation_results = {}
for retriever_name, eval_dataset in evaluation_datasets.items():
    result = evaluate(
        dataset=eval_dataset,
        metrics=[
            Faithfulness(),
            ResponseRelevancy(),
            ContextPrecision(),
            LLMContextRecall(),
        ],
        llm=evaluator_llm,
        run_config=RunConfig(timeout=360)
    )
    evaluation_results[retriever_name] = result

# 9. Generate comparative table
comparison_data = []
for retriever_name, result in evaluation_results.items():
    df = result.to_pandas()
    comparison_data.append({
        'Retriever': retriever_name,
        'Faithfulness': df['faithfulness'].mean(),
        'Answer Relevancy': df['answer_relevancy'].mean(),
        'Context Precision': df['context_precision'].mean(),
        'Context Recall': df['context_recall'].mean(),
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv("comparative_ragas_results.csv", index=False)
```

**Output Files:**
```
deliverables/evaluation_evidence/
├── comparative_ragas_results.csv          # Summary table
├── naive_evaluation_dataset.csv           # Full dataset per retriever
├── naive_detailed_results.csv             # Per-question metrics
├── bm25_evaluation_dataset.csv
├── bm25_detailed_results.csv
├── ensemble_evaluation_dataset.csv
├── ensemble_detailed_results.csv
├── cohere_rerank_evaluation_dataset.csv
├── cohere_rerank_detailed_results.csv
└── RUN_MANIFEST.json                      # Reproducibility manifest
```

---

### Script: upload_to_hf.py

**Source:** `/home/donbr/don-aie-cohort8/cert-challenge/scripts/upload_to_hf.py`

**Purpose:** Upload datasets to HuggingFace Hub with dataset cards.

**Entry Point:**
```bash
export HF_TOKEN=your_token_here
python scripts/upload_to_hf.py
```

#### Configuration

##### `HF_USERNAME`
**Source:** `scripts/upload_to_hf.py:21`

**Value:** `"dwb2023"`

HuggingFace username for dataset repos.

---

##### `SOURCES_DATASET_NAME`
**Source:** `scripts/upload_to_hf.py:22`

**Value:** `"dwb2023/gdelt-rag-sources"`

HuggingFace dataset repo for source documents.

---

##### `GOLDEN_TESTSET_NAME`
**Source:** `scripts/upload_to_hf.py:23`

**Value:** `"dwb2023/gdelt-rag-golden-testset"`

HuggingFace dataset repo for golden testset.

---

#### Functions

##### `create_sources_card() -> str`
**Source:** `scripts/upload_to_hf.py:34-110`

**Description:** Generates dataset card markdown for source documents.

**Returns:**
- `str`: Dataset card markdown with metadata header and documentation

**Example:**
```python
from scripts.upload_to_hf import create_sources_card

card = create_sources_card()
print(card[:200])  # Preview
```

**Card Contents:**
- License: Apache 2.0
- Task categories: text-retrieval, question-answering
- Dataset description and summary
- Data fields documentation
- Source data citation
- Usage examples

---

##### `create_golden_testset_card() -> str`
**Source:** `scripts/upload_to_hf.py:113-191`

**Description:** Generates dataset card markdown for golden testset.

**Returns:**
- `str`: Dataset card markdown

**Card Contents:**
- License: Apache 2.0
- Task categories: question-answering, text-generation
- RAGAS framework description
- Data fields documentation
- Evaluation metrics documentation
- Intended use cases

---

##### `load_manifest()`
**Source:** `scripts/upload_to_hf.py:194-197`

**Description:** Loads manifest.json file.

**Returns:**
- `dict`: Manifest data

---

##### `update_manifest(sources_repo: str, golden_testset_repo: str)`
**Source:** `scripts/upload_to_hf.py:200-216`

**Description:** Updates manifest with uploaded dataset repo IDs.

**Parameters:**
- `sources_repo` (str): Source dataset repo ID
- `golden_testset_repo` (str): Golden testset repo ID

**Updates:**
- `lineage.hf.dataset_repo_id`
- `lineage.hf.pending_upload` (set to False)
- `lineage.hf.uploaded_at` (ISO timestamp)

---

##### `main()`
**Source:** `scripts/upload_to_hf.py:219-292`

**Description:** Main upload workflow.

**Workflow:**
1. Check `HF_TOKEN` environment variable
2. Login to HuggingFace Hub
3. Load datasets from disk
4. Upload sources dataset
5. Upload sources dataset card
6. Upload golden testset dataset
7. Upload golden testset dataset card
8. Update manifest.json
9. Display success message with URLs

**Example:**
```bash
# Set token
export HF_TOKEN=hf_your_token_here

# Run upload
python scripts/upload_to_hf.py

# Output:
# 🔐 Logging in to Hugging Face...
# 📂 Loading datasets from data/interim...
# 📤 Uploading sources dataset to dwb2023/gdelt-rag-sources...
# ✅ Sources dataset uploaded successfully!
# ...
```

**Notes:**
- Requires `HF_TOKEN` environment variable
- Datasets are public by default (`private=False`)
- Creates comprehensive dataset cards automatically
- Updates manifest for lineage tracking

---

### Script: generate_run_manifest.py

**Source:** `/home/donbr/don-aie-cohort8/cert-challenge/scripts/generate_run_manifest.py`

**Purpose:** Generate reproducibility manifest capturing exact evaluation configuration.

**Entry Point:**
```bash
python scripts/generate_run_manifest.py
```

Or programmatically:
```python
from scripts.generate_run_manifest import generate_manifest

manifest = generate_manifest(
    output_path=Path("RUN_MANIFEST.json"),
    evaluation_results=evaluation_results,
    retrievers_config=retrievers_config
)
```

#### Functions

##### `generate_manifest(output_path: Path, evaluation_results: Optional[Dict[str, Any]] = None, retrievers_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]`
**Source:** `scripts/generate_run_manifest.py:21-170`

**Description:** Generates comprehensive reproducibility manifest.

**Parameters:**
- `output_path` (Path): Path to save manifest JSON
- `evaluation_results` (Optional[Dict[str, Any]]): RAGAS evaluation results (default: None)
- `retrievers_config` (Optional[Dict[str, Any]]): Retriever configurations (default: None)

**Returns:**
- `Dict[str, Any]`: Complete manifest dictionary

**Manifest Structure:**
```python
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
            "k": 5,
            "distance_metric": "cosine",
            "rerank": False
        },
        # ... (bm25, cohere_rerank, ensemble)
    ],
    "evaluation": {
        "golden_testset": "dwb2023/gdelt-rag-golden-testset",
        "golden_testset_size": 12,
        "source_dataset": "dwb2023/gdelt-rag-sources",
        "source_dataset_size": 38,
        "metrics": ["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
        "timeout_seconds": 360
    },
    "vector_store": {
        "type": "qdrant",
        "collection_name": "gdelt_comparative_eval",
        "host": "localhost",
        "port": 6333,
        "distance": "cosine",
        "vector_size": 1536
    },
    "results_summary": {
        "naive": {
            "faithfulness": 0.95,
            "answer_relevancy": 0.92,
            "context_precision": 0.88,
            "context_recall": 0.90,
            "average": 0.9125
        }
        # ... (other retrievers)
    },
    "generated_at": "2025-01-17T12:00:00Z",
    "generated_by": "scripts/generate_run_manifest.py"
}
```

**Example:**
```python
from pathlib import Path
from scripts.generate_run_manifest import generate_manifest

# Standalone (no results)
manifest = generate_manifest(Path("RUN_MANIFEST.json"))

# With evaluation results
manifest = generate_manifest(
    output_path=Path("output/RUN_MANIFEST.json"),
    evaluation_results=evaluation_results,
    retrievers_config=retrievers_config
)

print(f"RAGAS version: {manifest['ragas_version']}")
print(f"LLM: {manifest['llm']['model']}")
print(f"Retrievers: {len(manifest['retrievers'])}")
```

**Notes:**
- Captures all configuration for exact reproduction
- Optionally includes evaluation result summaries
- Documents all model versions and parameters
- Includes retriever-specific configurations

---

### Script: enrich_manifest.py

**Source:** `/home/donbr/don-aie-cohort8/cert-challenge/scripts/enrich_manifest.py`

**Purpose:** Enrich manifest.json with file checksums, row counts, schemas, and provenance metadata.

**Entry Point:**
```bash
python scripts/enrich_manifest.py [manifest_path]

# Default path: data/interim/manifest.json
python scripts/enrich_manifest.py
```

#### Functions

##### `sha256(path: Path) -> str`
**Source:** `scripts/enrich_manifest.py:7-12`

**Description:** Computes SHA256 hash of file.

**Parameters:**
- `path` (Path): File to hash

**Returns:**
- `str`: Hexadecimal hash digest

**Example:**
```python
from pathlib import Path
from scripts.enrich_manifest import sha256

checksum = sha256(Path("data/sources.jsonl"))
print(f"SHA256: {checksum}")
```

---

##### `file_bytes(path: Path) -> int`
**Source:** `scripts/enrich_manifest.py:15-16`

**Description:** Returns file size in bytes.

**Parameters:**
- `path` (Path): File path

**Returns:**
- `int`: File size in bytes

---

##### `count_jsonl_rows(path: Path) -> int`
**Source:** `scripts/enrich_manifest.py:19-21`

**Description:** Counts rows in JSONL file.

**Parameters:**
- `path` (Path): JSONL file path

**Returns:**
- `int`: Number of rows

---

##### `hfds_rows(path: Path) -> int`
**Source:** `scripts/enrich_manifest.py:24-34`

**Description:** Counts rows in HuggingFace dataset on disk.

**Parameters:**
- `path` (Path): HF dataset directory path

**Returns:**
- `int`: Number of rows (or None on error)

**Example:**
```python
from pathlib import Path
from scripts.enrich_manifest import hfds_rows

rows = hfds_rows(Path("data/interim/sources.hfds"))
print(f"Dataset contains {rows} rows")
```

---

##### `parquet_rows(path: Path) -> int`
**Source:** `scripts/enrich_manifest.py:37-48`

**Description:** Counts rows in Parquet file.

**Parameters:**
- `path` (Path): Parquet file path

**Returns:**
- `int`: Number of rows (or None on error)

**Implementation:**
- Tries PyArrow first (fast)
- Falls back to Pandas

---

##### `pandas_schema_from_parquet(path: Path)`
**Source:** `scripts/enrich_manifest.py:51-58`

**Description:** Extracts column schema from Parquet file.

**Parameters:**
- `path` (Path): Parquet file path

**Returns:**
- `list[dict]`: List of column definitions with name and dtype

**Example:**
```python
from scripts.enrich_manifest import pandas_schema_from_parquet

schema = pandas_schema_from_parquet(Path("data/sources.parquet"))
for col in schema:
    print(f"{col['name']}: {col['dtype']}")
```

---

##### `char_stats_jsonl(path: Path, field="page_content", max_scan=5000)`
**Source:** `scripts/enrich_manifest.py:61-78`

**Description:** Computes character statistics for JSONL field.

**Parameters:**
- `path` (Path): JSONL file path
- `field` (str): Field name to analyze (default: "page_content")
- `max_scan` (int): Maximum rows to scan (default: 5000)

**Returns:**
- `dict`: Statistics with `avg_chars`, `max_chars`, `scanned`

**Example:**
```python
from scripts.enrich_manifest import char_stats_jsonl

stats = char_stats_jsonl(Path("data/sources.jsonl"), "page_content")
print(f"Average: {stats['avg_chars']} chars")
print(f"Maximum: {stats['max_chars']} chars")
print(f"Scanned: {stats['scanned']} documents")
```

---

##### `main(manifest_path: Path)`
**Source:** `scripts/enrich_manifest.py:81-236`

**Description:** Main enrichment workflow.

**Parameters:**
- `manifest_path` (Path): Path to manifest.json file

**Enrichments:**
1. **Environment**: Python, OS, package versions
2. **Inputs**: Source directory detection
3. **Artifacts**: File bytes, checksums, row counts, schemas
4. **Metrics**: Document statistics, reference context stats
5. **Lineage**: HuggingFace, LangSmith, Phoenix scaffolding
6. **Compliance**: License, PII policy
7. **Run details**: Random seed, git commit SHA
8. **Path relativization**: Convert absolute paths to relative

**Example:**
```bash
# Enrich default manifest
python scripts/enrich_manifest.py

# Enrich custom manifest
python scripts/enrich_manifest.py data/custom/manifest.json
```

**Output:**
```
✅ Enriched manifest written: data/interim/manifest.json
```

**Enriched Manifest Structure:**
```json
{
  "id": "ragas_pipeline_abc123",
  "generated_at": "2025-01-17T12:00:00Z",
  "env": {
    "python": "3.11.5",
    "os": "Linux 6.6.87",
    "langchain": "0.3.19",
    "ragas": "0.2.10",
    "datasets": "3.2.0"
  },
  "artifacts": {
    "sources": {
      "jsonl": {"path": "...", "bytes": 12345, "sha256": "abc..."},
      "parquet": {"path": "...", "bytes": 8900, "sha256": "def...", "rows": 38},
      "hfds": {"path": "...", "rows": 38}
    },
    "golden_testset": {
      "jsonl": {"path": "...", "bytes": 5678, "sha256": "ghi..."},
      "parquet": {"path": "...", "bytes": 4500, "sha256": "jkl...", "rows": 12},
      "hfds": {"path": "...", "rows": 12}
    }
  },
  "metrics": {
    "sources": {
      "docs": 38,
      "page_content_stats": {"avg_chars": 2500, "max_chars": 5000, "scanned": 38}
    },
    "golden_testset": {
      "rows": 12,
      "avg_reference_contexts": 1.67
    }
  },
  "lineage": {
    "hf": {"dataset_repo_id": null, "pending_upload": true},
    "langsmith": {"project": null, "dataset_name": null},
    "phoenix": {"workspace": null, "dataset_name": null}
  },
  "compliance": {
    "license": "apache-2.0",
    "pii_present": "unknown",
    "pii_policy": "manual-review-before-publish"
  },
  "run": {
    "random_seed": 42,
    "git_commit_sha": "a0d4dd3"
  }
}
```

---

## Configuration Reference

### Environment Variables

All environment variables should be set in `.env` file or exported in shell.

#### Required Variables

##### `OPENAI_API_KEY`
**Description:** OpenAI API key for LLM and embeddings

**Used By:**
- All LLM calls (gpt-4.1-mini)
- All embedding calls (text-embedding-3-small)
- RAGAS evaluation

**Example:**
```bash
export OPENAI_API_KEY=sk-proj-...
```

---

##### `COHERE_API_KEY`
**Description:** Cohere API key for reranking

**Used By:**
- Compression retriever (rerank-v3.5)

**Example:**
```bash
export COHERE_API_KEY=...
```

---

##### `HF_TOKEN`
**Description:** HuggingFace access token

**Used By:**
- `upload_to_hf.py` script
- Dataset uploads to HuggingFace Hub

**Example:**
```bash
export HF_TOKEN=hf_...
```

---

#### Optional Variables

##### `TAVILY_API_KEY`
**Description:** Tavily API key (not currently used but in .env.example)

**Example:**
```bash
export TAVILY_API_KEY=...
```

---

##### `LANGSMITH_PROJECT`
**Description:** LangSmith project name for tracing

**Default:** `"certification-challenge"`

**Example:**
```bash
export LANGSMITH_PROJECT=certification-challenge
```

---

##### `LANGSMITH_TRACING`
**Description:** Enable LangSmith tracing

**Default:** `"true"`

**Example:**
```bash
export LANGSMITH_TRACING=true
```

---

##### `LANGSMITH_API_KEY`
**Description:** LangSmith API key

**Example:**
```bash
export LANGSMITH_API_KEY=...
```

---

##### `OPENAI_MODEL_NAME`
**Description:** OpenAI model name override

**Default:** `"gpt-4.1-mini"`

**Example:**
```bash
export OPENAI_MODEL_NAME=gpt-4o-mini
```

---

##### `EMBEDDING_MODEL_NAME`
**Description:** OpenAI embeddings model override

**Default:** `"text-embedding-3-small"`

**Example:**
```bash
export EMBEDDING_MODEL_NAME=text-embedding-3-large
```

---

##### `COHERE_RERANK_MODEL`
**Description:** Cohere rerank model name

**Default:** `"rerank-english-v3.0"` (code uses `"rerank-v3.5"`)

**Example:**
```bash
export COHERE_RERANK_MODEL=rerank-v3.5
```

---

##### `QDRANT_URL`
**Description:** Qdrant connection URL

**Default:** `"http://localhost:6333"`

**Example:**
```bash
export QDRANT_URL=http://localhost:6333
```

---

##### `HF_HUB_DISABLE_PROGRESS_BARS`
**Description:** Disable HuggingFace Hub progress bars

**Default:** `"1"` (disabled in scripts)

---

##### `HF_DATASETS_DISABLE_PROGRESS_BARS`
**Description:** Disable HuggingFace datasets progress bars

**Default:** `"1"` (disabled in scripts)

---

### Model Configuration

#### LLM Configuration

**Model:** `gpt-4.1-mini`

**Parameters:**
- `temperature`: 0 (deterministic)
- `timeout`: 60 seconds (in ingest.py)
- `max_retries`: 6 (in ingest.py)

**Usage:**
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    timeout=60,
    max_retries=6
)
```

---

#### Embeddings Configuration

**Model:** `text-embedding-3-small`

**Parameters:**
- Dimensions: 1536
- `timeout`: 60 seconds (in ingest.py)
- `max_retries`: 6 (in ingest.py)

**Usage:**
```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    timeout=60,
    max_retries=6
)
```

---

#### Reranker Configuration

**Model:** Cohere `rerank-v3.5`

**Usage:**
```python
from langchain_cohere import CohereRerank

compressor = CohereRerank(model="rerank-v3.5")
```

---

### Qdrant Configuration

**Host:** `localhost`
**Port:** `6333`
**Collection:** `gdelt_comparative_eval`

**Vector Configuration:**
- Size: 1536 (matches text-embedding-3-small)
- Distance: Cosine similarity

**Usage:**
```python
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

client = QdrantClient(host="localhost", port=6333)

# Create collection
client.create_collection(
    collection_name="gdelt_comparative_eval",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)
```

---

### Retriever Configurations

#### Naive (Baseline)
- **Type:** Dense vector search
- **k:** 5
- **Distance:** Cosine similarity
- **Rerank:** No

---

#### BM25
- **Type:** Sparse keyword matching
- **k:** 5
- **Algorithm:** BM25
- **Rerank:** No

---

#### Cohere Rerank
- **Type:** Contextual compression
- **Initial k:** 20
- **Top n:** 5 (after reranking)
- **Rerank model:** rerank-v3.5
- **Rerank:** Yes

---

#### Ensemble
- **Type:** Hybrid (dense + sparse)
- **Components:** Naive + BM25
- **Weights:** [0.5, 0.5]
- **k:** 5 (combined)
- **Rerank:** No

---

### RAGAS Configuration

**Version:** 0.2.10

**Metrics:**
- Faithfulness
- Answer Relevancy (ResponseRelevancy)
- Context Precision
- Context Recall (LLMContextRecall)

**Run Configuration:**
- Timeout: 360 seconds
- Max workers: 4 (in RUN_MANIFEST.json)

**Usage:**
```python
from ragas import evaluate, RunConfig
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    ContextPrecision,
    LLMContextRecall
)

result = evaluate(
    dataset=evaluation_dataset,
    metrics=[
        Faithfulness(),
        ResponseRelevancy(),
        ContextPrecision(),
        LLMContextRecall()
    ],
    llm=evaluator_llm,
    run_config=RunConfig(timeout=360)
)
```

---

## Type Definitions

### State TypedDict

**Source:** `src/state.py:7-10`

```python
from typing import List
from typing_extensions import TypedDict
from langchain_core.documents import Document

class State(TypedDict):
    question: str
    context: List[Document]
    response: str
```

**Fields:**

#### `question`
**Type:** `str`

**Description:** User input question to be answered by the RAG system.

**Example:**
```python
state: State = {
    "question": "What is GDELT Translingual?",
    "context": [],
    "response": ""
}
```

---

#### `context`
**Type:** `List[Document]`

**Description:** Retrieved LangChain Document objects containing relevant context.

**Example:**
```python
from langchain_core.documents import Document

state: State = {
    "question": "What is GDELT?",
    "context": [
        Document(
            page_content="GDELT is a global database...",
            metadata={"page": 5, "title": "GDELT Paper"}
        ),
        Document(
            page_content="The GDELT Project monitors...",
            metadata={"page": 6, "title": "GDELT Paper"}
        )
    ],
    "response": ""
}
```

---

#### `response`
**Type:** `str`

**Description:** Generated answer from the LLM based on retrieved context.

**Example:**
```python
state: State = {
    "question": "What is GDELT?",
    "context": [...],
    "response": "GDELT (Global Database of Events, Language, and Tone) is a comprehensive database that monitors world events..."
}
```

---

### RAGAS Schema

**Expected DataFrame Schema for RAGAS 0.2.10:**

```python
{
    'user_input': str,           # Question
    'response': str,              # Generated answer
    'retrieved_contexts': List[str],  # Retrieved document texts
    'reference': str              # Ground truth answer
}
```

**Alternate Column Names (auto-normalized):**
- `question` → `user_input`
- `answer` → `response`
- `contexts` → `retrieved_contexts`
- `ground_truth` → `reference`
- `ground_truths` → `reference`
- `reference_contexts` → `reference`

---

## Usage Patterns

### Pattern 1: Running RAG Evaluation

**Complete end-to-end evaluation workflow:**

```python
# 1. Set environment variables
import os
os.environ["OPENAI_API_KEY"] = "sk-proj-..."
os.environ["COHERE_API_KEY"] = "..."

# 2. Run the evaluation script
# Option A: Command line
# $ python scripts/single_file.py

# Option B: Programmatic
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd() / "scripts"))

# Import and run
# (single_file.py runs automatically when imported as __main__)
```

**Step-by-step manual evaluation:**

```python
from pathlib import Path
from datasets import load_dataset
from langchain_core.documents import Document
from src.graph import retrievers_config
from ragas import EvaluationDataset, evaluate, RunConfig
from ragas.metrics import Faithfulness, ResponseRelevancy, ContextPrecision, LLMContextRecall
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
import pandas as pd

# 1. Load golden testset
golden_dataset = load_dataset("dwb2023/gdelt-rag-golden-testset", split="train")
golden_df = golden_dataset.to_pandas()

# 2. Initialize evaluator LLM
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-mini", temperature=0))

# 3. Run questions through retriever
retriever_name = "naive"
graph = retrievers_config[retriever_name]

results = []
for _, row in golden_df.iterrows():
    question = row['user_input']
    result = graph.invoke({"question": question})

    results.append({
        'user_input': question,
        'response': result['response'],
        'retrieved_contexts': [doc.page_content for doc in result['context']],
        'reference': row['reference']
    })

# 4. Create evaluation dataset
eval_df = pd.DataFrame(results)
eval_dataset = EvaluationDataset.from_pandas(eval_df)

# 5. Run RAGAS evaluation
evaluation_result = evaluate(
    dataset=eval_dataset,
    metrics=[
        Faithfulness(),
        ResponseRelevancy(),
        ContextPrecision(),
        LLMContextRecall()
    ],
    llm=evaluator_llm,
    run_config=RunConfig(timeout=360)
)

# 6. View results
results_df = evaluation_result.to_pandas()
print(results_df[['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']].mean())
```

---

### Pattern 2: Ingesting Documents

**Complete ingestion pipeline:**

```python
# 1. Run the ingest script
# $ python scripts/ingest.py

# Or in notebook/REPL:
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from scripts.ingest import (
    docs_to_jsonl,
    docs_to_parquet,
    docs_to_hfds,
    build_testset,
    write_manifest
)
import uuid
from datetime import datetime

# 2. Load PDFs
project_root = Path.cwd()
raw_path = project_root / "data" / "raw"
interim_path = project_root / "data" / "interim"

loader = DirectoryLoader(str(raw_path), glob="*.pdf", loader_cls=PyMuPDFLoader)
docs = loader.load()

print(f"Loaded {len(docs)} documents")

# 3. Persist source documents
docs_to_jsonl(docs, interim_path / "sources.jsonl")
docs_to_parquet(docs, interim_path / "sources.parquet")
docs_to_hfds(docs, interim_path / "sources.hfds")

# 4. Generate RAGAS testset
golden_testset = build_testset(docs, size=12)

# 5. Persist testset
golden_testset.to_jsonl(str(interim_path / "golden_testset.jsonl"))
golden_testset.to_hf_dataset().save_to_disk(str(interim_path / "golden_testset.hfds"))

# 6. Create manifest
manifest = {
    "id": f"ragas_pipeline_{uuid.uuid4()}",
    "generated_at": datetime.utcnow().isoformat() + "Z",
    "paths": {
        "sources": {
            "jsonl": str(interim_path / "sources.jsonl"),
            "parquet": str(interim_path / "sources.parquet"),
            "hfds": str(interim_path / "sources.hfds")
        },
        "golden_testset": {
            "jsonl": str(interim_path / "golden_testset.jsonl"),
            "hfds": str(interim_path / "golden_testset.hfds")
        }
    }
}

write_manifest(interim_path / "manifest.json", manifest)

# 7. Enrich manifest
from scripts.enrich_manifest import main as enrich_main
enrich_main(interim_path / "manifest.json")
```

---

### Pattern 3: Uploading to HuggingFace

**Upload datasets with dataset cards:**

```python
# 1. Set HuggingFace token
import os
os.environ["HF_TOKEN"] = "hf_..."

# 2. Run upload script
# $ python scripts/upload_to_hf.py

# Or programmatically:
from scripts.upload_to_hf import main as upload_main

upload_main()

# Output:
# 🔐 Logging in to Hugging Face...
# 📂 Loading datasets from data/interim...
# 📤 Uploading sources dataset to dwb2023/gdelt-rag-sources...
# ✅ Sources dataset uploaded successfully!
# ...
```

**Custom upload workflow:**

```python
from pathlib import Path
from datasets import load_from_disk
from huggingface_hub import HfApi, login

# 1. Login
hf_token = os.environ["HF_TOKEN"]
login(token=hf_token)

# 2. Load dataset
dataset = load_from_disk("data/interim/sources.hfds")

# 3. Upload
dataset.push_to_hub(
    "username/dataset-name",
    private=False,
    token=hf_token
)

# 4. Upload dataset card
api = HfApi()
card_content = """---
license: apache-2.0
task_categories:
- text-retrieval
---

# Dataset Card
...
"""

api.upload_file(
    path_or_fileobj=card_content.encode(),
    path_in_repo="README.md",
    repo_id="username/dataset-name",
    repo_type="dataset",
    token=hf_token
)
```

---

### Pattern 4: Custom Retriever Implementation

**Add a new retriever to the system:**

```python
# 1. Create retriever in src/retrievers.py
from langchain.retrievers import ParentDocumentRetriever

# Add to src/retrievers.py:
parent_doc_retriever = ParentDocumentRetriever(
    vectorstore=vector_store,
    docstore=...,
    child_splitter=...,
    k=5
)

# 2. Create retrieval function in src/graph.py
def retrieve_parent_doc(state):
    """Parent document retrieval"""
    retrieved_docs = parent_doc_retriever.invoke(state["question"])
    return {"context": retrieved_docs}

# 3. Build LangGraph workflow
parent_doc_graph_builder = StateGraph(State).add_sequence([retrieve_parent_doc, generate])
parent_doc_graph_builder.add_edge(START, "retrieve_parent_doc")
parent_doc_graph = parent_doc_graph_builder.compile()

# 4. Add to retrievers_config
retrievers_config["parent_doc"] = parent_doc_graph

# 5. Now it will be included in evaluations automatically
```

---

### Pattern 5: Querying the RAG System

**Simple question answering:**

```python
from src.graph import baseline_graph

# Ask a question
result = baseline_graph.invoke({
    "question": "What is GDELT Translingual?"
})

print(f"Question: {result['question']}")
print(f"\nAnswer: {result['response']}")
print(f"\nRetrieved {len(result['context'])} documents:")
for i, doc in enumerate(result['context'], 1):
    print(f"\n{i}. Page {doc.metadata.get('page', '?')}")
    print(f"   {doc.page_content[:100]}...")
```

**Comparing retrievers:**

```python
from src.graph import retrievers_config

question = "What formats does GDELT support?"

print(f"Question: {question}\n")
print("=" * 80)

for retriever_name, graph in retrievers_config.items():
    result = graph.invoke({"question": question})

    print(f"\n{retriever_name.upper()}")
    print("-" * 80)
    print(f"Answer: {result['response'][:200]}...")
    print(f"Retrieved: {len(result['context'])} documents")
```

---

## Best Practices

### Retriever Selection

**When to use each retriever:**

1. **Naive (Baseline)**
   - Good starting point
   - Fast and simple
   - Use when: semantic similarity is primary concern
   - Limitations: misses exact keyword matches

2. **BM25**
   - Use when: exact keyword matching is important
   - Good for: technical terms, acronyms, specific phrases
   - Limitations: weak semantic understanding

3. **Ensemble**
   - Use when: want balance of semantic + keyword matching
   - Good for: general-purpose retrieval
   - Best for: diverse query types

4. **Cohere Rerank**
   - Use when: highest quality results needed
   - Good for: complex semantic queries
   - Limitations: slower, more expensive
   - Best for: production deployments where accuracy matters most

**Recommendation:** Start with Ensemble, optimize to Cohere Rerank if needed.

---

### Error Handling

**Common errors and solutions:**

#### 1. Missing API Keys
```python
# Error: openai.AuthenticationError
# Solution:
import os
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Set OPENAI_API_KEY environment variable")
```

#### 2. Qdrant Connection Error
```python
# Error: Cannot connect to Qdrant
# Solution:
# 1. Ensure Qdrant is running:
# $ docker-compose up -d qdrant

# 2. Check connection:
from qdrant_client import QdrantClient
try:
    client = QdrantClient(host="localhost", port=6333)
    print("Connected:", client.get_collections())
except Exception as e:
    print(f"Connection failed: {e}")
```

#### 3. RAGAS Schema Validation Error
```python
# Error: ValueError: Missing required columns
# Solution: Use validation function
from scripts.single_file import validate_and_normalize_ragas_schema

try:
    validated_df = validate_and_normalize_ragas_schema(df, retriever_name)
except ValueError as e:
    print(f"Schema validation failed: {e}")
    print("Available columns:", df.columns.tolist())
```

#### 4. Rate Limit Errors
```python
# Error: openai.RateLimitError
# Solution: Use retry logic (already in ingest.py)
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=20)
)
def call_api():
    # API call here
    pass
```

---

### Performance Optimization

#### 1. Batch Processing
```python
# Instead of:
for question in questions:
    result = graph.invoke({"question": question})

# Use batching where possible:
from langchain.schema import Document
results = llm.batch([
    {"messages": [{"role": "user", "content": q}]}
    for q in questions
])
```

#### 2. Caching Embeddings
```python
# Embeddings are expensive - cache them
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore

store = LocalFileStore("./cache/embeddings")
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    embeddings, store, namespace="text-embedding-3-small"
)
```

#### 3. Reduce k for Faster Retrieval
```python
# High k = slower but more context
retriever = vector_store.as_retriever(search_kwargs={"k": 20})

# Low k = faster but less context
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Sweet spot for most use cases: k=5
```

#### 4. Use Async for Concurrent Processing
```python
# Process multiple queries concurrently
import asyncio

async def process_questions(questions, graph):
    tasks = [graph.ainvoke({"question": q}) for q in questions]
    return await asyncio.gather(*tasks)

# Run
results = asyncio.run(process_questions(questions, baseline_graph))
```

---

### Dataset Management

#### 1. Version Control for Datasets
```python
# Always include checksums in manifest
from scripts.enrich_manifest import sha256

checksum = sha256(Path("data/sources.jsonl"))
print(f"Dataset checksum: {checksum}")

# Compare before/after
if checksum != expected_checksum:
    print("⚠️  Dataset has changed!")
```

#### 2. Multi-Format Persistence
```python
# Always save in multiple formats for flexibility
docs_to_jsonl(docs, path / "data.jsonl")      # Human-readable
docs_to_parquet(docs, path / "data.parquet")  # Analytics
docs_to_hfds(docs, path / "data.hfds")        # Fast loading

# Choose format based on use case:
# - JSONL: debugging, manual inspection
# - Parquet: pandas analysis, SQL queries
# - HFDS: training, fast iteration
```

#### 3. Manifest Best Practices
```python
# Always include:
manifest = {
    "id": f"run_{uuid.uuid4()}",              # Unique ID
    "generated_at": datetime.utcnow().isoformat(),
    "env": {...},                              # Package versions
    "fingerprints": {...},                     # File checksums
    "run": {"random_seed": 42}                 # Reproducibility
}
```

---

### LLM Configuration

#### 1. Temperature Settings
```python
# For RAG generation: temperature=0 (deterministic)
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

# For creative tasks: temperature>0
creative_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)
```

#### 2. Timeout and Retry Configuration
```python
# Production settings
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    timeout=60,        # Prevent hanging
    max_retries=6      # Handle transient errors
)
```

#### 3. Cost Optimization
```python
# Use cheaper models for evaluation
# gpt-4.1-mini is already cost-effective

# For large-scale evaluation, consider:
# 1. Caching LLM responses
# 2. Sampling subset of test cases
# 3. Using smaller testset during development
```

---

## Examples

### Complete Example: End-to-End Evaluation

```python
#!/usr/bin/env python3
"""
Complete RAG evaluation example - from scratch to results.
"""

import os
from pathlib import Path
from datasets import load_dataset
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
import pandas as pd

# Set API keys
os.environ["OPENAI_API_KEY"] = "sk-proj-..."
os.environ["COHERE_API_KEY"] = "..."

# Import project modules
from src.graph import retrievers_config
from ragas import EvaluationDataset, evaluate, RunConfig
from ragas.metrics import Faithfulness, ResponseRelevancy, ContextPrecision, LLMContextRecall
from ragas.llms import LangchainLLMWrapper
from scripts.single_file import validate_and_normalize_ragas_schema

def main():
    print("Starting RAG Evaluation...")

    # 1. Load golden testset
    print("\n1. Loading golden testset...")
    golden_dataset = load_dataset("dwb2023/gdelt-rag-golden-testset", split="train")
    golden_df = golden_dataset.to_pandas()
    print(f"   Loaded {len(golden_df)} test cases")

    # 2. Initialize evaluator
    print("\n2. Initializing evaluator...")
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-mini", temperature=0))

    # 3. Process questions through all retrievers
    print("\n3. Running questions through retrievers...")
    results = {}

    for retriever_name, graph in retrievers_config.items():
        print(f"\n   Processing {retriever_name}...")

        retriever_results = []
        for _, row in golden_df.iterrows():
            question = row['user_input']
            result = graph.invoke({"question": question})

            retriever_results.append({
                'user_input': question,
                'response': result['response'],
                'retrieved_contexts': [doc.page_content for doc in result['context']],
                'reference': row['reference']
            })

        results[retriever_name] = pd.DataFrame(retriever_results)
        print(f"   ✓ {retriever_name}: {len(retriever_results)} questions processed")

    # 4. Run RAGAS evaluation
    print("\n4. Running RAGAS evaluation...")
    evaluation_results = {}

    for retriever_name, df in results.items():
        print(f"\n   Evaluating {retriever_name}...")

        # Validate schema
        validated_df = validate_and_normalize_ragas_schema(df, retriever_name)
        eval_dataset = EvaluationDataset.from_pandas(validated_df)

        # Evaluate
        result = evaluate(
            dataset=eval_dataset,
            metrics=[
                Faithfulness(),
                ResponseRelevancy(),
                ContextPrecision(),
                LLMContextRecall()
            ],
            llm=evaluator_llm,
            run_config=RunConfig(timeout=360)
        )

        evaluation_results[retriever_name] = result
        print(f"   ✓ {retriever_name} evaluation complete")

    # 5. Generate comparative results
    print("\n5. Generating comparative results...")
    comparison_data = []

    for retriever_name, result in evaluation_results.items():
        df = result.to_pandas()

        comparison_data.append({
            'Retriever': retriever_name.replace('_', ' ').title(),
            'Faithfulness': df['faithfulness'].mean(),
            'Answer Relevancy': df['answer_relevancy'].mean(),
            'Context Precision': df['context_precision'].mean(),
            'Context Recall': df['context_recall'].mean()
        })

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df['Average'] = comparison_df[['Faithfulness', 'Answer Relevancy',
                                               'Context Precision', 'Context Recall']].mean(axis=1)
    comparison_df = comparison_df.sort_values('Average', ascending=False)

    # 6. Display results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print("\n", comparison_df.to_string(index=False))

    # 7. Save results
    output_dir = Path("output/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_df.to_csv(output_dir / "comparative_results.csv", index=False)
    print(f"\n✓ Results saved to {output_dir}")

    return comparison_df

if __name__ == "__main__":
    results = main()
```

---

### Example: Custom Retriever

```python
"""
Example: Adding a hybrid retriever with custom weighting.
"""

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from src.retrievers import vector_store, documents
from src.graph import generate
from src.state import State
from langgraph.graph import START, StateGraph

# 1. Create custom weighted ensemble
# Favor dense search (70%) over sparse (30%)
custom_ensemble = EnsembleRetriever(
    retrievers=[
        vector_store.as_retriever(search_kwargs={"k": 5}),
        BM25Retriever.from_documents(documents, k=5)
    ],
    weights=[0.7, 0.3]  # Custom weighting
)

# 2. Create retrieval function
def retrieve_custom_ensemble(state):
    """Custom weighted ensemble retrieval"""
    retrieved_docs = custom_ensemble.invoke(state["question"])
    return {"context": retrieved_docs}

# 3. Build LangGraph workflow
custom_graph_builder = StateGraph(State).add_sequence([
    retrieve_custom_ensemble,
    generate
])
custom_graph_builder.add_edge(START, "retrieve_custom_ensemble")
custom_graph = custom_graph_builder.compile()

# 4. Test the custom retriever
question = "What is GDELT Translingual?"
result = custom_graph.invoke({"question": question})

print(f"Question: {question}")
print(f"\nAnswer: {result['response']}")
print(f"\nRetrieved {len(result['context'])} documents")

# 5. Add to retrievers_config for evaluation
from src.graph import retrievers_config
retrievers_config["custom_ensemble_70_30"] = custom_graph
```

---

### Example: Analyzing Retrieval Quality

```python
"""
Example: Analyzing which documents are retrieved for each strategy.
"""

from src.graph import retrievers_config
import pandas as pd

def analyze_retrieval(question: str):
    """Compare retrieved documents across all retrievers."""

    print(f"Question: {question}\n")
    print("=" * 80)

    analysis = []

    for retriever_name, graph in retrievers_config.items():
        # Get retrieval results
        result = graph.invoke({"question": question})

        print(f"\n{retriever_name.upper()}")
        print("-" * 80)

        for i, doc in enumerate(result['context'], 1):
            page = doc.metadata.get('page', '?')
            snippet = doc.page_content[:100].replace('\n', ' ')

            print(f"{i}. Page {page}: {snippet}...")

            analysis.append({
                'retriever': retriever_name,
                'rank': i,
                'page': page,
                'content_preview': snippet
            })

    # Create analysis DataFrame
    df = pd.DataFrame(analysis)

    # Find common documents
    print("\n" + "=" * 80)
    print("COMMON DOCUMENTS")
    print("=" * 80)

    page_counts = df.groupby('page').size().sort_values(ascending=False)
    print("\nPages retrieved by multiple strategies:")
    for page, count in page_counts.items():
        if count > 1:
            retrievers = df[df['page'] == page]['retriever'].tolist()
            print(f"  Page {page}: {count} retrievers ({', '.join(retrievers)})")

    return df

# Run analysis
question = "What formats does GDELT support?"
analysis_df = analyze_retrieval(question)

# Save analysis
analysis_df.to_csv("output/retrieval_analysis.csv", index=False)
```

---

## Appendix

### Dependencies

**Core Dependencies (from pyproject.toml):**

| Package | Version | Purpose |
|---------|---------|---------|
| `langchain` | >=0.3.19 | Document loading, retrieval, prompts |
| `langchain-openai` | >=0.3.7 | OpenAI LLM and embeddings integration |
| `langchain-cohere` | ==0.4.4 | Cohere reranking integration |
| `langchain-qdrant` | >=0.2.0 | Qdrant vector store integration |
| `langgraph` | ==0.6.7 | State graph workflow orchestration |
| `ragas` | ==0.2.10 | RAG evaluation metrics |
| `qdrant-client` | >=1.13.2 | Qdrant vector database client |
| `rank-bm25` | >=0.2.2 | BM25 sparse retrieval |
| `pymupdf` | >=1.26.3 | PDF document loading |
| `huggingface-hub` | >=0.26.0 | HuggingFace dataset operations |
| `datasets` | >=3.2.0 | HuggingFace datasets library |
| `cohere` | >=5.12.0,<5.13.0 | Cohere API client |
| `jupyter` | >=1.1.1 | Notebook support |
| `streamlit` | >=1.40.0 | Web interface (optional) |

**Python Version:** >=3.11

---

### File Paths Reference

**Source Files:**

| File | Lines | Purpose |
|------|-------|---------|
| `src/__init__.py` | 1 | Package initialization (empty) |
| `src/config.py` | 11 | Configuration constants and global objects |
| `src/state.py` | 10 | State TypedDict definition |
| `src/prompts.py` | 12 | Prompt templates |
| `src/retrievers.py` | 58 | Retriever definitions |
| `src/graph.py` | 72 | LangGraph workflow definitions |
| `src/utils.py` | 2 | Utility functions (empty) |
| `scripts/ingest.py` | 336 | Data ingestion pipeline |
| `scripts/single_file.py` | 508 | RAG evaluation script |
| `scripts/upload_to_hf.py` | 293 | HuggingFace upload script |
| `scripts/generate_run_manifest.py` | 188 | Reproducibility manifest generator |
| `scripts/enrich_manifest.py` | 244 | Manifest enrichment script |

**Total Source Lines:** ~1,735 lines

---

### Project Structure

```
cert-challenge/
├── src/                          # Core library code
│   ├── __init__.py               # Package initialization
│   ├── config.py                 # Configuration
│   ├── state.py                  # State schema
│   ├── prompts.py                # Prompt templates
│   ├── retrievers.py             # Retriever definitions
│   ├── graph.py                  # LangGraph workflows
│   └── utils.py                  # Utilities
├── scripts/                      # Pipeline scripts
│   ├── ingest.py                 # Data ingestion
│   ├── single_file.py            # RAG evaluation
│   ├── upload_to_hf.py           # HuggingFace upload
│   ├── generate_run_manifest.py  # Manifest generation
│   └── enrich_manifest.py        # Manifest enrichment
├── data/                         # Data directory
│   ├── raw/                      # Source PDFs
│   ├── interim/                  # Processed datasets
│   │   ├── sources.jsonl
│   │   ├── sources.parquet
│   │   ├── sources.hfds/
│   │   ├── golden_testset.jsonl
│   │   ├── golden_testset.parquet
│   │   ├── golden_testset.hfds/
│   │   └── manifest.json
│   └── processed/                # Final outputs
├── deliverables/                 # Evaluation outputs
│   └── evaluation_evidence/
│       ├── comparative_ragas_results.csv
│       ├── *_evaluation_dataset.csv
│       ├── *_detailed_results.csv
│       └── RUN_MANIFEST.json
├── .env                          # Environment variables (gitignored)
├── .env.example                  # Environment template
├── pyproject.toml                # Project dependencies
├── docker-compose.yml            # Qdrant container
└── README.md                     # Project overview
```

---

### Quick Start Guide

**1. Setup Environment:**

```bash
# Clone repository
cd cert-challenge

# Install dependencies
pip install -e .

# Copy environment template
cp .env.example .env

# Edit .env and add API keys
nano .env
```

**2. Start Qdrant:**

```bash
docker-compose up -d qdrant
```

**3. Run Data Ingestion:**

```bash
python scripts/ingest.py
```

**4. Run Evaluation:**

```bash
python scripts/single_file.py
```

**5. Upload to HuggingFace (optional):**

```bash
python scripts/upload_to_hf.py
```

---

### Common Workflows

**Development Workflow:**
1. Modify retriever in `src/retrievers.py`
2. Update graph in `src/graph.py`
3. Test with single question
4. Run full evaluation
5. Compare results

**Evaluation Workflow:**
1. Ensure Qdrant is running
2. Run `single_file.py`
3. Review comparative results
4. Generate manifest
5. Save to deliverables

**Data Pipeline Workflow:**
1. Add PDFs to `data/raw/`
2. Run `ingest.py`
3. Verify outputs in `data/interim/`
4. Run `enrich_manifest.py`
5. Upload with `upload_to_hf.py`

---

### Support and Resources

**Documentation:**
- LangChain: https://python.langchain.com/
- LangGraph: https://langchain-ai.github.io/langgraph/
- RAGAS: https://docs.ragas.io/
- Qdrant: https://qdrant.tech/documentation/

**Datasets:**
- Sources: https://huggingface.co/datasets/dwb2023/gdelt-rag-sources
- Golden Testset: https://huggingface.co/datasets/dwb2023/gdelt-rag-golden-testset

**Repository:**
- GitHub: `/home/donbr/don-aie-cohort8/cert-challenge`

---

## Changelog

**Version 0.1.0** (Current)
- Initial API reference documentation
- Complete coverage of src/ and scripts/ modules
- Configuration reference
- Usage patterns and examples
- Best practices guide

---

*This documentation was generated on 2025-01-17 for the GDELT RAG Comparative Evaluation project.*

*For questions or issues, refer to the source code or project README.*
