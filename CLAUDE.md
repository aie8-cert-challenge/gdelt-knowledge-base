# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Certification challenge project for AI Engineering Bootcamp Cohort 8: a production-grade RAG system for GDELT (Global Database of Events, Language, and Tone) knowledge graphs with comparative evaluation of 4 retrieval strategies using RAGAS metrics.

**Project Goal**: Compare naive, BM25, ensemble, and Cohere rerank retrievers to determine optimal RAG configuration.

**Key Documentation**:
- `architecture/README.md` — Comprehensive architecture documentation (5,375 lines across 4 docs)
- `docs/deliverables.md` — Complete certification answers
- `README.md` — Project overview and evaluation results

## Essential Commands

### Environment Setup

```bash
# Create virtual environment (Python 3.11 required)
uv venv --python 3.11
source .venv/bin/activate

# Install all dependencies
uv pip install -e .

# Start infrastructure (Qdrant required for RAG system)
docker-compose up -d
```

### Running the RAG System

```bash
# Main evaluation pipeline (20-30 min runtime)
python scripts/single_file.py

# Interactive query via main entry point
python main.py

# Test individual retrievers
python -c "
from src.graph import retrievers_config
result = retrievers_config['naive'].invoke({'question': 'What is GDELT?'})
print(result['response'])
"
```

### Data Pipeline

```bash
# Ingest PDFs and generate RAGAS testset (5-10 min)
python scripts/ingest.py

# Upload datasets to HuggingFace
export HF_TOKEN=hf_...
python scripts/upload_to_hf.py

# Enrich manifest with checksums
python scripts/enrich_manifest.py

# Generate reproducibility manifest
python scripts/generate_run_manifest.py
```

### Development & Exploration

```bash
# Launch Jupyter for notebooks
jupyter notebook

# Check Qdrant status
docker-compose ps qdrant

# View evaluation results
ls -lh data/processed/*.csv
```

## Core Architecture

### Three-Layer Design

**Layer 1: Orchestration** (`scripts/`)
- `single_file.py` — Complete RAG evaluation pipeline (508 LOC)
- `ingest.py` — PDF extraction + RAGAS testset generation (336 LOC)
- `upload_to_hf.py` — HuggingFace dataset publisher (293 LOC)

**Layer 2: Core RAG System** (`src/`)
- `config.py` — Shared LLM, embeddings, Qdrant connection (11 LOC)
- `state.py` — TypedDict schema for LangGraph workflows (10 LOC)
- `prompts.py` — RAG prompt templates (12 LOC)
- `retrievers.py` — 4 retriever implementations (58 LOC)
- `graph.py` — LangGraph workflow compilation (72 LOC)

**Layer 3: External Services**
- OpenAI (GPT-4.1-mini + text-embedding-3-small)
- Cohere (rerank-v3.5)
- Qdrant (vector database at localhost:6333)
- HuggingFace (dataset hosting)

### Critical Architectural Pattern: LangGraph Workflow

**All retrievers follow the same two-node graph pattern:**

```python
# Pattern: retrieve → generate
from langgraph.graph import START, StateGraph
from src.state import State  # TypedDict: {question, context, response}

# Step 1: Define retrieval function (updates context)
def retrieve_naive(state):
    retrieved_docs = baseline_retriever.invoke(state["question"])
    return {"context": retrieved_docs}

# Step 2: Define generation function (updates response)
def generate(state):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = rag_prompt.format_messages(question=state["question"], context=docs_content)
    response = llm.invoke(messages)
    return {"response": response.content}

# Step 3: Build graph
graph_builder = StateGraph(State).add_sequence([retrieve_naive, generate])
graph_builder.add_edge(START, "retrieve_naive")
graph = graph_builder.compile()

# Step 4: Execute
result = graph.invoke({"question": "What is GDELT?"})
# result = {"question": "...", "context": [Document(...)], "response": "..."}
```

**Why this matters**: All 4 retrievers share the same `generate()` function and only differ in their retrieval strategy. This makes comparative evaluation fair—only the retrieval method changes.

### Retriever Strategy Implementations

```python
# src/retrievers.py exports 4 retrievers with uniform interface

# 1. Naive: Dense vector search (k=5)
baseline_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# 2. BM25: Sparse keyword matching (k=5)
bm25_retriever = BM25Retriever.from_documents(documents, k=5)

# 3. Ensemble: Hybrid (50% dense + 50% sparse)
ensemble_retriever = EnsembleRetriever(
    retrievers=[baseline_retriever, bm25_retriever],
    weights=[0.5, 0.5]
)

# 4. Cohere Rerank: Contextual compression (retrieve 20 → rerank to 5)
baseline_retriever_20 = vector_store.as_retriever(search_kwargs={"k": 20})
compressor = CohereRerank(model="rerank-v3.5")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=baseline_retriever_20
)
```

**Performance Results** (see `README.md` for full table):
- **Winner**: Cohere Rerank at 96.47% (+5.3% over baseline)
- **Runner-up**: BM25 at 94.14% (+2.8% over baseline)
- **Baseline**: Naive at 91.60%

### Configuration Management

**Centralized in `src/config.py`:**
```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Shared across all retrievers
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Qdrant connection
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "gdelt_comparative_eval"
```

**Important**: `src/retrievers.py` currently duplicates these constants instead of importing from `config.py`. This is a known improvement opportunity but doesn't affect functionality.

## Data Flow

### Critical Path: RAG Evaluation Pipeline

```
1. Load golden testset (12 QA pairs) from HuggingFace
2. Load source documents (38 docs) from HuggingFace
3. Create Qdrant vector store + embed all documents
4. Build 4 retriever strategies
5. Execute 48 RAG queries (4 retrievers × 12 questions)
6. Run RAGAS evaluation (4 metrics × 48 queries = 192 LLM calls)
7. Generate comparative summary
8. Save results to CSV + RUN_MANIFEST.json

Duration: 20-30 minutes (dominated by RAGAS LLM evaluation calls)
Cost: ~$5.65 per full evaluation run
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

### Multi-Format Persistence Strategy

All datasets persisted in 3 formats:
```
data/interim/
├── sources.jsonl          # Human-readable, line-by-line JSON
├── sources.parquet        # Analytics-optimized columnar format
├── sources.hfds/          # HuggingFace Dataset (fast loading + versioning)
├── golden_testset.jsonl
├── golden_testset.parquet
├── golden_testset.hfds/
└── manifest.json          # Checksums, versions, provenance
```

**Why 3 formats?**
- JSONL: Debugging and manual inspection
- Parquet: Pandas analytics and large-scale processing
- HFDS: Version control, fast Arrow-based loading, HF Hub integration

## Adding New Retrievers

### Step-by-Step Pattern

```python
# 1. Create retriever in src/retrievers.py
from langchain.retrievers import YourRetrieverClass

your_retriever = YourRetrieverClass(
    vectorstore=vector_store,
    k=5
)

# 2. Define retrieval function in src/graph.py
def retrieve_your_method(state):
    """Your retrieval description"""
    retrieved_docs = your_retriever.invoke(state["question"])
    return {"context": retrieved_docs}

# 3. Build LangGraph workflow in src/graph.py
your_graph_builder = StateGraph(State).add_sequence([retrieve_your_method, generate])
your_graph_builder.add_edge(START, "retrieve_your_method")
your_graph = your_graph_builder.compile()

# 4. Add to retrievers_config in src/graph.py
retrievers_config = {
    "naive": baseline_graph,
    "bm25": bm25_graph,
    "ensemble": ensemble_graph,
    "cohere_rerank": rerank_graph,
    "your_method": your_graph,  # <-- Add here
}

# 5. Re-run evaluation - new retriever automatically included
python scripts/single_file.py
```

**The system will automatically**:
- Evaluate your new retriever against all 12 test questions
- Compute 4 RAGAS metrics
- Include results in comparative summary CSV
- Calculate performance vs baseline

## RAGAS Evaluation Metrics

**Implemented in `scripts/single_file.py`:**

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

# All 4 metrics applied to each retriever
metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
```

**Metric Definitions**:
- **Faithfulness** (0-1): Is answer grounded in retrieved context? (detects hallucinations)
- **Answer Relevancy** (0-1): Does answer address the question?
- **Context Precision** (0-1): Are relevant contexts ranked higher than irrelevant?
- **Context Recall** (0-1): Is all ground truth information retrieved?

**Schema Requirements** (critical for RAGAS):
```python
# Required fields for evaluation dataset
{
    "question": str,         # User query
    "answer": str,           # System response
    "contexts": List[str],   # Retrieved doc.page_content (not Document objects!)
    "ground_truth": str      # Expected answer from golden testset
}
```

**Common Error**: Passing `List[Document]` instead of `List[str]` for `contexts` will cause RAGAS validation failure. Use `validate_and_normalize_ragas_schema()` in `single_file.py` to prevent this.

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
```

## File References

When working with the code, key line numbers:

**Core System**:
- [src/config.py:10-11](src/config.py#L10-L11) — LLM and embeddings initialization
- [src/retrievers.py:36-37](src/retrievers.py#L36-L37) — Baseline retriever
- [src/retrievers.py:40](src/retrievers.py#L40) — BM25 retriever (note: requires `documents` variable)
- [src/retrievers.py:46-49](src/retrievers.py#L46-L49) — Ensemble retriever
- [src/retrievers.py:52-58](src/retrievers.py#L52-L58) — Cohere rerank retriever
- [src/graph.py:20-23](src/graph.py#L20-L23) — Retrieval function pattern
- [src/graph.py:41-46](src/graph.py#L41-L46) — Shared generation function
- [src/graph.py:66-71](src/graph.py#L66-L71) — Retriever config dictionary

**Evaluation Pipeline**:
- [scripts/single_file.py](scripts/single_file.py) — Main evaluation script
- [scripts/ingest.py](scripts/ingest.py) — Data ingestion pipeline

## Infrastructure Services

Start Qdrant (required for vector search):
```bash
docker-compose up -d qdrant
```

**Full service stack** (optional, for advanced features):
```bash
docker-compose up -d
```

**Available Services**:
- Qdrant (6333/6334) — Vector database
- Redis (6379) — Caching layer
- Neo4j (7474/7687) — Graph database with APOC
- Phoenix (6006) — Arize observability
- MinIO (9000/9001) — S3-compatible storage
- Postgres (5432) — Relational database
- Adminer (8080) — Database admin UI

**Note**: Only Qdrant is required for baseline RAG system. Other services support advanced features (caching, graph queries, observability).

## Common Development Patterns

### Querying the RAG System

```python
from src.graph import retrievers_config

# Option 1: Use compiled graph
graph = retrievers_config["cohere_rerank"]  # or "naive", "bm25", "ensemble"
result = graph.invoke({"question": "What is GDELT GKG 2.1?"})
print(result["response"])

# Option 2: Access retriever directly
from src.retrievers import baseline_retriever
docs = baseline_retriever.invoke("What is GDELT?")
for doc in docs:
    print(doc.page_content)
```

### Loading Documents from HuggingFace

```python
from datasets import load_dataset
from langchain_core.documents import Document

dataset = load_dataset("dwb2023/gdelt-rag-sources", split="train")

documents = []
for item in dataset:
    doc = Document(
        page_content=item["page_content"],
        metadata=item["metadata"]  # Already a dict in HF dataset
    )
    documents.append(doc)
```

### Reproducibility Manifest Pattern

```python
# scripts/generate_run_manifest.py creates:
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

**All evaluation runs must generate manifest** for reproducibility.

## Known Limitations & Future Work

**Current Limitations**:
1. `src/retrievers.py` has undefined `documents` variable (line 40) — must be loaded before BM25 retriever creation
2. Configuration constants duplicated between `config.py` and `retrievers.py`
3. No async execution for parallel retriever evaluation (4x speedup opportunity)
4. No embedding cache (repeated API calls for same documents)
5. Ensemble weights hardcoded at 50/50 (should be tunable)

**Recommended Improvements** (post-certification):
1. Implement `CacheBackedEmbeddings` for embedding reuse
2. Async retriever evaluation with `asyncio.gather()`
3. Tune ensemble weights via grid search
4. Add query expansion and hypothetical document embeddings (HyDE)
5. Implement semantic caching with Redis
6. Add parent document retrieval strategy

## Additional Resources

- [architecture/README.md](architecture/README.md) — Complete architecture documentation (1,073 lines)
- [architecture/docs/01_component_inventory.md](architecture/docs/01_component_inventory.md) — Module catalog (521 lines)
- [architecture/diagrams/02_architecture_diagrams.md](architecture/diagrams/02_architecture_diagrams.md) — Visual diagrams (786 lines)
- [architecture/docs/03_data_flows.md](architecture/docs/03_data_flows.md) — Data flow analysis (947 lines)
- [architecture/docs/04_api_reference.md](architecture/docs/04_api_reference.md) — API documentation (3,121 lines)

**Total architecture documentation**: 5,375 lines across 5 files.