# GDELT Knowledge Graph RAG Assistant

> **Certification Challenge Project** — AI Engineering Bootcamp Cohort 8  
> An intelligent question-answering system for GDELT (Global Database of Events, Language, and Tone) documentation, powered by **Retrieval-Augmented Generation**.

---

## 📚 Documentation Overview

**Core Docs**
- 🏗️ **[CLAUDE.md](./CLAUDE.md)** — canonical developer guide for this repository  
- 📋 **[Full Deliverables](./docs/deliverables.md)** — complete certification submissions  
- 📊 **[Task Rubric](./docs/certification-challenge-task-list.md)** — 100-point grading breakdown  
- 🧠 **[docs/initial-architecture.md](./docs/initial-architecture.md)** — *early conceptual design (frozen)*  

**Architecture Documentation (Auto-Generated)**
Located in the **`architecture/`** directory — produced by the *Claude Agent SDK Analyzer*:
| File | Purpose |
|------|----------|
| `architecture/README.md` | Top-level architecture summary and system lifecycle |
| `architecture/docs/01_component_inventory.md` | Detailed module inventory |
| `architecture/docs/03_data_flows.md` | Ingestion → retrieval → evaluation flows |
| `architecture/docs/04_api_reference.md` | Public API reference |
| `architecture/diagrams/02_architecture_diagrams.md` | Layered system & runtime dependency diagrams |

> 🧠 **Note:** `docs/initial-architecture.md` captures the *original design sketch* before automation and is no longer updated.  
> The current `architecture/` tree is generated automatically from the live codebase.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11 +
- [`uv`](https://github.com/astral-sh/uv) package manager  
- `OPENAI_API_KEY` (required)  
- `COHERE_API_KEY` (optional — reranking)

### Installation
```bash
git clone https://github.com/<your-username>/cert-challenge.git
cd cert-challenge
uv venv --python 3.11
source .venv/bin/activate        # Linux/WSL/Mac
# .venv\Scripts\activate         # Windows
uv pip install -e .
````

### Environment Setup

```bash
cp .env.example .env
# Edit .env with:
# OPENAI_API_KEY=your_key_here
# COHERE_API_KEY=optional
```

### Run

```bash
# Interactive LangGraph Studio UI
uv add langgraph-cli[inmem]
uv run langgraph dev --allow-blocking
# → http://localhost:2024
# Studio: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024

# CLI evaluation
python scripts/run_eval_harness.py
# or
make eval

# Quick validation
make validate
```

---

## 🧩 Architecture Summary

This system follows a **5-layer architecture**:

| Layer             | Purpose                                               | Key Modules                    |
| ----------------- | ----------------------------------------------------- | ------------------------------ |
| **Configuration** | External services (OpenAI, Qdrant, Cohere)            | `src/config.py`                |
| **Data**          | Ingestion + persistence (HF datasets + manifest)      | `src/utils/`                   |
| **Retrieval**     | Multi-strategy search (naive, BM25, ensemble, rerank) | `src/retrievers.py`            |
| **Orchestration** | LangGraph workflows (retrieve → generate)             | `src/graph.py`, `src/state.py` |
| **Execution**     | Scripts and LangGraph Server entrypoints              | `scripts/`, `app/graph_app.py` |

**Design Principles**

* Factory pattern → deferred initialization of retrievers/graphs
* Singleton pattern → resource caching (`@lru_cache`)
* Strategy pattern → interchangeable retriever implementations

See the generated diagrams in
[`architecture/diagrams/02_architecture_diagrams.md`](./architecture/diagrams/02_architecture_diagrams.md).

---

## 🧠 Evaluation Workflow (Automated)

1. Load 12 QA pairs (golden testset)
2. Load 38 source docs from Hugging Face
3. Create Qdrant vector store
4. Build 4 retriever strategies
5. Execute 48 RAG queries
6. Evaluate with RAGAS metrics (Faithfulness, Answer Relevancy, Context Precision, Context Recall)
7. Persist results → `deliverables/evaluation_evidence/`

Cost ≈ $5 – 6 per run / 20–30 min.

---

## 🔏 Provenance and Manifest Integrity

Every major stage in this project — from **data ingestion** to **evaluation runs** — is signed by machine-readable manifests to ensure **traceability**, **data integrity**, and **AI assistant accountability**.

### 📄 Ingestion Manifest (`data/interim/manifest.json`)
Records the creation of intermediate artifacts (raw → interim → HF datasets) with:
- Environment fingerprint (`langchain`, `ragas`, `pyarrow`, etc.)
- Source & golden testset paths and SHA-256 hashes
- Quick schema preview of extracted columns
- Hugging Face lineage metadata linking to:
  - `dwb2023/gdelt-rag-sources-v2`
  - `dwb2023/gdelt-rag-golden-testset-v2`

This ensures that every RAGAS evaluation run references a reproducible, signed dataset snapshot.

### 🧾 Run Manifest (`deliverables/evaluation_evidence/RUN_MANIFEST.json`)
Documents the execution of each evaluation:
- Models: `gpt-4.1-mini` (RAG generation), `text-embedding-3-small`
- Retrievers: naive, BM25, ensemble, cohere_rerank
- Deterministic configuration (`temperature=0`)
- Metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall
- SHA links back to the ingestion manifest for complete lineage

Together, these manifests create a **verifiable audit trail** between:
> _source PDFs → testset → vector store → evaluation results_

By preserving SHA-256 fingerprints and HF lineage, this mechanism “signs” the dataset state — keeping automated assistants and evaluation scripts consistent, comparable, and honest.

---

## ⚙️ Technology Stack

| Component         | Technology                           | Purpose                        |
| ----------------- | ------------------------------------ | ------------------------------ |
| **LLM**           | OpenAI GPT-4.1-mini                  | Deterministic RAG generation   |
| **Embeddings**    | text-embedding-3-small               | 1536-dim semantic vectors      |
| **Vector DB**     | Qdrant                               | Fast cosine search             |
| **Orchestration** | LangGraph 0.6.7 + LangChain 0.3.19 + | Graph-based workflows          |
| **Evaluation**    | RAGAS 0.2.10 (pinned)                | Stable evaluation API          |
| **Monitoring**    | LangSmith                            | LLM trace observability        |
| **Data**          | Hugging Face Datasets                | Reproducible versioned sources |
| **UI**            | Streamlit                            | Prototype chat interface       |

---

## 🧮 Evaluation Results (Summary)

> Source of truth: `deliverables/evaluation_evidence/RUN_MANIFEST.json` (auto-generated each run).

| Retriever         | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Avg  |
|------------------|-------------:|-----------------:|------------------:|---------------:|-----:|
| **Cohere Rerank**| **95.8%**    | **94.8%**        | **93.1%**         | **96.7%**      | **95.1%** |
| Ensemble         | 93.4%        | 94.6%            | 87.5%             | **98.8%**      | 93.6% |
| BM25             | 94.2%        | 94.8%            | 85.8%             | **98.8%**      | 93.4% |
| Naive            | 94.0%        | 94.4%            | 88.5%             | **98.8%**      | 93.9% |

**Provenance:** dataset paths & SHA-256 fingerprints are recorded in `data/interim/manifest.json`.
**Reproducibility:** these numbers are written by the evaluation run into `RUN_MANIFEST.json`.
**TLDR summary:** `data/processed/comparative_ragas_results.csv`

---

## 📦 HuggingFace Datasets

This project publishes **4 datasets** to HuggingFace Hub for reproducibility and benchmarking:

### Interim Datasets (Raw Data)

These datasets contain the source documents and golden testset used for evaluation:

- **[dwb2023/gdelt-rag-sources-v2](https://huggingface.co/datasets/dwb2023/gdelt-rag-sources-v2)** - 38 GDELT documentation pages
  - GDELT GKG 2.1 architecture documentation
  - Knowledge graph construction guides
  - Baltimore Bridge Collapse case study
  - Format: Multiple formats (JSONL, Parquet, HuggingFace Dataset)

- **[dwb2023/gdelt-rag-golden-testset-v2](https://huggingface.co/datasets/dwb2023/gdelt-rag-golden-testset-v2)** - 12 QA pairs
  - Synthetically generated questions using RAGAS
  - Ground truth answers and reference contexts
  - Single-hop and multi-hop questions

### Processed Datasets (Evaluation Results)

These datasets contain consolidated evaluation results from all 5 retriever strategies:

- **[dwb2023/gdelt-rag-evaluation-datasets](https://huggingface.co/datasets/dwb2023/gdelt-rag-evaluation-datasets)** - 60 RAGAS evaluation inputs
  - Schema: `retriever`, `user_input`, `retrieved_contexts`, `reference_contexts`, `response`, `reference`
  - Use cases: Benchmarking retriever strategies, analyzing retrieval quality, reproducing evaluation results

- **[dwb2023/gdelt-rag-detailed-results](https://huggingface.co/datasets/dwb2023/gdelt-rag-detailed-results)** - 60 RAGAS metric scores
  - Schema: All evaluation dataset fields PLUS `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`
  - Use cases: Performance analysis, error analysis, training retrieval models with quality labels

### Loading Examples

```python
from datasets import load_dataset

# Load evaluation datasets
eval_ds = load_dataset("dwb2023/gdelt-rag-evaluation-datasets")

# Filter by retriever
cohere_evals = eval_ds['train'].filter(lambda x: x['retriever'] == 'cohere_rerank')
print(f"Cohere Rerank: {len(cohere_evals)} examples")

# Load detailed results with metrics
results_ds = load_dataset("dwb2023/gdelt-rag-detailed-results")

# Analyze performance by retriever
import pandas as pd
df = results_ds['train'].to_pandas()
performance = df.groupby('retriever')[
    ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']
].mean()
print(performance)
```

**Output**:
```
                 faithfulness  answer_relevancy  context_precision  context_recall
retriever
baseline             0.9351          0.9335           0.9459            0.9410
bm25                 0.9462          0.9583           0.9519            0.9511
cohere_rerank        0.9508          0.9321           0.9670            0.9668
ensemble             0.9424          0.9542           0.9477            0.9486
naive                0.9351          0.9335           0.9459            0.9410
```

---

## 🗂️ Repository Map

| Path | Purpose |
|------|----------|
| `src/` | Core modular RAG framework (`config`, `retrievers`, `graph`, `state`, `utils`) |
| `scripts/` | Executable workflows for data ingestion, evaluation, and validation |
| `architecture/` | **Auto-generated architecture snapshots** (Claude Agent SDK Analyzer) |
| ├── `00_README.md` | System overview and lifecycle summary |
| ├── `docs/` | Component inventory, data flows, and API reference |
| └── `diagrams/` | Mermaid dependency and system diagrams |
| `docs/` | Certification artifacts and legacy design documentation |
| └── `initial-architecture.md` | Original hand-drawn architecture sketch *(frozen, not updated)* |
| `data/` | Complete RAG dataset lineage and provenance chain |
| ├── `raw/` | Original GDELT PDFs |
| ├── `interim/` | Extracted text, Hugging Face datasets, and manifest fingerprints |
| │   ├── `manifest.json` → Ingestion provenance manifest (dataset lineage, SHA-256 hashes) |
| ├── `processed/` | Evaluation datasets (CSV, Parquet) |
| └── `deliverables/` | Final evaluation evidence and signed manifests |
|     ├── `evaluation_evidence/` | RAGAS results, raw datasets, and per-retriever outputs |
|     │   └── `RUN_MANIFEST.json` → Evaluation provenance manifest (retrievers, models, metrics) |
| `app/` | Lightweight LangGraph API (`graph_app.py`) and entrypoint |
| `deliverables/` | High-level evaluation reports and comparative analyses |
| `Makefile` | Task automation for environment setup, validation, and architecture snapshots |
| `docker-compose.yml` | Local container configuration for LangGraph + Qdrant stack |

---

## 🧱 Architecture Snapshot Generation

I created a **Claude Agent SDK** based multi-agent process to create reproducible architecture snapshots.  (Still a prototype, but has already helped refine my architectural workflow)

### Architecture Analysis

```bash
python -m ra_orchestrators.architecture_orchestrator "GDELT architecture"
```

Generates comprehensive repository analysis in `ra_output/`:
- Component inventory
- Architecture diagrams
- Data flow analysis
- API documentation
- Final synthesis

## 🤖 Claude Agent SDK Architecture

| Path | Purpose |
|------|----------|
| `ra_agents/` | Individual agent definitions for design and architecture workflows |
| `ra_orchestrators/` | Multi-agent orchestration logic coordinating Claude SDK agents |
| `ra_tools/` | Tools that extend agent capabilities via MCP and external APIs |
| `ra_output/` | Generated artifacts and agentic outputs (documentation drafts, diagrams, etc.) |

---

## 🧩 Key Design Patterns

| Pattern       | Purpose                         | Example                                 |
| ------------- | ------------------------------- | --------------------------------------- |
| **Factory**   | Deferred initialization         | `create_retrievers()` / `build_graph()` |
| **Singleton** | Cached resources (`@lru_cache`) | `get_llm()`, `get_qdrant()`             |
| **Strategy**  | Swap retrieval algorithms       | 4 retrievers share `.invoke()` API      |

---

## 🧾 License & Contact

Apache 2.0 — see `LICENSE`

**Contact:** Don Branson (`dwb2023`) – AI Engineering Bootcamp Cohort 8

---

### 🧩 About This Documentation

This repository distinguishes between **historical design artifacts** and **current architecture snapshots**:

* `docs/initial-architecture.md` → conceptual blueprint (frozen)
* `architecture/` → live system documentation (auto-generated)

This separation ensures long-term clarity and traceability of architectural evolution.
