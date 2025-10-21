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

This project publishes **4 datasets** to HuggingFace Hub for reproducibility and benchmarking.

> **Scientific Value**: These datasets provide the first publicly available evaluation suite for GDELT-focused RAG systems, enabling reproducible benchmarking of retrieval strategies with complete evaluation transparency.

### Interim Datasets (Raw Data)

**1. [dwb2023/gdelt-rag-sources-v2](https://huggingface.co/datasets/dwb2023/gdelt-rag-sources-v2)** - 38 GDELT documentation pages
  - **Content**: GDELT GKG 2.1 architecture docs, knowledge graph construction guides, Baltimore Bridge Collapse case study
  - **Format**: Parquet (analytics), JSONL (human-readable), HF Datasets (fast loading)
  - **Schema**: `page_content` (1.5k-5.2k chars), `metadata` (author, title, page, creation_date, etc.)
  - **Use**: Populate vector stores, document chunking experiments, GDELT research
  - **License**: Apache 2.0

**2. [dwb2023/gdelt-rag-golden-testset-v2](https://huggingface.co/datasets/dwb2023/gdelt-rag-golden-testset-v2)** - 12 QA pairs
  - **Content**: Synthetically generated questions (RAGAS 0.2.10), ground truth answers, reference contexts
  - **Topics**: GDELT data formats, Translingual features (65 languages), date extraction, proximity context, emotions
  - **Schema**: `user_input` (question), `reference_contexts` (ground truth passages), `reference` (answer), `synthesizer_name`
  - **Use**: Benchmark RAG systems using RAGAS metrics, validate retrieval performance
  - **License**: Apache 2.0

### Processed Datasets (Evaluation Results)

**3. [dwb2023/gdelt-rag-evaluation-inputs](https://huggingface.co/datasets/dwb2023/gdelt-rag-evaluation-inputs)** - 60 evaluation records
  - **Content**: Consolidated RAGAS inputs from 5 retrieval strategies (baseline, naive, BM25, ensemble, cohere_rerank)
  - **Schema**: `retriever`, `user_input`, `retrieved_contexts`, `reference_contexts`, `response`, `reference`, `synthesizer_name`
  - **Use**: Benchmark new retrievers, analyze retrieval quality, reproduce certification results, debug RAG pipelines
  - **License**: Apache 2.0

**4. [dwb2023/gdelt-rag-evaluation-metrics](https://huggingface.co/datasets/dwb2023/gdelt-rag-evaluation-metrics)** - 60 evaluation records with RAGAS scores
  - **Content**: Detailed RAGAS evaluation results with per-question metric scores
  - **Schema**: All evaluation-inputs fields PLUS `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall` (float64, 0-1)
  - **Key Findings**: Cohere Rerank winner (95.08% avg), Baseline (93.92% avg), Best Precision: Cohere (+4.55% vs baseline)
  - **Use**: Performance analysis, error analysis, train retrieval models with RAGAS scores as quality labels, RAG evaluation research
  - **License**: Apache 2.0

### Scientific Value & Research Impact

**Why These Datasets Matter:**

1. **Reproducibility**: Complete evaluation pipeline with versioned datasets and SHA-256 checksums
2. **Benchmarking**: Standard testset for comparing retrieval strategies across 4 RAGAS metrics
3. **Quality Labels**: RAGAS scores serve as training labels for learning-to-rank models
4. **Domain-Specific**: GDELT knowledge graph QA pairs rare in existing RAG datasets
5. **Evaluation Transparency**: Full evaluation inputs + metrics for analysis and debugging
6. **Multi-Format**: Parquet (analytics), JSONL (human-readable), HF Datasets (fast loading)

**Research Applications:**
- RAG Researchers: Benchmark retrieval strategies, analyze failure modes, validate hypotheses
- GDELT Analysts: Build Q&A systems, train domain-specific embeddings, extend to other GDELT resources
- Evaluation Researchers: Study RAGAS behavior, compare automatic vs human metrics, develop new methodologies
- Educators: Teach RAG best practices, demonstrate comparative analysis, illustrate data provenance

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

### Citation

If you use these datasets in your research, please cite:

```bibtex
@misc{branson2025gdelt-rag-datasets,
  author = {Branson, Don},
  title = {GDELT RAG Evaluation Datasets: Benchmarking Retrieval Strategies for Knowledge Graph Q\&A},
  year = {2025},
  publisher = {HuggingFace},
  howpublished = {\url{https://huggingface.co/dwb2023}},
  note = {Datasets: gdelt-rag-sources-v2, gdelt-rag-golden-testset-v2, gdelt-rag-evaluation-inputs, gdelt-rag-evaluation-metrics}
}

@article{myers2025gdelt,
  title={Talking to GDELT Through Knowledge Graphs},
  author={Myers, A. and Vargas, M. and Aksoy, S. G. and Joslyn, C. and Wilson, B. and Burke, L. and Grimes, T.},
  journal={arXiv preprint arXiv:2503.07584v3},
  year={2025}
}
```

### Dataset Provenance & Quality Assurance

**Provenance Chain:**
1. Source: arXiv:2503.07584v3 "Talking to GDELT Through Knowledge Graphs" (PDF)
2. Extraction: PyMuPDFLoader (page-level chunking)
3. Testset Generation: RAGAS 0.2.10 synthetic data generation
4. Evaluation: GPT-4.1-mini (LLM), text-embedding-3-small (embeddings), Cohere rerank-v3.5
5. Validation: SHA-256 checksums in `data/interim/manifest.json`

**Quality Guarantees:**
- ✅ RAGAS 0.2.10 schema validation
- ✅ SHA-256 fingerprints for data integrity
- ✅ Manifest tracking (timestamps, model versions, package versions)
- ✅ 100% validation pass rate (`make validate`)
- ✅ Apache 2.0 licensed (open access)

**Versioning:** `-v2` suffix indicates second iteration after fresh ingestion. Pin to specific revision for reproducibility: `load_dataset("dwb2023/gdelt-rag-sources-v2", revision="abc123")`

**Known Limitations:**
- Domain-specific (GDELT documentation), may not generalize to other domains
- Synthetic questions (RAGAS-generated, not human-authored)
- English-only (despite GDELT's multilingual capabilities)
- Small scale (12 evaluation questions - sufficient for comparative analysis, not large-scale benchmarking)
- Model bias (RAGAS metrics computed using GPT-4, inherits model biases)
- Temporal snapshot (based on GDELT documentation as of January 2025)

**Dataset Cards:** See HuggingFace for complete metadata and schemas
- [sources-v2 card](https://huggingface.co/datasets/dwb2023/gdelt-rag-sources-v2)
- [golden-testset-v2 card](https://huggingface.co/datasets/dwb2023/gdelt-rag-golden-testset-v2)
- [evaluation-inputs card](https://huggingface.co/datasets/dwb2023/gdelt-rag-evaluation-inputs)
- [evaluation-metrics card](https://huggingface.co/datasets/dwb2023/gdelt-rag-evaluation-metrics)

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
| `data/` | Complete RAG dataset lineage and provenance chain (Parquet-first) |
| ├── `raw/` | Original GDELT PDFs |
| ├── `interim/` | Extracted text, Hugging Face datasets, and manifest fingerprints |
| │   ├── `manifest.json` → Ingestion provenance manifest (dataset lineage, SHA-256 hashes) |
| ├── `processed/` | **Working data** (Parquet, ZSTD compressed) - evaluation results + RUN_MANIFEST.json |
| └── `deliverables/` | **Derived data** (CSV for human review, regenerable via `make deliverables`) |
|     ├── `evaluation_evidence/` | Human-readable CSV files generated from data/processed/ |
|     │   └── `RUN_MANIFEST.json` → Copied from data/processed/ |
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
