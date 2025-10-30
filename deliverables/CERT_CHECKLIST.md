## A. One‑Screen Summary
- **Project name**: GDELT Knowledge Graph RAG Assistant
- **1‑sentence problem**: Analysts struggle to query and understand GDELT’s complex knowledge graph documentation quickly and accurately.
- **Primary user**: Data/OSINT analysts and engineers working with GDELT.
- **Stack summary**: LLM: OpenAI GPT‑4.1‑mini; Embeddings: text‑embedding‑3‑small; Orchestrator: LangGraph + LangChain; Vector DB: Qdrant; Logging/Eval: LangSmith + RAGAS; Validators: manifest lineage + deterministic runs; UI/Serving: LangGraph Server + Streamlit.
- **Result summary**: Advanced retrieval (Cohere rerank) improved precision and average score vs naïve.

## B. Task Compliance Table
| Task | Status (Yes / Partial / No) | Evidence (file + section) | Scope / Clarity Note | One Improvement |
|---|-----|---|----|-----|
| 1. Problem & Audience | Yes | `README.md` — top lines 1–6; Docs purpose | Clear audience and problem captured | Add 1–2 paragraph narrative under `docs/` per spec |
| 2. Solution & Stack | Yes | `README.md` — Architecture Summary + Technology Stack | Tooling choices specified per a16z layers | Link each choice to 1‑sentence rationale inline |
| 3. Data & Chunking | Yes | `README.md` — HuggingFace Datasets; Provenance & Manifest | Describes sources, schemas, lineage | Add explicit default chunking rationale in `src/utils` docstring |
| 4. End‑to‑End Prototype | Yes | `app/graph_app.py` get_app; `src/graph.py` build_graph | Local LangGraph Server entrypoint present | Include minimal Streamlit UI link/run note in README |
| 5. Golden Test + RAGAS | Yes | `scripts/run_eval_harness.py` — RAGAS steps; `deliverables/evaluation_evidence/RUN_MANIFEST.json` | Metrics computed and saved | Add small table preview in `deliverables/README.md` |
| 6. Advanced Retrieval | Yes | `src/retrievers.py` (factory via usage); `README.md` Results table | Techniques include BM25, ensemble, Cohere rerank | Brief per‑technique “why useful” bullets in README |
| 7. Performance & Next Steps | Yes | `README.md` — Evaluation Results Summary | Winner, averages, provenance noted | Add explicit "next iteration" bullets in README |

## C. Metrics Snapshot (Tasks 5 & 7)
| Strategy | Faithfulness | Response Relevancy | Context Precision | Context Recall |
|----|-----|-----|----|----|
| naive | 94.0% | 94.4% | 88.5% | 98.8% |
| advanced | 95.8% | 94.8% | 93.1% | 96.7% |

- **1 Strength:** Context Precision improved most via reranking.
- **1 Next Step:** Document per‑query error analysis to guide retriever fusion tuning.

## D. Stack Alignment (a16z LLM App Stack)
| Layer | Tool Used | Why (≤ 12 words) |
|----|-----|---|
| LLM | OpenAI GPT‑4.1‑mini | Deterministic, reliable RAG generation |
| Embeddings | text‑embedding‑3‑small | Strong semantic vectors, cost‑effective |
| Orchestrator | LangGraph + LangChain | Graph workflows and LC integrations |
| Vector DB | Qdrant | Fast cosine search, simple local setup |
| Logging / Eval | LangSmith + RAGAS | Tracing plus standardized RAG metrics |
| Validators | Manifests + deterministic config | Reproducible lineage and runs |
| UI / Serving | LangGraph Server + Streamlit | Quick local API and prototype UI |

## E. Evidence Details (brief)
- **Task 1**: `README.md` — lines 1–6. One‑line problem, audience noted.
- **Task 2**: `README.md` — Architecture Summary, Technology Stack tables.
- **Task 3**: `README.md` — HuggingFace datasets; `data/interim/manifest.json` lineage.
- **Task 4**: `app/graph_app.py` get_app; `src/graph.py` build_graph/build_all_graphs.
- **Task 5**: `scripts/run_eval_harness.py` full RAGAS pipeline; outputs under `deliverables/evaluation_evidence/`.
- **Task 6**: `src/retrievers` usage in harness; README results comparing retrievers.
- **Task 7**: `README.md` — Evaluation Results table; `data/processed/comparative_ragas_results.parquet`.

# Certification Challenge Evidence Map

## A. One-Screen Summary

- **Project name**: GDELT Knowledge Graph RAG Assistant
- **1-sentence problem**: Researchers and analysts working with the GDELT dataset struggle to quickly find answers to complex questions about knowledge graph construction, data formats, and analytical techniques without manually searching through dense technical documentation.
- **Primary user**: GDELT researchers, data scientists, and analysts working with knowledge graphs
- **Stack summary**: GPT-4.1-mini (LLM) + text-embedding-3-small (Embeddings) + LangGraph (Orchestration) + Qdrant (Vector DB) + LangSmith (Logging) + RAGAS (Validators) + Streamlit (UI)
- **Result summary**: Advanced retrieval (Cohere Rerank at 94.4% average) outperformed naive retrieval (92.6% average) by 1.8 percentage points, with Context Precision showing the largest improvement (+6.3%).

## B. Task Compliance Table

| Task | Status | Evidence (file + section) | Scope / Clarity Note | One Improvement |
|------|--------|---------------------------|---------------------|-----------------|
| 1. Problem & Audience | ✅ Yes | `docs/deliverables.md` lines 128-134 | Clear problem statement and user identification | None needed |
| 2. Solution & Stack | ✅ Yes | `docs/deliverables.md` lines 172-184, `docs/initial-architecture.md` lines 9-18 | Complete stack mapping with rationale | None needed |
| 3. Data & Chunking | ✅ Yes | `docs/deliverables.md` lines 227-256 + [HF sources-v2](https://huggingface.co/datasets/dwb2023/gdelt-rag-sources-v2) | Page-level chunking strategy with metadata preservation, publicly verifiable | None needed |
| 4. End-to-End Prototype | ✅ Yes | `README.md` lines 31-74, `app/graph_app.py` | LangGraph Studio deployment with local endpoint | None needed |
| 5. Golden Test + RAGAS | ✅ Yes | `deliverables/evaluation_evidence/RUN_MANIFEST.json` lines 92-121 + [HF golden-testset-v2](https://huggingface.co/datasets/dwb2023/gdelt-rag-golden-testset-v2) + [HF evaluation-inputs](https://huggingface.co/datasets/dwb2023/gdelt-rag-evaluation-inputs) | Complete RAGAS evaluation with 4 metrics across 4 retrievers, publicly verifiable | None needed |
| 6. Advanced Retrieval | ✅ Yes | `src/retrievers.py` lines 20-89 + [HF evaluation-metrics](https://huggingface.co/datasets/dwb2023/gdelt-rag-evaluation-metrics) + [SQL Console](https://huggingface.co/datasets/dwb2023/gdelt-rag-evaluation-metrics/sql-console/LI3hpFs) | 4 retrieval strategies implemented and evaluated, publicly queryable | None needed |
| 7. Performance & Next Steps | ✅ Yes | [SQL Console](https://huggingface.co/datasets/dwb2023/gdelt-rag-evaluation-metrics/sql-console/LI3hpFs) + `docs/deliverables.md` lines 1110-1155 | Comparative analysis with clear performance improvements, publicly verifiable | None needed |

## C. Metrics Snapshot (Tasks 5 & 7)

**Primary Source**: [Interactive SQL Console](https://huggingface.co/datasets/dwb2023/gdelt-rag-evaluation-metrics/sql-console/LI3hpFs) - Query all 60 evaluation results live

| Strategy | Faithfulness | Response Relevancy | Context Precision | Context Recall |
|----------|-------------|-------------------|-------------------|----------------|
| naive | 92.2% | 94.8% | 84.7% | 98.8% |
| advanced | 95.5% | 94.6% | 91.0% | 96.7% |

**1 Strength:** Context Precision improved most significantly (+6.3% from 84.7% to 91.0%), demonstrating that Cohere Rerank effectively ranks relevant contexts higher than naive retrieval.

**1 Next Step:** Implement fine-tuned embeddings specifically for GDELT domain terminology to further improve semantic understanding of technical concepts like CAMEO codes and GKG fields.

**Public Verification**: Run this SQL query to validate results:
```sql
SELECT retriever,
       AVG(faithfulness) as avg_faith,
       AVG(answer_relevancy) as avg_relevancy,
       AVG(context_precision) as avg_precision,
       AVG(context_recall) as avg_recall
FROM train
GROUP BY retriever
ORDER BY (avg_faith + avg_relevancy + avg_precision + avg_recall) / 4 DESC;
```

## D. Stack Alignment (a16z LLM App Stack)

| Layer | Tool Used | Why (≤ 12 words) |
|-------|-----------|------------------|
| LLM | OpenAI GPT-4.1-mini | Optimal balance of reasoning capability and cost-effectiveness |
| Embeddings | text-embedding-3-small | Strong semantic search with 1536 dimensions at reasonable cost |
| Orchestrator | LangGraph | Production-grade framework for RAG chains and agentic workflows |
| Vector DB | Qdrant | High-performance vector similarity search with production-grade filtering |
| Logging / Eval | LangSmith + RAGAS | LLM trace observability and research-backed RAG evaluation metrics |
| Validators | RAGAS 0.2.10 | Comprehensive evaluation framework with faithfulness, relevancy, precision, recall |
| UI / Serving | Streamlit + LangGraph Studio | Interactive development interface and production serving capability |

## E. Evidence Details (brief)

**Task 1 - Problem & Audience**: `docs/deliverables.md` lines 128-134 - Clear 1-sentence problem statement identifying GDELT researchers as target users struggling with dense technical documentation.

**Task 2 - Solution & Stack**: `docs/initial-architecture.md` lines 9-18 - Complete a16z stack mapping with GPT-4.1-mini, text-embedding-3-small, LangGraph, Qdrant, LangSmith, RAGAS, and Streamlit with detailed rationale.

**Task 3 - Data & Chunking**: [HF sources-v2](https://huggingface.co/datasets/dwb2023/gdelt-rag-sources-v2) - Page-level chunking strategy using PyMuPDFLoader with metadata preservation, yielding 38 documents with SHA-256 fingerprints. Publicly verifiable.

**Task 4 - End-to-End Prototype**: `README.md` lines 31-74 - LangGraph Studio deployment with `uv run langgraph dev --allow-blocking` providing local endpoint at http://localhost:2024.

**Task 5 - Golden Test + RAGAS**: [HF golden-testset-v2](https://huggingface.co/datasets/dwb2023/gdelt-rag-golden-testset-v2) + [HF evaluation-inputs](https://huggingface.co/datasets/dwb2023/gdelt-rag-evaluation-inputs) - Complete RAGAS evaluation with 4 metrics (faithfulness, answer_relevancy, context_precision, context_recall) across 4 retrievers with 12 golden test questions. Publicly verifiable.

**Task 6 - Advanced Retrieval**: [HF evaluation-metrics](https://huggingface.co/datasets/dwb2023/gdelt-rag-evaluation-metrics) + [SQL Console](https://huggingface.co/datasets/dwb2023/gdelt-rag-evaluation-metrics/sql-console/LI3hpFs) - Factory function implementing 4 retrieval strategies: naive (dense), BM25 (sparse), ensemble (hybrid), and Cohere Rerank (contextual compression). Publicly queryable.

**Task 7 - Performance & Next Steps**: [SQL Console](https://huggingface.co/datasets/dwb2023/gdelt-rag-evaluation-metrics/sql-console/LI3hpFs) - Cohere Rerank achieved 94.4% average vs 92.6% naive baseline, with Context Precision showing largest improvement (+6.3%). Publicly verifiable.

---

## F. Public Verification (Scientific Contribution)

**First Publicly Available GDELT RAG Evaluation Suite** - All data and metrics published to HuggingFace for transparent validation:

**Dataset Cards** (complete metadata, schemas, provenance):
- [sources-v2](https://huggingface.co/datasets/dwb2023/gdelt-rag-sources-v2) - 38 source documents with SHA-256 fingerprints
- [golden-testset-v2](https://huggingface.co/datasets/dwb2023/gdelt-rag-golden-testset-v2) - 12 QA pairs (RAGAS synthetic generation)
- [evaluation-inputs](https://huggingface.co/datasets/dwb2023/gdelt-rag-evaluation-inputs) - 60 evaluation records (4 retrievers × 12 questions + baseline)
- [evaluation-metrics](https://huggingface.co/datasets/dwb2023/gdelt-rag-evaluation-metrics) - 60 RAGAS results (full metric scores)

**Interactive Verification**:
- [SQL Console](https://huggingface.co/datasets/dwb2023/gdelt-rag-evaluation-metrics/sql-console/LI3hpFs) - Query all 60 evaluation results live
- Anyone can validate performance claims by running SQL aggregations
- Full provenance chain: raw PDFs → chunked docs → golden testset → evaluation results

**Reproducibility**:
- SHA-256 checksums for all datasets
- RUN_MANIFEST.json with model versions, API costs, timestamps
- Complete evaluation pipeline available in `scripts/run_full_evaluation.py`
