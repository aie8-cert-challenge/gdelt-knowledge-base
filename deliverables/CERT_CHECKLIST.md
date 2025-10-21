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
| 3. Data & Chunking | ✅ Yes | `docs/deliverables.md` lines 227-256 | Page-level chunking strategy with metadata preservation | None needed |
| 4. End-to-End Prototype | ✅ Yes | `README.md` lines 31-74, `app/graph_app.py` | LangGraph Studio deployment with local endpoint | None needed |
| 5. Golden Test + RAGAS | ✅ Yes | `deliverables/evaluation_evidence/RUN_MANIFEST.json` lines 92-121 | Complete RAGAS evaluation with 4 metrics across 4 retrievers | None needed |
| 6. Advanced Retrieval | ✅ Yes | `src/retrievers.py` lines 20-89, `deliverables/evaluation_evidence/comparative_ragas_results.csv` | 4 retrieval strategies implemented and evaluated | None needed |
| 7. Performance & Next Steps | ✅ Yes | `deliverables/evaluation_evidence/comparative_ragas_results.csv`, `docs/deliverables.md` lines 1110-1155 | Comparative analysis with clear performance improvements | None needed |

## C. Metrics Snapshot (Tasks 5 & 7)

| Strategy | Faithfulness | Response Relevancy | Context Precision | Context Recall |
|----------|-------------|-------------------|-------------------|----------------|
| naive | 92.2% | 94.8% | 84.7% | 98.8% |
| advanced | 95.5% | 94.6% | 91.0% | 96.7% |

**1 Strength:** Context Precision improved most significantly (+6.3% from 84.7% to 91.0%), demonstrating that Cohere Rerank effectively ranks relevant contexts higher than naive retrieval.

**1 Next Step:** Implement fine-tuned embeddings specifically for GDELT domain terminology to further improve semantic understanding of technical concepts like CAMEO codes and GKG fields.

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

**Task 3 - Data & Chunking**: `docs/deliverables.md` lines 227-256 - Page-level chunking strategy using PyMuPDFLoader with metadata preservation, yielding 38 documents with SHA-256 fingerprints.

**Task 4 - End-to-End Prototype**: `README.md` lines 31-74 - LangGraph Studio deployment with `uv run langgraph dev --allow-blocking` providing local endpoint at http://localhost:2024.

**Task 5 - Golden Test + RAGAS**: `deliverables/evaluation_evidence/RUN_MANIFEST.json` lines 92-121 - Complete RAGAS evaluation with 4 metrics (faithfulness, answer_relevancy, context_precision, context_recall) across 4 retrievers with 12 golden test questions.

**Task 6 - Advanced Retrieval**: `src/retrievers.py` lines 20-89 - Factory function implementing 4 retrieval strategies: naive (dense), BM25 (sparse), ensemble (hybrid), and Cohere Rerank (contextual compression).

**Task 7 - Performance & Next Steps**: `deliverables/evaluation_evidence/comparative_ragas_results.csv` - Cohere Rerank achieved 94.4% average vs 92.6% naive baseline, with Context Precision showing largest improvement (+6.3%).
