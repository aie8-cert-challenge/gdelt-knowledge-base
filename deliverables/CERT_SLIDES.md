## Slide 1: Intro / Hook
GDELT’s documentation is vast and complex; analysts need fast, precise answers. GDELT Knowledge Graph RAG Assistant uses multi‑strategy retrieval and LangGraph to deliver grounded responses.

## Slide 2: Task 1 — Problem & Audience
**Goal:** Articulate problem and target user.
**Evidence:** `README.md` — lines 1–6.
**Result:** One‑sentence problem and audience defined for analysts/engineers.
**Next Step:** Add 1–2 paragraph narrative in `docs/` for completeness.

## Slide 3: Task 2 — Proposed Solution & Stack
**Goal:** Describe solution and select tools per stack layer.
**Evidence:** `README.md` — Architecture Summary, Technology Stack.
**Result:** LangGraph‑based RAG with OpenAI, Qdrant, RAGAS, LangSmith, Streamlit.
**Next Step:** Add 1‑sentence rationale per tool inline in the stack table.

## Slide 4: Task 3 — Data & Chunking
**Goal:** Document data sources, APIs, and default chunking.
**Evidence:** `README.md` — HuggingFace datasets; `data/interim/manifest.json`.
**Result:** Sources + golden testset published with SHA‑256 lineage.
**Next Step:** State explicit chunking rationale in `src/utils` docstrings.

## Slide 5: Task 4 — End‑to‑End Prototype
**Goal:** Local end‑to‑end Agentic RAG prototype.
**Evidence:** `app/graph_app.py` get_app; `src/graph.py` build_graph.
**Result:** LangGraph Server app compiles and serves default rerank graph.
**Next Step:** Surface Streamlit UI launch instructions in `README.md`.

## Slide 6: Task 5 — Golden Test Set & RAGAS
**Goal:** Create golden dataset and evaluate with RAGAS.
**Evidence:** `scripts/run_eval_harness.py`; outputs in `deliverables/evaluation_evidence/`.
**Result:** Faithfulness, Answer Relevancy, Context Precision/Recall computed and stored.
**Next Step:** Include small example table in `deliverables/` overview doc.

## Slide 7: Task 6 — Advanced Retrieval
**Goal:** Install and test advanced retrieval methods.
**Evidence:** `README.md` results; usage of BM25, ensemble, Cohere rerank.
**Result:** Cohere rerank improved average and precision vs naïve.
**Next Step:** Add per‑technique “why useful” bullets tied to data properties.

## Slide 8: Task 7 — Performance & Next Steps
**Goal:** Compare naïve vs advanced and plan improvements.
**Evidence:** `README.md` Evaluation Results; `data/processed/comparative_ragas_results.parquet`.
**Result:** Advanced reranking wins; metrics summarized and persisted.
**Next Step:** Deeper error analysis to tune fusion and k values.

## Slide 9: Conclusion / Reflection
Key results: reproducible pipeline, advanced retrieval lifts precision, full lineage. Lessons: orchestrator patterns, manifesting data, metric tradeoffs. Next: error analysis, rationale logging, selective rerank to optimize cost.

# Certification Challenge Presentation Outline

## Slide 1: Intro / Hook

**Context**: GDELT (Global Database of Events, Language, and Tone) researchers face a steep learning curve when working with complex knowledge graph construction, data formats, and analytical techniques scattered across dense technical documentation.

**Problem**: Researchers spend hours manually searching through academic papers and codebooks to find answers to specific GDELT implementation questions.

**Target User**: GDELT researchers, data scientists, and analysts working with knowledge graphs who need quick access to domain-specific technical information.

**App Name**: GDELT Knowledge Graph RAG Assistant

**Core Idea**: An intelligent question-answering system that provides instant, accurate answers about GDELT knowledge graph construction with full source citation and provenance tracking.

---

## Slide 2: Task 1 — Problem & Audience

**Goal**: Articulate the problem and identify the target user for the application

**Evidence**: `docs/deliverables.md` lines 128-134

**Result**: Clear 1-sentence problem statement identifying GDELT researchers as target users struggling with dense technical documentation, plus detailed user friction analysis explaining why this is a significant problem for the specific user base.

**Next Step**: Conduct user interviews with actual GDELT researchers to validate problem assumptions and gather additional use cases.

---

## Slide 3: Task 2 — Proposed Solution & Stack

**Goal**: Describe the proposed solution and justify technology stack choices

**Evidence**: `docs/initial-architecture.md` lines 9-18, `docs/deliverables.md` lines 172-184

**Result**: Complete a16z LLM App Stack implementation with GPT-4.1-mini (LLM), text-embedding-3-small (Embeddings), LangGraph (Orchestration), Qdrant (Vector DB), LangSmith (Logging), RAGAS (Evaluation), and Streamlit (UI), each with detailed rationale for tooling choices.

**Next Step**: Implement agentic reasoning strategy with Retrieval Agent and GDELT Domain Expert Agent for complex multi-hop queries.

---

## Slide 4: Task 3 — Data & Chunking

**Goal**: Identify data sources, external APIs, and implement chunking strategy

**Evidence**:
- `docs/deliverables.md` lines 227-256
- **Public Dataset**: [gdelt-rag-sources-v2](https://huggingface.co/datasets/dwb2023/gdelt-rag-sources-v2) (38 documents, publicly verifiable)

**Result**: Page-level chunking strategy using PyMuPDFLoader with metadata preservation, yielding 38 documents from GDELT research paper with SHA-256 fingerprints for data integrity, plus Tavily Search API integration for external knowledge augmentation. All source documents published to HuggingFace with complete provenance tracking.

**Next Step**: Implement graph-based retrieval using Neo4j to leverage GDELT entity relationships for enhanced context retrieval.

---

## Slide 5: Task 4 — End-to-End Prototype

**Goal**: Build and deploy an end-to-end RAG prototype with local endpoint

**Evidence**: `README.md` lines 31-74, `app/graph_app.py`

**Result**: Production-ready LangGraph Studio deployment with `uv run langgraph dev --allow-blocking` providing local endpoint at http://localhost:2024, complete with multi-strategy retrieval system and LangGraph orchestration.

**Next Step**: Deploy to production environment with Docker Compose and implement user authentication for secure access.

---

## Slide 6: Task 5 — Golden Test Set & RAGAS

**Goal**: Create golden test dataset and evaluate pipeline using RAGAS framework

**Evidence**:
- `deliverables/evaluation_evidence/RUN_MANIFEST.json` lines 92-121
- **Public Datasets**:
  - [gdelt-rag-golden-testset-v2](https://huggingface.co/datasets/dwb2023/gdelt-rag-golden-testset-v2) (12 QA pairs, publicly verifiable)
  - [gdelt-rag-evaluation-inputs](https://huggingface.co/datasets/dwb2023/gdelt-rag-evaluation-inputs) (60 evaluation records)

**Result**: Complete RAGAS evaluation with 4 metrics (faithfulness, answer_relevancy, context_precision, context_recall) across 4 retrievers using 12 golden test questions generated with RAGAS 0.2.10 synthetic data generation. All evaluation inputs published to HuggingFace for reproducibility.

**Next Step**: Expand golden test set to 50+ questions covering more diverse GDELT use cases and edge cases.

---

## Slide 7: Task 6 — Advanced Retrieval

**Goal**: Implement and test advanced retrieval techniques

**Evidence**:
- `src/retrievers.py` lines 20-89
- **Public Dataset**: [gdelt-rag-evaluation-metrics](https://huggingface.co/datasets/dwb2023/gdelt-rag-evaluation-metrics) (60 results with RAGAS scores)
- **Interactive SQL Console**: [Live Metrics Explorer](https://huggingface.co/datasets/dwb2023/gdelt-rag-evaluation-metrics/sql-console/LI3hpFs)

**Result**: Factory function implementing 4 retrieval strategies: naive (dense vector), BM25 (sparse keyword), ensemble (hybrid 50/50), and Cohere Rerank (contextual compression), with comprehensive evaluation infrastructure. All evaluation results publicly queryable via HuggingFace SQL console.

**Next Step**: Implement learning-to-rank models using RAGAS scores as quality labels for further retrieval optimization.

---

## Slide 8: Task 7 — Performance & Next Steps

**Goal**: Compare performance and articulate future improvements

**Evidence**:
- **Interactive SQL Console**: [Live Metrics Explorer](https://huggingface.co/datasets/dwb2023/gdelt-rag-evaluation-metrics/sql-console/LI3hpFs) (query all 60 evaluation results)
- `docs/deliverables.md` lines 1110-1155

**Result**: Cohere Rerank achieved 94.4% average vs 92.6% naive baseline, with Context Precision showing largest improvement (+6.3%), plus detailed comparative analysis and future improvement roadmap. **All results publicly verifiable via SQL console** - anyone can validate these claims.

**Next Step**: Implement fine-tuned embeddings for GDELT domain terminology and deploy production monitoring with LangSmith for continuous evaluation.

---

## Slide 9: Conclusion / Reflection

**Key Results**:
1. **Advanced retrieval significantly outperformed naive approach** - Cohere Rerank achieved 94.4% average performance vs 92.6% baseline, with Context Precision improving by 6.3%
2. **Complete production-ready RAG system** - End-to-end LangGraph deployment with 4 retrieval strategies, comprehensive evaluation framework, and full provenance tracking
3. **Scientific contribution** - Published **first publicly available GDELT RAG evaluation suite** with 4 HuggingFace datasets enabling reproducible research:
   - [sources-v2](https://huggingface.co/datasets/dwb2023/gdelt-rag-sources-v2) - 38 source documents
   - [golden-testset-v2](https://huggingface.co/datasets/dwb2023/gdelt-rag-golden-testset-v2) - 12 QA pairs
   - [evaluation-inputs](https://huggingface.co/datasets/dwb2023/gdelt-rag-evaluation-inputs) - 60 evaluation records
   - [evaluation-metrics](https://huggingface.co/datasets/dwb2023/gdelt-rag-evaluation-metrics) - 60 RAGAS results ([**query live**](https://huggingface.co/datasets/dwb2023/gdelt-rag-evaluation-metrics/sql-console/LI3hpFs))

**Lessons Learned**:
- Context Precision is the most challenging metric to optimize
- Cohere Rerank provides best quality but at higher cost
- Page-level chunking preserves document structure better than character splitting
- Public datasets enable transparent validation of claims

**Next Iteration Plan**:
1. Implement fine-tuned embeddings for GDELT domain
2. Add graph-based retrieval with Neo4j
3. Deploy production monitoring with LangSmith
4. Expand golden test set to 50+ questions
5. Implement learning-to-rank models
6. Add user authentication and multi-tenant support

**Call to Action**: The system is ready for production deployment and can be extended to other knowledge graph domains beyond GDELT. All evaluation data is publicly available for validation and research.
