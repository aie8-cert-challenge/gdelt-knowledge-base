# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is the certification challenge project for AI Engineering Bootcamp Cohort 8. The project follows a product-focused approach requiring you to define Problem, Solution, and Audience before building.

**Important**: See `docs/certification-challenge-task-list.md` for the complete certification challenge requirements across all 7 tasks. The README.md contains the high-level product management framework.

## Python Environment

- **Python Version**: 3.11 (specified in `.python-version`)
- **Package Manager**: `uv` for dependency management
- **Dependencies**: Managed via `pyproject.toml`

### Setup

```bash
# Create and activate virtual environment
uv venv --python 3.11
source .venv/bin/activate  # Linux/WSL/Mac

# Install dependencies
uv pip install -e .
```

### Running the Application

```bash
# Run the main script
python main.py
```

## Project Architecture

This certification challenge builds a production-grade Agentic RAG application across 7 tasks. The expected technology stack includes:

### Core Components

1. **LLM Layer** — Choice of commercial or OSS models (OpenAI API or local models)
2. **Embedding Model** — For vector representations of documents
3. **Vector Database** — Storage and retrieval (e.g., Qdrant, Pinecone, Weaviate)
4. **Orchestration** — LangChain/LangGraph for agent workflows
5. **External APIs** — At minimum: search API (Tavily/SERP) + domain-specific data sources
6. **Evaluation** — RAGAS framework for RAG assessment
7. **Monitoring** — LangSmith or similar for observability
8. **User Interface** — Delivery mechanism for end users

### Development Phases

- **Tasks 1-3**: Problem definition, solution design, data sourcing
- **Task 4**: End-to-end prototype with local deployment
- **Task 5**: Golden test dataset creation and baseline RAGAS evaluation
- **Task 6**: Advanced retrieval techniques implementation
- **Task 7**: Performance assessment and iteration planning

## Key Technologies from AIE8 Course

This project leverages patterns from previous AIE8 sessions:

- **RAG Implementation**: Session 02 (custom) and Session 04 (production patterns)
- **Agent Workflows**: Session 05-06 (LangGraph single/multi-agent patterns)
- **Vector Search**: Qdrant integration patterns
- **Evaluation**: RAGAS metrics (faithfulness, response relevancy, context precision/recall)

## Documentation

- `README.md` — High-level product management framework (Problem, Solution, Audience)
- `docs/certification-challenge-task-list.md` — Complete task list with all 7 tasks and deliverables
- `docs/cert-challenge.excalidraw` — Architecture diagram (editable in Excalidraw)

## Development Workflow

- Work on feature/task branches, not directly on main
- Follow standard Python .gitignore patterns (already configured)
- Each task should have clear deliverables documented in `docs/certification-challenge-task-list.md`

## Final Deliverable Requirements

The completed project must include:

1. Public GitHub repository
2. 5-minute (or less) Loom video demo
3. Written document addressing all deliverables
4. All application code
