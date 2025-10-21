# CLAUDE.md — Workflow Refactor Assistant

## Role and Objective
You are an expert AI engineer and Prefect 3 practitioner. Your job is to **convert the monolithic `run_full_evaluation.py` script** into a modular, maintainable, and type-safe pipeline using **Pydantic v2** for schemas and **Prefect 3** for orchestration.

---

## High‑Level Goal
Transform the current single script into a structured system:

```
.
├── schemas/                 # Pydantic v2 data contracts
│   ├── types.py
│   └── settings.py
├── tasks/                   # Prefect tasks (idempotent, typed)
│   ├── ingest.py
│   ├── index_qdrant.py
│   ├── run_retriever.py
│   ├── evaluate_ragas.py
│   └── summarize.py
├── flows/
│   └── eval_flow.py         # Orchestration of the whole evaluation
└── utils/
    ├── io.py
    └── token_pack.py
```

The pipeline should behave identically to the original version—same outputs, metrics, and files—but gain validation, observability, and modularity.

---

## Core Principles

1. **Pydantic v2 for All Contracts**
   - Define strict schemas for each logical artifact:
     - `QuestionItem`, `ContextChunk`, `RetrievalResult`, `GenerationResult`, `JudgeScore`, `EvaluationRow`, `ComparativeRow`, `RunManifest`, `ModelRef`, `RetrieverConfig`, `QdrantConfig`.
   - Validate only at boundaries (ingestion, before persistence, pre‑evaluation).  
   - Use `TypeAdapter(list[T])` for batch validation.

2. **Prefect 3 for Orchestration**
   - Wrap each major step as a Prefect task with clear input/output schemas.
   - The flow should orchestrate all tasks deterministically, fan‑out per retriever, and persist after each phase.

3. **Parity and Determinism**
   - Metrics within ±0.5 % of baseline.
   - `temperature=0`, consistent `k` and token budgets.
   - Immediate persistence after inference.

4. **Artifacts**
   - `*_evaluation_inputs.parquet`
   - `*_evaluation_metrics.parquet`
   - `comparative_ragas_results.parquet`
   - `RUN_MANIFEST.json`

---

## Conversion Tasks (T1 → T5)

### **T1 – Schemas & Settings**
- Create `schemas/types.py` and `schemas/settings.py`.
- Implement all core models and a Pydantic‑Settings class for env‑driven configuration.
- Write minimal unit tests for instantiation and validation.

### **T2 – IO & Token Packer**
- `utils/io.py`: JSONL/Parquet helpers using orjson + Arrow.
- `utils/token_pack.py`: deterministic token‑capped context builder.
- Add unit tests verifying deterministic token counts.

### **T3 – Prefect Tasks**
- Refactor logic from `run_full_evaluation.py` into the five tasks listed above.
- Each task produces validated artifacts and logs key metadata (latency, cost, tokens).

### **T4 – Flow Orchestration**
- Build `flows/eval_flow.py` that chains the tasks, maps over retrievers, and persists all results.
- Expose a CLI entrypoint for local runs: `python -m flows.eval_flow`.

### **T5 – Parity & CI**
- Add pytest verifying metric parity (±0.5 %) and artifact presence.
- Optional GitHub Action: run flow on a small fixture dataset.

---

## Agent Work Pattern

1. **Plan → Implement → Verify**
   - Before coding, list new/changed files and functions.
   - Apply atomic diffs per task (T1–T5).

2. **Use Typed Boundaries**
   - Validate external inputs/outputs only.
   - Internal transformations may stay unvalidated for performance.

3. **Logs & Manifest**
   - Every run writes a `RunManifest` capturing:
     - git SHA, dataset ref, model names, prompt hashes, retriever params, total cost, timestamps.

4. **Parity Testing**
   - Compare new Parquet outputs against baseline (`run_full_evaluation.py`) metrics.

---

## Ready‑to‑Use Prompts

### **Planning**
> “Decompose `run_full_evaluation.py` into the modular file layout above. List each new module, its functions, and type signatures.”

### **Implementation**
> “Implement `schemas/types.py` with the Pydantic v2 models described in the CLAUDE.md. Include minimal tests validating coercion and serialization.”

### **Refactor → Tasks**
> “Move dataset loading into `tasks/ingest.py` returning `list[QuestionItem]`; add tests using a 3‑row synthetic dataset.”

### **Flow Integration**
> “Create `flows/eval_flow.py` to orchestrate all tasks with Prefect 3. Ensure outputs match the existing artifacts.”

### **CI Parity**
> “Write a test comparing metrics from old vs. new pipeline and assert difference < 0.005 for each RAGAS metric.”

---

## Runbook

```bash
# Install & setup
uv sync
export OPENAI_API_KEY=...
export APP_OUT_DIR="data/processed/runs"

# Local test run
pytest -q
python -m flows.eval_flow
```

---

## Acceptance Criteria

| Category | Requirement |
|-----------|--------------|
| **Correctness** | Output parity ± 0.5 % on metrics |
| **Durability** | Artifacts persisted after each retriever |
| **Observability** | RunManifest includes all metadata |
| **Modularity** | Tasks isolated and idempotent |
| **Compliance** | Pydantic v2 + Prefect 3 only |

---

## Notes
- Keep the naming conventions and column headers identical to preserve compatibility.
- Add docstrings to each Prefect task summarizing its contract (`Input → Output → Artifacts → Side Effects`).
- Favor small atomic commits with descriptive messages (`feat(tasks): add ingest.py with schema validation`).

---
