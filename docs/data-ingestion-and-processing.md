# Data Ingestion & Processing Flow

> **Goal:** Establish a clear, reproducible, Parquet‑first pipeline with manifests and strict directory boundaries. All machine outputs land in `data/interim/` or `data/processed/`. The `deliverables/` tree is **derived only** via a manual script and never used as a working sink.

---

## Scope & Principles

* **Parquet‑first** for all tabular artifacts (no CSV in working dirs).
* **Two working sinks only:**

  * `data/interim/` – ephemeral, re‑generable staging artifacts produced by ingestion & light transforms.
  * `data/processed/` – durable, versionable, post‑processing & evaluation artifacts ready for publishing.
* **Deliverables are derived:** `deliverables/evaluation_evidence/` is generated **only** by `scripts/generate_deliverables.py` from `data/*` (never written to directly by ingestion/eval steps).
* **Manifests sign every stage** (content hashes, schema, params, environment) to ensure provenance & reproducibility.

---

## Directory Contract

```text
data/
  raw/              # immutable inputs (PDFs, external dumps)
  interim/          # ingestion outputs (Parquet + manifest)
  processed/        # evaluation & consolidation outputs (Parquet only)

deliverables/
  evaluation_evidence/  # human-friendly reports & run manifests (derived)
```

### Allowed Formats

* `data/interim/`: Parquet (`*.parquet`), HF dataset dir (`*.hfds`), JSON manifest only
* `data/processed/`: Parquet only (primary), optional HF export mirrors
* `deliverables/`: Markdown/CSV/Parquet images ok (presentation), **but always derived**

---

## Pipeline Overview (Mermaid)

```mermaid
flowchart LR
  A[raw PDFs / external data\n data/raw/*.pdf] --> B[Ingest]
  B -->|extract, split, normalize| C[data/interim\n sources.parquet\n golden_testset.parquet\n manifest.json]
  C --> D[Evaluate]
  D -->|per retriever| E[data/processed\n *_evaluation_dataset.parquet\n *_detailed_results.parquet]
  E --> F[Publish (optional)]
  E --> G[Generate Deliverables]
  F -->|HF| H[(Hub)]
  G -->|manual script| I[deliverables/evaluation_evidence\n reports, RUN_MANIFEST.json, tables]
```

---

## Stages, Triggers & Outputs

### 1) Ingestion

* **Entry points:**

  * `scripts/ingest_raw_pdfs.py` (canonical)
  * Optional: `scripts/publish_interim_datasets.py` (to HF mirrors)
* **Reads:** `data/raw/*.pdf`
* **Writes (Parquet‑first):**

  * `data/interim/sources.parquet` (documents)
  * `data/interim/golden_testset.parquet` (QA pairs)
  * `data/interim/manifest.json` (env, params, paths, SHA‑256 fingerprints, quick schema)
* **Notes:**

  * No CSV in interim.
  * If HF mirrors are created, they are *secondary* to Parquet.

### 2) Evaluation / Processing

* **Entry points:**

  * `scripts/run_eval_harness.py` (single, deterministic run)
  * `scripts/run_full_evaluation.py` (batch/all retrievers)
* **Reads:** `data/interim/*.parquet`
* **Writes (per retriever):**

  * `data/processed/{retriever}_evaluation_inputs.parquet`
  * `data/processed/{retriever}_evaluation_metrics.parquet`
* **Consolidations (optional):**

  * `scripts/publish_processed_datasets.py` → pushes consolidated HF datasets from `processed/`
* **Notes:**

  * Need to confirm trigger for running generate_manifest (`src/utils/manifest.py`)
  * Numeric metrics are `float64`.
  * Lists are `list<string>` columns (`retrieved_contexts`, `reference_contexts`).

### 3) Deliverables (Derived Only)

* **Entry point:** `scripts/generate_deliverables.py` (manual)
* **Reads:** `data/interim/*.parquet`, `data/processed/*.parquet`, `data/interim/manifest.json`, `data/processed/run_manifest.json`
* **Writes:** `deliverables/evaluation_evidence/`

  * Markdown reports, comparison tables, plots, and any CSVs intended for presentation

* **Rule:** No process other than `generate_deliverables.py` writes into `deliverables/`.

---

## Schemas (Authoritative)

### Ingestion Artifacts (Parquet)

* **`sources.parquet`**

  * `metadata` (struct or flattened columns like `metadata.title`, `metadata.page`, …)
  * `page_content` (string)

* **`golden_testset.parquet`**

  * `user_input` (string)
  * `reference` (string)
  * `reference_contexts` (list<string>)
  * `synthesizer_name` (string)

### Evaluation Artifacts (Parquet per retriever)

* **`*_evaluation_inputs.parquet`**

  * `retriever` (string)
  * `user_input` (string)
  * `retrieved_contexts` (list<string>)
  * `reference_contexts` (list<string>)
  * `response` (string)
  * `reference` (string)

* **`*_evaluation_metrics.parquet`**

  * All columns above
  * `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall` (float64)

---

## File Naming & Conventions

* **Per‑retriever prefix:** `naive_`, `bm25_`, `ensemble_`, `cohere_rerank_`
* **Suffixes:** `_evaluation_inputs.parquet`, `_evaluation_metrics.parquet`
* **No indexes** embedded; Parquet written with `preserve_index=False`.
* **Compression:** ZSTD (`compression="zstd"`).

---

## Manifests & Provenance

* **`data/interim/manifest.json`** – the ingestion signature

  * env (python, libs), params, paths, SHA‑256 of Parquet artifacts, quick schema
* **`data/processed/run_manifest.json`** – the human‑oriented run ledger

  * models, retrievers, metrics summary, pointers back to interim manifest fingerprints

---

## Guardrails & Checks

* **Schema validation** before write: enforce Arrow `Features` or `pyarrow.schema`.
* **Determinism:** LLM `temperature=0`, fixed seeds for any sampling.
* **Directory boundaries:**

  * Ingestion/eval **never** write to `deliverables/`.
  * `generate_deliverables.py` **only reads** from `data/*` and writes to `deliverables/`.
* **HF Preview sanity:** publish as `DatasetDict({"train": ds})`, provide `Features`, shard `<50MB` if needed.

---

## Make Targets (Suggested)

```make
.PHONY: ingest eval deliverables publish-interim publish-processed

ingest:
	uv run python scripts/ingest_raw_pdfs.py

eval:
	uv run python scripts/run_full_evaluation.py

publish-interim:
	uv run python scripts/publish_interim_datasets.py

publish-processed:
	uv run python scripts/publish_processed_datasets.py

deliverables:
	uv run python scripts/generate_deliverables.py
```

---

## Open Questions (for future iteration)

* Should `RUN_MANIFEST.json` also be mirrored in `data/processed/` for machine workflows?
* Add automated README sync from `RUN_MANIFEST.json`? (`make readme-sync`)
* Add a schema registry (pydantic models + `pyarrow.schema`) to prevent drift across repos.
