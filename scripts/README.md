# Scripts Directory

All evaluation, validation, and data pipeline scripts for the GDELT RAG certification challenge.

## Quick Reference

| Script | Purpose | Duration | Cost | Output |
|--------|---------|----------|------|--------|
| **run_app_validation.py** | Validate application stack (100% pass required) | 2 min | $0 | Terminal: 23/23 checks |
| **run_eval_harness.py** | RAGAS evaluation (modular, uses src/) | 20-30 min | $5-6 | 16 CSV files + manifest |
| **run_full_evaluation.py** | RAGAS evaluation (standalone reference) | 20-30 min | $5-6 | 16 CSV files + manifest |
| **ingest_raw_pdfs.py** | Extract raw PDFs → interim datasets | 5-10 min | $2-3 | data/interim/* |
| **publish_interim_datasets.py** | Upload interim datasets to HuggingFace | 1 min | $0 | HF repos |

## Naming Convention

Scripts follow a consistent naming pattern that indicates execution frequency:

- **`run_*`** - REPEATABLE operations (evaluation, validation)
- **`ingest_*`** - ONE-TIME data ingestion
- **`publish_*`** - ONE-TIME publishing/deployment

This makes it immediately clear which scripts are idempotent operations vs. one-time setup tasks.

## Common Workflows

### Standard Development Workflow

```bash
# 1. Validate environment and src/ modules (MUST PASS 100%)
source .venv/bin/activate
PYTHONPATH=. python scripts/run_app_validation.py

# 2. Run comparative evaluation
PYTHONPATH=. python scripts/run_eval_harness.py
# or use make command:
make eval

# 3. Publish datasets (optional, one-time)
python scripts/publish_interim_datasets.py
```

### Quick Validation (before deployment)

```bash
# Fast check - validates environment + module imports + factory patterns
make validate
# Expected: 23/23 checks PASS (100%)
```

### Standalone Evaluation (works without src/)

```bash
# Self-contained reference implementation
python scripts/run_full_evaluation.py
```

---

## Script Details

### run_app_validation.py

**Purpose**: Validates the complete application stack (environment, src/ modules, LangGraph workflows).

**Usage**:
```bash
PYTHONPATH=. python scripts/run_app_validation.py
```

**Validation Checks** (23 total):
1. **Environment validation** (9 checks):
   - API keys set (OPENAI_API_KEY, COHERE_API_KEY)
   - Qdrant connectivity (localhost:6333)
   - Package imports (langchain, langgraph, qdrant_client, datasets, ragas)

2. **Module import validation** (6 checks):
   - All src/ modules importable without errors
   - Verifies factory pattern (retrievers not created at import time)

3. **Factory pattern validation** (3 checks):
   - Documents load from HuggingFace
   - Vector store created successfully
   - All 4 retrievers created via factory

4. **Graph compilation** (4 checks):
   - Each of 4 graphs compiles successfully
   - StateGraph structure valid

5. **Functional testing** (1 check):
   - Test query executes through all 4 retrievers

**Exit Codes**:
- `0`: All checks passed (ready for deployment)
- `1`: One or more checks failed (fix before proceeding)

**Output**: Colored terminal output with pass/fail status and diagnostic recommendations.

**Why "app_validation"?** - Clarifies this validates the APPLICATION stack, not just generic "validation". Distinguishes from potential future scripts like `run_data_validation.py` or `run_model_validation.py`.

---

### run_eval_harness.py vs run_full_evaluation.py

**What's the difference?**

`run_eval_harness.py` is a **simplified version** of `run_full_evaluation.py` that uses `src/` modules instead of inline code. Same inputs, same outputs, same evaluation results - just cleaner code.

| Aspect | run_full_evaluation.py | run_eval_harness.py |
|--------|------------------------|---------------------|
| **What it does** | Full RAGAS evaluation | Full RAGAS evaluation |
| **Retrievers** | naive, bm25, ensemble, cohere_rerank | naive, bm25, ensemble, cohere_rerank |
| **Test questions** | 12 (from golden testset) | 12 (from golden testset) |
| **RAGAS metrics** | 4 metrics | 4 metrics |
| **Output files** | 16 files (12 CSVs + 4 parquet + manifest) | 16 files (12 CSVs + 4 parquet + manifest) |
| **Models used** | gpt-4.1-mini, text-embedding-3-small | gpt-4.1-mini, text-embedding-3-small |
| **Code** | 508 lines (inline implementations) | 268 lines (uses src/ modules) |
| **Results** | Identical | Identical |

**When to use which**:

- **Use `run_eval_harness.py` if**:
  - ✅ `make validate` shows 100% pass rate
  - ✅ You want cleaner code (268 vs 508 lines)
  - ✅ You trust the refactored src/ modules

- **Use `run_full_evaluation.py` if**:
  - ⚠️ You want a standalone reference (works without src/)
  - ⚠️ You want to see full implementation details inline
  - ⚠️ You're comparing results between old and new versions

---

### Controlling Vector Store Recreation

**By default, evaluation scripts REUSE the existing Qdrant collection** (faster).

```bash
# Reuse existing collection (default, faster)
make eval

# Force recreate collection (slower, ensures fresh embeddings)
make eval recreate=true

# Or directly:
PYTHONPATH=. python scripts/run_eval_harness.py --recreate
```

**Why this matters**:
- `recreate=false` (default): Saves ~15 minutes by reusing embeddings
- `recreate=true`: Ensures fresh embeddings if you changed documents/models

---

### Freezing Dataset Revisions (Reproducibility)

**Pin HuggingFace dataset revisions to prevent score drift over time.**

```bash
# Pin to specific dataset commits
export HF_SOURCES_REV=main@abc123  # Replace abc123 with actual commit SHA
export HF_GOLDEN_REV=main@def456   # Replace def456 with actual commit SHA

make eval
```

**Without pinning**: Dataset updates on HuggingFace can change your eval scores
**With pinning**: Same datasets every time → reproducible results

**To get current revision SHAs**:
```python
from datasets import load_dataset
ds = load_dataset("dwb2023/gdelt-rag-sources", split="train")
print(ds.info.download_checksums)  # Shows revision info
```

---

### Output Files

Both evaluation scripts generate the same 16 files:

```
deliverables/evaluation_evidence/
├── naive_raw_dataset.parquet              # Immediate save (pre-RAGAS)
├── naive_evaluation_dataset.csv           # Full RAGAS dataset
├── naive_detailed_results.csv             # Per-question metrics
├── bm25_raw_dataset.parquet
├── bm25_evaluation_dataset.csv
├── bm25_detailed_results.csv
├── ensemble_raw_dataset.parquet
├── ensemble_evaluation_dataset.csv
├── ensemble_detailed_results.csv
├── cohere_rerank_raw_dataset.parquet
├── cohere_rerank_evaluation_dataset.csv
├── cohere_rerank_detailed_results.csv
├── comparative_ragas_results.csv          # Main summary table
└── RUN_MANIFEST.json                      # Reproducibility metadata
```

**File Descriptions**:
- **`*_raw_dataset.parquet`**: Raw retrieval results (questions + contexts) saved immediately before RAGAS evaluation
- **`*_evaluation_dataset.csv`**: Full RAGAS-compatible dataset with all fields
- **`*_detailed_results.csv`**: Per-question breakdown of all RAGAS metrics
- **`comparative_ragas_results.csv`**: Summary table comparing all 4 retrievers (the "money shot")
- **`RUN_MANIFEST.json`**: Complete provenance (models, datasets, checksums, versions)

---

## ingest_raw_pdfs.py - Data Pipeline

**Purpose**: Extract raw PDFs → generate RAGAS golden testset → persist in multiple formats.

**Usage**:
```bash
python scripts/ingest_raw_pdfs.py
```

**Pipeline Steps**:
1. Extract PDFs from `data/raw/` → LangChain Documents (PyMuPDF)
2. Sanitize metadata for Arrow/JSON compatibility
3. Persist sources to `data/interim/` (JSONL, Parquet, HFDS)
4. Generate synthetic test questions via RAGAS (12 QA pairs)
5. Persist golden testset to `data/interim/` (JSONL, Parquet, HFDS)
6. Create `manifest.json` with checksums + schema + provenance

**Output Artifacts**:
- `data/interim/sources.{jsonl,parquet,hfds}`
- `data/interim/golden_testset.{jsonl,parquet,hfds}`
- `data/interim/manifest.json`

**Duration**: 5-10 minutes
**Cost**: ~$2-3 in OpenAI API calls (RAGAS testset generation)

**When to run**:
- After adding new PDFs to `data/raw/`
- When regenerating golden testset with different parameters
- During initial project setup

**Why "ingest_raw_pdfs"?** - Clarifies this ingests RAW PDFs specifically (not other data sources). The `ingest_*` prefix indicates this is a ONE-TIME operation, not a repeatable workflow.

**See also**: [data/README.md](../data/README.md) for data flow and manifest schema.

---

## publish_interim_datasets.py - HuggingFace Upload

**Purpose**: Upload `data/interim/` datasets to HuggingFace Hub for versioning and sharing.

**Prerequisites**:
```bash
# Set HuggingFace token
export HF_TOKEN=hf_...

# Install huggingface-hub
pip install huggingface-hub
```

**Usage**:
```bash
python scripts/publish_interim_datasets.py
```

**What it uploads**:
- `dwb2023/gdelt-rag-sources` (38 documents)
- `dwb2023/gdelt-rag-golden-testset` (12 QA pairs)

**Output**:
- Updates `manifest.json` with dataset repo IDs and upload timestamp
- Creates dataset cards on HuggingFace Hub
- Prints dataset URLs for verification

**Duration**: 1-2 minutes
**Cost**: $0 (HuggingFace Hub free tier)

**When to run**:
- After initial ingestion (`scripts/ingest_raw_pdfs.py`)
- When updating datasets with new source material
- For certification submission to make datasets publicly accessible

**Why "publish_interim_datasets"?** - Clarifies this publishes INTERIM datasets (from `data/interim/`), not processed evaluation results. The `publish_*` prefix indicates this is a ONE-TIME operation.

**Repository URLs**:
- [dwb2023/gdelt-rag-sources](https://huggingface.co/datasets/dwb2023/gdelt-rag-sources)
- [dwb2023/gdelt-rag-golden-testset](https://huggingface.co/datasets/dwb2023/gdelt-rag-golden-testset)

**See also**: [scripts/ingest_raw_pdfs.py](#ingest_raw_pdfspy---data-pipeline) - Generate interim datasets first.

---

## Make Commands

For convenience, use Makefile targets instead of running scripts directly:

```bash
make validate   # Run application validation (run_app_validation.py)
make eval       # Run evaluation (run_eval_harness.py)
make clean      # Clean Python cache
make help       # Show all commands
```

---

## Troubleshooting

**Issue**: `No module named 'src'`
- **Fix**: Set `PYTHONPATH=.` before running scripts
- **Or**: Use `make` commands which handle this automatically

**Issue**: `QdrantException: Connection refused`
- **Fix**: Start Qdrant: `docker-compose up -d qdrant`

**Issue**: `OpenAI API key not set`
- **Fix**: `export OPENAI_API_KEY=sk-...` or add to `.env` file

**Issue**: Validation fails at module imports
- **Fix**: Install dependencies: `uv pip install -e .`
- **Verify**: Check virtual environment is activated

**Issue**: RAGAS evaluation stalls
- **Cause**: Rate limiting or network issues
- **Fix**: Results are saved incrementally - just re-run the script

**Issue**: Script not found after renaming
- **Check**: Ensure you're using the new names:
  - ✅ `run_app_validation.py` (not `run_validation.py`)
  - ✅ `ingest_raw_pdfs.py` (not `ingest.py`)
  - ✅ `publish_interim_datasets.py` (not `publish_datasets.py`)

---

## Reference

For deeper understanding of the evaluation pipeline, factory patterns, and architecture:
- **[CLAUDE.md](../CLAUDE.md)** - Complete technical reference
- **[src/README.md](../src/README.md)** - Factory pattern guide
- **[data/README.md](../data/README.md)** - Data flow documentation
