# Data Directory

Data flow, artifact storage, and provenance tracking for the GDELT RAG evaluation pipeline.

## Directory Structure

```
data/
├── raw/                 # Source PDFs (immutable)
│   └── 2503.07584v3.pdf     # "Talking to GDELT Through Knowledge Graphs" paper
├── interim/             # Processed documents + golden testset + manifest
│   ├── sources.jsonl            # Human-readable document format
│   ├── sources.parquet          # Analytics-optimized format
│   ├── sources.hfds/            # HuggingFace Dataset format
│   ├── golden_testset.jsonl     # RAGAS test questions
│   ├── golden_testset.parquet
│   ├── golden_testset.hfds/
│   └── manifest.json            # Provenance + checksums
└── processed/           # (DEPRECATED) Old evaluation results
```

**Note**: Current evaluation results are in `deliverables/evaluation_evidence/`.

## Data Flow

### Complete Pipeline

```
raw/2503.07584v3.pdf (12 pages, source material)
    ↓
[scripts/ingest_raw_pdfs.py]
    - PyMuPDF extraction
    - Page-level chunking (38 documents)
    - RAGAS synthetic testset generation (12 QA pairs)
    - Multi-format persistence
    ↓
interim/sources.{jsonl,parquet,hfds}
interim/golden_testset.{jsonl,parquet,hfds}
interim/manifest.json
    ↓
[scripts/publish_interim_datasets.py]
    ↓
dwb2023/gdelt-rag-sources (HuggingFace)
dwb2023/gdelt-rag-golden-testset (HuggingFace)
    ↓
[scripts/run_eval_harness.py]
    - Load from HuggingFace
    - Create Qdrant vector store
    - Run 4 retrievers × 12 questions = 48 queries
    - RAGAS evaluation (4 metrics)
    ↓
deliverables/evaluation_evidence/
    ├── naive_raw_dataset.parquet
    ├── naive_evaluation_dataset.csv
    ├── naive_detailed_results.csv
    ├── (same for bm25, ensemble, cohere_rerank)
    ├── comparative_ragas_results.csv
    └── RUN_MANIFEST.json
```

## File Formats

### Why Three Formats?

| Format | Purpose | Use Case | Tools |
|--------|---------|----------|-------|
| **JSONL** | Human-readable, git-friendly | Code review, debugging | `cat`, `jq` |
| **Parquet** | Analytics-optimized, columnar | Data analysis | Pandas, DuckDB |
| **HFDS** | HuggingFace Dataset format | Fast loading, versioning | `datasets` library |

**Example Usage**:
```python
# JSONL
import json
with open("data/interim/sources.jsonl") as f:
    for line in f:
        doc = json.loads(line)

# Parquet
import pandas as pd
df = pd.read_parquet("data/interim/sources.parquet")

# HuggingFace Dataset
from datasets import load_from_disk
ds = load_from_disk("data/interim/sources.hfds")
```

## Manifest Schema

`interim/manifest.json` provides complete provenance:

```json
{
  "id": "ragas_pipeline_...",
  "generated_at": "2025-10-17T07:12:55Z",
  "run": {"random_seed": 42, "git_commit_sha": "a0d4dd3"},
  "env": {"python": "3.11.13", "ragas": "0.2.10"},
  "params": {"OPENAI_MODEL": "gpt-4.1-mini", "TESTSET_SIZE": 10},
  "paths": {
    "sources": {"jsonl": "...", "parquet": "...", "hfds": "..."},
    "golden_testset": {"jsonl": "...", "parquet": "...", "hfds": "..."}
  },
  "checksums": {"sources_jsonl": "sha256:...", ...}
}
```

## HuggingFace Datasets

### dwb2023/gdelt-rag-sources

**Size**: 38 documents (pages from PDF)

**Schema**:
```python
{
    "page_content": str,       # Full text of the page
    "metadata": {
        "source": str,         # PDF filename
        "page": int,           # Page number
        "total_pages": int,
        "title": str,
        "author": str
    }
}
```

### dwb2023/gdelt-rag-golden-testset

**Size**: 12 question-answer pairs

**Schema** (RAGAS-compatible):
```python
{
    "user_input": str,              # Question
    "reference": str,               # Ground truth answer
    "reference_contexts": [str]     # Reference contexts
}
```

## Evaluation Output Files

### deliverables/evaluation_evidence/

**Per-Retriever Files** (12 files total):
- `<retriever>_raw_dataset.parquet` - Raw results (pre-RAGAS)
- `<retriever>_evaluation_dataset.csv` - Full RAGAS dataset
- `<retriever>_detailed_results.csv` - Per-question breakdown

**Summary Files**:
- `comparative_ragas_results.csv` - Main comparison table
- `RUN_MANIFEST.json` - Complete provenance

Example comparative results:
```csv
Retriever,Faithfulness,Answer Relevancy,Context Precision,Context Recall,Average
Cohere Rerank,0.9615,0.9501,0.9999,0.9833,0.9737
Ensemble,0.9545,0.9625,0.8566,0.9833,0.9392
Bm25,0.9386,0.9560,0.8425,0.9833,0.9301
Naive,0.9653,0.9468,0.7999,0.9833,0.9238
```

## Common Operations

### Loading Documents Locally

```python
# Option 1: From HuggingFace (recommended)
from src.utils import load_documents_from_huggingface
documents = load_documents_from_huggingface()

# Option 2: From local JSONL
import json
from langchain_core.documents import Document

documents = []
with open("data/interim/sources.jsonl") as f:
    for line in f:
        item = json.loads(line)
        doc = Document(page_content=item["page_content"], metadata=item["metadata"])
        documents.append(doc)
```

### Verifying Data Integrity

```python
import hashlib
import json

# Load manifest
with open("data/interim/manifest.json") as f:
    manifest = json.load(f)

# Verify checksum
with open("data/interim/sources.jsonl", "rb") as f:
    actual = hashlib.sha256(f.read()).hexdigest()

expected = manifest["checksums"]["sources_jsonl"]
assert actual == expected, "Checksum mismatch!"
```

### Regenerating Datasets

```bash
# Re-run ingestion pipeline
python scripts/ingest_raw_pdfs.py

# Verify new manifest
cat data/interim/manifest.json | jq '.schema'
```

## Data Lineage

### Full Lineage Chain

1. `raw/2503.07584v3.pdf` → SHA256 checksum
2. `interim/manifest.json` → Source checksum + ingestion params
3. HuggingFace datasets → Git revision IDs
4. `RUN_MANIFEST.json` → HF revisions + eval params + results

This enables **bit-perfect reproducibility**.

## Best Practices

### DO:
✅ Check `manifest.json` for dataset provenance
✅ Pin HuggingFace dataset revisions for reproducibility
✅ Verify checksums before evaluation runs
✅ Use HuggingFace datasets for loading (faster)

### DON'T:
❌ Modify files in `data/raw/` (immutable source)
❌ Manually edit JSONL/Parquet (regenerate via `ingest_raw_pdfs.py`)
❌ Delete `manifest.json` (breaks provenance)
❌ Use `data/processed/` for new evaluations (deprecated)

## Troubleshooting

**Issue**: `FileNotFoundError: data/interim/sources.jsonl`
**Fix**: Run `python scripts/ingest_raw_pdfs.py`

**Issue**: Checksum mismatch after loading from HuggingFace
**Fix**: Pin to specific revision: `load_documents_from_huggingface(revision="abc123")`

**Issue**: Manifest missing required fields
**Fix**: Re-run `python scripts/ingest_raw_pdfs.py` with latest code

## Further Reading

- **[scripts/README.md](../scripts/README.md)** - Ingestion and evaluation scripts
- **[src/utils/loaders.py](../src/utils/loaders.py)** - HuggingFace loading functions
- **[CLAUDE.md](../CLAUDE.md)** - Complete data flow documentation
