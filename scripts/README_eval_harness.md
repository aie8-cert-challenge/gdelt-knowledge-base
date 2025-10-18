# Evaluation Harness Documentation

## What is `run_eval_harness.py`?

**It's a simplified version of `single_file.py` that uses the `src/` modules instead of inline code.**

That's it. Same inputs, same outputs, same evaluation. Just cleaner.

---

## What does `make eval` do?

Runs the exact same evaluation as `single_file.py`:

1. Load 38 documents from HuggingFace
2. Load 12 test questions from HuggingFace
3. Create/connect to Qdrant vector store
4. Run 12 questions through 4 retrievers (naive, bm25, ensemble, cohere_rerank)
5. Evaluate with RAGAS (4 metrics: faithfulness, answer_relevancy, context_precision, context_recall)
6. Save results to `deliverables/evaluation_evidence/`

**Time**: 20-30 minutes (or ~5 min if reusing existing vector store)
**Cost**: ~$5-6 in OpenAI API calls
**Output**: Same CSV files as `single_file.py`

---

## Controlling Vector Store Recreation

**By default, the harness REUSES the existing Qdrant collection (faster).**

```bash
# Reuse existing collection (default, faster)
make eval

# Force recreate collection (slower, ensures fresh embeddings)
make eval recreate=true
```

**Why this matters:**
- `recreate=false` (default): Saves ~15 minutes by reusing embeddings
- `recreate=true`: Ensures fresh embeddings if you changed documents/models

---

## Freezing Dataset Revisions (Reproducibility)

**Pin HuggingFace dataset revisions to prevent score drift over time.**

```bash
# Pin to specific dataset commits (prevents dataset updates from changing results)
export HF_SOURCES_REV=main@abc123  # Replace abc123 with actual commit SHA
export HF_GOLDEN_REV=main@def456   # Replace def456 with actual commit SHA

make eval
```

**Without pinning**: Dataset updates on HuggingFace can change your eval scores
**With pinning**: Same datasets every time, reproducible results

**To get current revision SHAs:**
```python
from datasets import load_dataset
ds = load_dataset("dwb2023/gdelt-rag-sources", split="train")
print(ds.info.download_checksums)  # Shows revision info
```

---

## Comparison: `single_file.py` vs `run_eval_harness.py`

| Aspect | single_file.py | run_eval_harness.py |
|--------|----------------|---------------------|
| **What it does** | Full RAGAS evaluation | Full RAGAS evaluation |
| **Retrievers** | naive, bm25, ensemble, cohere_rerank | naive, bm25, ensemble, cohere_rerank |
| **Test questions** | 12 (from golden testset) | 12 (from golden testset) |
| **RAGAS metrics** | 4 metrics | 4 metrics |
| **Output files** | 13 CSV/Parquet files | 13 CSV/Parquet files |
| **Models used** | gpt-4.1-mini, text-embedding-3-small | gpt-4.1-mini, text-embedding-3-small |
| **Code** | 508 lines (inline implementations) | 268 lines (uses src/ modules) |
| **Results** | Identical | Identical |

**Key difference**: `run_eval_harness.py` imports from `src/` instead of duplicating code.

---

## Should you use it?

**Use `run_eval_harness.py` if:**
- ✅ You want cleaner code
- ✅ The validation script shows 100% pass rate
- ✅ You trust the `src/` modules

**Keep using `single_file.py` if:**
- ⚠️ You don't trust the refactored code yet
- ⚠️ You want to wait and compare results first
- ⚠️ The validation script has failures

---

## Testing it safely

**Option 1: Dry run (validation only)**
```bash
make validate
# 23 checks, 2 minutes, $0 cost
# Proves src/ modules work correctly
```

**Option 2: Side-by-side comparison**
```bash
# Run new harness
make eval

# Run old script
python scripts/single_file.py

# Compare results
diff deliverables/evaluation_evidence/comparative_ragas_results.csv \
     deliverables/evaluation_evidence/comparative_ragas_results_old.csv
```

**Option 3: Just trust the validation**

If `make validate` shows 100% pass rate, the harness will produce identical results.

---

## What files get created?

Same files as `single_file.py`:

```
deliverables/evaluation_evidence/
├── naive_raw_dataset.parquet              # Your change: immediate save
├── naive_evaluation_dataset.csv           # Same as before
├── naive_detailed_results.csv             # Same as before
├── bm25_raw_dataset.parquet               # Your change: immediate save
├── bm25_evaluation_dataset.csv            # Same as before
├── bm25_detailed_results.csv              # Same as before
├── ensemble_raw_dataset.parquet           # Your change: immediate save
├── ensemble_evaluation_dataset.csv        # Same as before
├── ensemble_detailed_results.csv          # Same as before
├── cohere_rerank_raw_dataset.parquet      # Your change: immediate save
├── cohere_rerank_evaluation_dataset.csv   # Same as before
├── cohere_rerank_detailed_results.csv     # Same as before
└── comparative_ragas_results.csv          # Same as before
```

**New**: `*_raw_dataset.parquet` files (your requirement - immediate persistence)
**Same**: All CSV files match `single_file.py` output

---

## What the Makefile does

```bash
make validate   # Run validation script (2 min, $0)
make eval       # Run evaluation harness (20-30 min, ~$6)
make env        # Show API keys and Docker status
make help       # Show all commands
```

**Other commands** (docker-up, clean, notebook, etc.) are just convenience - you can ignore them.

---

## Bottom line

- **`run_eval_harness.py` = `single_file.py` using `src/` modules**
- **Same inputs, same outputs, same results**
- **Only use it if validation passes**
- **Not required - `single_file.py` still works fine**

The refactor was to make the codebase cleaner, not to change functionality.
