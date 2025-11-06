# Makefile Documentation

Complete guide to the GDELT RAG evaluation pipeline automation via Makefile.

## Quick Reference

```bash
make help    # Show all available commands
make config  # Show current configuration
```

## Essential Shortcuts

| Shortcut | Command | Description |
|----------|---------|-------------|
| `make v` | `make validate` | Run validation (23 checks) |
| `make e` | `make eval` | Run full evaluation pipeline |
| `make d` | `make docker-up` | Start Docker infrastructure |
| `make i` | `make ingest` | Ingest PDFs (one-time) |

## End-to-End Workflow

### Pipeline Steps (Base Commands)

The complete evaluation pipeline in logical order:

| Step | Command | What It Does | Cost | Time | Creates |
|------|---------|--------------|------|------|---------|
| **0** | `make clean-all` | Resets workspace by removing all generated data | $0 | <1s | Clean slate |
| **1** | `make ingest` | Extracts text from PDFs and generates test questions | ~$2-3 | 5-10m | `data/interim/` datasets |
| **2** | `make publish-interim` | Uploads source docs & testset to HuggingFace | $0 | 1-2m | HF datasets (sources, testset) |
| **3** | `make validate` | Verifies environment setup (23 checks must pass) | $0 | <1s | Validation report |
| **4** | `make qdrant-up` | Starts vector database Docker container | $0 | <1s | Qdrant on port 6333 |
| **5a** | `make inference` | Runs RAG queries using all 4 retrievers | ~$3-4 | 5-10m | Inference results (Parquet) |
| **5b** | `make eval-metrics` | Scores RAG outputs with RAGAS metrics | ~$2 | 5-10m | Evaluation scores (Parquet) |
| **5c** | `make summarize` | Aggregates results across all retrievers | $0 | <1s | Comparative analysis |
| **6** | `make deliverables` | Converts Parquet to human-readable CSVs | $0 | <1s | `deliverables/` CSVs |
| **7** | `make publish-processed` | Uploads evaluation results to HuggingFace | $0 | 1-2m | HF evaluation datasets |

**Shortcut:** Steps 5a-5c can be combined with `make eval` (runs all three phases sequentially)

**Total:** ~$7-9, 30-45 minutes (for full pipeline from scratch)

### Parameter Guide

Control pipeline behavior with these parameters:

| Parameter | Commands | Purpose | Values | Example | When to Use |
|-----------|----------|---------|--------|---------|-------------|
| **VERSION** | `ingest`<br>`publish-*`<br>`inference`<br>`summarize`<br>`eval` | Dataset version tag | `v3` (default)<br>`v4`, `v5`, etc. | `make ingest VERSION=v4` | • New dataset creation<br>• Version tracking<br>• A/B testing |
| **RECREATE** | `inference`<br>`eval` | Force new Qdrant collection | `false` (default)<br>`true` | `make inference RECREATE=true` | • Collection corrupted<br>• Schema changes<br>• Fresh start needed |
| **HF_TOKEN** | `publish-*` | HuggingFace authentication | Your token | `export HF_TOKEN=hf_...` | Required for publishing |
| **EMBED_MODEL** | `inference`<br>`eval` | Embedding model | `text-embedding-3-small` (default) | `make eval EMBED_MODEL=text-embedding-3-large` | Testing different embeddings |
| **LLM_MODEL** | `eval-metrics`<br>`eval` | LLM for generation | `gpt-4.1-mini` (default) | `make eval LLM_MODEL=gpt-4o` | Testing different LLMs |
| **HF_NAMESPACE** | All publishing | HuggingFace org/user | `dwb2023` (default) | `make publish HF_NAMESPACE=myorg` | Publishing to your account |

### Common Scenarios

| Scenario | Command | Why |
|----------|---------|-----|
| **First-time setup** | `make eval RECREATE=true` | Creates new Qdrant collection and runs full pipeline |
| **Re-run with existing data** | `make eval` | Uses existing Qdrant collection, faster startup |
| **Test new version** | `make eval VERSION=v5 RECREATE=true` | Creates v5 datasets with fresh collection |
| **Fix failed RAGAS** | `make eval-metrics` | Re-runs only RAGAS scoring (inference preserved) |
| **Update CSVs only** | `make deliverables` | Regenerates CSVs from existing Parquet files |
| **Publish v4 results** | `make publish-processed VERSION=v4` | Uploads v4 evaluation to HuggingFace |
| **Test different LLM** | `make eval-metrics LLM_MODEL=gpt-4o` | Evaluate with different model (reuses inference) |
| **Test different embeddings** | `make inference EMBED_MODEL=text-embedding-3-large RECREATE=true` | Requires recreating collection |

## Command Categories

### Development Commands

```bash
make validate          # Validate environment + modules (23 checks, must pass 100%)
make eval             # Run full evaluation: inference → RAGAS → summarize
make deliverables     # Generate human-readable CSV files from Parquet
```

**Three-Phase Evaluation** (for cost control):
```bash
make inference        # Phase 1: RAG inference only (~$3-4)
make eval-metrics     # Phase 2: RAGAS scoring (~$2)
make summarize        # Phase 3: Create summary + manifest ($0)
```

**Legacy Monolithic** (for comparison):
```bash
make eval-monolithic  # Old single-script version (same results)
```

### Infrastructure Commands

```bash
make docker-up        # Start all services (Qdrant, Redis, etc.)
make docker-down      # Stop all services
make qdrant-up        # Start Qdrant only
make notebook         # Launch Jupyter
```

### Publishing Commands

```bash
export HF_TOKEN=hf_...

make publish-interim     # Upload sources + testset to HuggingFace
make publish-processed   # Upload evaluation results to HuggingFace
```

### Cleaning Commands

```bash
make clean                # Python cache clean (__pycache__, *.pyc)
make clean-deliverables   # Remove CSV files (regenerable)
make clean-processed      # Remove evaluation results (Parquet)
make clean-all            # Reset everything (interim + processed + deliverables)
```

## Configuration

View current configuration:

```bash
make config
```

Output:
```
⚙️  Pipeline Configuration
-------------------------------------
VERSION:           v3
HF_NAMESPACE:      dwb2023
EMBED_MODEL:       text-embedding-3-small
LLM_MODEL:         gpt-4.1-mini
RECREATE:          false
PYTHONPATH_OPT:    .

Datasets:
  Sources:         dwb2023/gdelt-rag-sources-v3
  Golden:          dwb2023/gdelt-rag-golden-testset-v3
  Eval Inputs:     dwb2023/gdelt-rag-evaluation-inputs-v3
  Eval Metrics:    dwb2023/gdelt-rag-evaluation-metrics-v3
```

## Internal Implementation

### PYTHONPATH Handling

The Makefile automatically sets `PYTHONPATH=.` for all Python scripts, so you don't need to set it manually:

```makefile
PYTHONPATH_OPT ?= .

validate:
	@PYTHONPATH=$(PYTHONPATH_OPT) uv run python scripts/run_app_validation.py
```

### Docker Compose Detection

The Makefile auto-detects whether to use `docker compose` (new) or `docker-compose` (legacy):

```makefile
DOCKER_COMPOSE := $(shell command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1 && echo "docker compose" || echo "docker-compose")

docker-up:
	$(DOCKER_COMPOSE) up -d
```

### Dataset Name Construction

Dataset names are automatically constructed from `HF_NAMESPACE` and `VERSION`:

```makefile
VERSION ?= v3
HF_NAMESPACE ?= dwb2023

INTERIM_SOURCES  := $(HF_NAMESPACE)/gdelt-rag-sources-$(VERSION)
INTERIM_GOLDEN   := $(HF_NAMESPACE)/gdelt-rag-golden-testset-$(VERSION)
PROCESSED_INPUTS := $(HF_NAMESPACE)/gdelt-rag-evaluation-inputs-$(VERSION)
PROCESSED_METRICS:= $(HF_NAMESPACE)/gdelt-rag-evaluation-metrics-$(VERSION)
```

## Workflow Examples

### Standard Development Workflow

```bash
# 1. Start infrastructure
make qdrant-up

# 2. Validate environment (must pass 100%)
make validate

# 3. Run evaluation
make eval

# 4. Generate deliverables
make deliverables

# 5. Review results
cat deliverables/evaluation_evidence/comparative_ragas_results.csv
```

### Cost-Conscious Evaluation

```bash
# Only run inference if you need fresh results
make inference

# Run evaluation on saved inference results
make eval-metrics

# Aggregate results (always free)
make summarize

# Generate human-readable files
make deliverables
```

### Creating a New Dataset Version

```bash
# 1. Ingest with new version
make ingest VERSION=v4

# 2. Publish interim datasets
export HF_TOKEN=hf_...
make publish-interim VERSION=v4

# 3. Run evaluation with new version
make eval VERSION=v4 RECREATE=true

# 4. Publish results
make publish-processed VERSION=v4
```

### Testing Different Models

```bash
# Test different embedding model (requires recreating collection)
make inference EMBED_MODEL=text-embedding-3-large RECREATE=true

# Test different LLM (reuses existing inference)
make eval-metrics LLM_MODEL=gpt-4o

# Summarize and compare
make summarize
```

### Complete Fresh Start

```bash
# Reset everything
make clean-all

# Recreate from scratch
make ingest
make eval RECREATE=true
make deliverables
```

## Troubleshooting

### Command Not Found

**Issue**: `make: command not found`

**Fix**: Install make (usually pre-installed on Linux/Mac)
```bash
# Ubuntu/Debian
sudo apt-get install build-essential

# Mac
xcode-select --install
```

### Qdrant Connection Refused

**Issue**: Evaluation fails with `QdrantException: Connection refused`

**Fix**: Start Qdrant first
```bash
make qdrant-up
curl http://localhost:6333/collections  # Verify it's running
```

### HuggingFace Token Error

**Issue**: `HF_TOKEN not set` when publishing

**Fix**: Export your HuggingFace token
```bash
export HF_TOKEN=hf_...
# Or add to .env file
echo "HF_TOKEN=hf_..." >> .env
```

### Validation Failures

**Issue**: `make validate` shows failures

**Fix**:
```bash
# Install dependencies
uv pip install -e .

# Activate environment
source .venv/bin/activate

# Re-run validation
make validate
```

### Stale Results

**Issue**: Results don't match expected output

**Fix**: Force recreation
```bash
make clean-processed
make eval RECREATE=true
```

## Advanced Usage

### Override Multiple Parameters

```bash
make eval \
  VERSION=v5 \
  EMBED_MODEL=text-embedding-3-large \
  LLM_MODEL=gpt-4o \
  RECREATE=true
```

### Custom HuggingFace Namespace

```bash
make publish-interim \
  HF_NAMESPACE=myorganization \
  VERSION=v1
```

### Run Single Script Directly

If you need to run scripts directly (not recommended, use make instead):

```bash
PYTHONPATH=. uv run python scripts/run_eval_harness.py \
  --recreate true \
  --version v3 \
  --embed-model text-embedding-3-small \
  --llm-model gpt-4.1-mini
```

## Best Practices

**DO**:
- ✅ Use `make` commands instead of running scripts directly
- ✅ Run `make validate` before committing (must pass 100%)
- ✅ Use `make config` to verify settings before expensive operations
- ✅ Use `make clean-deliverables && make deliverables` to regenerate CSVs
- ✅ Use versioning (`VERSION=v4`) for dataset iterations

**DON'T**:
- ❌ Run scripts directly without `PYTHONPATH=.` (use make instead)
- ❌ Skip `make validate` before deployment
- ❌ Delete Parquet files manually (use `make clean-processed`)
- ❌ Forget to export `HF_TOKEN` before publishing
- ❌ Mix dataset versions without tracking

## Reference

- **[CLAUDE.md](./CLAUDE.md)** - High-level architecture guide
- **[scripts/README.md](./scripts/README.md)** - Detailed script documentation
- **[Makefile](./Makefile)** - Source of truth for all commands