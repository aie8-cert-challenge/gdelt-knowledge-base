# Makefile for GDELT RAG Evaluation System

.PHONY: help validate eval deliverables ingest publish-interim publish-processed clean clean-deliverables clean-processed clean-all env docker-up docker-down test notebook

# Default target
help:
	@echo "GDELT RAG Evaluation System - Available Commands"
	@echo ""
	@echo "Data Preparation (one-time setup):"
	@echo "  make ingest      - Extract PDFs and generate golden testset (~5-10 min, \$$2-3)"
	@echo ""
	@echo "Development:"
	@echo "  make validate    - Validate src/ module implementation (100% pass required)"
	@echo "  make eval        - Run full RAGAS evaluation harness (~20-30 min, \$$5-6)"
	@echo "  make deliverables - Generate human-friendly CSV files from Parquet data"
	@echo "  make test        - Run quick validation test"
	@echo ""
	@echo "Publishing (optional, requires HF_TOKEN):"
	@echo "  make publish-interim    - Upload sources & golden testset to HuggingFace Hub"
	@echo "  make publish-processed  - Upload evaluation results to HuggingFace Hub"
	@echo ""
	@echo "Infrastructure:"
	@echo "  make docker-up   - Start all infrastructure services (Qdrant, Redis, Neo4j, etc.)"
	@echo "  make docker-down - Stop all infrastructure services"
	@echo "  make qdrant-up   - Start only Qdrant (minimal requirement)"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean       - Clean Python cache and temporary files"
	@echo "  make clean-deliverables - Clean derived deliverables (regenerable)"
	@echo "  make clean-processed    - Clean processed data (requires re-eval)"
	@echo "  make clean-all   - Full cleanup (cache + interim + processed + deliverables)"
	@echo ""
	@echo "Environment:"
	@echo "  make env         - Show environment variables"
	@echo ""
	@echo "Jupyter:"
	@echo "  make notebook    - Start Jupyter notebook server"
	@echo ""

# Validate src/ module implementation
validate:
	@echo "🔍 Validating src/ module implementation..."
	@PYTHONPATH=. uv run python scripts/run_app_validation.py
	@echo ""
	@echo "🔐 Validating manifest files and SHA-256 checksums..."
	@PYTHONPATH=. uv run python scripts/validate_manifests.py

# Run full RAGAS evaluation (same as run_full_evaluation.py but uses src/ modules)
# Usage: make eval              (reuses existing Qdrant collection)
#        make eval recreate=true (recreates Qdrant collection)
recreate ?= false
eval:
	@echo "🚀 Running RAGAS evaluation harness..."
	@echo ""
	@echo "This does the SAME thing as scripts/run_full_evaluation.py:"
	@echo "  - 12 questions × 4 retrievers = 48 queries"
	@echo "  - RAGAS evaluation with 4 metrics"
	@echo "  - Saves to deliverables/evaluation_evidence/"
	@echo ""
	@echo "⏱️  Time: 20-30 minutes"
	@echo "💰 Cost: ~\$$5-6 in OpenAI API calls"
	@echo ""
	@echo "Vector store: recreate=$(recreate)"
	@if [ "$(recreate)" = "true" ]; then \
		echo "  ⚠️  Will DELETE and recreate Qdrant collection"; \
	else \
		echo "  ✓ Will reuse existing Qdrant collection (faster)"; \
	fi
	@echo ""
	@PYTHONPATH=. uv run python scripts/run_eval_harness.py --recreate=$(recreate)

# Quick test (validation only, no full eval)
test: validate

# Start all infrastructure services
docker-up:
	@echo "🐳 Starting all infrastructure services..."
	docker-compose up -d
	@echo "✅ Services started. Access points:"
	@echo "  - Qdrant: http://localhost:6333"
	@echo "  - Redis: localhost:6379"
	@echo "  - Neo4j: http://localhost:7474"
	@echo "  - Phoenix: http://localhost:6006"
	@echo "  - MinIO: http://localhost:9001"

# Stop all infrastructure services
docker-down:
	@echo "🛑 Stopping all infrastructure services..."
	docker-compose down

# Start only Qdrant (minimal requirement)
qdrant-up:
	@echo "🐳 Starting Qdrant..."
	docker-compose up -d qdrant
	@echo "✅ Qdrant started at http://localhost:6333"

# Show environment configuration
env:
	@echo "Environment Configuration:"
	@echo ""
	@echo "API Keys:"
	@if [ -n "$$OPENAI_API_KEY" ]; then echo "  ✅ OPENAI_API_KEY: set"; else echo "  ❌ OPENAI_API_KEY: not set"; fi
	@if [ -n "$$COHERE_API_KEY" ]; then echo "  ✅ COHERE_API_KEY: set"; else echo "  ⚠️  COHERE_API_KEY: not set (cohere_rerank will fail)"; fi
	@if [ -n "$$LANGCHAIN_API_KEY" ]; then echo "  ✅ LANGCHAIN_API_KEY: set"; else echo "  ℹ️  LANGCHAIN_API_KEY: not set (tracing disabled)"; fi
	@echo ""
	@echo "Python:"
	@python --version 2>/dev/null || echo "  ❌ Python not found"
	@echo ""
	@echo "Infrastructure:"
	@docker-compose ps 2>/dev/null || echo "  ℹ️  Docker Compose not running"

# Clean Python cache and temporary files
clean:
	@echo "🧹 Cleaning Python cache and temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Clean complete"

# Generate human-friendly deliverables from Parquet data
deliverables:
	@echo "📂 Generating deliverables from data/processed/..."
	uv run python scripts/generate_deliverables.py

# Clean derived deliverables (can be regenerated)
clean-deliverables:
	@echo "🧹 Cleaning deliverables/evaluation_evidence/..."
	@rm -f deliverables/evaluation_evidence/*.csv 2>/dev/null || true
	@rm -f deliverables/evaluation_evidence/*.parquet 2>/dev/null || true
	@rm -f deliverables/evaluation_evidence/RUN_MANIFEST.json 2>/dev/null || true
	@echo "✅ Deliverables cleaned (regenerate with 'make deliverables')"

# Clean processed data (requires re-running evaluation)
clean-processed:
	@echo "🧹 Cleaning data/processed/..."
	@rm -f data/processed/*.parquet 2>/dev/null || true
	@rm -f data/processed/*.csv 2>/dev/null || true
	@rm -f data/processed/RUN_MANIFEST.json 2>/dev/null || true
	@echo "⚠️  Processed data cleaned (re-run evaluation with 'make eval')"

# Full clean (interim + processed + deliverables + cache)
clean-all: clean clean-deliverables clean-processed
	@echo "🧹 Cleaning data/interim/..."
	@rm -f data/interim/*.parquet 2>/dev/null || true
	@rm -f data/interim/*.jsonl 2>/dev/null || true
	@rm -f data/interim/manifest.json 2>/dev/null || true
	@echo "✅ Full cleanup complete (cache + interim + processed + deliverables)"

# Start Jupyter notebook
notebook:
	@echo "📓 Starting Jupyter notebook..."
	jupyter notebook

# Data preparation (one-time ingestion from raw PDFs)
ingest:
	@echo "📄 Ingesting raw PDFs and generating golden testset..."
	@echo ""
	@echo "This extracts PDFs from data/raw/ and generates:"
	@echo "  - 38 source documents (page-level chunks)"
	@echo "  - 12 RAGAS golden testset QA pairs"
	@echo "  - Persisted to data/interim/ (JSONL, Parquet, HFDS)"
	@echo "  - manifest.json with checksums and provenance"
	@echo ""
	@echo "⏱️  Time: 5-10 minutes"
	@echo "💰 Cost: ~\$$2-3 in OpenAI API calls"
	@echo ""
	@PYTHONPATH=. uv run python scripts/ingest_raw_pdfs.py

# Publish interim datasets to HuggingFace Hub
publish-interim:
	@echo "📤 Publishing interim datasets to HuggingFace Hub..."
	@echo ""
	@echo "Uploads to HuggingFace Hub:"
	@echo "  - dwb2023/gdelt-rag-sources-v3 (38 documents)"
	@echo "  - dwb2023/gdelt-rag-golden-testset-v3 (12 QA pairs)"
	@echo ""
	@echo "⏱️  Time: 1-2 minutes"
	@echo "⚠️  Requires: HF_TOKEN environment variable"
	@echo ""
	@PYTHONPATH=. uv run python scripts/publish_interim_datasets.py

# Publish processed evaluation results to HuggingFace Hub
publish-processed:
	@echo "📤 Publishing evaluation results to HuggingFace Hub..."
	@echo ""
	@echo "Uploads to HuggingFace Hub:"
	@echo "  - dwb2023/gdelt-rag-evaluation-inputs-v3 (48 records)"
	@echo "  - dwb2023/gdelt-rag-evaluation-metrics-v3 (48 records with RAGAS scores)"
	@echo ""
	@echo "⏱️  Time: 1-2 minutes"
	@echo "⚠️  Requires: HF_TOKEN environment variable"
	@echo ""
	@PYTHONPATH=. uv run python scripts/publish_processed_datasets.py

# Convenience aliases
v: validate
e: eval
d: docker-up
i: ingest
