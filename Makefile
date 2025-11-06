# ===============================================
# GDELT RAG Evaluation System — Reproducible Pipeline
# ===============================================

.DEFAULT_GOAL := help
SHELL := /bin/bash
.SHELLFLAGS := -euo pipefail -c

.PHONY: help config ingest validate eval deliverables \
        publish-interim publish-processed \
        docker-up docker-down qdrant-up \
        clean clean-deliverables clean-processed clean-all \
        test notebook v e d i

# -------- Global Parameters --------
VERSION ?= v3
HF_NAMESPACE ?= dwb2023

# Embedding + LLM (optional overrides)
EMBED_MODEL ?= text-embedding-3-small
LLM_MODEL ?= gpt-4.1-mini
RECREATE ?= false

# If your scripts already import modules via a package (e.g. src/),
# you can remove PYTHONPATH entirely.
PYTHONPATH_OPT ?= .

# -------- Derived HuggingFace Dataset Names --------
INTERIM_SOURCES  := $(HF_NAMESPACE)/gdelt-rag-sources-$(VERSION)
INTERIM_GOLDEN   := $(HF_NAMESPACE)/gdelt-rag-golden-testset-$(VERSION)
PROCESSED_INPUTS := $(HF_NAMESPACE)/gdelt-rag-evaluation-inputs-$(VERSION)
PROCESSED_METRICS:= $(HF_NAMESPACE)/gdelt-rag-evaluation-metrics-$(VERSION)

# -------- Help --------
help:
	@echo "🔎 GDELT RAG Pipeline Commands"
	@echo ""
	@echo "🧠 Core Pipeline"
	@echo "  make ingest               Extract PDFs + generate golden testset"
	@echo "  make eval                 Run full evaluation pipeline (3 phases)"
	@echo "  make deliverables         Generate CSV artifacts"
	@echo ""
	@echo "🔄 Three-Phase Pipeline (NEW)"
	@echo "  make inference            Phase 1: Run inference only (~\$$3-4)"
	@echo "  make eval-metrics         Phase 2: Run RAGAS evaluation (~\$$2)"
	@echo "  make summarize            Phase 3: Create summary & manifest (\$$0)"
	@echo "  make eval-monolithic      Legacy: Run old single-script version"
	@echo ""
	@echo "📦 Publishing"
	@echo "  make publish-interim      Upload sources & golden set to HF Hub"
	@echo "  make publish-processed    Upload eval outputs to HF Hub"
	@echo ""
	@echo "🧰 Dev & Infra"
	@echo "  make validate             Validate src + manifests"
	@echo "  make docker-up            Start all infra (Qdrant, Redis, etc.)"
	@echo "  make qdrant-up            Start Qdrant only"
	@echo "  make notebook             Launch Jupyter"
	@echo ""
	@echo "🧽 Cleanups"
	@echo "  make clean                Python cache clean"
	@echo "  make clean-deliverables   Remove CSV artifacts"
	@echo "  make clean-processed      Remove eval outputs"
	@echo "  make clean-all            Reset everything"
	@echo ""
	@echo "🧪 Shortcuts"
	@echo "  make v   -> validate"
	@echo "  make e   -> eval"
	@echo "  make d   -> docker-up"
	@echo "  make i   -> ingest"

# -------- Config Printer --------
config:
	@echo "⚙️  Pipeline Configuration"
	@echo "-------------------------------------"
	@echo "VERSION:           $(VERSION)"
	@echo "HF_NAMESPACE:      $(HF_NAMESPACE)"
	@echo "EMBED_MODEL:       $(EMBED_MODEL)"
	@echo "LLM_MODEL:         $(LLM_MODEL)"
	@echo "RECREATE:          $(RECREATE)"
	@echo "PYTHONPATH_OPT:    $(PYTHONPATH_OPT)"
	@echo ""
	@echo "Datasets:"
	@echo "  Sources:         $(INTERIM_SOURCES)"
	@echo "  Golden:          $(INTERIM_GOLDEN)"
	@echo "  Eval Inputs:     $(PROCESSED_INPUTS)"
	@echo "  Eval Metrics:    $(PROCESSED_METRICS)"

# -------- Pipeline Steps --------

ingest:
	@echo "📄 Ingesting PDFs + generating golden testset (VERSION=$(VERSION))"
	@PYTHONPATH=$(PYTHONPATH_OPT) uv run python scripts/ingest_raw_pdfs.py \
		--version $(VERSION)

validate:
	@echo "🔍 Validating src + manifests"
	@PYTHONPATH=$(PYTHONPATH_OPT) uv run python scripts/run_app_validation.py
	@PYTHONPATH=$(PYTHONPATH_OPT) uv run python scripts/validate_manifests.py

# -------- Three-Phase Evaluation Pipeline --------

# Phase 1: Run inference only (most expensive, ~$3-4)
inference:
	@echo "🤖 Phase 1: Running inference (RECREATE=$(RECREATE))"
	@PYTHONPATH=$(PYTHONPATH_OPT) uv run python scripts/run_inference.py \
		--sources $(INTERIM_SOURCES) \
		--golden $(INTERIM_GOLDEN) \
		--recreate $(RECREATE) \
		--embed-model $(EMBED_MODEL)

# Phase 2: Run RAGAS evaluation on saved inputs (~$2)
eval-metrics:
	@echo "📊 Phase 2: Running RAGAS evaluation"
	@PYTHONPATH=$(PYTHONPATH_OPT) uv run python scripts/run_evaluation.py \
		--llm-model $(LLM_MODEL)

# Phase 3: Summarize results and create manifest (no API calls)
summarize:
	@echo "📈 Phase 3: Summarizing results (VERSION=$(VERSION))"
	@PYTHONPATH=$(PYTHONPATH_OPT) uv run python scripts/summarize_results.py \
		--version $(VERSION)

# Combined target: run all three phases (backward compatibility)
eval: inference eval-metrics summarize
	@echo "✅ Full evaluation pipeline complete"

# Legacy target using monolithic script (for comparison)
eval-monolithic:
	@echo "🚀 Running monolithic evaluation (RECREATE=$(RECREATE), VERSION=$(VERSION))"
	@PYTHONPATH=$(PYTHONPATH_OPT) uv run python scripts/run_eval_harness.py \
		--recreate $(RECREATE) \
		--version $(VERSION) \
		--embed-model $(EMBED_MODEL) \
		--llm-model $(LLM_MODEL)

deliverables:
	@echo "📊 Generating deliverables"
	uv run python scripts/generate_deliverables.py

# -------- Hugging Face Publishing --------

publish-interim:
	@echo "📤 Publishing interim datasets (VERSION=$(VERSION))"
	@test -n "$$HF_TOKEN" || { echo "❌ HF_TOKEN not set"; exit 1; }
	@echo "→ $(INTERIM_SOURCES)"
	@echo "→ $(INTERIM_GOLDEN)"
	@PYTHONPATH=$(PYTHONPATH_OPT) uv run python scripts/publish_interim_datasets.py \
		--sources $(INTERIM_SOURCES) \
		--golden $(INTERIM_GOLDEN)

publish-processed:
	@echo "📤 Publishing processed eval outputs (VERSION=$(VERSION))"
	@test -n "$$HF_TOKEN" || { echo "❌ HF_TOKEN not set"; exit 1; }
	@echo "→ $(PROCESSED_INPUTS)"
	@echo "→ $(PROCESSED_METRICS)"
	@PYTHONPATH=$(PYTHONPATH_OPT) uv run python scripts/publish_processed_datasets.py \
		--inputs $(PROCESSED_INPUTS) \
		--metrics $(PROCESSED_METRICS)

# -------- Infra Commands --------
# Prefer 'docker compose'; fall back to 'docker-compose' if needed.
DOCKER_COMPOSE := $(shell command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1 && echo "docker compose" || echo "docker-compose")

docker-up:
	@echo "🐳 Starting infrastructure"
	$(DOCKER_COMPOSE) up -d

docker-down:
	$(DOCKER_COMPOSE) down

qdrant-up:
	@echo "🐳 Starting Qdrant only"
	$(DOCKER_COMPOSE) up -d qdrant

notebook:
	jupyter notebook

# -------- Cleaning --------

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

clean-deliverables:
	rm -f deliverables/evaluation_evidence/*

clean-processed:
	rm -f data/processed/*

clean-all: clean clean-deliverables clean-processed
	rm -rf data/interim/*

# -------- Aliases --------
v: validate
e: eval
d: docker-up
i: ingest
