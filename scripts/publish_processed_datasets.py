#!/usr/bin/env python3
"""
Upload GDELT RAG processed evaluation datasets to Hugging Face Hub.

This script:
1. Loads evaluation_dataset.csv files from all retrievers (baseline, naive, bm25, ensemble, cohere_rerank)
2. Loads detailed_results.csv files from all retrievers with RAGAS metric scores
3. Adds 'retriever' column to identify source retriever
4. Creates consolidated datasets with comprehensive metadata
5. Uploads to Hugging Face Hub with dataset cards

Datasets Created:
- dwb2023/gdelt-rag-evaluation-inputs: RAGAS input datasets (questions, contexts, responses)
- dwb2023/gdelt-rag-evaluation-metrics: RAGAS evaluation results with metric scores
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from datasets import Dataset
from huggingface_hub import HfApi, login

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_retrievers

retrievers = get_retrievers()

# Defaults (can be overridden via CLI)
HF_USERNAME = "dwb2023"
DEFAULT_VERSION = "v3"
EVALUATION_DATASETS_NAME = f"{HF_USERNAME}/gdelt-rag-evaluation-inputs-{DEFAULT_VERSION}"
DETAILED_RESULTS_NAME = f"{HF_USERNAME}/gdelt-rag-evaluation-metrics-{DEFAULT_VERSION}"

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"


def create_evaluation_datasets_card(sources_version="v2", golden_version="v2") -> str:
    """Create dataset card for consolidated evaluation datasets."""
    return f"""---
license: apache-2.0
task_categories:
- question-answering
- text-retrieval
tags:
- rag
- ragas
- evaluation
- gdelt
- retrieval-comparison
- benchmark
pretty_name: GDELT RAG Evaluation Datasets
size_categories:
- 1K<n<10K
---

# GDELT RAG Evaluation Datasets

## Dataset Description

This dataset contains consolidated RAGAS evaluation input datasets from 4 different retrieval strategies tested on the GDELT (Global Database of Events, Language, and Tone) RAG system. Each strategy was evaluated on the same golden testset of 12 questions, providing a direct comparison of retrieval performance.

### Dataset Summary

- **Total Examples**: 48 evaluation records (12 questions × 4 retrievers)
- **Retrievers Compared**:
  1. Naive (Dense vector search baseline, k=5)
  2. BM25 (Sparse keyword matching)
  3. Ensemble (50% dense + 50% sparse hybrid)
  4. Cohere Rerank (Dense retrieval with rerank-v3.5 compression)
- **Questions Per Retriever**: 12 test questions
- **Purpose**: RAG system comparative evaluation
- **Framework**: RAGAS (Retrieval-Augmented Generation Assessment)
- **Domain**: GDELT Knowledge Graphs

### Data Fields

- `retriever` (string): Source retriever strategy (naive | bm25 | ensemble | cohere_rerank)
- `user_input` (string): The question or query
- `retrieved_contexts` (list[string]): Document chunks retrieved by the retriever
- `reference_contexts` (list[string]): Ground truth context passages containing the answer
- `response` (string): LLM-generated answer using retrieved contexts
- `reference` (string): Ground truth answer from golden testset

### Retriever Strategies Explained

**Naive (Baseline)**:
- Simple dense vector similarity search
- OpenAI text-embedding-3-small embeddings
- Top-k=5 documents
- This is the baseline strategy for comparison

**BM25**:
- Sparse keyword-based retrieval
- Statistical term frequency scoring
- No semantic understanding

**Ensemble**:
- Hybrid approach combining dense + sparse
- 50% weight to naive retriever, 50% to BM25
- Balances semantic and keyword matching

**Cohere Rerank**:
- Two-stage retrieval pipeline
- Stage 1: Dense retrieval (k=20 candidates)
- Stage 2: Cohere rerank-v3.5 compression to top-5
- Most sophisticated strategy tested

### Performance Results

Based on RAGAS evaluation metrics (see `gdelt-rag-evaluation-metrics` dataset):

### Data Splits

This dataset contains a single split with all evaluation records from all 5 retrievers.

### Intended Use

This dataset is intended for:
- Benchmarking RAG retrieval strategies on GDELT documentation
- Comparing dense, sparse, hybrid, and reranking approaches
- Analyzing retrieval quality across different query types
- Reproducing RAGAS evaluation results
- Training retrieval models (retrieved_contexts as weak supervision)

### Source Data

**Golden Testset**: `dwb2023/gdelt-rag-golden-testset-{golden_version}` (12 QA pairs)
- Generated using RAGAS synthetic test data generation
- Based on "Talking to GDELT Through Knowledge Graphs" (arXiv:2503.07584v3)

**Source Documents**: `dwb2023/gdelt-rag-sources-{sources_version}` (38 documents)
- GDELT GKG 2.1 architecture documentation
- Knowledge graph construction guides
- Baltimore Bridge Collapse case study

### Evaluation Methodology

1. Load 38 source documents from HuggingFace
2. Create Qdrant vector store with text-embedding-3-small embeddings
3. Build 5 retriever strategies (baseline, naive, BM25, ensemble, cohere_rerank)
4. Execute 12 queries per retriever
5. Generate answers using gpt-4.1-mini with retrieved contexts
6. Evaluate using RAGAS metrics (faithfulness, answer_relevancy, context_precision, context_recall)

### Licensing

This dataset is released under the Apache 2.0 license.

### Citation

If you use this dataset, please cite the original paper and reference this evaluation work.

### Dataset Creation

This dataset was created as part of the AI Engineering Bootcamp Cohort 8 certification challenge project comparing retrieval strategies for GDELT documentation Q&A.

### Related Datasets

- **Evaluation Results**: `dwb2023/gdelt-rag-evaluation-metrics-{sources_version}` (RAGAS metric scores)
- **Golden Testset**: `dwb2023/gdelt-rag-golden-testset-{golden_version}` (ground truth QA pairs)
- **Source Documents**: `dwb2023/gdelt-rag-sources-{sources_version}` (knowledge base)

### Contact

For questions or issues, please open an issue on the GitHub repository.
"""


def create_detailed_results_card(sources_version="v2", golden_version="v2") -> str:
    """Create dataset card for consolidated detailed results."""
    return f"""---
license: apache-2.0
task_categories:
- question-answering
- text-retrieval
tags:
- rag
- ragas
- evaluation
- metrics
- gdelt
- retrieval-comparison
pretty_name: GDELT RAG Detailed Evaluation Results
size_categories:
- 1K<n<10K
---

# GDELT RAG Detailed Evaluation Results

## Dataset Description

This dataset contains detailed RAGAS evaluation results with per-question metric scores for 4 different retrieval strategies tested on the GDELT RAG system. Each record includes the full evaluation context (question, contexts, response) plus 4 RAGAS metric scores.

### Dataset Summary

- **Total Examples**: 48 evaluation records with metric scores (12 questions × 4 retrievers)
- **Retrievers Evaluated**: Naive (baseline), BM25, Ensemble, Cohere Rerank
- **Metrics Per Record**: 4 RAGAS metrics (faithfulness, answer_relevancy, context_precision, context_recall)
- **Questions Per Retriever**: 12 test questions from golden testset
- **Purpose**: Detailed RAG performance analysis and metric comparison

### Data Fields

- `retriever` (string): Source retriever strategy (naive | bm25 | ensemble | cohere_rerank)
- `user_input` (string): The question or query
- `retrieved_contexts` (list[string]): Document chunks retrieved by the retriever
- `reference_contexts` (list[string]): Ground truth context passages
- `response` (string): LLM-generated answer
- `reference` (string): Ground truth answer
- `faithfulness` (float): Score 0-1, measures if answer is grounded in retrieved contexts (detects hallucinations)
- `answer_relevancy` (float): Score 0-1, measures if answer addresses the question
- `context_precision` (float): Score 0-1, measures if relevant contexts are ranked higher
- `context_recall` (float): Score 0-1, measures if ground truth information was retrieved

### RAGAS Metrics Explained

**Faithfulness** (Higher is Better):
- Evaluates if the generated answer is factually grounded in retrieved contexts
- Detects hallucinations and unsupported claims
- Score of 1.0 means every claim in the answer is supported by contexts

**Answer Relevancy** (Higher is Better):
- Measures how well the answer addresses the specific question
- Penalizes generic or off-topic responses
- Score of 1.0 means answer is perfectly relevant to question

**Context Precision** (Higher is Better):
- Evaluates retrieval ranking quality
- Measures if relevant contexts appear earlier in results
- Score of 1.0 means all relevant contexts ranked at top

**Context Recall** (Higher is Better):
- Measures if ground truth information was successfully retrieved
- Evaluates retrieval coverage and completeness
- Score of 1.0 means all reference contexts were retrieved

### Aggregate Performance Results

- based on prior evaluation results and experience what we expect to see

| Retriever | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Overall |
|-----------|--------------|------------------|-------------------|----------------|---------|
| Cohere Rerank | 0.9844 | 0.9717 | 0.9999 | 0.9136 | 96.47% |
| BM25 | 0.9528 | 0.9641 | 0.9461 | 0.9058 | 94.14% |
| Ensemble | 0.9520 | 0.9582 | 0.9442 | 0.9056 | 93.96% |
| Naive | 0.9249 | 0.9432 | 0.9152 | 0.8904 | 91.60% |

**Key Insights - from prior evaluations**:
- Cohere Rerank achieves near-perfect context precision (99.99%)
- All retrievers score >0.89 on context recall (good coverage)
- Cohere Rerank leads in faithfulness (98.44%, fewest hallucinations)
- BM25 surprisingly competitive with ensemble approach

### Data Splits

This dataset contains a single split with all detailed evaluation records.

### Use Cases

**RAG Research**:
- Analyze which retrieval strategies work best for specific question types
- Study correlation between retrieval quality and answer quality
- Identify failure modes (low precision vs. low recall)

**Model Development**:
- Train retrieval models using RAGAS scores as quality labels
- Fine-tune rerankers using context precision scores
- Develop ensemble weighting strategies based on per-question performance

**Benchmarking**:
- Compare new retrieval strategies against 5 baseline approaches
- Validate RAGAS evaluation framework on domain-specific documentation
- Reproduce certification challenge evaluation results

**Error Analysis**:
- Filter for low-scoring examples
- Identify question patterns that challenge specific retrievers
- Debug retrieval failures using retrieved_contexts field

### Evaluation Configuration

**Models**:
- LLM: gpt-4.1-mini (temperature=0)
- Embeddings: text-embedding-3-small
- Reranker: rerank-v3.5 (Cohere)
- RAGAS: v0.2.10

**Infrastructure**:
- Vector Store: Qdrant (localhost:6333)
- Collection: gdelt_rag (cosine similarity)
- Chunk Strategy: Page-level (38 documents)

**Evaluation Cost**: Approximately $5-6 per full run (192 LLM calls for RAGAS metrics)

### Source Data

**Golden Testset**: dwb2023/gdelt-rag-golden-testset-{golden_version}
- 12 synthetically generated QA pairs
- Single-hop and multi-hop questions
- GDELT-specific technical questions

**Source Documents**: dwb2023/gdelt-rag-sources-{sources_version}
- 38 pages from GDELT research paper
- Topics: GKG 2.1 architecture, event encoding, knowledge graphs

### Licensing

This dataset is released under the Apache 2.0 license.

### Citation

If you use this dataset, please cite the original GDELT paper and reference this evaluation work.

### Dataset Creation

Created as part of AI Engineering Bootcamp Cohort 8 certification challenge (January 2025).

### Related Datasets

- **Evaluation Inputs**: dwb2023/gdelt-rag-evaluation-inputs-{sources_version} (without metric scores)
- **Golden Testset**: dwb2023/gdelt-rag-golden-testset-{golden_version}
- **Source Documents**: dwb2023/gdelt-rag-sources-{sources_version}

### Contact

For questions or issues, please open an issue on the GitHub repository.
"""


def load_parquet_with_retriever_column(file_path: Path, retriever_name: str) -> pd.DataFrame:
    """Load Parquet and add retriever column."""
    df = pd.read_parquet(file_path)
    df.insert(0, "retriever", retriever_name)
    return df


def load_and_consolidate_datasets(pattern: str, retrievers: list[str]) -> pd.DataFrame:
    """Load and consolidate all Parquet files matching pattern with retriever column."""
    dfs = []

    for retriever in retrievers:
        file_path = DATA_DIR / f"{retriever}_{pattern}.parquet"

        if not file_path.exists():
            print(f"   Warning: {file_path.name} not found, skipping...")
            continue

        print(f"   • Loading {file_path.name}...")
        df = load_parquet_with_retriever_column(file_path, retriever)
        dfs.append(df)
        print(f"      Loaded {len(df)} rows from {retriever}")

    if not dfs:
        raise ValueError(f"No Parquet files found matching pattern: *_{pattern}.parquet")

    consolidated = pd.concat(dfs, ignore_index=True)
    print(f"   ✅ Consolidated {len(consolidated)} total rows from {len(dfs)} retrievers")
    return consolidated


def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="Publish processed HF datasets")
    # Option A: pass full repo IDs (highest precedence)
    p.add_argument("--inputs", type=str, help="Full HF repo id for eval inputs (e.g. dwb2023/gdelt-rag-evaluation-inputs-v4)")
    p.add_argument("--metrics", type=str, help="Full HF repo id for eval metrics (e.g. dwb2023/gdelt-rag-evaluation-metrics-v4)")
    # Option B: build from namespace+version if full ids not provided
    p.add_argument("--namespace", type=str, default=HF_USERNAME, help="HF namespace (default: dwb2023)")
    p.add_argument("--version", type=str, default=DEFAULT_VERSION, help="version string (default: v3)")
    return p.parse_args()


def main():
    """Main upload function."""
    args = parse_args()

    # Resolve final repo ids
    namespace = args.namespace
    version = args.version
    inputs_repo = args.inputs or f"{namespace}/gdelt-rag-evaluation-inputs-{version}"
    metrics_repo = args.metrics or f"{namespace}/gdelt-rag-evaluation-metrics-{version}"

    # Check for HF token
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not set")

    # Login to Hugging Face
    print("🔐 Logging in to Hugging Face...")
    login(token=hf_token)

    # Initialize API
    api = HfApi()

    # ========================================
    # Dataset 1: Evaluation Datasets
    # ========================================
    print(f"\n📂 Loading evaluation datasets from {DATA_DIR}...")
    eval_df = load_and_consolidate_datasets("evaluation_inputs", retrievers)

    print("\n🔄 Converting evaluation datasets to HuggingFace Dataset...")
    eval_dataset = Dataset.from_pandas(eval_df)
    print(f"   • Dataset size: {len(eval_dataset)} examples")
    print(f"   • Features: {list(eval_dataset.features.keys())}")

    print(f"\n📤 Uploading evaluation datasets to {inputs_repo}...")
    eval_dataset.push_to_hub(
        inputs_repo,
        private=False,
        token=hf_token
    )

    # Create and upload dataset card
    print("   • Creating dataset card...")
    api.upload_file(
        path_or_fileobj=create_evaluation_datasets_card(sources_version=version, golden_version=version).encode(),
        path_in_repo="README.md",
        repo_id=inputs_repo,
        repo_type="dataset",
        token=hf_token
    )
    print("   ✅ Evaluation datasets uploaded successfully!")
    print(f"      View at: https://huggingface.co/datasets/{inputs_repo}")

    # ========================================
    # Dataset 2: Detailed Results
    # ========================================
    print(f"\n📂 Loading detailed results from {DATA_DIR}...")
    results_df = load_and_consolidate_datasets("evaluation_metrics", retrievers)

    print("\n🔄 Converting detailed results to HuggingFace Dataset...")
    results_dataset = Dataset.from_pandas(results_df)
    print(f"   • Dataset size: {len(results_dataset)} examples")
    print(f"   • Features: {list(results_dataset.features.keys())}")

    print(f"\n📤 Uploading detailed results to {metrics_repo}...")
    results_dataset.push_to_hub(
        metrics_repo,
        private=False,
        token=hf_token
    )

    # Create and upload dataset card
    print("   • Creating dataset card...")
    api.upload_file(
        path_or_fileobj=create_detailed_results_card(sources_version=version, golden_version=version).encode(),
        path_in_repo="README.md",
        repo_id=metrics_repo,
        repo_type="dataset",
        token=hf_token
    )
    print("   ✅ Detailed results uploaded successfully!")
    print(f"      View at: https://huggingface.co/datasets/{metrics_repo}")

    # ========================================
    # Summary
    # ========================================
    print("\n🎉 All datasets uploaded successfully!")
    print("\n📊 Dataset URLs:")
    print(f"   • Evaluation Datasets: https://huggingface.co/datasets/{inputs_repo}")
    print(f"   • Detailed Results: https://huggingface.co/datasets/{metrics_repo}")

    print("\n📈 Dataset Statistics:")
    print(f"   • Evaluation Datasets: {len(eval_dataset)} examples across {len(eval_df['retriever'].unique())} retrievers")
    print(f"   • Detailed Results: {len(results_dataset)} examples with RAGAS metric scores")

    print("\n✨ Next Steps:")
    print("   1. Verify datasets on HuggingFace Hub")
    print("   2. Update README.md with new dataset references")
    print("   3. Update docs/deliverables.md with dataset URLs")


if __name__ == "__main__":
    main()
