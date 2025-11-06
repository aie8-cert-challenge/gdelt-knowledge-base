#!/usr/bin/env python3
"""
Upload GDELT RAG datasets to Hugging Face Hub.

This script:
1. Loads source documents and golden testset datasets from local storage
2. Creates dataset cards with metadata
3. Uploads datasets to Hugging Face Hub
4. Updates manifest.json with dataset repo IDs
"""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_from_disk
from huggingface_hub import HfApi, login

# Defaults (can be overridden via CLI)
HF_USERNAME = "dwb2023"
DEFAULT_VERSION = "v3"
SOURCES_DATASET_NAME = f"{HF_USERNAME}/gdelt-rag-sources-{DEFAULT_VERSION}"
GOLDEN_TESTSET_NAME = f"{HF_USERNAME}/gdelt-rag-golden-testset-{DEFAULT_VERSION}"

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "interim"
MANIFEST_PATH = DATA_DIR / "manifest.json"

SOURCES_PATH = DATA_DIR / "sources.hfds"
GOLDEN_TESTSET_PATH = DATA_DIR / "golden_testset.hfds"


def create_sources_card() -> str:
    """Create dataset card for sources dataset."""
    return """---
license: apache-2.0
task_categories:
- text-retrieval
- question-answering
tags:
- rag
- gdelt
- knowledge-graphs
- retrieval
pretty_name: GDELT RAG Source Documents
size_categories:
- n<1K
---

# GDELT RAG Source Documents

## Dataset Description

This dataset contains source documents extracted from the research paper "Talking to GDELT Through Knowledge Graphs"
(arXiv:2503.07584v3). The documents are used as the knowledge base for a Retrieval-Augmented Generation (RAG) system
focused on GDELT (Global Database of Events, Language, and Tone) analysis.

### Dataset Summary

- **Total Documents**: 38 pages
- **Source**: Research paper on GDELT Knowledge Graphs
- **Format**: PDF pages with extracted text and metadata
- **Use Case**: RAG knowledge base for GDELT-related queries

### Data Fields

- `page_content` (string): Extracted text content from the PDF page
- `metadata` (dict): Document metadata including:
  - `title`: Paper title
  - `author`: Paper authors
  - `page`: Page number
  - `total_pages`: Total pages in source document
  - `file_path`: Original file path
  - `format`: Document format (PDF)
  - `producer`, `creator`: PDF metadata
  - Other PDF metadata fields

### Data Splits

This dataset contains a single split with all 38 documents.

### Source Data

The source material is the research paper:
- **Title**: "Talking to GDELT Through Knowledge Graphs"
- **Authors**: Audun Myers, Max Vargas, Sinan G. Aksoy, Cliff Joslyn, Benjamin Wilson, Lee Burke, Tom Grimes
- **arXiv ID**: 2503.07584v3

### Licensing

This dataset is released under the Apache 2.0 license.

### Citation

If you use this dataset, please cite the original paper:

```
@article{myers2025talking,
  title={Talking to GDELT Through Knowledge Graphs},
  author={Myers, Audun and Vargas, Max and Aksoy, Sinan G and Joslyn, Cliff and Wilson, Benjamin and Burke, Lee and Grimes, Tom},
  journal={arXiv preprint arXiv:2503.07584},
  year={2025}
}
```

### Dataset Creation

This dataset was created as part of the AI Engineering Bootcamp Cohort 8 certification challenge project.
"""


def create_golden_testset_card() -> str:
    """Create dataset card for golden testset."""
    return """---
license: apache-2.0
task_categories:
- question-answering
- text-generation
tags:
- rag
- evaluation
- ragas
- gdelt
- knowledge-graphs
pretty_name: GDELT RAG Golden Test Set
size_categories:
- n<1K
---

# GDELT RAG Golden Test Set

## Dataset Description

This dataset contains a curated set of question-answering pairs designed for evaluating RAG (Retrieval-Augmented Generation)
systems focused on GDELT (Global Database of Events, Language, and Tone) analysis. The dataset was generated using the
RAGAS framework for synthetic test data generation.

### Dataset Summary

- **Total Examples**: 12 QA pairs
- **Purpose**: RAG system evaluation
- **Framework**: RAGAS (Retrieval-Augmented Generation Assessment)
- **Domain**: GDELT Knowledge Graphs

### Data Fields

- `user_input` (string): The question or query
- `reference_contexts` (list[string]): Ground truth context passages that contain the answer
- `reference` (string): Ground truth answer
- `synthesizer_name` (string): Name of the RAGAS synthesizer used to generate the example
  - `single_hop_specifc_query_synthesizer`: Single-hop specific queries
  - `multi_hop_abstract_query_synthesizer`: Multi-hop abstract queries

### Data Splits

This dataset contains a single split with all 12 evaluation examples.

### Example Queries

The dataset includes questions about:
- GDELT data formats (JSON, CSV)
- GDELT Translingual features
- Date mentions in news articles
- Proximity context in GKG 2.1
- Emotion and theme measurement across languages

### Intended Use

This dataset is intended for:
- Evaluating RAG systems on GDELT-related queries
- Benchmarking retrieval quality using RAGAS metrics:
  - Context Precision
  - Context Recall
  - Faithfulness
  - Answer Relevancy

### Licensing

This dataset is released under the Apache 2.0 license.

### Dataset Creation

This dataset was created using RAGAS synthetic test data generation as part of the AI Engineering Bootcamp Cohort 8
certification challenge project. The source documents come from the research paper "Talking to GDELT Through Knowledge Graphs"
(arXiv:2503.07584v3).

### Evaluation Metrics

Average reference contexts per question: 1.67
"""


def load_manifest():
    """Load manifest.json."""
    with open(MANIFEST_PATH, "r") as f:
        return json.load(f)


def update_manifest(sources_repo: str, golden_testset_repo: str):
    """Update manifest with dataset repo IDs."""
    manifest = load_manifest()

    # Create lineage structure if it doesn't exist
    if "lineage" not in manifest:
        manifest["lineage"] = {}

    if "hf" not in manifest["lineage"]:
        manifest["lineage"]["hf"] = {}

    # Update lineage section
    manifest["lineage"]["hf"]["dataset_repo_id"] = {
        "sources": sources_repo,
        "golden_testset": golden_testset_repo
    }
    manifest["lineage"]["hf"]["pending_upload"] = False
    manifest["lineage"]["hf"]["uploaded_at"] = datetime.now(timezone.utc).isoformat()

    # Write updated manifest
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✅ Updated manifest.json with dataset repo IDs")


def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="Publish interim HF datasets")
    # Option A: pass full repo IDs (highest precedence)
    p.add_argument("--sources", type=str, help="Full HF repo id for sources dataset (e.g. dwb2023/gdelt-rag-sources-v4)")
    p.add_argument("--golden", type=str, help="Full HF repo id for golden testset (e.g. dwb2023/gdelt-rag-golden-testset-v4)")
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
    sources_repo = args.sources or f"{namespace}/gdelt-rag-sources-{version}"
    golden_repo = args.golden or f"{namespace}/gdelt-rag-golden-testset-{version}"

    # Check for HF token
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not set")

    # Login to Hugging Face
    print("🔐 Logging in to Hugging Face...")
    login(token=hf_token)

    # Initialize API
    api = HfApi()

    # Load datasets
    print(f"\n📂 Loading datasets from {DATA_DIR}...")
    sources_dataset = load_from_disk(str(SOURCES_PATH))
    golden_testset_dataset = load_from_disk(str(GOLDEN_TESTSET_PATH))

    print(f"   • Sources dataset: {len(sources_dataset)} documents")
    print(f"   • Golden testset: {len(golden_testset_dataset)} examples")

    # Upload sources dataset
    print(f"\n📤 Uploading sources dataset to {sources_repo}...")
    sources_dataset.push_to_hub(
        sources_repo,
        private=False,
        token=hf_token
    )

    # Create and upload sources dataset card
    print(f"   • Creating dataset card...")
    api.upload_file(
        path_or_fileobj=create_sources_card().encode(),
        path_in_repo="README.md",
        repo_id=sources_repo,
        repo_type="dataset",
        token=hf_token
    )
    print(f"   ✅ Sources dataset uploaded successfully!")
    print(f"      View at: https://huggingface.co/datasets/{sources_repo}")

    # Upload golden testset dataset
    print(f"\n📤 Uploading golden testset to {golden_repo}...")
    golden_testset_dataset.push_to_hub(
        golden_repo,
        private=False,
        token=hf_token
    )

    # Create and upload golden testset dataset card
    print(f"   • Creating dataset card...")
    api.upload_file(
        path_or_fileobj=create_golden_testset_card().encode(),
        path_in_repo="README.md",
        repo_id=golden_repo,
        repo_type="dataset",
        token=hf_token
    )
    print(f"   ✅ Golden testset uploaded successfully!")
    print(f"      View at: https://huggingface.co/datasets/{golden_repo}")

    # Update manifest
    print(f"\n📝 Updating manifest...")
    update_manifest(sources_repo, golden_repo)

    print("\n🎉 All datasets uploaded successfully!")
    print(f"\n📊 Dataset URLs:")
    print(f"   • Sources: https://huggingface.co/datasets/{sources_repo}")
    print(f"   • Golden Testset: https://huggingface.co/datasets/{golden_repo}")


if __name__ == "__main__":
    main()
