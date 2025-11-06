#!/usr/bin/env python3
"""
Phase 1: Run Inference Only

This script runs the inference phase of the evaluation pipeline:
1. Loads source documents and golden testset from HuggingFace
2. Creates/connects to vector store
3. Builds all retrievers and graphs
4. Runs inference for each retriever
5. Saves *_evaluation_inputs.parquet files

This is the most expensive phase (~$3-4 in LLM calls).
Re-running evaluation or summary doesn't require re-running this.

Usage:
    make inference
    # or
    python scripts/run_inference.py --recreate true
"""

import os
import sys
import copy
import json
import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_documents_from_huggingface, load_golden_testset_from_huggingface
from src.config import create_vector_store, get_llm
from src.retrievers import create_retrievers
from src.graph import build_all_graphs


def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="Run inference for all retrievers; writes *_evaluation_inputs.parquet")
    # Option A: pass full repo IDs (highest precedence)
    p.add_argument("--sources", type=str,
                   help="Full HF repo ID for sources dataset (e.g. dwb2023/gdelt-rag-sources-v3)")
    p.add_argument("--golden", type=str,
                   help="Full HF repo ID for golden testset (e.g. dwb2023/gdelt-rag-golden-testset-v3)")
    # Option B: build from namespace+version if full ids not provided
    p.add_argument("--namespace", default="dwb2023",
                   help="HF namespace (default: dwb2023)")
    p.add_argument("--version", default="v3",
                   help="Dataset version (default: v3)")
    p.add_argument("--split", default="train",
                   help="HF split to use")
    p.add_argument("--k", type=int, default=5,
                   help="top-k documents per retriever")
    p.add_argument("--recreate", choices=["true", "false"], default="false",
                   help="recreate vector collection (true/false)")
    p.add_argument("--embed-model", default="text-embedding-3-small",
                   help="OpenAI embedding model")
    p.add_argument("--outdir", default=str(Path(__file__).parent.parent / "data" / "processed"),
                   help="Output directory for parquet files")
    return p.parse_args()


def main():
    """Run inference phase."""
    args = parse_args()
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    recreate = args.recreate.lower() == "true"

    # Resolve dataset names (full IDs take precedence)
    sources_dataset = args.sources or f"{args.namespace}/gdelt-rag-sources-{args.version}"
    golden_dataset = args.golden or f"{args.namespace}/gdelt-rag-golden-testset-{args.version}"

    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("❌ OPENAI_API_KEY not set")

    print("="*80)
    print("PHASE 1: INFERENCE")
    print("="*80)
    print(f"Configuration:")
    print(f"  Sources: {sources_dataset}")
    print(f"  Golden: {golden_dataset}")
    print(f"  Split: {args.split}")
    print(f"  K: {args.k}")
    print(f"  Recreate: {recreate}")
    print(f"  Embed Model: {args.embed_model}")
    print(f"  Output Dir: {out_dir}")

    # Optional Cohere check
    if os.getenv("COHERE_API_KEY"):
        print("  ✓ Cohere API key configured (cohere_rerank will work)")
    else:
        print("  ⚠ Cohere API key not set (cohere_rerank may fail)")

    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)

    print(f"\nLoading source documents from {sources_dataset}/{args.split}...")
    docs = load_documents_from_huggingface(sources_dataset, args.split)
    print(f"✓ Loaded {len(docs)} source documents")

    print(f"\nLoading golden testset from {golden_dataset}/{args.split}...")
    golden_ds = load_golden_testset_from_huggingface(golden_dataset, args.split)
    golden_df = golden_ds.to_pandas()
    print(f"✓ Loaded {len(golden_df)} test questions")
    print(f"  Columns: {golden_df.columns.tolist()}")

    # Build RAG stack
    print("\n" + "="*80)
    print("BUILDING RAG STACK")
    print("="*80)

    print(f"\nCreating vector store (recreate={recreate})...")
    vs = create_vector_store(docs, recreate_collection=recreate)
    if recreate:
        print("✓ Vector store recreated and populated")
    else:
        print("✓ Vector store connected (reusing existing collection)")

    print(f"\nCreating retrievers (k={args.k})...")
    retrievers = create_retrievers(docs, vs, k=args.k)
    print(f"✓ Created {len(retrievers)} retrievers: {list(retrievers.keys())}")

    print("\nBuilding LangGraph workflows...")
    graphs = build_all_graphs(retrievers, llm=get_llm())
    print(f"✓ Built {len(graphs)} compiled graphs")

    # Run inference
    print("\n" + "="*80)
    print("RUNNING INFERENCE")
    print("="*80)
    print(f"Processing {len(golden_df)} questions × {len(graphs)} retrievers...")

    start_time = datetime.now()
    files_written = []

    for name, graph in graphs.items():
        print(f"\n📊 Processing {name} retriever...")

        # Create deep copy of golden dataset
        df = copy.deepcopy(golden_df)

        # Initialize columns
        df["response"] = None
        df["retrieved_contexts"] = None

        # Run inference
        for idx, row in df.iterrows():
            q = row["user_input"]
            result = graph.invoke({"question": q})
            df.at[idx, "response"] = result["response"]
            df.at[idx, "retrieved_contexts"] = [d.page_content for d in result["context"]]

            # Progress indicator
            if (idx + 1) % 3 == 0 or idx == len(df) - 1:
                print(f"   Progress: {idx + 1}/{len(df)} questions")

        # Save results
        out_path = out_dir / f"{name}_evaluation_inputs.parquet"
        df.to_parquet(out_path, compression="zstd", index=False)
        files_written.append(out_path.name)
        print(f"   💾 Saved: {out_path.name}")

    # Summary
    elapsed = (datetime.now() - start_time).total_seconds()
    print("\n" + "="*80)
    print("✅ INFERENCE COMPLETE")
    print("="*80)
    print(f"  Time: {elapsed:.1f} seconds")
    print(f"  Files written: {len(files_written)}")
    for f in files_written:
        print(f"    • {f}")
    print(f"\n💡 Next step: make eval-metrics")


if __name__ == "__main__":
    main()