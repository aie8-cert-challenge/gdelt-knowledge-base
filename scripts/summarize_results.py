#!/usr/bin/env python3
"""
Phase 3: Summarize Results and Generate Manifest

This script aggregates RAGAS evaluation results from Phase 2 and generates
the reproducibility manifest. This is a free phase (no API calls) that reads
pre-computed Parquet files.

What it does:
1. Loads *_evaluation_metrics.parquet files from Phase 2
2. Computes comparative analysis (mean metrics per retriever)
3. Saves comparative_ragas_results.parquet
4. Generates RUN_MANIFEST.json for reproducibility

This is the cheapest phase ($0, <1 second) and can be re-run freely.

Usage:
    make summarize
    # or
    python scripts/summarize_results.py --version v3
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Optional, Any
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import generate_run_manifest


# ==============================================================================
# COMMAND LINE ARGUMENTS
# ==============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 3: Summarize evaluation results and generate manifest"
    )
    parser.add_argument(
        "--indir",
        type=str,
        default=str(Path(__file__).parent.parent / "data" / "processed"),
        help="Input directory containing *_evaluation_metrics.parquet files"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(Path(__file__).parent.parent / "data" / "processed"),
        help="Output directory for comparative results and manifest"
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v3",
        help="Dataset version for provenance tracking (default: v3)"
    )
    return parser.parse_args()


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_evaluation_results(indir: Path) -> Dict[str, pd.DataFrame]:
    """Load pre-computed RAGAS metrics from Parquet files.

    Args:
        indir: Directory containing *_evaluation_metrics.parquet files

    Returns:
        Dict of {retriever_name → DataFrame with RAGAS scores}

    Raises:
        RuntimeError: If no evaluation results found
    """
    retrievers = ["naive", "bm25", "ensemble", "cohere_rerank"]
    results = {}

    print("\n" + "="*80)
    print("STEP 1: LOADING EVALUATION RESULTS")
    print("="*80)
    print("\n📊 Reading evaluation metrics...")

    for name in retrievers:
        path = indir / f"{name}_evaluation_metrics.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            results[name] = df
            print(f"   ✓ {path.name} ({len(df)} examples)")
        else:
            print(f"   ⚠  Missing: {path.name}")

    if not results:
        raise RuntimeError(f"❌ No evaluation results found in {indir}")

    return results


# ==============================================================================
# COMPARATIVE ANALYSIS (Extracted from run_eval_harness.py lines 275-317)
# ==============================================================================

def compute_and_display_results(results: Dict[str, pd.DataFrame], out_dir: Path) -> pd.DataFrame:
    """Compute comparative analysis across retrievers.

    This function is extracted verbatim from run_eval_harness.py (STEP 5).
    The only change: `rdf` is already a DataFrame (not `res.to_pandas()`).

    Args:
        results: Dict of {retriever_name → DataFrame with RAGAS scores}
        out_dir: Output directory for parquet file

    Returns:
        Comparative DataFrame sorted by average score
    """
    print("\n" + "="*80)
    print("STEP 2: COMPARATIVE ANALYSIS")
    print("="*80)

    comp = []
    for name, rdf in results.items():
        row = {
            "Retriever": name.replace("_", " ").title(),
            "Faithfulness": rdf["faithfulness"].mean(),
            "Answer Relevancy": rdf["answer_relevancy"].mean(),
            "Context Precision": rdf["context_precision"].mean(),
            "Context Recall": rdf["context_recall"].mean(),
        }
        row["Average"] = (
            row["Faithfulness"] +
            row["Answer Relevancy"] +
            row["Context Precision"] +
            row["Context Recall"]
        ) / 4
        comp.append(row)

    comp_df = pd.DataFrame(comp).sort_values("Average", ascending=False).reset_index(drop=True)

    # Save comparative table
    comp_parquet = out_dir / "comparative_ragas_results.parquet"
    comp_df.to_parquet(comp_parquet, compression="zstd", index=False)
    print(f"\n💾 Saved comparative results: {comp_parquet.name}")

    # Display results
    print("\n" + "="*80)
    print("COMPARATIVE RAGAS RESULTS")
    print("="*80)
    print()
    print(comp_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()

    # Find winner
    winner = comp_df.iloc[0]
    baseline = comp_df[comp_df["Retriever"] == "Naive"].iloc[0] if "Naive" in comp_df["Retriever"].values else None

    if baseline is not None and winner["Retriever"] != "Naive":
        improvement = ((winner["Average"] - baseline["Average"]) / baseline["Average"]) * 100
        print(f"🏆 Winner: {winner['Retriever']} with {winner['Average']:.2%} average score")
        print(f"   Improvement over baseline: +{improvement:.1f}%")
    else:
        print(f"🏆 Best performer: {winner['Retriever']} with {winner['Average']:.2%} average score")

    return comp_df


# ==============================================================================
# MANIFEST GENERATION (Adapted from run_eval_harness.py lines 362-381)
# ==============================================================================

def generate_and_save_manifest(out_dir: Path, data_provenance: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate RUN_MANIFEST.json for reproducibility.

    Adapted from run_eval_harness.py (STEP 6). Key changes:
    - evaluation_results=None (Phase 3 doesn't have Result objects)
    - retrievers_config=None (Phase 3 doesn't instantiate graphs)

    Args:
        out_dir: Output directory for manifest file
        data_provenance: Optional dict linking to ingestion manifest

    Returns:
        Generated manifest dict
    """
    print("\n" + "="*80)
    print("STEP 3: GENERATING RUN MANIFEST")
    print("="*80)

    manifest_path = out_dir / "RUN_MANIFEST.json"
    manifest = generate_run_manifest(
        output_path=manifest_path,
        evaluation_results=None,      # Phase 3 doesn't have Result objects
        retrievers_config=None,       # Phase 3 doesn't instantiate graphs
        data_provenance=data_provenance
    )

    print(f"\n💾 Saved run manifest: {manifest_path.name}")
    print(f"   ✓ RAGAS version: {manifest['ragas_version']}")
    print(f"   ✓ Python version: {manifest['python_version']}")
    print(f"   ✓ Retriever configs: {len(manifest['retrievers'])}")
    print(f"   ✓ Evaluation settings captured")

    return manifest


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Phase 3 main execution."""
    args = parse_args()
    in_dir = Path(args.indir)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("PHASE 3: SUMMARIZATION & MANIFEST GENERATION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Input Dir: {in_dir}")
    print(f"  Output Dir: {out_dir}")
    print(f"  Version: {args.version}")

    # Load optional data provenance from ingestion manifest
    ingest_manifest_path = Path(__file__).parent.parent / "data/interim/manifest.json"
    data_provenance = None
    if ingest_manifest_path.exists():
        with open(ingest_manifest_path) as f:
            ingest_manifest = json.load(f)
        data_provenance = {
            "ingest_manifest_id": ingest_manifest["id"],
            "ingest_timestamp": ingest_manifest["generated_at"],
            "sources_sha256": ingest_manifest["fingerprints"]["sources"]["jsonl_sha256"],
            "golden_testset_sha256": ingest_manifest["fingerprints"]["golden_testset"]["jsonl_sha256"],
            "source_pdfs_count": ingest_manifest["params"]["MAX_DOCS"] or "all",
            "ragas_testset_size": ingest_manifest["params"]["TESTSET_SIZE"]
        }
        print(f"  ✓ Linked to ingestion manifest: {ingest_manifest['id'][:8]}...")
    else:
        print(f"  ⚠  No ingestion manifest found at {ingest_manifest_path}")

    # Execute pipeline
    start_time = datetime.now()

    # Step 1: Load evaluation results from Parquet files
    results = load_evaluation_results(in_dir)

    # Step 2: Compute comparative analysis and display results
    comp_df = compute_and_display_results(results, out_dir)

    # Step 3: Generate reproducibility manifest
    manifest = generate_and_save_manifest(out_dir, data_provenance)

    # Final summary
    elapsed = (datetime.now() - start_time).total_seconds()
    print("\n" + "="*80)
    print("✅ SUMMARIZATION COMPLETE")
    print("="*80)
    print(f"\nTime: {elapsed:.1f} seconds")
    print(f"Retrievers analyzed: {len(results)}")
    print(f"\nOutput Files ({out_dir}):")
    print(f"  - comparative_ragas_results.parquet (4 rows, 6 columns)")
    print(f"  - RUN_MANIFEST.json (reproducibility manifest)")
    print(f"\n💡 Next step: make deliverables")


if __name__ == "__main__":
    main()
