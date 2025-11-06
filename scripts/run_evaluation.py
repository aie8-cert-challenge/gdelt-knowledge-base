#!/usr/bin/env python3
"""
Phase 2: Run RAGAS Evaluation

This script runs RAGAS evaluation on existing inference results:
1. Reads *_evaluation_inputs.parquet files
2. Runs RAGAS metrics (Faithfulness, ResponseRelevancy, ContextPrecision, LLMContextRecall)
3. Saves *_evaluation_metrics.parquet files

This phase uses the evaluation LLM (~$2 in API calls).
Can be re-run without re-running expensive inference.

Usage:
    make eval-metrics
    # or
    python scripts/run_evaluation.py --retrievers naive bm25
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.metrics import Faithfulness, ResponseRelevancy, ContextPrecision, LLMContextRecall
from ragas.llms import LangchainLLMWrapper

from src.config import get_llm
from src.config import RETRIEVERS

def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="Evaluate existing inference inputs; writes *_evaluation_metrics.parquet")
    p.add_argument("--indir", default=str(Path(__file__).parent.parent / "data" / "processed"),
                   help="Input directory containing *_evaluation_inputs.parquet files")
    p.add_argument("--retrievers", nargs="*", default=RETRIEVERS,
                   help="Subset of retrievers to evaluate (default: all)")
    p.add_argument("--llm-model", default="gpt-4.1-mini",
                   help="OpenAI LLM model for evaluation")
    p.add_argument("--timeout", type=int, default=360,
                   help="Timeout per evaluation in seconds")
    return p.parse_args()


def main():
    """Run evaluation phase."""
    args = parse_args()
    in_dir = Path(args.indir)

    print("="*80)
    print("PHASE 2: RAGAS EVALUATION")
    print("="*80)
    print(f"Configuration:")
    print(f"  Input Dir: {in_dir}")
    print(f"  Retrievers: {args.retrievers}")
    print(f"  LLM Model: {args.llm_model}")
    print(f"  Timeout: {args.timeout}s")

    # Check for required files
    missing = []
    for name in args.retrievers:
        in_path = in_dir / f"{name}_evaluation_inputs.parquet"
        if not in_path.exists():
            missing.append(in_path.name)

    if missing:
        print("\n⚠ Missing input files:")
        for f in missing:
            print(f"  • {f}")
        print("\n💡 Run 'make inference' first to generate these files")
        return 1

    # Set up evaluation
    evaluator_llm = LangchainLLMWrapper(get_llm())
    run_cfg = RunConfig(timeout=args.timeout)

    print("\n" + "="*80)
    print("RUNNING RAGAS EVALUATION")
    print("="*80)

    metrics = [
        Faithfulness(),
        ResponseRelevancy(),
        ContextPrecision(),
        LLMContextRecall()
    ]
    print(f"Metrics: {[m.__class__.__name__ for m in metrics]}")

    start_time = datetime.now()
    files_written = []
    results_summary = []

    for name in args.retrievers:
        in_path = in_dir / f"{name}_evaluation_inputs.parquet"

        if not in_path.exists():
            print(f"\n⚠ Skipping {name}: {in_path.name} not found")
            continue

        print(f"\n📊 Evaluating {name} retriever...")
        print(f"   Reading: {in_path.name}")

        # Load inference results
        df = pd.read_parquet(in_path)
        print(f"   Loaded {len(df)} examples")

        # Convert to RAGAS dataset
        eval_ds = EvaluationDataset.from_pandas(df)

        # Run evaluation
        print(f"   Running RAGAS evaluation...")
        res = evaluate(
            dataset=eval_ds,
            metrics=metrics,
            llm=evaluator_llm,
            run_config=run_cfg,
        )

        # Save results
        out_path = in_dir / f"{name}_evaluation_metrics.parquet"
        results_df = res.to_pandas()
        results_df.to_parquet(out_path, compression="zstd", index=False)
        files_written.append(out_path.name)
        print(f"   💾 Saved: {out_path.name}")

        # Collect summary statistics
        summary = {
            "retriever": name,
            "faithfulness": results_df["faithfulness"].mean(),
            "answer_relevancy": results_df["answer_relevancy"].mean(),
            "context_precision": results_df["context_precision"].mean(),
            "context_recall": results_df["context_recall"].mean(),
        }
        summary["average"] = (
            summary["faithfulness"] +
            summary["answer_relevancy"] +
            summary["context_precision"] +
            summary["context_recall"]
        ) / 4
        results_summary.append(summary)

        print(f"   📈 Scores:")
        print(f"      • Faithfulness:      {summary['faithfulness']:.4f}")
        print(f"      • Answer Relevancy:  {summary['answer_relevancy']:.4f}")
        print(f"      • Context Precision: {summary['context_precision']:.4f}")
        print(f"      • Context Recall:    {summary['context_recall']:.4f}")
        print(f"      • Average:           {summary['average']:.4f}")

    # Print summary table
    if results_summary:
        elapsed = (datetime.now() - start_time).total_seconds()

        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)

        # Create summary DataFrame and sort by average score
        summary_df = pd.DataFrame(results_summary)
        summary_df = summary_df.sort_values("average", ascending=False)

        print("\n📊 Comparative Results (sorted by average score):")
        print("-" * 80)
        print(f"{'Retriever':<15} {'Faithful':<10} {'Relevant':<10} {'Precision':<10} {'Recall':<10} {'Average':<10}")
        print("-" * 80)

        for _, row in summary_df.iterrows():
            print(f"{row['retriever']:<15} "
                  f"{row['faithfulness']:<10.4f} "
                  f"{row['answer_relevancy']:<10.4f} "
                  f"{row['context_precision']:<10.4f} "
                  f"{row['context_recall']:<10.4f} "
                  f"{row['average']:<10.4f}")

        print("-" * 80)
        print(f"\n🏆 Best performer: {summary_df.iloc[0]['retriever']} (avg: {summary_df.iloc[0]['average']:.4f})")

    # Final summary
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE")
    print("="*80)
    print(f"  Time: {elapsed:.1f} seconds")
    print(f"  Files written: {len(files_written)}")
    for f in files_written:
        print(f"    • {f}")
    print(f"\n💡 Next step: make summarize")


if __name__ == "__main__":
    main()