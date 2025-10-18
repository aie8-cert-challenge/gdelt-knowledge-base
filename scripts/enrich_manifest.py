#!/usr/bin/env python3
import json, os, sys, hashlib, platform, re
from pathlib import Path
from datetime import datetime


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_bytes(path: Path) -> int:
    return path.stat().st_size


def count_jsonl_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def hfds_rows(path: Path) -> int:
    # Lazy: infer from arrow files if present; else leave None
    try:
        from datasets import load_from_disk

        ds = load_from_disk(str(path))
        return sum(
            len(split) for split in (ds.values() if hasattr(ds, "values") else [ds])
        )
    except Exception:
        return None


def parquet_rows(path: Path) -> int:
    try:
        import pyarrow.parquet as pq

        return pq.ParquetFile(str(path)).metadata.num_rows
    except Exception:
        try:
            import pandas as pd

            return len(pd.read_parquet(path))
        except Exception:
            return None


def pandas_schema_from_parquet(path: Path):
    try:
        import pandas as pd

        dtypes = pd.read_parquet(path).dtypes
        return [{"name": c, "dtype": str(dtypes[c])} for c in dtypes.index]
    except Exception:
        return None


def char_stats_jsonl(path: Path, field="page_content", max_scan=5000):
    tot = 0
    mx = 0
    n = 0
    import json as _json

    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_scan:
                break
            obj = _json.loads(line)
            s = obj.get(field, "")
            if isinstance(s, str):
                L = len(s)
                tot += L
                mx = max(mx, L)
                n += 1
    return {"avg_chars": (tot / n if n else 0), "max_chars": mx, "scanned": n}


def main(manifest_path: Path):
    m = json.loads(manifest_path.read_text())
    paths = m.get("paths", {})
    src = paths.get("sources", {})
    gt = paths.get("golden_testset", {})

    # ---- env
    def ver(pkg):
        try:
            import importlib.metadata as im

            return im.version(pkg)
        except Exception:
            return None

    m["env"] = {
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}",
        "langchain": ver("langchain"),
        "ragas": ver("ragas"),
        "datasets": ver("datasets"),
        "pyarrow": ver("pyarrow"),
        "huggingface_hub": ver("huggingface-hub"),
    }

    # ---- inputs (best-effort from sources.jsonl metadata.sample)
    # If you want exact PDFs list+hash, compute under your data/raw and store here.
    if "inputs" not in m:
        m["inputs"] = {}
    if "source_dir" not in m["inputs"]:
        # Try to infer common prefix of 'metadata.source' in samples
        samples = m.get("quick_schema", {}).get("sources_jsonl", {}).get("sample") or []
        if samples:
            first = samples[0]
            meta = first.get("metadata", {})
            src_path = meta.get("file_path") or meta.get("source")
            if src_path:
                p = Path(src_path)
                m["inputs"]["source_dir"] = str(p.parent)

    # ---- artifacts: add bytes/rows for each file
    def add_artifact(meta: dict, key: str, is_dir=False):
        p = meta.get(key)
        if not p:
            return None
        P = Path(p)
        out = {"path": p}
        if is_dir:
            out["rows"] = hfds_rows(P)
        else:
            out["bytes"] = file_bytes(P)
            out["sha256"] = sha256(P)
            # rows + schema for parquet/jsonl
            if P.suffix == ".jsonl":
                out["rows"] = count_jsonl_rows(P)
            elif P.suffix == ".parquet":
                out["rows"] = parquet_rows(P)
                out["schema"] = pandas_schema_from_parquet(P)
        return out

    m.setdefault("artifacts", {})
    m["artifacts"]["sources"] = {
        "jsonl": add_artifact(src, "jsonl"),
        "parquet": add_artifact(src, "parquet"),
        "hfds": add_artifact(src, "hfds", is_dir=True),
    }
    m["artifacts"]["golden_testset"] = {
        "jsonl": add_artifact(gt, "jsonl"),
        "parquet": add_artifact(gt, "parquet"),
        "hfds": add_artifact(gt, "hfds", is_dir=True),
    }

    # ---- metrics quick stats
    src_jsonl = src.get("jsonl")
    gt_jsonl = gt.get("jsonl")
    metrics = {"sources": {}, "golden_testset": {}}
    if src_jsonl and Path(src_jsonl).exists():
        metrics["sources"].update(
            {
                "docs": count_jsonl_rows(Path(src_jsonl)),
                "page_content_stats": char_stats_jsonl(Path(src_jsonl), "page_content"),
            }
        )
    if gt_jsonl and Path(gt_jsonl).exists():
        # avg reference_contexts length
        import json as _json

        total = 0
        n = 0
        with Path(gt_jsonl).open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                obj = _json.loads(line)
                rc = obj.get("reference_contexts") or []
                if isinstance(rc, list):
                    total += len(rc)
                    n += 1
        metrics["golden_testset"].update(
            {"rows": n, "avg_reference_contexts": (total / n if n else 0)}
        )
    m["metrics"] = metrics

    # ---- lineage scaffold (leave values if already set)
    m.setdefault(
        "lineage",
        {
            "hf": {"dataset_repo_id": None, "pending_upload": True},
            "langsmith": {"project": None, "dataset_name": None},
            "phoenix": {"workspace": None, "dataset_name": None},
        },
    )

    # ---- compliance scaffold
    m.setdefault(
        "compliance",
        {
            "license": "apache-2.0",
            "pii_present": "unknown",
            "pii_policy": "manual-review-before-publish",
            "notes": None,
        },
    )

    # ---- run details scaffold (opt-in)
    m.setdefault("run", {})
    m["run"].setdefault("random_seed", 42)
    # git commit (best-effort)
    try:
        import subprocess

        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent
            )
            .decode()
            .strip()
        )
        m["run"]["git_commit_sha"] = sha
    except Exception:
        pass

    # Prefer relative paths (optional): make any absolute path under repo root relative
    repo_root = Path.cwd()

    def relativize(obj):
        if isinstance(obj, dict):
            return {k: relativize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [relativize(v) for v in obj]
        if isinstance(obj, str) and obj.startswith(str(repo_root)):
            return str(Path(obj).relative_to(repo_root))
        return obj

    m = relativize(m)

    manifest_path.write_text(json.dumps(m, indent=2, ensure_ascii=False))
    print(f"✅ Enriched manifest written: {manifest_path}")


if __name__ == "__main__":
    path = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/interim/manifest.json")
    )
    main(path)
