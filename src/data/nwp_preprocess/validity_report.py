"""
Validity report generator for preprocessed NWP data.

Scans preprocessed CSV directories and reports the ``is_valid=True``
ratio per column, grouped by subdirectory (coordinate or pressure level).

Usage::

    from src.data.nwp_preprocess.validity_report import generate_validity_report

    report = generate_validity_report("data/preprocess_new/dongbok/ECMWF")
    # {'33.56_126.69': {'wspd10': 0.98, 'wdir10': 0.98, ...}, ...}

    # Or save to JSON:
    generate_validity_report("data/preprocess_new/dongbok/GDAPS", save=True)
    # → writes validity_report.json next to preprocess_log.json
"""
import json
import logging
from pathlib import Path
from typing import Dict

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


def generate_validity_report(
    preprocess_dir: str | Path,
    *,
    save: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Compute per-column ``is_valid=True`` ratio for each subdirectory.

    Automatically discovers the directory structure:
    - ECMWF-style: ``{root}/{lat}_{lon}/*.csv`` → 1-level grouping
    - KMA-style:   ``{root}/{group}/{level}/*.csv`` → 2-level grouping

    Args:
        preprocess_dir: Root directory of preprocessed NWP CSVs.
        save: If ``True``, write ``validity_report.json`` in *preprocess_dir*.

    Returns:
        ``{subdir_name: {column: valid_ratio}}``
        where ``valid_ratio`` is the fraction of rows with ``is_valid=True``
        (1.0 = all valid, 0.0 = all invalid/filled).
    """
    root = Path(preprocess_dir)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")

    # Discover leaf directories (those containing CSVs)
    leaf_dirs = _find_leaf_dirs(root)
    if not leaf_dirs:
        raise FileNotFoundError(f"No CSV files found under {root}")

    logger.info("Scanning %d directories under %s", len(leaf_dirs), root)

    report: Dict[str, Dict[str, float]] = {}

    for leaf in tqdm(leaf_dirs, desc="Analyzing validity"):
        rel_path = str(leaf.relative_to(root))
        col_stats = _analyze_directory(leaf)
        if col_stats:
            report[rel_path] = col_stats

    if save:
        out_path = root / "validity_report.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info("Report saved to %s", out_path)

    # Print summary
    _print_summary(report, root.name)

    return report


def _find_leaf_dirs(root: Path) -> list[Path]:
    """Find directories that directly contain CSV files."""
    leaf_dirs = []
    for path in sorted(root.rglob("*.csv")):
        parent = path.parent
        if parent not in leaf_dirs:
            leaf_dirs.append(parent)
    return leaf_dirs


def _analyze_directory(directory: Path) -> Dict[str, float]:
    """Compute is_valid=True ratio per column for all CSVs in a directory."""
    csv_files = sorted(directory.glob("*.csv"))
    if not csv_files:
        return {}

    total_rows = 0
    valid_rows = 0
    col_valid_counts: Dict[str, int] = {}
    col_total_counts: Dict[str, int] = {}

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path, index_col=0)
        except Exception:
            continue

        if "is_valid" not in df.columns:
            continue

        n = len(df)
        total_rows += n
        valid_mask = df["is_valid"].astype(bool)
        valid_rows += int(valid_mask.sum())

        data_cols = [
            c for c in df.columns
            if c not in {"is_valid", "basis_time", "pressure_level"}
        ]
        for col in data_cols:
            col_total_counts[col] = col_total_counts.get(col, 0) + n
            col_valid_counts[col] = (
                col_valid_counts.get(col, 0) + int(valid_mask.sum())
            )

    if total_rows == 0:
        return {}

    result = {}
    for col in sorted(col_total_counts.keys()):
        total = col_total_counts[col]
        valid = col_valid_counts[col]
        result[col] = round(valid / total, 6) if total > 0 else 0.0

    result["_total_rows"] = int(total_rows)
    result["_valid_rows"] = int(valid_rows)
    result["_valid_ratio"] = round(valid_rows / total_rows, 6)

    return result


def _print_summary(
    report: Dict[str, Dict[str, float]],
    source_name: str,
) -> None:
    """Print a human-readable summary to the logger."""
    if not report:
        logger.warning("No data found to report.")
        return

    logger.info("=" * 60)
    logger.info("Validity Report: %s (%d directories)", source_name, len(report))
    logger.info("=" * 60)

    for subdir, stats in sorted(report.items()):
        total = stats.get("_total_rows", 0)
        ratio = stats.get("_valid_ratio", 0.0)
        invalid = total - stats.get("_valid_rows", 0)

        col_ratios = {
            k: v for k, v in stats.items() if not k.startswith("_")
        }
        min_col = min(col_ratios, key=col_ratios.get) if col_ratios else "-"
        min_val = col_ratios.get(min_col, 0.0)

        logger.info(
            "  %s: %d rows, valid=%.2f%%, lowest=%s (%.2f%%)",
            subdir, total, ratio * 100, min_col, min_val * 100,
        )
