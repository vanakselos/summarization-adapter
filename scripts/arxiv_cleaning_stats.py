#!/usr/bin/env python3
"""Compute ArXiv cleaning statistics and export charts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from adapter.data import _clean_arxiv_text, _load_hf_parquet, _resolve_dataset_spec


def _word_count(text: str) -> int:
    return len(text.split())


def _token_count(text: str, tokenizer: Any) -> int:
    encoded = tokenizer(text, add_special_tokens=False, truncation=False)
    return len(encoded.get("input_ids", []))


def _placeholder_counts(text: str) -> tuple[int, int]:
    return text.count("@xcite"), sum(1 for token in text.split() if token.startswith("@xmath"))


def _removed_ratio(before_words: int, after_words: int) -> float:
    if before_words <= 0:
        return 0.0
    return max(0.0, (before_words - after_words) / before_words)


def _percentiles(values: list[int] | list[float], points: list[int]) -> dict[str, float]:
    arr = np.array(values)
    return {f"p{point}": float(np.percentile(arr, point)) for point in points}


def _plot_histogram(values: list[int], title: str, xlabel: str, output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=40, color="#3b82f6", edgecolor="white")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_boxplot(series: dict[str, list[int]], title: str, ylabel: str, output_path: Path) -> None:
    labels = list(series.keys())
    values = [series[label] for label in labels]
    plt.figure(figsize=(10, 6))
    plt.boxplot(values, tick_labels=labels, vert=True)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _load_arxiv_rows(split: str, max_samples: int | None) -> list[dict[str, Any]]:
    _, spec = _resolve_dataset_spec("arxiv")
    try:
        kwargs: dict[str, Any] = {"split": split, "streaming": True}
        if spec.config_name:
            ds = load_dataset(spec.hf_id, spec.config_name, **kwargs)
        else:
            ds = load_dataset(spec.hf_id, **kwargs)
    except (RuntimeError, ValueError) as exc:
        message = str(exc)
        if "Dataset scripts are no longer supported" not in message:
            raise
        ds = _load_hf_parquet(
            lambda *args, **kwargs: load_dataset(*args, **kwargs, streaming=True),
            spec,
            split,
        )

    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(ds):
        if max_samples is not None and idx >= max_samples:
            break
        rows.append(row)
    return rows


def _summarize(rows: list[dict[str, Any]], tokenizer: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        raw_source = str(row.get("article", "")).strip()
        raw_summary = str(row.get("abstract", "")).strip()
        if not raw_source or not raw_summary:
            continue

        clean_source = _clean_arxiv_text(raw_source)
        clean_summary = _clean_arxiv_text(raw_summary)
        source_cite_before, source_math_before = _placeholder_counts(raw_source)
        source_cite_after, source_math_after = _placeholder_counts(clean_source)
        summary_cite_before, summary_math_before = _placeholder_counts(raw_summary)
        summary_cite_after, summary_math_after = _placeholder_counts(clean_summary)

        raw_source_words = _word_count(raw_source)
        clean_source_words = _word_count(clean_source)
        raw_summary_words = _word_count(raw_summary)
        clean_summary_words = _word_count(clean_summary)
        raw_source_tokens = _token_count(raw_source, tokenizer)
        clean_source_tokens = _token_count(clean_source, tokenizer)
        raw_summary_tokens = _token_count(raw_summary, tokenizer)
        clean_summary_tokens = _token_count(clean_summary, tokenizer)

        metrics.append(
            {
                "id": str(row.get("id", idx)),
                "raw_source_words": raw_source_words,
                "clean_source_words": clean_source_words,
                "raw_summary_words": raw_summary_words,
                "clean_summary_words": clean_summary_words,
                "raw_source_tokens": raw_source_tokens,
                "clean_source_tokens": clean_source_tokens,
                "raw_summary_tokens": raw_summary_tokens,
                "clean_summary_tokens": clean_summary_tokens,
                "source_placeholders_before": source_cite_before + source_math_before,
                "source_placeholders_after": source_cite_after + source_math_after,
                "summary_placeholders_before": summary_cite_before + summary_math_before,
                "summary_placeholders_after": summary_cite_after + summary_math_after,
                "source_removed_ratio": _removed_ratio(raw_source_words, clean_source_words),
                "summary_removed_ratio": _removed_ratio(raw_summary_words, clean_summary_words),
            }
        )

    if not metrics:
        raise RuntimeError("No valid ArXiv rows were available for statistics.")

    clean_source_words = [row["clean_source_words"] for row in metrics]
    clean_summary_words = [row["clean_summary_words"] for row in metrics]
    clean_source_tokens = [row["clean_source_tokens"] for row in metrics]
    clean_summary_tokens = [row["clean_summary_tokens"] for row in metrics]
    source_placeholders_after = [row["source_placeholders_after"] for row in metrics]
    source_removed_ratio = [row["source_removed_ratio"] for row in metrics]

    summary = {
        "num_rows": len(metrics),
        "recommended_thresholds": {
            "min_source_tokens_p1": int(np.percentile(np.array(clean_source_tokens), 1)),
            "min_summary_tokens_p1": int(np.percentile(np.array(clean_summary_tokens), 1)),
            "max_source_tokens_p99": int(np.percentile(np.array(clean_source_tokens), 99)),
            "max_summary_tokens_p99": int(np.percentile(np.array(clean_summary_tokens), 99)),
            "source_placeholders_after_must_be_zero": False,
        },
        "clean_source_tokens": {
            "min": int(min(clean_source_tokens)),
            "max": int(max(clean_source_tokens)),
            "mean": float(np.mean(clean_source_tokens)),
            **_percentiles(clean_source_tokens, [1, 5, 25, 50, 75, 95, 99]),
        },
        "clean_summary_tokens": {
            "min": int(min(clean_summary_tokens)),
            "max": int(max(clean_summary_tokens)),
            "mean": float(np.mean(clean_summary_tokens)),
            **_percentiles(clean_summary_tokens, [1, 5, 25, 50, 75, 95, 99]),
        },
        "clean_source_words": {
            "min": int(min(clean_source_words)),
            "max": int(max(clean_source_words)),
            "mean": float(np.mean(clean_source_words)),
            **_percentiles(clean_source_words, [1, 5, 25, 50, 75, 95, 99]),
        },
        "clean_summary_words": {
            "min": int(min(clean_summary_words)),
            "max": int(max(clean_summary_words)),
            "mean": float(np.mean(clean_summary_words)),
            **_percentiles(clean_summary_words, [1, 5, 25, 50, 75, 95, 99]),
        },
        "source_placeholders_after": {
            "max": int(max(source_placeholders_after)),
            "rows_with_nonzero": int(sum(1 for value in source_placeholders_after if value > 0)),
        },
        "source_removed_ratio": {
            "mean": float(np.mean(source_removed_ratio)),
            **_percentiles(source_removed_ratio, [1, 5, 25, 50, 75, 95, 99]),
        },
    }
    return summary, metrics


def _write_outputs(output_dir: Path, summary: dict[str, Any], metrics: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "arxiv_cleaning_summary.json"
    metrics_path = output_dir / "arxiv_cleaning_metrics.csv"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)

    clean_source_words = [row["clean_source_words"] for row in metrics]
    clean_summary_words = [row["clean_summary_words"] for row in metrics]
    raw_source_words = [row["raw_source_words"] for row in metrics]
    raw_summary_words = [row["raw_summary_words"] for row in metrics]
    clean_source_tokens = [row["clean_source_tokens"] for row in metrics]
    clean_summary_tokens = [row["clean_summary_tokens"] for row in metrics]
    raw_source_tokens = [row["raw_source_tokens"] for row in metrics]
    raw_summary_tokens = [row["raw_summary_tokens"] for row in metrics]

    _plot_histogram(
        clean_source_words,
        title="ArXiv Clean Source Length Distribution",
        xlabel="Words per cleaned source",
        output_path=output_dir / "arxiv_clean_source_hist.png",
    )
    _plot_histogram(
        clean_summary_words,
        title="ArXiv Clean Summary Length Distribution",
        xlabel="Words per cleaned summary",
        output_path=output_dir / "arxiv_clean_summary_hist.png",
    )
    _plot_boxplot(
        {
            "raw_source_words": raw_source_words,
            "clean_source_words": clean_source_words,
            "raw_summary_words": raw_summary_words,
            "clean_summary_words": clean_summary_words,
        },
        title="ArXiv Length Comparison Before and After Cleaning",
        ylabel="Words",
        output_path=output_dir / "arxiv_length_boxplot.png",
    )
    _plot_histogram(
        clean_source_tokens,
        title="ArXiv Clean Source Token Distribution",
        xlabel="Tokens per cleaned source",
        output_path=output_dir / "arxiv_clean_source_token_hist.png",
    )
    _plot_histogram(
        clean_summary_tokens,
        title="ArXiv Clean Summary Token Distribution",
        xlabel="Tokens per cleaned summary",
        output_path=output_dir / "arxiv_clean_summary_token_hist.png",
    )
    _plot_boxplot(
        {
            "raw_source_tokens": raw_source_tokens,
            "clean_source_tokens": clean_source_tokens,
            "raw_summary_tokens": raw_summary_tokens,
            "clean_summary_tokens": clean_summary_tokens,
        },
        title="ArXiv Token Comparison Before and After Cleaning",
        ylabel="Tokens",
        output_path=output_dir / "arxiv_token_length_boxplot.png",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ArXiv cleaning statistics and charts.")
    parser.add_argument("--split", default="train", help="Dataset split to analyze.")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1000,
        help="Maximum number of ArXiv samples to analyze.",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/arxiv-cleaning-stats",
        help="Directory for exported metrics and charts.",
    )
    parser.add_argument(
        "--tokenizer_model",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Tokenizer model id or local tokenizer path for token statistics.",
    )
    args = parser.parse_args()

    rows = _load_arxiv_rows(split=args.split, max_samples=args.max_samples)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model, use_fast=True)
    summary, metrics = _summarize(rows, tokenizer)
    _write_outputs(Path(args.output_dir), summary, metrics)
    print(json.dumps(summary["recommended_thresholds"], indent=2))
    print(f"Exported statistics to {args.output_dir}")


if __name__ == "__main__":
    main()
