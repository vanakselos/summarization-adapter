"""Evaluation entrypoint for summarization predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

from .config import load_config
from .llm_judge import judge_summary_openrouter
from .metrics import (
    bertscore_f1,
    compression_efficiency,
    compression_ratio,
    composite_score,
    coverage_score,
    faithfulness_proxy,
    rouge_1_f1,
    rouge_2_f1,
    rouge_l_f1,
)
from .utils import load_jsonl


def _print_evaluation_progress(index: int, total: int, row_id: str) -> None:
    print(f"[{index}/{total}] Evaluate id={row_id}")


def _index_by_id(rows: list[dict]) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for i, row in enumerate(rows):
        row_id = str(row.get("id", i))
        indexed[row_id] = row
    return indexed


def _mean_or_zero(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def evaluate_predictions(
    pred_rows: list[dict],
    ref_rows: list[dict],
    enable_bertscore: bool,
    enable_rouge: bool = True,
    enable_faithfulness_proxy: bool = True,
    enable_llm_judge: bool = False,
    llm_judge_model: str = "google/gemini-3.1-pro-preview",
    llm_judge_api_base: str = "https://openrouter.ai/api/v1",
    llm_judge_timeout_seconds: int = 120,
    llm_judge_max_tokens: int = 400,
    llm_judge_temperature: float = 0.0,
) -> dict[str, Any]:
    pred_by_id = _index_by_id(pred_rows)
    ref_by_id = _index_by_id(ref_rows)

    common_ids = sorted(set(pred_by_id) & set(ref_by_id))
    if not common_ids:
        raise ValueError("No overlapping IDs between predictions and references.")

    per_example: list[dict[str, Any]] = []

    for idx, row_id in enumerate(common_ids, start=1):
        _print_evaluation_progress(idx, len(common_ids), row_id)
        pred = pred_by_id[row_id]
        ref = ref_by_id[row_id]

        summary = str(pred.get("summary", "")).strip()
        reference = str(ref.get("summary", "")).strip()
        document = str(ref.get("document", pred.get("document", ""))).strip()

        r1 = rouge_1_f1(summary, reference) if enable_rouge else 0.0
        r2 = rouge_2_f1(summary, reference) if enable_rouge else 0.0
        rl = rouge_l_f1(summary, reference) if enable_rouge else 0.0
        sem = bertscore_f1(summary, reference) if enable_bertscore else None
        if sem is None:
            sem = r1

        faith = faithfulness_proxy(summary, document) if enable_faithfulness_proxy else 0.0
        cov = coverage_score(summary, reference)
        comp_ratio = compression_ratio(summary, document)
        comp_eff = compression_efficiency(summary, document)
        final_score = composite_score(faith, sem, cov, comp_eff)
        llm_judge: dict[str, Any] | None = None
        if enable_llm_judge:
            llm_judge = judge_summary_openrouter(
                document=document,
                summary=summary,
                reference=reference,
                model=llm_judge_model,
                api_base=llm_judge_api_base,
                timeout_seconds=llm_judge_timeout_seconds,
                max_tokens=llm_judge_max_tokens,
                temperature=llm_judge_temperature,
            )

        row: dict[str, Any] = {
            "id": row_id,
            "rouge1_f1": r1,
            "rouge2_f1": r2,
            "rougeL_f1": rl,
            "bertscore_f1": sem,
            "faithfulness_proxy": faith,
            "coverage": cov,
            "compression_ratio": comp_ratio,
            "compression_efficiency": comp_eff,
            "composite_score": final_score,
            "summary": summary,
            "reference": reference,
            "document": document,
        }
        if llm_judge is not None:
            row.update(
                {
                    "llm_judge_model": llm_judge_model,
                    "llm_judge_faithfulness_1_5": llm_judge["faithfulness_1_5"],
                    "llm_judge_coverage_1_5": llm_judge["coverage_1_5"],
                    "llm_judge_coherence_1_5": llm_judge["coherence_1_5"],
                    "llm_judge_conciseness_1_5": llm_judge["conciseness_1_5"],
                    "llm_judge_overall_1_5": llm_judge["overall_1_5"],
                    "llm_judge_rationale": llm_judge["rationale"],
                }
            )
        per_example.append(row)

    aggregate = {
        "num_examples": len(per_example),
        "rouge1_f1": _mean_or_zero([row["rouge1_f1"] for row in per_example]),
        "rouge2_f1": _mean_or_zero([row["rouge2_f1"] for row in per_example]),
        "rougeL_f1": _mean_or_zero([row["rougeL_f1"] for row in per_example]),
        "bertscore_f1": _mean_or_zero([row["bertscore_f1"] for row in per_example]),
        "faithfulness_proxy": _mean_or_zero([row["faithfulness_proxy"] for row in per_example]),
        "coverage": _mean_or_zero([row["coverage"] for row in per_example]),
        "compression_ratio": _mean_or_zero([row["compression_ratio"] for row in per_example]),
        "compression_efficiency": _mean_or_zero(
            [row["compression_efficiency"] for row in per_example]
        ),
        "composite_score": _mean_or_zero([row["composite_score"] for row in per_example]),
    }
    if enable_llm_judge:
        aggregate.update(
            {
                "llm_judge_model": llm_judge_model,
                "llm_judge_faithfulness_1_5": _mean_or_zero(
                    [row["llm_judge_faithfulness_1_5"] for row in per_example]
                ),
                "llm_judge_coverage_1_5": _mean_or_zero(
                    [row["llm_judge_coverage_1_5"] for row in per_example]
                ),
                "llm_judge_coherence_1_5": _mean_or_zero(
                    [row["llm_judge_coherence_1_5"] for row in per_example]
                ),
                "llm_judge_conciseness_1_5": _mean_or_zero(
                    [row["llm_judge_conciseness_1_5"] for row in per_example]
                ),
                "llm_judge_overall_1_5": _mean_or_zero(
                    [row["llm_judge_overall_1_5"] for row in per_example]
                ),
            }
        )

    return {"aggregate": aggregate, "per_example": per_example}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_human_review_template(path: Path, rows: list[dict[str, Any]], sample_size: int) -> None:
    sampled = rows[:sample_size]
    template_rows = []
    for row in sampled:
        template_rows.append(
            {
                "id": row["id"],
                "summary": row["summary"],
                "reference": row["reference"],
                "document": row["document"],
                "faithfulness_rating_1_5": "",
                "coherence_rating_1_5": "",
                "conciseness_rating_1_5": "",
                "coverage_rating_1_5": "",
                "notes": "",
            }
        )
    _write_csv(path, template_rows)


def run_evaluation(
    config_path: str,
    pred_file: str,
    ref_file: str,
    report_file: str,
    per_example_file: str,
    human_review_file: str,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    pred_rows = load_jsonl(pred_file)
    ref_rows = load_jsonl(ref_file)
    print(f"Loaded predictions: {len(pred_rows)}")
    print(f"Loaded references: {len(ref_rows)}")

    result = evaluate_predictions(
        pred_rows=pred_rows,
        ref_rows=ref_rows,
        enable_bertscore=cfg.eval.enable_bertscore,
        enable_rouge=cfg.eval.enable_rouge,
        enable_faithfulness_proxy=cfg.eval.enable_faithfulness_proxy,
        enable_llm_judge=cfg.eval.enable_llm_judge,
        llm_judge_model=cfg.eval.llm_judge_model,
        llm_judge_api_base=cfg.eval.llm_judge_api_base,
        llm_judge_timeout_seconds=cfg.eval.llm_judge_timeout_seconds,
        llm_judge_max_tokens=cfg.eval.llm_judge_max_tokens,
        llm_judge_temperature=cfg.eval.llm_judge_temperature,
    )

    report_path = Path(report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(result["aggregate"], handle, indent=2)
    print(f"Wrote aggregate report: {report_file}")

    _write_csv(Path(per_example_file), result["per_example"])
    print(f"Wrote per-example report: {per_example_file}")
    _write_human_review_template(
        Path(human_review_file),
        result["per_example"],
        sample_size=cfg.eval.human_review_samples,
    )
    print(f"Wrote human review template: {human_review_file}")

    return result["aggregate"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate summarization outputs")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--pred_file", required=True, help="Path to predictions jsonl")
    parser.add_argument("--ref_file", required=True, help="Path to references jsonl")
    parser.add_argument("--report_file", required=True, help="Output JSON report path")
    parser.add_argument("--per_example_file", required=True, help="Output CSV path")
    parser.add_argument(
        "--human_review_file",
        required=True,
        help="Output CSV template for manual evaluation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aggregate = run_evaluation(
        config_path=args.config,
        pred_file=args.pred_file,
        ref_file=args.ref_file,
        report_file=args.report_file,
        per_example_file=args.per_example_file,
        human_review_file=args.human_review_file,
    )
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
