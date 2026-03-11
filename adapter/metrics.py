"""Evaluation metrics for summarization quality."""

from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache

TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def _tokens(text: str) -> list[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(text)]


def _f1_from_counters(pred: Counter, ref: Counter) -> float:
    overlap = sum((pred & ref).values())
    if overlap == 0:
        return 0.0
    pred_total = sum(pred.values())
    ref_total = sum(ref.values())
    precision = overlap / pred_total if pred_total else 0.0
    recall = overlap / ref_total if ref_total else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


@lru_cache(maxsize=1)
def _get_rouge_scorer():
    try:
        from rouge_score import rouge_scorer
    except ImportError as exc:
        raise RuntimeError(
            "Missing ROUGE dependency. Install requirements first: pip install -r requirements.txt"
        ) from exc
    return rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def _rouge_fmeasure(metric_name: str, prediction: str, reference: str) -> float:
    scores = _get_rouge_scorer().score(reference, prediction)
    return float(scores[metric_name].fmeasure)


def rouge_1_f1(prediction: str, reference: str) -> float:
    return _rouge_fmeasure("rouge1", prediction, reference)


def rouge_2_f1(prediction: str, reference: str) -> float:
    return _rouge_fmeasure("rouge2", prediction, reference)


def rouge_l_f1(prediction: str, reference: str) -> float:
    return _rouge_fmeasure("rougeL", prediction, reference)


def bertscore_f1(prediction: str, reference: str) -> float | None:
    """Compute BERTScore when optional dependency is available."""
    try:
        from bert_score import score as bert_score
    except ImportError:
        return None

    _, _, f1 = bert_score([prediction], [reference], lang="en", verbose=False)
    return float(f1.mean().item())


def coverage_score(summary: str, reference: str) -> float:
    summary_toks = set(_tokens(summary))
    ref_toks = set(_tokens(reference))
    if not ref_toks:
        return 0.0
    return len(summary_toks & ref_toks) / len(ref_toks)


def faithfulness_proxy(summary: str, document: str) -> float:
    """Token-overlap proxy against source document (0-1)."""
    summary_toks = set(_tokens(summary))
    doc_toks = set(_tokens(document))
    if not summary_toks:
        return 0.0
    return len(summary_toks & doc_toks) / len(summary_toks)


def compression_ratio(summary: str, document: str) -> float:
    doc_len = max(len(_tokens(document)), 1)
    return len(_tokens(summary)) / doc_len


def compression_efficiency(summary: str, document: str, target_ratio: float = 0.2) -> float:
    """Higher score is better when compression is near desired ratio."""
    ratio = compression_ratio(summary, document)
    score = 1.0 - abs(ratio - target_ratio) / max(target_ratio, 1e-6)
    return max(0.0, min(1.0, score))


def composite_score(
    faithfulness: float,
    semantic_similarity: float,
    coverage: float,
    compression_eff: float,
) -> float:
    return (
        0.4 * faithfulness
        + 0.3 * semantic_similarity
        + 0.2 * coverage
        + 0.1 * compression_eff
    )
