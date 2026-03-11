import unittest
from unittest.mock import patch

from adapter.metrics import (
    composite_score,
    compression_ratio,
    coverage_score,
    faithfulness_proxy,
    rouge_1_f1,
    rouge_2_f1,
    rouge_l_f1,
)


class _FakeScore:
    def __init__(self, fmeasure: float) -> None:
        self.fmeasure = fmeasure


class _FakeRougeScorer:
    def score(self, reference: str, prediction: str):
        ref_len = max(len(reference.split()), 1)
        pred_len = max(len(prediction.split()), 1)
        overlap = min(ref_len, pred_len) / max(ref_len, pred_len)
        return {
            "rouge1": _FakeScore(overlap),
            "rouge2": _FakeScore(overlap / 2),
            "rougeL": _FakeScore(overlap),
        }


class MetricTests(unittest.TestCase):
    def test_rouge_scores_non_negative(self) -> None:
        pred = "the cat sat on mat"
        ref = "cat sat on the mat"
        with patch("adapter.metrics._get_rouge_scorer", return_value=_FakeRougeScorer()):
            self.assertGreaterEqual(rouge_1_f1(pred, ref), 0.0)
            self.assertGreaterEqual(rouge_2_f1(pred, ref), 0.0)
            self.assertGreaterEqual(rouge_l_f1(pred, ref), 0.0)

    def test_proxy_metrics(self) -> None:
        doc = "alice visited paris and wrote a report"
        summary = "alice wrote a report"
        ref = "alice wrote report about paris"
        self.assertGreater(faithfulness_proxy(summary, doc), 0.5)
        self.assertGreater(coverage_score(summary, ref), 0.3)
        self.assertGreater(compression_ratio(summary, doc), 0.0)

    def test_composite_score_bounds(self) -> None:
        value = composite_score(0.8, 0.7, 0.6, 0.9)
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)


if __name__ == "__main__":
    unittest.main()
