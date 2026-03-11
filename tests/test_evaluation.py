import unittest
from unittest.mock import patch

from adapter.evaluation import evaluate_predictions


class EvaluationTests(unittest.TestCase):
    def test_evaluate_predictions_with_overlap_ids(self) -> None:
        preds = [{"id": "1", "summary": "alice wrote report", "document": "alice wrote report in paris"}]
        refs = [{"id": "1", "summary": "alice wrote a report", "document": "alice wrote report in paris"}]

        with patch("adapter.metrics._get_rouge_scorer") as fake_rouge:
            fake_rouge.return_value.score.return_value = {
                "rouge1": type("Score", (), {"fmeasure": 0.5})(),
                "rouge2": type("Score", (), {"fmeasure": 0.3})(),
                "rougeL": type("Score", (), {"fmeasure": 0.4})(),
            }
            result = evaluate_predictions(pred_rows=preds, ref_rows=refs, enable_bertscore=False)

        self.assertEqual(result["aggregate"]["num_examples"], 1)
        self.assertIn("composite_score", result["aggregate"])

    def test_evaluate_predictions_without_overlap_ids_raises(self) -> None:
        preds = [{"id": "1", "summary": "a"}]
        refs = [{"id": "2", "summary": "b"}]

        with self.assertRaises(ValueError):
            evaluate_predictions(pred_rows=preds, ref_rows=refs, enable_bertscore=False)

    def test_evaluate_predictions_with_llm_judge(self) -> None:
        preds = [{"id": "1", "summary": "alice wrote report", "document": "alice wrote report in paris"}]
        refs = [{"id": "1", "summary": "alice wrote a report", "document": "alice wrote report in paris"}]

        with patch(
            "adapter.evaluation.judge_summary_openrouter",
            return_value={
                "faithfulness_1_5": 5,
                "coverage_1_5": 4,
                "coherence_1_5": 4,
                "conciseness_1_5": 5,
                "overall_1_5": 4,
                "rationale": "Looks faithful and concise.",
            },
        ), patch("adapter.metrics._get_rouge_scorer") as fake_rouge:
            fake_rouge.return_value.score.return_value = {
                "rouge1": type("Score", (), {"fmeasure": 0.5})(),
                "rouge2": type("Score", (), {"fmeasure": 0.3})(),
                "rougeL": type("Score", (), {"fmeasure": 0.4})(),
            }
            result = evaluate_predictions(
                pred_rows=preds,
                ref_rows=refs,
                enable_bertscore=False,
                enable_llm_judge=True,
            )

        self.assertEqual(result["aggregate"]["llm_judge_overall_1_5"], 4.0)
        self.assertEqual(result["per_example"][0]["llm_judge_faithfulness_1_5"], 5)


if __name__ == "__main__":
    unittest.main()
