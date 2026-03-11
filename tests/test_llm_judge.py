import unittest

from adapter.llm_judge import _extract_json_object, _normalize_judge_result


class LLMJudgeTests(unittest.TestCase):
    def test_extract_json_object_from_code_fence(self) -> None:
        text = """```json
{"faithfulness_1_5": 5, "coverage_1_5": 4, "coherence_1_5": 4, "conciseness_1_5": 5, "overall_1_5": 4, "rationale": "ok"}
```"""
        parsed = _extract_json_object(text)
        self.assertEqual(parsed["overall_1_5"], 4)

    def test_normalize_judge_result_rejects_out_of_range_scores(self) -> None:
        with self.assertRaises(ValueError):
            _normalize_judge_result(
                {
                    "faithfulness_1_5": 0,
                    "coverage_1_5": 4,
                    "coherence_1_5": 4,
                    "conciseness_1_5": 5,
                    "overall_1_5": 4,
                    "rationale": "ok",
                }
            )


if __name__ == "__main__":
    unittest.main()
