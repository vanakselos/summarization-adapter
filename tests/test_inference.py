import unittest

from adapter.config import AdapterConfig
from adapter.inference import (
    _build_general_prompt_functions,
    _resolve_inference_dataset,
    arxiv_lead_refine_summarize,
    hierarchical_summarize,
)


class InferenceTests(unittest.TestCase):
    def test_resolve_inference_dataset_for_single_dataset_config(self) -> None:
        cfg = AdapterConfig()
        cfg.data.train_datasets = ["govreport"]
        cfg.data.eval_datasets = ["ccdv/govreport-summarization"]
        self.assertEqual(_resolve_inference_dataset(cfg), "govreport")

    def test_resolve_inference_dataset_returns_none_for_mixed_config(self) -> None:
        cfg = AdapterConfig()
        cfg.data.train_datasets = ["xsum", "govreport"]
        cfg.data.eval_datasets = ["xsum"]
        self.assertIsNone(_resolve_inference_dataset(cfg))

    def test_build_general_prompt_functions_routes_xsum(self) -> None:
        build_chunk_prompt, build_merge_prompt, build_final_prompt = _build_general_prompt_functions(
            "xsum",
            90,
        )
        self.assertIn("news article", build_chunk_prompt("article section"))
        self.assertIn("Final News Notes", build_merge_prompt("note block"))
        self.assertIn("single-sentence news summary", build_final_prompt("full article"))

    def test_build_general_prompt_functions_routes_govreport(self) -> None:
        build_chunk_prompt, build_merge_prompt, build_final_prompt = _build_general_prompt_functions(
            "govreport",
            150,
        )
        self.assertIn("government report", build_chunk_prompt("report section"))
        self.assertIn("executive summary", build_merge_prompt("note block").lower())
        self.assertIn("### Executive Summary:", build_final_prompt("full report"))

    def test_single_pass_for_short_input(self) -> None:
        calls = []

        def fake_summarize(text: str) -> str:
            calls.append(text)
            return "short-summary"

        result = hierarchical_summarize(
            document="one two three",
            chunk_tokens=10,
            chunk_overlap=2,
            summarize_fn=fake_summarize,
        )

        self.assertFalse(result["stats"]["hierarchical"])
        self.assertEqual(result["final_summary"], "short-summary")
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["merged_summary"], "")
        self.assertEqual(result["stats"]["inference_strategy"], "hierarchical")

    def test_hierarchical_for_long_input(self) -> None:
        calls = []

        def fake_summarize(text: str) -> str:
            calls.append(text)
            return f"sum-{len(text.split())}"

        doc = " ".join([f"w{i}" for i in range(12)])
        result = hierarchical_summarize(
            document=doc,
            chunk_tokens=5,
            chunk_overlap=1,
            summarize_fn=fake_summarize,
        )

        self.assertTrue(result["stats"]["hierarchical"])
        self.assertEqual(result["stats"]["num_chunks"], 3)
        self.assertEqual(len(result["partial_summaries"]), 3)
        self.assertEqual(len(calls), 5)
        self.assertTrue(result["merged_summary"])
        self.assertEqual(result["stats"]["inference_strategy"], "hierarchical")

    def test_hierarchical_accepts_batched_chunk_summarizer(self) -> None:
        calls = {"chunks": [], "merge": [], "final": []}

        def fake_summarize_chunks(chunks: list[str]) -> list[str]:
            calls["chunks"].append(list(chunks))
            return [f"sum-{idx}" for idx, _ in enumerate(chunks, start=1)]

        def fake_merge(text: str) -> str:
            calls["merge"].append(text)
            return "merged"

        def fake_final(text: str) -> str:
            calls["final"].append(text)
            return "final"

        doc = " ".join([f"w{i}" for i in range(12)])
        result = hierarchical_summarize(
            document=doc,
            chunk_tokens=5,
            chunk_overlap=1,
            summarize_chunks_fn=fake_summarize_chunks,
            summarize_merge_fn=fake_merge,
            summarize_final_fn=fake_final,
        )

        self.assertEqual(len(calls["chunks"]), 1)
        self.assertEqual(len(calls["chunks"][0]), 3)
        self.assertEqual(result["partial_summaries"], ["sum-1", "sum-2", "sum-3"])
        self.assertEqual(len(calls["merge"]), 1)
        self.assertIn("Section 1 Summary:", calls["merge"][0])
        self.assertEqual(calls["final"], ["merged"])
        self.assertEqual(result["final_summary"], "final")

    def test_arxiv_lead_refine_uses_intro_findings_conclusion_and_final(self) -> None:
        calls = {"intro": [], "findings": [], "conclusion": [], "final": []}

        def fake_intro(text: str) -> str:
            calls["intro"].append(text)
            return f"intro-{len(text.split())}"

        def fake_findings(text: str) -> str:
            calls["findings"].append(text)
            return f"finding-{len(text.split())}"

        def fake_conclusion(text: str) -> str:
            calls["conclusion"].append(text)
            return f"conclusion-{len(text.split())}"

        def fake_final(text: str) -> str:
            calls["final"].append(text)
            return "final-abstract"

        doc = " ".join([f"w{i}" for i in range(16)])
        result = arxiv_lead_refine_summarize(
            document=doc,
            chunk_tokens=5,
            chunk_overlap=1,
            summarize_intro_fn=fake_intro,
            summarize_findings_fn=fake_findings,
            summarize_conclusion_fn=fake_conclusion,
            summarize_final_fn=fake_final,
        )

        self.assertTrue(result["stats"]["hierarchical"])
        self.assertEqual(result["stats"]["num_chunks"], 4)
        self.assertEqual(result["stats"]["inference_strategy"], "arxiv_lead_refine")
        self.assertEqual(len(calls["intro"]), 1)
        self.assertEqual(len(calls["findings"]), 2)
        self.assertEqual(len(calls["conclusion"]), 1)
        self.assertEqual(len(calls["final"]), 1)
        self.assertEqual(result["partial_summaries"], ["intro-5", "finding-5", "finding-5", "conclusion-4"])
        self.assertIn("Introduction Context:", result["merged_summary"])
        self.assertIn("Key Findings:", result["merged_summary"])
        self.assertIn("Conclusion:", result["merged_summary"])
        self.assertEqual(result["final_summary"], "final-abstract")

    def test_arxiv_lead_refine_accepts_batched_findings_summarizer(self) -> None:
        calls = {"intro": [], "findings_batch": [], "conclusion": [], "final": []}

        def fake_intro(text: str) -> str:
            calls["intro"].append(text)
            return "intro"

        def fake_findings_batch(texts: list[str]) -> list[str]:
            calls["findings_batch"].append(list(texts))
            return [f"finding-{idx}" for idx, _ in enumerate(texts, start=1)]

        def fake_conclusion(text: str) -> str:
            calls["conclusion"].append(text)
            return "conclusion"

        def fake_final(text: str) -> str:
            calls["final"].append(text)
            return "final"

        doc = " ".join([f"w{i}" for i in range(16)])
        result = arxiv_lead_refine_summarize(
            document=doc,
            chunk_tokens=5,
            chunk_overlap=1,
            summarize_intro_fn=fake_intro,
            summarize_findings_fn=lambda text: "unused",
            summarize_conclusion_fn=fake_conclusion,
            summarize_final_fn=fake_final,
            summarize_findings_batch_fn=fake_findings_batch,
        )

        self.assertEqual(len(calls["intro"]), 1)
        self.assertEqual(len(calls["findings_batch"]), 1)
        self.assertEqual(len(calls["findings_batch"][0]), 2)
        self.assertEqual(len(calls["conclusion"]), 1)
        self.assertEqual(result["partial_summaries"], ["intro", "finding-1", "finding-2", "conclusion"])
        self.assertEqual(result["final_summary"], "final")

    def test_arxiv_lead_refine_short_input_stays_single_pass(self) -> None:
        calls = {"document": [], "final": []}

        def fake_document(text: str) -> str:
            calls["document"].append(text)
            return "single-pass-abstract"

        result = arxiv_lead_refine_summarize(
            document="one two three",
            chunk_tokens=10,
            chunk_overlap=2,
            summarize_intro_fn=lambda text: "unused",
            summarize_findings_fn=lambda text: "unused",
            summarize_conclusion_fn=lambda text: "unused",
            summarize_final_fn=lambda text: calls["final"].append(text) or "unused-final",
            summarize_document_fn=fake_document,
        )

        self.assertFalse(result["stats"]["hierarchical"])
        self.assertEqual(result["stats"]["inference_strategy"], "arxiv_lead_refine")
        self.assertEqual(result["merged_summary"], "")
        self.assertEqual(result["partial_summaries"], [])
        self.assertEqual(result["final_summary"], "single-pass-abstract")
        self.assertEqual(calls["document"], ["one two three"])
        self.assertEqual(calls["final"], [])


if __name__ == "__main__":
    unittest.main()
