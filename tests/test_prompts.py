import unittest

from adapter.prompts import (
    build_arxiv_final_abstract_prompt,
    build_arxiv_findings_prompt,
    build_arxiv_intro_prompt,
    build_arxiv_paper_summary_prompt,
    build_chunk_summary_prompt,
    build_general_summary_prompt,
    build_govreport_chunk_prompt,
    build_govreport_merge_prompt,
    build_govreport_prompt,
    build_merge_summary_prompt,
    build_summary_prompt,
    build_xsum_chunk_prompt,
    build_xsum_merge_prompt,
    build_xsum_prompt,
    strip_prompt_from_generation,
)


class PromptTests(unittest.TestCase):
    def test_build_summary_prompt_contains_required_sections(self) -> None:
        prompt = build_summary_prompt("alpha beta", max_words=42)
        self.assertIn("### Instruction:", prompt)
        self.assertIn("### Document:", prompt)
        self.assertIn("### Summary:", prompt)
        self.assertIn("42 words", prompt)

    def test_chunk_prompt_contains_section_marker(self) -> None:
        prompt = build_chunk_summary_prompt("section text", max_words=60)
        self.assertIn("### Section:", prompt)
        self.assertIn("### Section Summary:", prompt)
        self.assertNotIn("research paper", prompt.lower())

    def test_merge_prompt_contains_notes_marker(self) -> None:
        prompt = build_merge_summary_prompt("Section 1 Summary: ...", max_words=80)
        self.assertIn("### Section Summaries:", prompt)
        self.assertIn("### Consolidated Notes:", prompt)
        self.assertNotIn("final abstract", prompt.lower())

    def test_arxiv_intro_prompt_targets_research_goal_and_approach(self) -> None:
        prompt = build_arxiv_intro_prompt("section text", max_words=70)
        self.assertIn("research objective", prompt)
        self.assertIn("main approach", prompt)
        self.assertIn("### Introduction Context:", prompt)

    def test_arxiv_findings_prompt_extracts_findings_instead_of_generic_summary(self) -> None:
        prompt = build_arxiv_findings_prompt("section text", max_words=55)
        self.assertIn("Extract concise research notes", prompt)
        self.assertIn("Do not summarize the whole section", prompt)
        self.assertIn("### Key Findings:", prompt)

    def test_arxiv_final_prompt_emphasizes_dedup_and_limited_math_notation(self) -> None:
        prompt = build_arxiv_final_abstract_prompt("notes", max_words=90)
        self.assertIn("Remove repetition", prompt)
        self.assertIn("avoid unnecessary mathematical notation", prompt)
        self.assertIn("### Research Notes:", prompt)

    def test_arxiv_paper_prompt_uses_document_marker(self) -> None:
        prompt = build_arxiv_paper_summary_prompt("paper text", max_words=90)
        self.assertIn("Summarize the following paper", prompt)
        self.assertIn("### Document:", prompt)

    def test_xsum_prompt_uses_news_style_instruction(self) -> None:
        prompt = build_xsum_prompt("news text", max_words=30)
        self.assertIn("single-sentence news summary", prompt)
        self.assertIn("### Article:", prompt)
        self.assertIn("### Summary:", prompt)
        self.assertIn("30 words", prompt)

    def test_xsum_chunk_and_merge_prompts_use_news_language(self) -> None:
        chunk_prompt = build_xsum_chunk_prompt("news section", max_words=40)
        merge_prompt = build_xsum_merge_prompt("Section 1 Summary: ...", max_words=60)
        self.assertIn("news article", chunk_prompt)
        self.assertIn("### News Notes:", chunk_prompt)
        self.assertIn("one-sentence news summary", merge_prompt)
        self.assertIn("### Final News Notes:", merge_prompt)

    def test_govreport_prompt_uses_executive_summary_style(self) -> None:
        prompt = build_govreport_prompt("report text", max_words=180)
        self.assertIn("executive summary", prompt.lower())
        self.assertIn("government report", prompt)
        self.assertIn("### Report:", prompt)
        self.assertIn("### Executive Summary:", prompt)

    def test_govreport_chunk_and_merge_prompts_use_report_language(self) -> None:
        chunk_prompt = build_govreport_chunk_prompt("report section", max_words=90)
        merge_prompt = build_govreport_merge_prompt("Section 1 Summary: ...", max_words=140)
        self.assertIn("government report", chunk_prompt)
        self.assertIn("### Report Section:", chunk_prompt)
        self.assertIn("executive summary", merge_prompt.lower())
        self.assertIn("### Executive Summary Notes:", merge_prompt)

    def test_general_summary_prompt_is_not_academic_specific(self) -> None:
        prompt = build_general_summary_prompt("doc text", max_words=80)
        self.assertIn("concise summary of the following document", prompt)
        self.assertIn("### Summary:", prompt)
        self.assertNotIn("academic abstract", prompt.lower())

    def test_strip_prompt_from_generation(self) -> None:
        prompt = build_summary_prompt("doc")
        generated = prompt + "short answer"
        self.assertEqual(strip_prompt_from_generation(generated, prompt), "short answer")


if __name__ == "__main__":
    unittest.main()
