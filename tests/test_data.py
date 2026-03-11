import unittest
from unittest import mock

from adapter.data import (
    _load_hf_dataset,
    _load_hf_parquet,
    _resolve_dataset_spec,
    filter_arxiv_records_by_tokens,
    load_normalized_dataset,
    is_arxiv_dataset_name,
    normalize_example,
)


class DataTests(unittest.TestCase):
    class _FakeTokenizer:
        def __call__(self, text, add_special_tokens=False, truncation=False):
            return {"input_ids": text.split()}

    def test_resolve_arxiv(self) -> None:
        key, spec = _resolve_dataset_spec("arxiv")
        self.assertEqual(key, "arxiv")
        self.assertEqual(spec.hf_id, "scientific_papers")
        self.assertEqual(spec.config_name, "arxiv")

    def test_is_arxiv_dataset_name(self) -> None:
        self.assertTrue(is_arxiv_dataset_name("arxiv"))
        self.assertTrue(is_arxiv_dataset_name("arxiv:default"))
        self.assertFalse(is_arxiv_dataset_name("xsum"))

    def test_normalize_arxiv_example(self) -> None:
        article = (
            "introduction [ sec : data ] this paper studies x - ray clusters with @xcite and @xmath0 "
            "markers removed . " * 40
        )
        abstract = (
            "we study x - ray clusters and show the cleaning step removes artifacts while preserving "
            "the main research contribution, the observational setup, the measurement pipeline, and the "
            "main conclusions of the paper."
        )
        normalized = normalize_example("arxiv", {"id": "a1", "article": article, "abstract": abstract})
        self.assertEqual(normalized["id"], "a1")
        self.assertNotIn("@xcite", normalized["source"])
        self.assertNotIn("@xmath0", normalized["source"])
        self.assertNotIn("[ sec : data ]", normalized["source"])
        self.assertIn("x-ray clusters", normalized["source"])
        self.assertIn("x-ray clusters", normalized["summary"])

    def test_normalize_arxiv_removes_broken_citation_and_inline_markup(self) -> None:
        article = (
            "background * ? ? ? * the _ rosat _ all - sky survey is described here . " * 40
        )
        abstract = (
            "the rosat all-sky survey provides the main observational context for this cleaned example, "
            "and the abstract remains long enough to resemble a realistic paper summary after cleanup."
        )
        normalized = normalize_example("arxiv", {"id": "a2", "article": article, "abstract": abstract})
        self.assertNotIn("* ? ? ?", normalized["source"])
        self.assertNotIn("_ rosat _", normalized["source"])
        self.assertIn("rosat all-sky survey", normalized["source"])

    def test_normalize_arxiv_filters_extremely_long_documents(self) -> None:
        article = "word " * 20050
        abstract = "this abstract is long enough to pass the minimum summary length requirement for testing."
        normalized = normalize_example("arxiv", {"id": "a3", "article": article, "abstract": abstract})
        self.assertTrue(normalized["source"].startswith("word"))

    def test_normalize_arxiv_filters_short_summary_after_cleaning(self) -> None:
        article = ("useful scientific content " * 250).strip()
        with self.assertRaises(ValueError):
            normalize_example("arxiv", {"id": "a4", "article": article, "abstract": "@xcite @xmath0"})

    def test_filter_arxiv_records_by_tokens(self) -> None:
        tokenizer = self._FakeTokenizer()
        records = [
            {"id": "1", "source": "a b c d", "summary": "x y"},
            {"id": "2", "source": "a b c d e f", "summary": "x y z"},
            {"id": "3", "source": "a b c d e", "summary": "x y z w v"},
            {"id": "4", "source": "a b @xmath0 d e", "summary": "x y z"},
        ]
        filtered = filter_arxiv_records_by_tokens(
            records,
            tokenizer,
            min_source_tokens=5,
            min_summary_tokens=3,
            max_source_tokens=5,
            max_summary_tokens=5,
            require_placeholder_free=True,
        )
        self.assertEqual([row["id"] for row in filtered], ["3"])

    def test_resolve_xlsum_default_english(self) -> None:
        key, spec = _resolve_dataset_spec("XLSum")
        self.assertEqual(key, "xlsum")
        self.assertEqual(spec.hf_id, "csebuetnlp/xlsum")
        self.assertEqual(spec.config_name, "english")

    def test_resolve_xlsum_language_override(self) -> None:
        key, spec = _resolve_dataset_spec("xlsum:vietnamese")
        self.assertEqual(key, "xlsum")
        self.assertEqual(spec.config_name, "vietnamese")

        key2, spec2 = _resolve_dataset_spec("xlsum/indonesian")
        self.assertEqual(key2, "xlsum")
        self.assertEqual(spec2.config_name, "indonesian")

    def test_normalize_xlsum_example(self) -> None:
        normalized = normalize_example("xlsum", {"id": "42", "text": "Doc", "summary": "Sum"})
        self.assertEqual(normalized["id"], "42")
        self.assertEqual(normalized["source"], "Doc")
        self.assertEqual(normalized["summary"], "Sum")

    def test_build_instruction_record_uses_xsum_prompt(self) -> None:
        from adapter.data import build_instruction_record

        record = build_instruction_record(
            {"id": "x1", "dataset": "xsum", "source": "News article body", "summary": "Short summary"},
            max_words=150,
        )
        self.assertIn("single-sentence news summary", record["prompt"])
        self.assertIn("### Article:", record["prompt"])
        self.assertIn("35 words", record["prompt"])

    def test_build_instruction_record_uses_govreport_prompt(self) -> None:
        from adapter.data import build_instruction_record

        record = build_instruction_record(
            {"id": "g1", "dataset": "govreport", "source": "Government report body", "summary": "Short summary"},
            max_words=150,
        )
        self.assertIn("executive summary", record["prompt"].lower())
        self.assertIn("### Report:", record["prompt"])
        self.assertIn("### Executive Summary:", record["prompt"])

    def test_load_hf_parquet_builds_expected_hf_paths(self) -> None:
        _, spec = _resolve_dataset_spec("xlsum:vietnamese")
        calls = {}

        def fake_loader(name, data_files=None, split=None):
            calls["name"] = name
            calls["data_files"] = data_files
            calls["split"] = split
            return "ok"

        out = _load_hf_parquet(fake_loader, spec, split="validation")
        self.assertEqual(out, "ok")
        self.assertEqual(calls["name"], "parquet")
        self.assertEqual(calls["split"], "validation")
        self.assertIn("hf://datasets/csebuetnlp/xlsum@refs/convert/parquet/vietnamese/train/", calls["data_files"]["train"])

    def test_load_hf_parquet_for_arxiv_uses_config_subpath(self) -> None:
        _, spec = _resolve_dataset_spec("arxiv")
        calls = {}

        def fake_loader(name, data_files=None, split=None):
            calls["name"] = name
            calls["data_files"] = data_files
            calls["split"] = split
            return "ok"

        out = _load_hf_parquet(fake_loader, spec, split="train")
        self.assertEqual(out, "ok")
        self.assertEqual(calls["name"], "parquet")
        self.assertEqual(calls["split"], "train")
        self.assertIn(
            "hf://datasets/scientific_papers@refs/convert/parquet/arxiv/partial-train/",
            calls["data_files"]["train"],
        )

    def test_load_hf_dataset_arxiv_enables_trust_remote_code(self) -> None:
        _, spec = _resolve_dataset_spec("arxiv")
        calls = {}

        def fake_loader(hf_id, config_name=None, **kwargs):
            calls["hf_id"] = hf_id
            calls["config_name"] = config_name
            calls["kwargs"] = kwargs
            return "ok"

        out = _load_hf_dataset(fake_loader, "arxiv", spec, split="train")
        self.assertEqual(out, "ok")
        self.assertEqual(calls["hf_id"], "scientific_papers")
        self.assertEqual(calls["config_name"], "arxiv")
        self.assertTrue(calls["kwargs"]["trust_remote_code"])
        self.assertEqual(calls["kwargs"]["split"], "train")

    def test_load_normalized_dataset_arxiv_prefers_parquet(self) -> None:
        class FakeDataset:
            def __init__(self, rows):
                self._rows = rows

            def select(self, indices):
                return FakeDataset([self._rows[idx] for idx in indices])

            def __len__(self):
                return len(self._rows)

            def __iter__(self):
                return iter(self._rows)

        rows = FakeDataset(
            [{"id": "a1", "article": "scientific content " * 300, "abstract": "summary text " * 30}]
        )

        with mock.patch("datasets.load_dataset") as fake_loader, mock.patch(
            "adapter.data._load_hf_parquet", return_value=rows
        ) as fake_parquet, mock.patch("adapter.data._load_hf_dataset") as fake_script:
            normalized = load_normalized_dataset("arxiv", split="train", max_samples=1)

        self.assertEqual(len(normalized), 1)
        fake_parquet.assert_called_once()
        fake_script.assert_not_called()

if __name__ == "__main__":
    unittest.main()
