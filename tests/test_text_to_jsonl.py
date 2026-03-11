import importlib.util
import tempfile
import unittest
from pathlib import Path

from adapter.utils import load_jsonl


_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "text_to_jsonl.py"
_SPEC = importlib.util.spec_from_file_location("text_to_jsonl_script", _SCRIPT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Could not load script module from {_SCRIPT_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class TextToJsonlTests(unittest.TestCase):
    def test_convert_single_text_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "sample.txt"
            output_path = root / "sample.jsonl"
            input_path.write_text("  Hello world.\nSecond line.  ", encoding="utf-8")

            count = _MODULE.convert_text_path_to_jsonl(input_path, output_path)

            self.assertEqual(count, 1)
            rows = load_jsonl(output_path)
            self.assertEqual(rows, [{"id": "sample", "document": "Hello world.\nSecond line."}])

    def test_convert_directory_of_text_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "docs"
            output_path = root / "batch.jsonl"
            input_dir.mkdir()
            (input_dir / "b.txt").write_text("Document B", encoding="utf-8")
            (input_dir / "a.txt").write_text("Document A", encoding="utf-8")
            (input_dir / "ignore.md").write_text("Ignore me", encoding="utf-8")

            count = _MODULE.convert_text_path_to_jsonl(input_dir, output_path)

            self.assertEqual(count, 2)
            rows = load_jsonl(output_path)
            self.assertEqual(
                rows,
                [
                    {"id": "a", "document": "Document A"},
                    {"id": "b", "document": "Document B"},
                ],
            )


if __name__ == "__main__":
    unittest.main()
