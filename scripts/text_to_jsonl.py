"""Convert plain text files into inference-ready JSONL rows."""

from __future__ import annotations

import argparse
from pathlib import Path

from adapter.utils import write_jsonl


def _read_text_document(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Input file is empty after stripping whitespace: {path}")
    return {"id": path.stem, "document": text}


def load_text_documents(input_path: str | Path) -> list[dict[str, str]]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    if path.is_file():
        return [_read_text_document(path)]

    text_files = sorted(child for child in path.iterdir() if child.is_file() and child.suffix.lower() == ".txt")
    if not text_files:
        raise ValueError(f"No .txt files found in directory: {path}")
    return [_read_text_document(file_path) for file_path in text_files]


def convert_text_path_to_jsonl(input_path: str | Path, output_path: str | Path) -> int:
    rows = load_text_documents(input_path)
    write_jsonl(output_path, rows)
    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert plain text input into JSONL rows")
    parser.add_argument("--input_path", required=True, help="Path to a .txt file or a directory of .txt files")
    parser.add_argument("--output_path", required=True, help="Path to output .jsonl file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = convert_text_path_to_jsonl(args.input_path, args.output_path)
    print(f"Wrote {count} document(s) to: {args.output_path}")


if __name__ == "__main__":
    main()
