"""Dataset loading and normalization utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .prompts import build_govreport_prompt, build_summary_prompt, build_xsum_prompt


@dataclass(frozen=True)
class DatasetSpec:
    hf_id: str
    config_name: str | None


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "cnn_dailymail": DatasetSpec("cnn_dailymail", "3.0.0"),
    "xsum": DatasetSpec("xsum", None),
    "arxiv": DatasetSpec("scientific_papers", "arxiv"),
    "govreport": DatasetSpec("ccdv/govreport-summarization", None),
    "xlsum": DatasetSpec("csebuetnlp/xlsum", "english"),
}

_ARXIV_PLACEHOLDER_RE = re.compile(r"@xcite\b|@xmath\d+\b", re.IGNORECASE)
_ARXIV_SECTION_REF_RE = re.compile(r"\[\s*sec\s*:[^\]]+\]", re.IGNORECASE)
_ARXIV_BROKEN_CITATION_RE = re.compile(r"\*\s*(?:\?\s*){2,}\*?", re.IGNORECASE)
_ARXIV_INLINE_MARKUP_RE = re.compile(r"_\s*([^_\n]+?)\s*_")
_ARXIV_BACK_MATTER_RE = re.compile(
    r"(?:^|\n)\s*(acknowledg(?:e)?ments?|references|bibliography|appendix)\b",
    re.IGNORECASE,
)
_ARXIV_MIN_SOURCE_TOKENS = 900
_ARXIV_MIN_SUMMARY_TOKENS = 50
_ARXIV_MAX_SOURCE_TOKENS = 8192
_ARXIV_MAX_SUMMARY_TOKENS = 1024
_ARXIV_REQUIRE_PLACEHOLDER_FREE = False


def _resolve_dataset_spec(dataset_name: str) -> tuple[str, DatasetSpec]:
    raw = dataset_name.strip()
    lowered = raw.lower()

    key = lowered
    config_override: str | None = None
    if ":" in lowered:
        key, config_override = [part.strip() for part in lowered.split(":", 1)]
    elif lowered.startswith("xlsum/"):
        key, config_override = "xlsum", lowered.split("/", 1)[1].strip()

    if key not in DATASET_REGISTRY:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    base = DATASET_REGISTRY[key]
    if config_override:
        return key, DatasetSpec(base.hf_id, config_override)
    return key, base


def _load_hf_parquet(load_dataset_fn: Any, spec: DatasetSpec, split: str) -> Any:
    base = f"hf://datasets/{spec.hf_id}@refs/convert/parquet"
    if spec.config_name:
        base = f"{base}/{spec.config_name}"
    split_names = {
        "train": "train",
        "validation": "validation",
        "test": "test",
    }
    if spec.hf_id == "scientific_papers" and spec.config_name == "arxiv":
        split_names = {
            "train": "partial-train",
            "validation": "partial-validation",
            "test": "partial-test",
        }
    data_files = {
        "train": f"{base}/{split_names['train']}/*.parquet",
        "validation": f"{base}/{split_names['validation']}/*.parquet",
        "test": f"{base}/{split_names['test']}/*.parquet",
    }
    return load_dataset_fn("parquet", data_files=data_files, split=split)


def _load_hf_dataset(load_dataset_fn: Any, key: str, spec: DatasetSpec, split: str) -> Any:
    kwargs: dict[str, Any] = {"split": split}
    # scientific_papers requires custom dataset code on many Kaggle images.
    if key == "arxiv":
        kwargs["trust_remote_code"] = True

    if spec.config_name:
        return load_dataset_fn(spec.hf_id, spec.config_name, **kwargs)
    return load_dataset_fn(spec.hf_id, **kwargs)


def _extract_text(example: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        val = example.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def is_arxiv_dataset_name(dataset_name: str) -> bool:
    return dataset_name.strip().lower().split(":", 1)[0] == "arxiv"


def _clean_arxiv_text(text: str) -> str:
    cleaned = text
    cleaned = _ARXIV_BACK_MATTER_RE.split(cleaned, maxsplit=1)[0]
    cleaned = _ARXIV_PLACEHOLDER_RE.sub(" ", cleaned)
    cleaned = _ARXIV_SECTION_REF_RE.sub(" ", cleaned)
    cleaned = _ARXIV_BROKEN_CITATION_RE.sub(" ", cleaned)
    cleaned = _ARXIV_INLINE_MARKUP_RE.sub(lambda match: match.group(1).strip(), cleaned)
    cleaned = re.sub(r"[ \t]*\n[ \t]*", "\n", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"([(\[])\s+", r"\1", cleaned)
    cleaned = re.sub(r"\s+([)\]])", r"\1", cleaned)
    cleaned = re.sub(r"\s*-\s*", "-", cleaned)
    cleaned = re.sub(r"\s*/\s*", "/", cleaned)
    cleaned = _normalize_whitespace(cleaned)
    return cleaned


def _count_tokens(text: str, tokenizer: Any) -> int:
    encoded = tokenizer(text, add_special_tokens=False, truncation=False)
    input_ids = encoded.get("input_ids", [])
    return len(input_ids)


def filter_arxiv_records_by_tokens(
    records: list[dict[str, str]],
    tokenizer: Any,
    min_source_tokens: int = _ARXIV_MIN_SOURCE_TOKENS,
    min_summary_tokens: int = _ARXIV_MIN_SUMMARY_TOKENS,
    max_source_tokens: int = _ARXIV_MAX_SOURCE_TOKENS,
    max_summary_tokens: int = _ARXIV_MAX_SUMMARY_TOKENS,
    require_placeholder_free: bool = _ARXIV_REQUIRE_PLACEHOLDER_FREE,
) -> list[dict[str, str]]:
    filtered: list[dict[str, str]] = []
    for row in records:
        source = str(row.get("source", "")).strip()
        summary = str(row.get("summary", "")).strip()
        if not source or not summary:
            continue

        if require_placeholder_free and (
            _ARXIV_PLACEHOLDER_RE.search(source) or _ARXIV_PLACEHOLDER_RE.search(summary)
        ):
            continue

        source_tokens = _count_tokens(source, tokenizer)
        summary_tokens = _count_tokens(summary, tokenizer)
        if min_source_tokens is not None and source_tokens < min_source_tokens:
            continue
        if min_summary_tokens is not None and summary_tokens < min_summary_tokens:
            continue
        if max_source_tokens is not None and source_tokens > max_source_tokens:
            continue
        if max_summary_tokens is not None and summary_tokens > max_summary_tokens:
            continue
        filtered.append(row)
    return filtered


def normalize_example(dataset_name: str, example: dict[str, Any]) -> dict[str, str]:
    dataset_name = dataset_name.lower()
    if dataset_name == "cnn_dailymail":
        source = _extract_text(example, ["article", "document", "source"])
        summary = _extract_text(example, ["highlights", "summary"])
    elif dataset_name == "xsum":
        source = _extract_text(example, ["document", "article", "source"])
        summary = _extract_text(example, ["summary", "highlights"])
    elif dataset_name == "arxiv":
        source = _extract_text(example, ["article", "document", "source"])
        summary = _extract_text(example, ["abstract", "summary", "highlights"])
    elif dataset_name == "xlsum":
        source = _extract_text(example, ["text", "document", "article", "source"])
        summary = _extract_text(example, ["summary", "highlights"])
    elif dataset_name == "govreport":
        source = _extract_text(example, ["report", "document", "source"])
        summary = _extract_text(example, ["summary", "highlights"])
    else:
        source = _extract_text(example, ["document", "article", "report", "source"])
        summary = _extract_text(example, ["summary", "highlights", "target"])

    if not source or not summary:
        raise ValueError(f"Failed to normalize record for dataset={dataset_name}")

    if is_arxiv_dataset_name(dataset_name):
        source = _clean_arxiv_text(source)
        summary = _clean_arxiv_text(summary)
        if not source or not summary:
            raise ValueError("Failed to normalize cleaned ArXiv record.")

    sample_id = str(example.get("id", example.get("guid", "")))
    return {
        "id": sample_id,
        "dataset": dataset_name,
        "source": source,
        "summary": summary,
    }


def load_normalized_dataset(
    dataset_name: str,
    split: str,
    max_samples: int | None = None,
) -> list[dict[str, str]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install 'datasets' to load HF corpora.") from exc

    key, spec = _resolve_dataset_spec(dataset_name)
    if key == "arxiv":
        try:
            ds = _load_hf_parquet(load_dataset, spec, split)
        except Exception as parquet_exc:
            raise RuntimeError(
                f"Failed to load dataset '{dataset_name}' from HF parquet shards. "
                "Ensure internet access to Hugging Face parquet mirrors."
            ) from parquet_exc
    else:
        try:
            ds = _load_hf_dataset(load_dataset, key, spec, split)
        except (RuntimeError, ValueError) as exc:
            # Datasets>=4 removes support for python dataset scripts. XL-Sum still
            # depends on that path, so fall back to HF parquet conversion.
            message = str(exc)
            if key == "xlsum" and (
                "Dataset scripts are no longer supported" in message
                or "trust_remote_code=True" in message
                or "contains custom code" in message
            ):
                try:
                    ds = _load_hf_parquet(load_dataset, spec, split)
                except Exception as parquet_exc:
                    raise RuntimeError(
                        f"Failed to load dataset '{dataset_name}' with both HF script and parquet fallback. "
                        "On Kaggle/Colab, try `pip install -U datasets` and ensure internet access to HF parquet mirrors."
                    ) from parquet_exc
            else:
                raise

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    normalized: list[dict[str, str]] = []
    for row in ds:
        try:
            normalized.append(normalize_example(key, row))
        except ValueError:
            continue
    return normalized


def build_instruction_record(example: dict[str, str], max_words: int = 150) -> dict[str, str]:
    dataset_name = str(example.get("dataset", "")).strip().lower()
    if dataset_name == "xsum":
        prompt = build_xsum_prompt(example["source"], max_words=min(max_words, 35))
    elif dataset_name == "govreport":
        prompt = build_govreport_prompt(example["source"], max_words=max(max_words, 180))
    else:
        prompt = build_summary_prompt(example["source"], max_words=max_words)
    return {
        "id": example["id"],
        "dataset": example["dataset"],
        "source": example["source"],
        "summary": example["summary"],
        "prompt": prompt,
    }
