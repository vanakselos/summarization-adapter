"""Inference entrypoint with hierarchical long-document summarization."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

from .config import AdapterConfig, load_config
from .prompts import (
    build_arxiv_conclusion_prompt,
    build_arxiv_final_abstract_prompt,
    build_arxiv_findings_prompt,
    build_arxiv_intro_prompt,
    build_arxiv_paper_summary_prompt,
    build_chunk_summary_prompt,
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
from .utils import chunk_text_by_words, load_jsonl, write_jsonl


_PROMPT_BATCH_SIZE = 2
_FINAL_WORD_BUDGET_RATIO = 0.75
_MIN_FINAL_WORDS = 32
_MIN_SECTION_WORDS = 40
_MAX_SECTION_WORDS = 120
_SECTION_WORD_DIVISOR = 2
_MIN_FINDINGS_WORDS = 30
_MAX_FINDINGS_WORDS = 90
_FINDINGS_WORD_DIVISOR = 3


def _import_inference_dependencies() -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "Missing inference dependencies. Install requirements first: pip install -r requirements.txt"
        ) from exc

    return {
        "torch": torch,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
    }


def estimate_token_count(text: str) -> int:
    """Approximate token count for routing. Uses words as a lightweight proxy."""
    return len(text.split())


def _format_section_summaries(partial_summaries: list[str]) -> str:
    blocks: list[str] = []
    for idx, summary in enumerate(partial_summaries, start=1):
        blocks.append(f"Section {idx} Summary:\n{summary.strip()}")
    return "\n\n".join(blocks)


def _format_arxiv_research_notes(
    intro_summary: str,
    key_findings: list[str],
    conclusion_summary: str,
) -> str:
    blocks = [f"Introduction Context:\n{intro_summary.strip()}"]
    findings_block = "\n".join(
        f"- {finding.strip()}" for finding in key_findings if finding.strip()
    ) or "- None"
    blocks.append(f"Key Findings:\n{findings_block}")
    blocks.append(f"Conclusion:\n{conclusion_summary.strip()}")
    return "\n\n".join(blocks)


def _batched(items: list[str], batch_size: int) -> list[list[str]]:
    return [items[idx : idx + batch_size] for idx in range(0, len(items), batch_size)]


def _derive_prompt_word_budgets(max_new_tokens: int) -> tuple[int, int, int]:
    """Map generation length to prompt word budgets for final, section, and findings stages."""
    max_words = max(_MIN_FINAL_WORDS, int(max_new_tokens * _FINAL_WORD_BUDGET_RATIO))
    section_max_words = max(
        _MIN_SECTION_WORDS,
        min(_MAX_SECTION_WORDS, max_words // _SECTION_WORD_DIVISOR),
    )
    findings_max_words = max(
        _MIN_FINDINGS_WORDS,
        min(_MAX_FINDINGS_WORDS, max_words // _FINDINGS_WORD_DIVISOR),
    )
    return max_words, section_max_words, findings_max_words


def _canonical_dataset_name(dataset_name: str) -> str:
    normalized = dataset_name.strip().lower()
    if normalized == "ccdv/govreport-summarization":
        return "govreport"
    return normalized


def _resolve_inference_dataset(cfg: AdapterConfig) -> str | None:
    dataset_names = {
        _canonical_dataset_name(name)
        for name in [*cfg.data.train_datasets, *cfg.data.eval_datasets]
        if isinstance(name, str) and name.strip()
    }
    if len(dataset_names) != 1:
        return None
    return next(iter(dataset_names))


def _build_general_prompt_functions(
    dataset_name: str | None,
    max_words: int,
) -> tuple[Callable[[str], str], Callable[[str], str], Callable[[str], str]]:
    if dataset_name == "xsum":
        return (
            lambda text: build_xsum_chunk_prompt(text, max_words=min(max_words, 60)),
            lambda text: build_xsum_merge_prompt(text, max_words=min(max_words, 80)),
            lambda text: build_xsum_prompt(text, max_words=min(max_words, 35)),
        )
    if dataset_name == "govreport":
        return (
            lambda text: build_govreport_chunk_prompt(text, max_words=max(max_words, 120)),
            lambda text: build_govreport_merge_prompt(text, max_words=max(max_words, 180)),
            lambda text: build_govreport_prompt(text, max_words=max(max_words, 180)),
        )
    return (
        lambda text: build_chunk_summary_prompt(text, max_words=max_words),
        lambda text: build_merge_summary_prompt(text, max_words=max_words),
        lambda text: build_summary_prompt(text, max_words=max_words),
    )


def _print_doc_progress(index: int, total: int, row_id: str, strategy_name: str) -> None:
    print(f"[{index}/{total}] Inference id={row_id} strategy={strategy_name}")


def hierarchical_summarize(
    document: str,
    chunk_tokens: int,
    chunk_overlap: int,
    summarize_fn: Callable[[str], str] | None = None,
    summarize_chunk_fn: Callable[[str], str] | None = None,
    summarize_chunks_fn: Callable[[list[str]], list[str]] | None = None,
    summarize_merge_fn: Callable[[str], str] | None = None,
    summarize_final_fn: Callable[[str], str] | None = None,
    strategy_name: str = "hierarchical",
) -> dict[str, Any]:
    if summarize_fn is not None:
        summarize_chunk_fn = summarize_chunk_fn or summarize_fn
        summarize_merge_fn = summarize_merge_fn or summarize_fn
        summarize_final_fn = summarize_final_fn or summarize_fn

    if summarize_merge_fn is None or summarize_final_fn is None:
        raise ValueError(
            "Provide summarize_fn or summarize_merge_fn and summarize_final_fn."
        )
    if summarize_chunk_fn is None and summarize_chunks_fn is None:
        raise ValueError("Provide summarize_fn, summarize_chunk_fn, or summarize_chunks_fn.")

    token_count = estimate_token_count(document)

    if token_count <= chunk_tokens:
        final_summary = summarize_final_fn(document)
        return {
            "chunks": [document],
            "partial_summaries": [],
            "merged_summary": "",
            "final_summary": final_summary,
            "stats": {
                "input_tokens": token_count,
                "num_chunks": 1,
                "hierarchical": False,
                "inference_strategy": strategy_name,
            },
        }

    chunks = chunk_text_by_words(document, chunk_size=chunk_tokens, overlap=chunk_overlap)
    if summarize_chunks_fn is not None:
        partial_summaries = summarize_chunks_fn(chunks)
    else:
        partial_summaries = [summarize_chunk_fn(chunk) for chunk in chunks]
    merged_input = _format_section_summaries(partial_summaries)
    merged_summary = summarize_merge_fn(merged_input)
    final_summary = summarize_final_fn(merged_summary)

    return {
        "chunks": chunks,
        "partial_summaries": partial_summaries,
        "merged_summary": merged_summary,
        "final_summary": final_summary,
        "stats": {
            "input_tokens": token_count,
            "num_chunks": len(chunks),
            "hierarchical": True,
            "inference_strategy": strategy_name,
        },
    }


def arxiv_lead_refine_summarize(
    document: str,
    chunk_tokens: int,
    chunk_overlap: int,
    summarize_intro_fn: Callable[[str], str],
    summarize_findings_fn: Callable[[str], str],
    summarize_conclusion_fn: Callable[[str], str],
    summarize_final_fn: Callable[[str], str],
    summarize_document_fn: Callable[[str], str] | None = None,
    summarize_findings_batch_fn: Callable[[list[str]], list[str]] | None = None,
) -> dict[str, Any]:
    if summarize_document_fn is None:
        summarize_document_fn = summarize_final_fn

    token_count = estimate_token_count(document)

    if token_count <= chunk_tokens:
        final_summary = summarize_document_fn(document)
        return {
            "chunks": [document],
            "partial_summaries": [],
            "merged_summary": "",
            "final_summary": final_summary,
            "stats": {
                "input_tokens": token_count,
                "num_chunks": 1,
                "hierarchical": False,
                "inference_strategy": "arxiv_lead_refine",
            },
        }

    chunks = chunk_text_by_words(document, chunk_size=chunk_tokens, overlap=chunk_overlap)
    intro_summary = summarize_intro_fn(chunks[0])
    middle_chunks = chunks[1:-1]
    if summarize_findings_batch_fn is not None:
        middle_findings = summarize_findings_batch_fn(middle_chunks)
    else:
        middle_findings = [summarize_findings_fn(chunk) for chunk in middle_chunks]
    conclusion_summary = summarize_conclusion_fn(chunks[-1])

    partial_summaries = [intro_summary, *middle_findings, conclusion_summary]
    merged_summary = _format_arxiv_research_notes(
        intro_summary=intro_summary,
        key_findings=middle_findings,
        conclusion_summary=conclusion_summary,
    )
    final_summary = summarize_final_fn(merged_summary)

    return {
        "chunks": chunks,
        "partial_summaries": partial_summaries,
        "merged_summary": merged_summary,
        "final_summary": final_summary,
        "stats": {
            "input_tokens": token_count,
            "num_chunks": len(chunks),
            "hierarchical": True,
            "inference_strategy": "arxiv_lead_refine",
        },
    }


def _build_model_prompt_generators(
    model: Any,
    tokenizer: Any,
    cfg: AdapterConfig,
) -> tuple[Callable[[str], str], Callable[[list[str]], list[str]]]:
    prompt_max_length = max(128, cfg.model.context_window_tokens - cfg.inference.max_new_tokens)
    tokenizer_model_max_length = getattr(tokenizer, "model_max_length", None)
    if (
        isinstance(tokenizer_model_max_length, int)
        and tokenizer_model_max_length > 0
        and tokenizer_model_max_length < 1_000_000
    ):
        prompt_max_length = min(prompt_max_length, tokenizer_model_max_length)

    def _encode_prompts(prompts: list[str]) -> dict[str, Any]:
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=prompt_max_length,
        )
        model_device = getattr(model, "device", None)
        if model_device is not None:
            encoded = {k: v.to(model_device) for k, v in encoded.items()}
        return encoded

    def _generate(encoded: dict[str, Any]) -> Any:
        import torch

        do_sample = cfg.inference.temperature > 0
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": cfg.inference.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = cfg.inference.temperature
            generate_kwargs["top_p"] = cfg.inference.top_p

        try:
            with torch.inference_mode():
                generated = model.generate(**encoded, **generate_kwargs)
        except RuntimeError as exc:
            msg = str(exc)
            if "CUDA error" not in msg:
                raise

            # Retry once with greedy decoding, which is typically more stable than sampling.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            fallback_kwargs = {
                "max_new_tokens": cfg.inference.max_new_tokens,
                "do_sample": False,
                "pad_token_id": tokenizer.eos_token_id,
            }
            try:
                with torch.inference_mode():
                    generated = model.generate(**encoded, **fallback_kwargs)
            except RuntimeError as retry_exc:
                raise RuntimeError(
                    "Inference failed on CUDA during generation. "
                    "Try runtime.device='cuda:0', runtime.bfloat16=false, and inference.temperature=0.0. "
                    "For debugging, rerun with CUDA_LAUNCH_BLOCKING=1."
                ) from retry_exc
        return generated

    def _generate_from_prompt(prompt: str) -> str:
        generated = _generate(_encode_prompts([prompt]))
        output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        return strip_prompt_from_generation(output_text, prompt)

    def _generate_from_prompts(prompts: list[str]) -> list[str]:
        if not prompts:
            return []
        generated = _generate(_encode_prompts(prompts))
        output_texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
        return [
            strip_prompt_from_generation(output_text, prompt)
            for prompt, output_text in zip(prompts, output_texts)
        ]

    return _generate_from_prompt, _generate_from_prompts


def load_model_and_tokenizer(cfg: AdapterConfig, adapter_path: str | None = None) -> tuple[Any, Any]:
    deps = _import_inference_dependencies()
    torch = deps["torch"]
    use_cuda = torch.cuda.is_available()

    tokenizer = deps["AutoTokenizer"].from_pretrained(cfg.model.base_model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    dtype = torch.bfloat16 if cfg.runtime.bfloat16 else torch.float16
    use_4bit = cfg.runtime.load_in_4bit and use_cuda
    if use_4bit:
        quantization_config = deps["BitsAndBytesConfig"](
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

    # Keep adapter loading off the accelerate offload path because PEFT can fail
    # with KeyError on some transformer/accelerate combinations.
    model_device_map: Any = None if adapter_path else cfg.runtime.device

    model = deps["AutoModelForCausalLM"].from_pretrained(
        cfg.model.base_model_id,
        torch_dtype=dtype,
        quantization_config=quantization_config,
        device_map=model_device_map,
        low_cpu_mem_usage=not adapter_path,
    )

    if adapter_path:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("Install 'peft' to load adapter weights.") from exc

        if use_cuda and not use_4bit:
            model = model.to("cuda:0")

        peft_kwargs: dict[str, Any] = {"low_cpu_mem_usage": False}

        try:
            model = PeftModel.from_pretrained(model, adapter_path, **peft_kwargs)
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                "Adapter load failed due to a PEFT/accelerate offload compatibility issue. "
                "Run with runtime.load_in_4bit=false and runtime.device='cuda:0' for inference."
            ) from exc
        except Exception as exc:
            message = str(exc)
            if "get_balanced_memory" in message or "unhashable type" in message or "offload" in message:
                model = PeftModel.from_pretrained(model, adapter_path, low_cpu_mem_usage=False)
            else:
                raise

    model.eval()
    return model, tokenizer


def _read_input(path: Path) -> list[dict[str, str]]:
    if path.suffix.lower() == ".jsonl":
        rows = load_jsonl(path)
        normalized: list[dict[str, str]] = []
        for i, row in enumerate(rows):
            doc = row.get("document")
            if isinstance(doc, str) and doc.strip():
                normalized.append({"id": str(row.get("id", i)), "document": doc.strip()})
        return normalized

    text = path.read_text(encoding="utf-8")
    return [{"id": path.stem, "document": text.strip()}]


def run_general_inference(
    cfg: AdapterConfig,
    input_path: str,
    output_path: str,
    adapter_path: str | None = None,
) -> None:
    model, tokenizer = load_model_and_tokenizer(cfg, adapter_path=adapter_path)
    generate_from_prompt, generate_from_prompts = _build_model_prompt_generators(
        model,
        tokenizer,
        cfg,
    )
    max_words, _, _ = _derive_prompt_word_budgets(cfg.inference.max_new_tokens)
    dataset_name = _resolve_inference_dataset(cfg)
    build_chunk_prompt, build_merge_prompt, build_final_prompt = _build_general_prompt_functions(
        dataset_name,
        max_words,
    )

    def summarize_chunk(section_text: str) -> str:
        prompt = build_chunk_prompt(section_text)
        return generate_from_prompt(prompt)

    def summarize_chunks(section_texts: list[str]) -> list[str]:
        outputs: list[str] = []
        for batch in _batched(section_texts, cfg.inference.prompt_batch_size):
            prompts = [build_chunk_prompt(text) for text in batch]
            outputs.extend(generate_from_prompts(prompts))
        return outputs

    def summarize_merge(section_summaries: str) -> str:
        prompt = build_merge_prompt(section_summaries)
        return generate_from_prompt(prompt)

    def summarize_final(text: str) -> str:
        prompt = build_final_prompt(text)
        return generate_from_prompt(prompt)

    docs = _read_input(Path(input_path))
    outputs: list[dict[str, Any]] = []

    for idx, row in enumerate(docs, start=1):
        _print_doc_progress(idx, len(docs), row["id"], cfg.inference.strategy)
        result = hierarchical_summarize(
            row["document"],
            chunk_tokens=cfg.inference.chunk_tokens,
            chunk_overlap=cfg.inference.chunk_overlap,
            summarize_chunk_fn=summarize_chunk,
            summarize_chunks_fn=summarize_chunks,
            summarize_merge_fn=summarize_merge,
            summarize_final_fn=summarize_final,
            strategy_name=cfg.inference.strategy,
        )
        outputs.append(
            {
                "id": row["id"],
                "document": row["document"],
                "chunks": result["chunks"],
                "partial_summaries": result["partial_summaries"],
                "merged_summary": result["merged_summary"],
                "summary": result["final_summary"],
                "stats": result["stats"],
            }
        )

    write_jsonl(output_path, outputs)


def run_arxiv_lead_refine_inference(
    cfg: AdapterConfig,
    input_path: str,
    output_path: str,
    adapter_path: str | None = None,
) -> None:
    model, tokenizer = load_model_and_tokenizer(cfg, adapter_path=adapter_path)
    generate_from_prompt, generate_from_prompts = _build_model_prompt_generators(
        model,
        tokenizer,
        cfg,
    )
    max_words, section_max_words, findings_max_words = _derive_prompt_word_budgets(
        cfg.inference.max_new_tokens
    )

    def summarize_arxiv_intro(section_text: str) -> str:
        prompt = build_arxiv_intro_prompt(section_text, max_words=section_max_words)
        return generate_from_prompt(prompt)

    def summarize_arxiv_findings(section_text: str) -> str:
        prompt = build_arxiv_findings_prompt(section_text, max_words=findings_max_words)
        return generate_from_prompt(prompt)

    def summarize_arxiv_findings_batch(section_texts: list[str]) -> list[str]:
        outputs: list[str] = []
        for batch in _batched(section_texts, _PROMPT_BATCH_SIZE):
            prompts = [build_arxiv_findings_prompt(text, max_words=findings_max_words) for text in batch]
            outputs.extend(generate_from_prompts(prompts))
        return outputs

    def summarize_arxiv_conclusion(section_text: str) -> str:
        prompt = build_arxiv_conclusion_prompt(section_text, max_words=section_max_words)
        return generate_from_prompt(prompt)

    def summarize_arxiv_final(notes: str) -> str:
        prompt = build_arxiv_final_abstract_prompt(notes, max_words=max_words)
        return generate_from_prompt(prompt)

    def summarize_arxiv_document(document: str) -> str:
        prompt = build_arxiv_paper_summary_prompt(document, max_words=max_words)
        return generate_from_prompt(prompt)
    
    docs = _read_input(Path(input_path))
    outputs: list[dict[str, Any]] = []

    for idx, row in enumerate(docs, start=1):
        _print_doc_progress(idx, len(docs), row["id"], cfg.inference.strategy)
        result = arxiv_lead_refine_summarize(
            row["document"],
            chunk_tokens=cfg.inference.chunk_tokens,
            chunk_overlap=cfg.inference.chunk_overlap,
            summarize_intro_fn=summarize_arxiv_intro,
            summarize_findings_fn=summarize_arxiv_findings,
            summarize_conclusion_fn=summarize_arxiv_conclusion,
            summarize_final_fn=summarize_arxiv_final,
            summarize_document_fn=summarize_arxiv_document,
            summarize_findings_batch_fn=summarize_arxiv_findings_batch,
        )
        outputs.append(
            {
                "id": row["id"],
                "document": row["document"],
                "chunks": result["chunks"],
                "partial_summaries": result["partial_summaries"],
                "merged_summary": result["merged_summary"],
                "summary": result["final_summary"],
                "stats": result["stats"],
            }
        )
    write_jsonl(output_path, outputs)


def run_inference(
    cfg: AdapterConfig,
    input_path: str,
    output_path: str,
    adapter_path: str | None = None,
) -> None:
    
    if cfg.inference.strategy == "arxiv_lead_refine":
        run_arxiv_lead_refine_inference(cfg, input_path, output_path, adapter_path=adapter_path)
    else:
        run_general_inference(cfg, input_path, output_path, adapter_path=adapter_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run summarization inference")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--input_file", required=True, help="Path to text or jsonl input")
    parser.add_argument("--output_file", required=True, help="Path to output jsonl")
    parser.add_argument("--adapter_path", default=None, help="Optional LoRA adapter path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_inference(cfg, args.input_file, args.output_file, adapter_path=args.adapter_path)
    print(f"Inference complete. Output saved to: {args.output_file}")


if __name__ == "__main__":
    main()
