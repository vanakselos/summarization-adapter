"""LoRA fine-tuning entrypoint for Qwen-based summarization."""

from __future__ import annotations

import argparse
import inspect
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import AdapterConfig, load_config
from .data import (
    build_instruction_record,
    filter_arxiv_records_by_tokens,
    is_arxiv_dataset_name,
    load_normalized_dataset,
)
from .utils import ensure_dir, set_seed, write_jsonl


def _import_training_dependencies() -> dict[str, Any]:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )
    return {
        "torch": torch,
        "Dataset": Dataset,
        "LoraConfig": LoraConfig,
        "get_peft_model": get_peft_model,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "DataCollatorForSeq2Seq": DataCollatorForSeq2Seq,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
    }


def _tokenize_example(example: dict[str, str], tokenizer: Any, cfg: AdapterConfig) -> dict[str, Any]:
    # Truncate at tokenization time to avoid massive intermediate sequences.
    prompt_ids = tokenizer(
        example["prompt"],
        add_special_tokens=False,
        truncation=True,
        max_length=cfg.data.max_source_tokens,
    )["input_ids"]
    target_ids = tokenizer(
        example["summary"],
        add_special_tokens=False,
        truncation=True,
        max_length=cfg.data.max_target_tokens,
    )["input_ids"]

    eos_id = tokenizer.eos_token_id
    full_ids = prompt_ids + target_ids + ([eos_id] if eos_id is not None else [])
    labels = ([-100] * len(prompt_ids)) + target_ids + ([eos_id] if eos_id is not None else [])
    max_len = cfg.data.max_source_tokens + cfg.data.max_target_tokens
    full_ids = full_ids[:max_len]
    labels = labels[:max_len]
    attn = [1] * len(full_ids)

    return {
        "input_ids": full_ids,
        "attention_mask": attn,
        "labels": labels,
    }


def _load_and_prepare_records(
    cfg: AdapterConfig,
    tokenizer: Any,
    split: str,
    max_samples: int | None,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    datasets = cfg.data.train_datasets if split == cfg.data.train_split else cfg.data.eval_datasets

    for name in datasets:
        subset = load_normalized_dataset(name, split=split, max_samples=max_samples)
        if is_arxiv_dataset_name(name):
            subset = filter_arxiv_records_by_tokens(subset, tokenizer)
        records.extend(build_instruction_record(item) for item in subset)
    return records


def _write_eval_artifacts(
    run_dir: Path,
    cfg: AdapterConfig,
    eval_records: list[dict[str, str]],
) -> dict[str, Any]:
    """Persist fixed eval input/reference files for reproducible inference/evaluation."""
    seen_ids: set[str] = set()
    input_rows: list[dict[str, str]] = []
    ref_rows: list[dict[str, str]] = []

    for idx, row in enumerate(eval_records):
        dataset = str(row.get("dataset", "unknown")).strip() or "unknown"
        raw_id = str(row.get("id", "")).strip()
        row_id = raw_id if raw_id else f"{dataset}-{idx}"
        if row_id in seen_ids:
            row_id = f"{row_id}-{idx}"
        seen_ids.add(row_id)

        document = str(row.get("source", "")).strip()
        summary = str(row.get("summary", "")).strip()
        if not document or not summary:
            continue

        input_rows.append({"id": row_id, "document": document})
        ref_rows.append(
            {
                "id": row_id,
                "document": document,
                "summary": summary,
            }
        )

    input_path = run_dir / "input_eval.jsonl"
    refs_path = run_dir / "refs_eval.jsonl"
    manifest_path = run_dir / "eval_manifest.json"

    write_jsonl(input_path, input_rows)
    write_jsonl(refs_path, ref_rows)

    manifest = {
        "eval_datasets": cfg.data.eval_datasets,
        "eval_split": cfg.data.eval_split,
        "max_eval_samples_per_dataset": cfg.data.max_eval_samples_per_dataset,
        "seed": cfg.train.seed,
        "num_examples": len(input_rows),
        "input_file": input_path.name,
        "refs_file": refs_path.name,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    return {
        "input_file": input_path.name,
        "refs_file": refs_path.name,
        "manifest_file": manifest_path.name,
        "num_examples": len(input_rows),
    }


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise RuntimeError(f"Missing required {label}: {path}")


def _copy_config_snapshot(config_path: str | None, run_dir: Path) -> str | None:
    if not config_path:
        return None

    src = Path(config_path)
    if not src.exists():
        raise ValueError(f"Config path does not exist: {src}")

    dst = run_dir / "config_used.yaml"
    shutil.copy2(src, dst)
    return dst.name


def _run_post_train_generation(
    cfg: AdapterConfig,
    config_path: str,
    run_dir: Path,
    eval_artifacts: dict[str, Any],
) -> dict[str, Any]:
    input_path = run_dir / str(eval_artifacts["input_file"])
    refs_path = run_dir / str(eval_artifacts["refs_file"])
    _require_file(input_path, "eval input file")
    _require_file(refs_path, "eval reference file")

    if int(eval_artifacts.get("num_examples", 0)) <= 0:
        return {
            "status": "skipped",
            "reason": "no_eval_examples",
        }

    from .evaluation import run_evaluation
    from .inference import run_inference

    pred_path = run_dir / "preds_eval.jsonl"
    report_path = run_dir / "eval_report.json"
    per_example_path = run_dir / "eval_per_example.csv"
    human_review_path = run_dir / "human_review_template.csv"

    run_inference(
        cfg,
        input_path=str(input_path),
        output_path=str(pred_path),
        adapter_path=str(run_dir),
    )
    _require_file(pred_path, "prediction file")

    aggregate = run_evaluation(
        config_path=config_path,
        pred_file=str(pred_path),
        ref_file=str(refs_path),
        report_file=str(report_path),
        per_example_file=str(per_example_path),
        human_review_file=str(human_review_path),
    )
    _require_file(report_path, "evaluation report")
    _require_file(per_example_path, "per-example evaluation file")
    _require_file(human_review_path, "human review template")

    return {
        "status": "completed",
        "pred_file": pred_path.name,
        "report_file": report_path.name,
        "per_example_file": per_example_path.name,
        "human_review_file": human_review_path.name,
        "aggregate": aggregate,
    }


def _latest_checkpoint_in_run(run_dir: Path) -> Path | None:
    checkpoints = [p for p in run_dir.glob("checkpoint-*") if p.is_dir()]
    if not checkpoints:
        return None

    def _step(p: Path) -> int:
        try:
            return int(p.name.split("-")[-1])
        except ValueError:
            return -1

    checkpoints.sort(key=_step)
    return checkpoints[-1]


def _resolve_resume_checkpoint(path_like: str | Path) -> Path:
    path = Path(path_like)
    if not path.exists():
        raise ValueError(f"resume_from_checkpoint path does not exist: {path}")

    if path.is_dir() and path.name.startswith("checkpoint-"):
        return path

    # If pointing to a run directory, pick its latest checkpoint.
    ckpt = _latest_checkpoint_in_run(path)
    if ckpt is not None:
        return ckpt

    # If pointing to an output root, pick latest run then latest checkpoint.
    runs = [p for p in path.glob("run-*") if p.is_dir()]
    runs.sort(key=lambda p: p.name)
    if runs:
        ckpt = _latest_checkpoint_in_run(runs[-1])
        if ckpt is not None:
            return ckpt

    raise ValueError(f"No checkpoint-* directory found under: {path}")


def _resolve_latest_checkpoint_from_output(output_root: Path) -> Path:
    return _resolve_resume_checkpoint(output_root)


def train(
    cfg: AdapterConfig,
    output_dir: str | None = None,
    config_path: str | None = None,
    resume_from_checkpoint: str | None = None,
    resume_latest: bool = False,
) -> Path:
    deps = _import_training_dependencies()
    torch = deps["torch"]
    use_cuda = torch.cuda.is_available()

    set_seed(cfg.train.seed)

    out_dir = ensure_dir(output_dir or cfg.train.output_dir)

    resolved_resume_ckpt: Path | None = None
    if resume_from_checkpoint:
        resolved_resume_ckpt = _resolve_resume_checkpoint(resume_from_checkpoint)
    elif resume_latest:
        resolved_resume_ckpt = _resolve_latest_checkpoint_from_output(out_dir)

    if resolved_resume_ckpt is not None:
        run_dir = ensure_dir(resolved_resume_ckpt.parent)
    else:
        run_dir = ensure_dir(out_dir / datetime.utcnow().strftime("run-%Y%m%d-%H%M%S"))
    config_snapshot = _copy_config_snapshot(config_path, run_dir)

    tokenizer = deps["AutoTokenizer"].from_pretrained(cfg.model.base_model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    model_dtype = torch.bfloat16 if cfg.runtime.bfloat16 else torch.float16
    use_4bit = cfg.runtime.load_in_4bit and use_cuda
    if use_4bit:
        quantization_config = deps["BitsAndBytesConfig"](
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=model_dtype,
            bnb_4bit_use_double_quant=True,
        )

    model = deps["AutoModelForCausalLM"].from_pretrained(
        cfg.model.base_model_id,
        torch_dtype=model_dtype,
        quantization_config=quantization_config,
        device_map=cfg.runtime.device,
    )

    if use_4bit:
        prep_sig = inspect.signature(deps["prepare_model_for_kbit_training"]).parameters
        prep_kwargs: dict[str, Any] = {}
        if "use_gradient_checkpointing" in prep_sig:
            prep_kwargs["use_gradient_checkpointing"] = cfg.train.gradient_checkpointing
        model = deps["prepare_model_for_kbit_training"](model, **prep_kwargs)

    lora_config = deps["LoraConfig"](
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        target_modules=cfg.lora.target_modules,
        lora_dropout=cfg.lora.dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = deps["get_peft_model"](model, lora_config)
    if cfg.train.gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    train_records = _load_and_prepare_records(
        cfg,
        tokenizer,
        split=cfg.data.train_split,
        max_samples=cfg.data.max_train_samples_per_dataset,
    )

    eval_records = _load_and_prepare_records(
        cfg,
        tokenizer,
        split=cfg.data.eval_split,
        max_samples=cfg.data.max_eval_samples_per_dataset,
    )
    eval_artifacts = _write_eval_artifacts(run_dir, cfg, eval_records)

    train_features = [_tokenize_example(rec, tokenizer, cfg) for rec in train_records]
    eval_features = [_tokenize_example(rec, tokenizer, cfg) for rec in eval_records]

    train_ds = deps["Dataset"].from_list(train_features)
    eval_ds = deps["Dataset"].from_list(eval_features) if eval_features else None

    ta_kwargs = {
        "output_dir": str(run_dir),
        "per_device_train_batch_size": cfg.train.batch_size,
        "per_device_eval_batch_size": cfg.train.batch_size,
        "gradient_accumulation_steps": cfg.train.grad_accum,
        "learning_rate": cfg.train.lr,
        "num_train_epochs": cfg.train.epochs,
        "warmup_ratio": cfg.train.warmup_ratio,
        "lr_scheduler_type": cfg.train.lr_scheduler_type,
        "weight_decay": cfg.train.weight_decay,
        "logging_steps": cfg.train.logging_steps,
        "save_steps": cfg.train.save_steps,
        "eval_steps": cfg.train.eval_steps,
        "bf16": cfg.runtime.bfloat16 and use_cuda,
        "fp16": (not cfg.runtime.bfloat16) and use_cuda,
        "report_to": [],
        "remove_unused_columns": False,
    }

    # Transformers 4.x uses `evaluation_strategy`; 5.x uses `eval_strategy`.
    ta_sig = inspect.signature(deps["TrainingArguments"].__init__).parameters
    optional_ta_kwargs = {
        "group_by_length": True,
        "dataloader_num_workers": 2,
        "save_total_limit": 2,
        "gradient_checkpointing": cfg.train.gradient_checkpointing,
    }
    for key, value in optional_ta_kwargs.items():
        if key in ta_sig:
            ta_kwargs[key] = value

    eval_value = "steps" if eval_ds is not None else "no"
    if "evaluation_strategy" in ta_sig:
        ta_kwargs["evaluation_strategy"] = eval_value
    elif "eval_strategy" in ta_sig:
        ta_kwargs["eval_strategy"] = eval_value

    # Last-pass compatibility filter for mixed Transformers versions.
    ta_kwargs = {k: v for k, v in ta_kwargs.items() if k in ta_sig}

    training_args = deps["TrainingArguments"](**ta_kwargs)

    trainer = deps["Trainer"](
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=deps["DataCollatorForSeq2Seq"](
            tokenizer=tokenizer,
            model=model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,
        ),
    )

    if resolved_resume_ckpt is not None:
        try:
            trainer.train(resume_from_checkpoint=str(resolved_resume_ckpt))
        except ValueError as exc:
            trainer._load_from_checkpoint(str(resolved_resume_ckpt))
            trainer.train()
    else:
        trainer.train()

    trainer.save_model(str(run_dir))
    tokenizer.save_pretrained(str(run_dir))

    auto_outputs: dict[str, Any] = {"status": "disabled"}
    if cfg.train.auto_generate_outputs:
        if not config_path:
            raise ValueError("config_path is required when train.auto_generate_outputs is enabled.")
        auto_outputs = _run_post_train_generation(cfg, config_path, run_dir, eval_artifacts)

    metadata = {
        "base_model_id": cfg.model.base_model_id,
        "config_snapshot": config_snapshot,
        "train_samples": len(train_records),
        "eval_samples": len(eval_records),
        "eval_artifacts": eval_artifacts,
        "resumed_from_checkpoint": str(resolved_resume_ckpt) if resolved_resume_ckpt else None,
        "auto_outputs": auto_outputs,
        "lora": {
            "r": cfg.lora.r,
            "alpha": cfg.lora.alpha,
            "dropout": cfg.lora.dropout,
            "target_modules": cfg.lora.target_modules,
        },
    }
    with (run_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LoRA adapter for summarization")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--output_dir", default=None, help="Optional override for output dir")
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="Checkpoint, run dir, or output root to resume from.",
    )
    parser.add_argument(
        "--resume_latest",
        action="store_true",
        help="Resume from latest checkpoint found under output_dir.",
    )
    parser.add_argument(
        "--auto_generate_outputs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run inference + evaluation on saved eval files after training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.resume_from_checkpoint and args.resume_latest:
        raise ValueError("Use only one of --resume_from_checkpoint or --resume_latest.")
    if args.auto_generate_outputs is not None:
        cfg.train.auto_generate_outputs = args.auto_generate_outputs
    run_path = train(
        cfg,
        output_dir=args.output_dir,
        config_path=args.config,
        resume_from_checkpoint=args.resume_from_checkpoint,
        resume_latest=args.resume_latest,
    )
    print(f"Training complete. Artifacts saved to: {run_path}")


if __name__ == "__main__":
    main()
