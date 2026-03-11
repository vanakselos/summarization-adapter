"""Configuration schema and loader for TextAdapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    base_model_id: str = "Qwen/Qwen2.5-3B-Instruct"
    context_window_tokens: int = 32768


@dataclass
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.1
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])


@dataclass
class DataConfig:
    train_datasets: list[str] = field(
        default_factory=lambda: ["cnn_dailymail", "xsum", "govreport"]
    )
    eval_datasets: list[str] = field(default_factory=lambda: ["cnn_dailymail", "xsum"])
    train_split: str = "train"
    eval_split: str = "validation"
    max_train_samples_per_dataset: int | None = None
    max_eval_samples_per_dataset: int | None = 500
    max_source_tokens: int = 2048
    max_target_tokens: int = 256


@dataclass
class TrainConfig:
    output_dir: str = "outputs/train"
    batch_size: int = 2
    grad_accum: int = 8
    lr: float = 2e-4
    epochs: int = 2
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "linear"
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    seed: int = 42
    gradient_checkpointing: bool = False
    auto_generate_outputs: bool = True


@dataclass
class InferenceConfig:
    strategy: str = "hierarchical"
    chunk_tokens: int = 2048
    chunk_overlap: int = 256
    max_new_tokens: int = 256
    temperature: float = 0.4
    top_p: float = 0.9
    prompt_batch_size: int = 1


@dataclass
class EvalConfig:
    enable_rouge: bool = True
    enable_bertscore: bool = True
    enable_faithfulness_proxy: bool = True
    human_review_samples: int = 20
    enable_llm_judge: bool = False
    llm_judge_model: str = "google/gemini-3.1-pro-preview"
    llm_judge_api_base: str = "https://openrouter.ai/api/v1"
    llm_judge_timeout_seconds: int = 120
    llm_judge_max_tokens: int = 400
    llm_judge_temperature: float = 0.0


@dataclass
class RuntimeConfig:
    load_in_4bit: bool = True
    bfloat16: bool = True
    device: str = "auto"


@dataclass
class AdapterConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def _as_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    raise ValueError("Expected config section to be a dictionary.")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Top-level config must be a dictionary.")
    return loaded


def load_config(path: str | Path) -> AdapterConfig:
    """Load and validate an adapter config from YAML."""
    raw = _load_yaml(Path(path))

    cfg = AdapterConfig(
        model=ModelConfig(**_as_dict(raw.get("model", {}))),
        lora=LoRAConfig(**_as_dict(raw.get("lora", {}))),
        data=DataConfig(**_as_dict(raw.get("data", {}))),
        train=TrainConfig(**_as_dict(raw.get("train", {}))),
        inference=InferenceConfig(**_as_dict(raw.get("inference", {}))),
        eval=EvalConfig(**_as_dict(raw.get("eval", {}))),
        runtime=RuntimeConfig(**_as_dict(raw.get("runtime", {}))),
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: AdapterConfig) -> None:
    """Raise ValueError on invalid config values."""
    if cfg.lora.r <= 0:
        raise ValueError("lora.r must be > 0")
    if cfg.lora.alpha <= 0:
        raise ValueError("lora.alpha must be > 0")
    if not 0 <= cfg.lora.dropout < 1:
        raise ValueError("lora.dropout must be in [0, 1)")

    if cfg.data.max_source_tokens <= 0 or cfg.data.max_target_tokens <= 0:
        raise ValueError("data.max_source_tokens and data.max_target_tokens must be > 0")

    allowed_inference_strategies = {"hierarchical", "arxiv_lead_refine"}
    if cfg.inference.strategy not in allowed_inference_strategies:
        raise ValueError(
            "inference.strategy must be one of: "
            + ", ".join(sorted(allowed_inference_strategies))
        )
    if cfg.inference.chunk_tokens <= 0:
        raise ValueError("inference.chunk_tokens must be > 0")
    if cfg.inference.chunk_overlap < 0:
        raise ValueError("inference.chunk_overlap must be >= 0")
    if cfg.inference.chunk_overlap >= cfg.inference.chunk_tokens:
        raise ValueError("inference.chunk_overlap must be smaller than chunk_tokens")

    if cfg.train.batch_size <= 0 or cfg.train.grad_accum <= 0:
        raise ValueError("train.batch_size and train.grad_accum must be > 0")
    if cfg.train.lr <= 0:
        raise ValueError("train.lr must be > 0")
    if cfg.train.epochs <= 0:
        raise ValueError("train.epochs must be > 0")
    if not 0 <= cfg.train.warmup_ratio < 1:
        raise ValueError("train.warmup_ratio must be in [0, 1)")
    allowed_schedulers = {
        "linear",
        "cosine",
    }
    if cfg.train.lr_scheduler_type not in allowed_schedulers:
        raise ValueError(
            "train.lr_scheduler_type must be one of: "
            + ", ".join(sorted(allowed_schedulers))
        )

    if cfg.model.context_window_tokens <= 0:
        raise ValueError("model.context_window_tokens must be > 0")
    if cfg.eval.enable_llm_judge:
        if not cfg.eval.llm_judge_model.strip():
            raise ValueError("eval.llm_judge_model must be non-empty when LLM judge is enabled")
        if not cfg.eval.llm_judge_api_base.strip():
            raise ValueError("eval.llm_judge_api_base must be non-empty when LLM judge is enabled")
        if cfg.eval.llm_judge_timeout_seconds <= 0:
            raise ValueError("eval.llm_judge_timeout_seconds must be > 0")
        if cfg.eval.llm_judge_max_tokens <= 0:
            raise ValueError("eval.llm_judge_max_tokens must be > 0")
        if cfg.eval.llm_judge_temperature < 0:
            raise ValueError("eval.llm_judge_temperature must be >= 0")
