"""Merge a trained LoRA adapter into standalone model weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import AdapterConfig, load_config


def _import_merge_dependencies() -> dict[str, Any]:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Missing merge dependencies. Install requirements first: pip install -r requirements.txt"
        ) from exc

    return {
        "torch": torch,
        "PeftModel": PeftModel,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
    }


def _read_adapter_base_model_id(adapter_path: Path) -> str | None:
    config_path = adapter_path / "adapter_config.json"
    if not config_path.exists():
        return None

    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    base_model_id = raw.get("base_model_name_or_path")
    if isinstance(base_model_id, str) and base_model_id.strip():
        return base_model_id.strip()
    return None


def _resolve_base_model_id(
    adapter_path: Path,
    cfg: AdapterConfig | None = None,
    explicit_base_model_id: str | None = None,
) -> str:
    if explicit_base_model_id and explicit_base_model_id.strip():
        return explicit_base_model_id.strip()

    adapter_base_model_id = _read_adapter_base_model_id(adapter_path)
    if adapter_base_model_id:
        return adapter_base_model_id

    if cfg is not None and cfg.model.base_model_id.strip():
        return cfg.model.base_model_id.strip()

    raise ValueError(
        "Could not resolve base model id. Provide --base_model_id, provide --config, "
        "or ensure adapter_config.json contains base_model_name_or_path."
    )


def _resolve_merge_dtype(torch_module: Any, use_cuda: bool, cfg: AdapterConfig | None = None) -> Any:
    if not use_cuda:
        return torch_module.float32

    use_bfloat16 = True if cfg is None else cfg.runtime.bfloat16
    return torch_module.bfloat16 if use_bfloat16 else torch_module.float16


def _prepare_output_dir(output_dir: Path) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"Output path exists and is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def merge_adapter(
    adapter_path: str | Path,
    output_dir: str | Path,
    config_path: str | None = None,
    base_model_id: str | None = None,
) -> Path:
    adapter_dir = Path(adapter_path)
    if not adapter_dir.exists() or not adapter_dir.is_dir():
        raise ValueError(f"Adapter path must be an existing directory: {adapter_dir}")

    cfg = load_config(config_path) if config_path else None
    resolved_base_model_id = _resolve_base_model_id(
        adapter_dir,
        cfg=cfg,
        explicit_base_model_id=base_model_id,
    )

    deps = _import_merge_dependencies()
    torch = deps["torch"]
    use_cuda = torch.cuda.is_available()
    dtype = _resolve_merge_dtype(torch, use_cuda=use_cuda, cfg=cfg)

    model_kwargs: dict[str, Any] = {"torch_dtype": dtype}
    if use_cuda:
        model_kwargs["device_map"] = "auto"

    base_model = deps["AutoModelForCausalLM"].from_pretrained(
        resolved_base_model_id,
        **model_kwargs,
    )
    model = deps["PeftModel"].from_pretrained(base_model, str(adapter_dir))
    merged_model = model.merge_and_unload(safe_merge=True)

    merged_dir = Path(output_dir)
    _prepare_output_dir(merged_dir)
    merged_model.save_pretrained(str(merged_dir), safe_serialization=True)

    try:
        tokenizer = deps["AutoTokenizer"].from_pretrained(str(adapter_dir), use_fast=True)
    except Exception:
        tokenizer = deps["AutoTokenizer"].from_pretrained(resolved_base_model_id, use_fast=True)
    tokenizer.save_pretrained(str(merged_dir))

    return merged_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into a standalone model.")
    parser.add_argument("--adapter_path", required=True, help="Path to a trained adapter directory.")
    parser.add_argument("--output_dir", required=True, help="Directory for the merged model.")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional config file used only to resolve defaults such as runtime dtype.",
    )
    parser.add_argument(
        "--base_model_id",
        default=None,
        help="Optional explicit base model id or local path. Overrides adapter metadata and config.",
    )
    args = parser.parse_args()

    merged_dir = merge_adapter(
        adapter_path=args.adapter_path,
        output_dir=args.output_dir,
        config_path=args.config,
        base_model_id=args.base_model_id,
    )
    print(f"Merged model saved to {merged_dir}")


if __name__ == "__main__":
    main()
