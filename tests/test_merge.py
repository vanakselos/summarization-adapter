import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from adapter.config import AdapterConfig
from adapter.merge import (
    _read_adapter_base_model_id,
    _resolve_base_model_id,
    _resolve_merge_dtype,
    merge_adapter,
)


class _FakeCuda:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _FakeTorch:
    bfloat16 = "bf16"
    float16 = "fp16"
    float32 = "fp32"

    def __init__(self, cuda_available: bool) -> None:
        self.cuda = _FakeCuda(cuda_available)


class MergeTests(unittest.TestCase):
    def test_read_adapter_base_model_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir)
            (adapter_dir / "adapter_config.json").write_text(
                json.dumps({"base_model_name_or_path": "Qwen/Qwen2.5-3B-Instruct"}),
                encoding="utf-8",
            )
            self.assertEqual(
                _read_adapter_base_model_id(adapter_dir),
                "Qwen/Qwen2.5-3B-Instruct",
            )

    def test_resolve_base_model_id_prefers_explicit_value(self) -> None:
        cfg = AdapterConfig()
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir)
            (adapter_dir / "adapter_config.json").write_text(
                json.dumps({"base_model_name_or_path": "from-adapter"}),
                encoding="utf-8",
            )
            resolved = _resolve_base_model_id(
                adapter_dir,
                cfg=cfg,
                explicit_base_model_id="from-arg",
            )
            self.assertEqual(resolved, "from-arg")

    def test_resolve_base_model_id_prefers_adapter_metadata_over_config(self) -> None:
        cfg = AdapterConfig()
        cfg.model.base_model_id = "from-config"
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir)
            (adapter_dir / "adapter_config.json").write_text(
                json.dumps({"base_model_name_or_path": "from-adapter"}),
                encoding="utf-8",
            )
            self.assertEqual(_resolve_base_model_id(adapter_dir, cfg=cfg), "from-adapter")

    def test_resolve_merge_dtype_uses_float32_on_cpu(self) -> None:
        dtype = _resolve_merge_dtype(_FakeTorch(cuda_available=False), use_cuda=False, cfg=None)
        self.assertEqual(dtype, "fp32")

    def test_resolve_merge_dtype_respects_config_on_cuda(self) -> None:
        cfg = AdapterConfig()
        cfg.runtime.bfloat16 = False
        dtype = _resolve_merge_dtype(_FakeTorch(cuda_available=True), use_cuda=True, cfg=cfg)
        self.assertEqual(dtype, "fp16")

    def test_merge_adapter_merges_and_saves_outputs(self) -> None:
        calls: dict[str, object] = {}

        class FakeAutoModelForCausalLM:
            @staticmethod
            def from_pretrained(model_id: str, **kwargs):
                calls["base_model_id"] = model_id
                calls["model_kwargs"] = kwargs
                return object()

        class FakeMergedModel:
            def save_pretrained(self, output_dir: str, safe_serialization: bool = False) -> None:
                calls["save_output_dir"] = output_dir
                calls["safe_serialization"] = safe_serialization

        class FakePeftLoadedModel:
            def merge_and_unload(self, safe_merge: bool = False):
                calls["safe_merge"] = safe_merge
                return FakeMergedModel()

        class FakePeftModel:
            @staticmethod
            def from_pretrained(base_model, adapter_path: str):
                calls["adapter_path"] = adapter_path
                return FakePeftLoadedModel()

        class FakeTokenizer:
            def save_pretrained(self, output_dir: str) -> None:
                calls["tokenizer_output_dir"] = output_dir

        class FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(source: str, use_fast: bool = True):
                calls["tokenizer_source"] = source
                calls["tokenizer_use_fast"] = use_fast
                return FakeTokenizer()

        fake_deps = {
            "torch": _FakeTorch(cuda_available=True),
            "PeftModel": FakePeftModel,
            "AutoModelForCausalLM": FakeAutoModelForCausalLM,
            "AutoTokenizer": FakeAutoTokenizer,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            adapter_dir = root / "adapter"
            adapter_dir.mkdir()
            (adapter_dir / "adapter_config.json").write_text(
                json.dumps({"base_model_name_or_path": "resolved-from-adapter"}),
                encoding="utf-8",
            )
            merged_dir = root / "merged"

            with patch("adapter.merge._import_merge_dependencies", return_value=fake_deps):
                result = merge_adapter(adapter_path=adapter_dir, output_dir=merged_dir)

            self.assertEqual(result, merged_dir)
            self.assertTrue(merged_dir.exists())
            self.assertEqual(calls["base_model_id"], "resolved-from-adapter")
            self.assertEqual(calls["model_kwargs"], {"torch_dtype": "bf16", "device_map": "auto"})
            self.assertEqual(calls["adapter_path"], str(adapter_dir))
            self.assertEqual(calls["safe_merge"], True)
            self.assertEqual(calls["save_output_dir"], str(merged_dir))
            self.assertEqual(calls["safe_serialization"], True)
            self.assertEqual(calls["tokenizer_source"], str(adapter_dir))
            self.assertEqual(calls["tokenizer_output_dir"], str(merged_dir))


if __name__ == "__main__":
    unittest.main()
