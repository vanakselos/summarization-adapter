import tempfile
import unittest
from pathlib import Path

from adapter.config import load_config


class ConfigTests(unittest.TestCase):
    def test_load_default_config_file(self) -> None:
        cfg = load_config("adapter/config.yaml")
        self.assertEqual(cfg.model.base_model_id, "Qwen/Qwen2.5-3B-Instruct")
        self.assertEqual(cfg.lora.r, 16)
        self.assertEqual(cfg.inference.strategy, "hierarchical")

    def test_invalid_overlap_raises(self) -> None:
        config_text = """
inference:
  chunk_tokens: 100
  chunk_overlap: 100
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.yaml"
            path.write_text(config_text, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)

    def test_invalid_lr_scheduler_raises(self) -> None:
        config_text = """
train:
  lr_scheduler_type: not_a_scheduler
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad_scheduler.yaml"
            path.write_text(config_text, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)

    def test_invalid_inference_strategy_raises(self) -> None:
        config_text = """
inference:
  strategy: unsupported
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad_strategy.yaml"
            path.write_text(config_text, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)

    def test_invalid_llm_judge_config_raises(self) -> None:
        config_text = """
eval:
  enable_llm_judge: true
  llm_judge_model: ""
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad_llm_judge.yaml"
            path.write_text(config_text, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)

if __name__ == "__main__":
    unittest.main()
