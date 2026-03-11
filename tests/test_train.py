import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from adapter.config import AdapterConfig
from adapter.train import (
    _copy_config_snapshot,
    _resolve_latest_checkpoint_from_output,
    _resolve_resume_checkpoint,
    _run_post_train_generation,
    _write_eval_artifacts,
)
from adapter.utils import load_jsonl, write_jsonl


class TrainArtifactTests(unittest.TestCase):
    def test_copy_config_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src = tmp / "config.yaml"
            src.write_text("train:\n  seed: 42\n", encoding="utf-8")
            run_dir = tmp / "run-1"
            run_dir.mkdir(parents=True)

            name = _copy_config_snapshot(str(src), run_dir)
            self.assertEqual(name, "config_used.yaml")
            self.assertTrue((run_dir / "config_used.yaml").exists())

    def test_write_eval_artifacts_outputs_expected_files(self) -> None:
        cfg = AdapterConfig()
        eval_records = [
            {"id": "1", "dataset": "xsum", "source": "doc one", "summary": "sum one"},
            {"id": "1", "dataset": "xsum", "source": "doc two", "summary": "sum two"},
            {"id": "", "dataset": "xsum", "source": "doc three", "summary": "sum three"},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            files = _write_eval_artifacts(run_dir, cfg, eval_records)

            input_path = run_dir / files["input_file"]
            refs_path = run_dir / files["refs_file"]
            manifest_path = run_dir / files["manifest_file"]

            self.assertTrue(input_path.exists())
            self.assertTrue(refs_path.exists())
            self.assertTrue(manifest_path.exists())

            input_rows = load_jsonl(input_path)
            ref_rows = load_jsonl(refs_path)
            self.assertEqual(len(input_rows), 3)
            self.assertEqual(len(ref_rows), 3)
            self.assertEqual(len({row["id"] for row in input_rows}), 3)

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["num_examples"], 3)
            self.assertEqual(manifest["input_file"], "input_eval.jsonl")
            self.assertEqual(manifest["refs_file"], "refs_eval.jsonl")

    def test_run_post_train_generation_creates_expected_files(self) -> None:
        cfg = AdapterConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            write_jsonl(run_dir / "input_eval.jsonl", [{"id": "1", "document": "doc"}])
            write_jsonl(run_dir / "refs_eval.jsonl", [{"id": "1", "document": "doc", "summary": "sum"}])

            eval_artifacts = {
                "input_file": "input_eval.jsonl",
                "refs_file": "refs_eval.jsonl",
                "num_examples": 1,
            }

            def fake_inference(config, input_path, output_path, adapter_path=None):
                write_jsonl(output_path, [{"id": "1", "document": "doc", "summary": "pred"}])

            def fake_eval(
                config_path,
                pred_file,
                ref_file,
                report_file,
                per_example_file,
                human_review_file,
            ):
                Path(report_file).write_text('{"composite_score": 0.5}', encoding="utf-8")
                Path(per_example_file).write_text("id,score\n1,0.5\n", encoding="utf-8")
                Path(human_review_file).write_text("id,notes\n1,\n", encoding="utf-8")
                return {"composite_score": 0.5}

            with patch("adapter.inference.run_inference", side_effect=fake_inference), patch(
                "adapter.evaluation.run_evaluation", side_effect=fake_eval
            ):
                result = _run_post_train_generation(
                    cfg,
                    config_path="configs/train_smoke.yaml",
                    run_dir=run_dir,
                    eval_artifacts=eval_artifacts,
                )

            self.assertEqual(result["status"], "completed")
            self.assertTrue((run_dir / "preds_eval.jsonl").exists())
            self.assertTrue((run_dir / "eval_report.json").exists())
            self.assertTrue((run_dir / "eval_per_example.csv").exists())
            self.assertTrue((run_dir / "human_review_template.csv").exists())

    def test_resolve_resume_checkpoint_from_checkpoint_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "checkpoint-12"
            ckpt.mkdir(parents=True)
            resolved = _resolve_resume_checkpoint(ckpt)
            self.assertEqual(resolved, ckpt)

    def test_resolve_resume_checkpoint_from_run_dir_picks_latest_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run = Path(tmpdir) / "run-1"
            (run / "checkpoint-10").mkdir(parents=True)
            (run / "checkpoint-50").mkdir(parents=True)
            resolved = _resolve_resume_checkpoint(run)
            self.assertEqual(resolved.name, "checkpoint-50")

    def test_resolve_latest_checkpoint_from_output_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            run_old = out / "run-20260101-010101"
            run_new = out / "run-20260101-020202"
            (run_old / "checkpoint-10").mkdir(parents=True)
            (run_new / "checkpoint-20").mkdir(parents=True)
            resolved = _resolve_latest_checkpoint_from_output(out)
            self.assertEqual(resolved.name, "checkpoint-20")


if __name__ == "__main__":
    unittest.main()
