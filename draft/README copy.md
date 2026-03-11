# TextAdapter

LoRA-based long-document summarization on top of `Qwen/Qwen2.5-3B-Instruct`.

## Train An Adapter

```bash
cd /home/vananh/Apple/TextAdapter
uv run python -m adapter.train --config configs/train_gorvreport_full.yaml
```

Other example configs:

```bash
cd /home/vananh/Apple/TextAdapter
uv run python -m adapter.train --config configs/train_smoke.yaml
uv run python -m adapter.train --config configs/train_pilot.yaml
uv run python -m adapter.train --config configs/train_full.yaml
```

## Run Inference

Base model only:

```bash
cd /home/vananh/Apple/TextAdapter
uv run python -m adapter.inference \
  --config outputs/govreport-5k/run-20260310-164841/config_used.yaml \
  --input_file outputs/govreport-5k/run-20260310-164841/input_50.jsonl \
  --output_file outputs/govreport-5k/run-20260310-164841/preds_eval_50_a.jsonl
```

Base model plus LoRA adapter:

```bash
cd /home/vananh/Apple/TextAdapter
uv run python -m adapter.inference \
  --config outputs/govreport-5k/run-20260310-164841/config_used.yaml \
  --input_file outputs/govreport-5k/run-20260310-164841/input_eval.jsonl \
  --output_file outputs/govreport-5k/run-20260310-164841/preds_eval.jsonl \
  --adapter_path outputs/govreport-5k/run-20260310-164841
```

sed -n '51,100p' outputs/govreport-5k/run-20260310-164841/input_eval.jsonl > outputs/govreport-5k/run-20260310-164841/input_50_b.jsonl
head -n 100 outputs/govreport-5k/run-20260310-164841/preds_eval.jsonl > outputs/govreport-5k/run-20260310-164841/pred_100_lora.jsonl

## Merge Base Model And Adapter

This project supports adapter loading at inference time via `--adapter_path`, and also a standalone merge command:

```bash
cd /home/vananh/Apple/TextAdapter
uv run python -m adapter.merge \
  --adapter_path outputs/run-20260307-172145 \
  --output_dir outputs/run-20260307-172145-merged
```

Notes:

- The merge command resolves the base model from `adapter_config.json` by default.
- Use `--base_model_id` if you need to override the base model explicitly.
- Merge from full-precision or bf16/fp16 weights, not a 4-bit loaded base model.
- The merged directory is much larger than the adapter directory.

To run inference from the merged model:

1. Copy your config file to a new file.
2. Set `model.base_model_id` to the merged directory, for example `outputs/run-20260307-172145-merged`.
3. Run `python -m adapter.inference` without `--adapter_path`.

Example:

```bash
cd /home/vananh/Apple/TextAdapter
uv run python -m adapter.inference \
  --config outputs/xsum-5k/run-20260311-084211/config_merged.yaml \
  --input_file outputs/xsum-5k/run-20260311-084211/input_1000.jsonl \
  --output_file outputs/xsum-5k/run-20260311-084211/preds_100_merged.jsonl
```

## Evaluate Predictions

```bash
cd /home/vananh/Apple/TextAdapter
uv run python -m adapter.evaluation \
  --config outputs/govreport-5k/run-20260310-164841/config_used.yaml \
  --pred_file outputs/govreport-5k/run-20260310-164841/pred_100_lora.jsonl \
  --ref_file outputs/govreport-5k/run-20260310-164841/refs_eval.jsonl \
  --report_file outputs/govreport-5k/run-20260310-164841/report_100_lora.json \
  --per_example_file outputs/govreport-5k/run-20260310-164841/per_100_lora.csv \
  --human_review_file outputs/govreport-5k/run-20260310-164841/human_100_lora.csv
```

## Prompt Example

```bash
cd /home/vananh/Apple/TextAdapter
uv run python -m adapter.prompt_qwen \
  --config configs/train_xlsum_smoke.yaml \
  --prompt "let summary this

document: In a groundbreaking study, researchers at the University of California have
discovered a new method to significantly enhance the efficiency of solar panels. The
method involves the use of a novel material that can absorb a broader spectrum of light,
leading to a 20% increase in energy conversion rates. This breakthrough could
revolutionize the renewable energy sector and pave the way for more sustainable and
cost-effective solutions." \
  --max_new_tokens 128 \
  --temperature 0.0
```
