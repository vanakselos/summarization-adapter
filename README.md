
# TextAdapter

LoRA-based long-document summarization built on top of **`Qwen/Qwen2.5-3B-Instruct`**.

This project implements a parameter-efficient summarization adapter using **LoRA** to improve performance on domain-specific datasets such as **XSum** and **GovReport**.

---

# Project Scope

This repository contains:

* Source code for **LoRA training, inference, and evaluation**
* Dataset normalization and **prompt construction logic**
* Evaluation scripts and metric reporting
* Example training configurations
* Trained adapters for experiment reproduction

The system supports:

* Extreme summarization (XSum)
* Long-document summarization (GovReport)
* Adapter-based inference
* Base-model comparison

---

# Base Model

All adapters are trained on:

`Qwen/Qwen2.5-3B-Instruct`

---

# Available Adapters

Two trained LoRA adapters are provided.

| Adapter             | Dataset   | Task                        |
| ------------------- | --------- | --------------------------- |
| `xsum_adapter`      | XSum      | extreme summarization       |
| `govreport_adapter` | GovReport | long-document summarization |

---

# Repository Structure

```id="h0tnbf"
TextAdapter
│
├── adapter
│   ├── train.py
│   ├── inference.py
│   └── evaluation.py
│
├── configs
│   ├── train_smoke.yaml
│   ├── train_pilot.yaml
│   ├── train_full.yaml
│   └── train_arxiv_full.yaml
│
├── scripts
│   └── text_to_jsonl.py
│
├── outputs
│   └── training logs and experiment outputs
│
└── requirements.txt
```

---

# Environment

Tested environment:

```
Python 3.10
PyTorch 2.2
CUDA 12.1
transformers
peft
datasets
```

Install dependencies:

```bash id="e61p70"
pip install -r requirements.txt
```

---

# Clone The Project

```bash id="3xg61u"
git clone <your-github-repo-url>
cd TextAdapter
```

---

# Download Trained Adapters

Adapters and evaluation files are hosted on Hugging Face.

Download example:

```bash id="n16pxx"
huggingface-cli download <hf-repo>/xsum_adapter --local-dir ./artifacts/xsum
```

```bash id="mb6apg"
huggingface-cli download <hf-repo>/govreport_adapter --local-dir ./artifacts/govreport
```

Each adapter folder contains:

```
adapter_model.safetensors
adapter_config.json
config_used.yaml
eval_data/
    input.jsonl
    refs.jsonl
```

---

# Inference

## Run Inference with Adapter

Example for **XSum adapter**:

```bash id="1xxcrs"
python -m adapter.inference \
  --config artifacts/xsum/config_used.yaml \
  --input_file artifacts/xsum/eval_data/input.jsonl \
  --output_file preds_xsum.jsonl \
  --adapter_path artifacts/xsum
```

Example for **GovReport adapter**:

```bash id="5e1o2f"
python -m adapter.inference \
  --config artifacts/govreport/config_used.yaml \
  --input_file artifacts/govreport/eval_data/input.jsonl \
  --output_file preds_govreport.jsonl \
  --adapter_path artifacts/govreport
```

---

# Base Model Inference (Without Adapter)

```bash id="k67l98"
python -m adapter.inference \
  --config artifacts/xsum/config_used.yaml \
  --input_file artifacts/xsum/eval_data/input.jsonl \
  --output_file preds_base.jsonl
```

---

# Inference Input Format

The inference input must be a **JSONL file**.

Example:

```json id="8lhyyr"
{"id": "1", "document": "your document text here"}
```

For plain text files, `.txt` inputs are also supported.

To convert `.txt` files to JSONL:

```bash id="s1t8nl"
python scripts/text_to_jsonl.py \
  --input_path path/to/file-or-folder \
  --output_path input.jsonl
```

---

# Evaluate Predictions

```bash id="5yuzcs"
python -m adapter.evaluation \
  --config artifacts/xsum/config_used.yaml \
  --pred_file preds_xsum.jsonl \
  --ref_file artifacts/xsum/eval_data/refs.jsonl \
  --report_file report.json \
  --per_example_file per_example.csv
```

---

# Evaluation Metrics

The evaluation pipeline computes the following metrics:

* **ROUGE**
* **BERTScore**
* **Faithfulness proxy**
* **Coverage**

The evaluation module aligns predictions and references using the example `id`.

---

# Train a LoRA Adapter

Example training command:

```bash id="hsug49"
python -m adapter.train --config configs/train_govreport_full.yaml
```

Other available configs:

```bash id="svkrgq"
python -m adapter.train --config configs/train_smoke.yaml
python -m adapter.train --config configs/train_pilot.yaml
python -m adapter.train --config configs/train_full.yaml
python -m adapter.train --config configs/train_arxiv_full.yaml
```

---

# Reproducing the Experiments

To reproduce the experimental results:

1. Download the trained adapter from Hugging Face.
2. Run inference using `adapter.inference`.
3. Evaluate predictions using `adapter.evaluation`.

All required evaluation files (`input.jsonl` and `refs.jsonl`) are provided in the adapter repository.

---

# Hardware Requirements

Typical training configuration:

```
GPU: 24GB – 80GB VRAM
QLoRA: enabled
Gradient checkpointing: enabled
Batch size: 2
```

---

# Notes

* Long-document inference uses **chunk-based processing**.
* QLoRA is used to reduce memory usage during training.
* Gradient accumulation is applied to simulate larger batch sizes.

---

# Model License

This project is built on top of:

`Qwen/Qwen2.5-3B-Instruct`

Please refer to the original model license for usage restrictions.

---

# Contact

For questions regarding the implementation or experiments, please refer to the source code or open an issue in the repository.

---

