Below is a **practical, production-oriented design** for a summarization adapter module built on top of **Alibaba Cloud’s Qwen2.5-3B** (3B, runs locally on 16GB GPU or quantized CPU). I’ll also mention alternatives like **Llama 3 3B** and **Mistral 3B** where relevant.

No fluff. This is how you’d build it in real life.

---

# 1️⃣ System Goal

Build a **parameter-efficient summarization adapter** that:

* Runs on-device
* Handles long documents (5k–50k tokens)
* Produces concise, faithful summaries
* Can be fine-tuned cheaply
* Supports evaluation across multiple dimensions

---

# 2️⃣ High-Level Architecture

### Core Components

```
Long Document
    ↓
Chunking Module
    ↓
Adapter-Enhanced Qwen 2.5 3B
    ↓
Hierarchical Aggregation
    ↓
Final Summary
```

---

# 3️⃣ Model Strategy

We do NOT fully fine-tune 3B parameters.

We use:

### ✔ LoRA (Low-Rank Adaptation)

Efficient, low VRAM
Only train small rank matrices

Recommended:

* Rank: 8–32
* Alpha: 16
* Target modules: q_proj, v_proj, k_proj, o_proj, optionally MLP

Framework:

* HuggingFace PEFT

---

# 4️⃣ Handling Long Documents

Qwen 2.5 3B context ~32k (depending on variant).

For very long documents:

### Option A — Hierarchical Summarization (Most Stable)

1. Split document into chunks (e.g., 2048 tokens)
2. Summarize each chunk
3. Merge chunk summaries
4. Summarize merged summary

This avoids:

* Memory explosion
* Attention quadratic cost

---

# 5️⃣ Adapter Design

### Adapter = LoRA layer injected into:

```
Attention projections
MLP feedforward blocks
```

Training objective:

```
Causal LM loss
```

Prompt format:

```
### Instruction:
Summarize the following document concisely.

### Document:
<chunk>

### Summary:
```

---

# 6️⃣ Datasets

You need both:

### General summarization:

* CNN/DailyMail
* XSum
* GovReport (long-form)
* PubMed

### For long-document robustness:

* arXiv summarization
* BookSum
* BigPatent

All open source.

---

# 7️⃣ Training Pipeline

### Step 1 — Tokenization

Use model tokenizer.

### Step 2 — Data Formatting

Convert to instruction format.

### Step 3 — LoRA Fine-Tuning

Example config:

```python
LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
```

Training hyperparameters:

| Parameter  | Value    |
| ---------- | -------- |
| LR         | 2e-4     |
| Batch size | 16       |
| Epochs     | 2–3      |
| Warmup     | 5%       |
| Optimizer  | AdamW    |
| Precision  | bfloat16 |

---

# 8️⃣ Inference Pipeline

For long doc:

```python
def summarize_long_doc(doc):
    chunks = chunk(doc)
    partial_summaries = [model.generate(chunk) for chunk in chunks]
    merged = " ".join(partial_summaries)
    final = model.generate(merged)
    return final
```

Optional:

* Add length control token
* Add style control token

---

# 9️⃣ Evaluation Strategy

Don’t rely on one metric. That’s amateur.

### A. ROUGE (n-gram overlap)

Standard
ROUGE-1, ROUGE-2, ROUGE-L

### B. BERTScore

Semantic similarity

### C. Faithfulness (Critical)

Use:

* QAEval
* Factual consistency model
* GPT-based evaluator

### D. Compression Ratio

```
len(summary) / len(document)
```

### E. Human Evaluation

Rate:

* Faithfulness
* Coherence
* Conciseness
* Coverage

---

# 🔟 Custom Metric (Recommended)

Design:

```
Score = 
0.4 * Faithfulness +
0.3 * SemanticSimilarity +
0.2 * Coverage +
0.1 * CompressionEfficiency
```

This prevents:

* Hallucination
* Overly short summaries
* Verbose summaries

---

# 1️⃣1️⃣ Improving Quality

If performance is weak:

### Improve Data

* Add domain-specific dataset
* Use synthetic augmentation
* Distill from larger model (teacher → student)

### Improve Model

* Increase LoRA rank
* Add MLP layers to adapter
* Use instruction tuning dataset

### Improve Decoding

* Use temperature 0.3–0.7
* Top-p 0.9
* Length penalty

---

# 1️⃣2️⃣ On-Device Deployment

Options:

### 4-bit Quantization

* Use bitsandbytes
* Reduces memory drastically

### GGUF conversion

* Run via llama.cpp
* CPU-friendly

---

# 1️⃣3️⃣ Possible Extensions

### 1. Query-Focused Summarization

Add question into prompt.

### 2. Multi-document summarization

Add attention pooling layer.

### 3. Streaming summarization

For live transcripts.

---

# 1️⃣4️⃣ Why Qwen 2.5 3B?

Pros:

* Strong multilingual ability
* Large context
* Efficient inference

If using:

* Llama 3B → slightly weaker instruction following
* Mistral 3B → faster inference

---

# 1️⃣5️⃣ Production Considerations

* Add guardrail against hallucination
* Cache chunk summaries
* Add timeout fallback
* Log summary entropy for uncertainty detection

---

# Final Deliverables

Your module should include:

```
/adapter/
    train.py
    inference.py
    evaluation.py
    config.yaml
    lora_weights.bin
```

---

# Summary

A solid solution:

* Use Qwen 2.5 3B
* Inject LoRA adapters
* Use hierarchical summarization
* Evaluate with ROUGE + BERTScore + Faithfulness
* Quantize for deployment

This design is:

* Realistic
* Efficient
* Production-ready
* Interview-strong

If you want, I can now:

* Provide full training code
* Provide evaluation code
* Or help you turn this into a portfolio-grade ML system design write-up
