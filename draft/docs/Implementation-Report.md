# Implementation Report

## Project Title

Implementation Report for TextAdapter Summarization Pipeline

## 1. Introduction

This report describes the implemented process in the current source code of the TextAdapter project. The system is designed for long-document summarization with LoRA adapter training, inference, and evaluation. The explanation below follows the actual code structure and current implementation status.

## 2. Overall Process

Based on the source code, the implemented process is:

1. Dataset loading and normalization
2. Prompt-based data preparation
3. Tokenization for training
4. LoRA adapter training
5. Inference for summary generation
6. Evaluation with automatic metrics

There is one important clarification: document chunking is implemented in inference stage, not in training data preprocessing stage.

## 3. Dataset Preprocessing

Dataset preprocessing is implemented in `adapter/data.py`.

The system supports these datasets:

- `cnn_dailymail`
- `xsum`
- `arxiv`
- `govreport`
- `xlsum`

Each dataset has different field names. Because of this, the code normalizes all records into one common format:

- `id`
- `dataset`
- `source`
- `summary`

This normalization is implemented in the `normalize_example()` function.

For example:

- `arxiv` uses article text as source and abstract as summary
- `govreport` uses report text as source and summary as target

After normalization, the code creates instruction-style training records by building a prompt from the source document. This is implemented in `build_instruction_record()`.

## 4. Document Chunking

Document chunking is implemented in `adapter/utils.py` and used in `adapter/inference.py`.

The function `chunk_text_by_words()` splits a long document into overlapping chunks using:

- `chunk_size`
- `overlap`

This chunking is used only during inference for long documents. It is not used in training preprocessing.

The current code supports two inference strategies:

- `hierarchical`
- `arxiv_lead_refine`

In both strategies, chunking is applied when the document length is larger than the configured threshold.

So, from implementation point of view:

- Training: truncation
- Inference: chunking for long document

## 5. Tokenization

Tokenization is implemented in `_tokenize_example()` in `adapter/train.py`.

The process is:

- Tokenize prompt text
- Tokenize target summary text
- Truncate prompt to `max_source_tokens`
- Truncate target to `max_target_tokens`
- Concatenate prompt tokens and target tokens
- Add EOS token if available

For training labels:

- Prompt tokens are masked with `-100`
- Only target summary tokens contribute to loss

This means the model is trained in instruction-following style, where it reads the prompt and learns to generate the summary.

## 6. Adapter Training

Adapter training is implemented in `adapter/train.py`.

The base model is loaded with Hugging Face `AutoModelForCausalLM`.
Then LoRA adapter is applied with PEFT using:

- rank `r`
- `lora_alpha`
- `lora_dropout`
- `target_modules`

If 4-bit loading is enabled, the model is prepared with `prepare_model_for_kbit_training()` before LoRA wrapping.

Training is performed with Hugging Face `Trainer`.

At the end of training:

- adapter weights are saved
- tokenizer is saved
- metadata is written to run directory

So the implemented training method is adapter fine-tuning, not full model fine-tuning.

## 7. Training Configuration

Training configuration is defined in `adapter/config.py`.

The main configuration sections are:

- `model`
- `lora`
- `data`
- `train`
- `inference`
- `eval`
- `runtime`

Important configurable training parameters include:

- base model id
- LoRA rank and dropout
- train and eval dataset names
- max source tokens
- max target tokens
- batch size
- gradient accumulation
- learning rate
- number of epochs
- warmup ratio
- scheduler type
- gradient checkpointing
- runtime precision and device

The config file is validated before training starts.

## 8. Evaluation

Evaluation is implemented in `adapter/evaluation.py`.

The evaluation pipeline:

- load prediction file
- load reference file
- align prediction and reference by `id`
- compute metrics
- save aggregate report and per-example outputs

Implemented metrics include:

- ROUGE-1 F1
- ROUGE-2 F1
- ROUGE-L F1
- BERTScore F1
- faithfulness proxy
- coverage
- compression ratio
- compression efficiency
- composite score

The evaluation stage also creates a human review template for manual inspection.

## 9. Important Implementation Note

The implemented system is not fully symmetric between training and inference.

Training side:

- uses truncation on tokenized input
- does not use hierarchical chunk summarization

Inference side:

- uses chunking for long documents
- supports hierarchical summarization logic

This is an important design characteristic of the current codebase.

## 10. Conclusion

Yes, based on the source code, this is the implemented process:

- Dataset preprocessing is implemented
- Document chunking is implemented for inference
- Tokenization is implemented for instruction-style training
- Adapter training with LoRA is implemented
- Training configuration is implemented through validated config schema
- Evaluation is implemented with automatic metrics and review artifacts

The only correction is that document chunking belongs to inference process, not training preprocessing process.

## 11. Final Confirmation

I can confirm that this is the actual implementation process in the current codebase.
