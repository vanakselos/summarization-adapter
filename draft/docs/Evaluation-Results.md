# Evaluation Results

## Overview

This note records the current evaluation results shared during development and provides a short interpretation for each dataset.

Important note:

- `composite_score` is an internal heuristic aggregate.
- It should not be treated as the primary reported metric in a formal submission.
- The main metrics to report are `ROUGE`, `BERTScore`, and qualitative faithfulness assessment such as human review or LLM-as-a-judge.

## Metric Semantics

- `rouge1_f1`, `rouge2_f1`, `rougeL_f1`: compare prediction against the reference summary.
- `bertscore_f1`: semantic similarity between prediction and reference summary.
- `coverage`: token-set overlap of prediction against the reference summary.
- `faithfulness_proxy`: token-set overlap of prediction against the source document.
- `compression_ratio` and `compression_efficiency`: compare summary length against source length.

## GovReport

Evaluation summary:

```json
{
  "num_examples": 200,
  "rouge1_f1": 0.4778384521019463,
  "rouge2_f1": 0.18527329272896037,
  "rougeL_f1": 0.22711623127239786,
  "bertscore_f1": 0.8584522208571435,
  "faithfulness_proxy": 0.9157626783512143,
  "coverage": 0.3205683164023358,
  "compression_ratio": 0.06259323281207493,
  "compression_efficiency": 0.31153279405031575,
  "composite_score": 0.7191076802831275
}
```

Interpretation:

- The model is strong on source-grounding for GovReport, with a high `faithfulness_proxy` of `0.916`.
- `BERTScore F1 = 0.858` suggests that the generated summaries are semantically close to the references.
- `ROUGE-2 F1 = 0.185` and `ROUGE-L F1 = 0.227` show that phrasing and structure still differ substantially from the references.
- `coverage = 0.321` indicates that the summaries capture only part of the information present in the reference summaries.
- The model appears conservative: it tends to produce safe, source-grounded summaries, but may omit important findings, actions, recommendations, or impacts.

Recommended interpretation for reporting:

> On GovReport, the model produces source-grounded summaries with strong semantic alignment to the reference summaries, but only moderate coverage and low structural overlap, indicating that it captures the central content while still missing some report-specific details.

## XSum

Evaluation summary:

```json
{
  "num_examples": 999,
  "rouge1_f1": 0.34069114436497194,
  "rouge2_f1": 0.13065694043361012,
  "rougeL_f1": 0.2751265781847083,
  "bertscore_f1": 0.9015048532275943,
  "faithfulness_proxy": 0.6229006901900904,
  "coverage": 0.33122435730810934,
  "compression_ratio": 0.08996489845240892,
  "compression_efficiency": 0.36723486069307054,
  "composite_score": 0.6225800895752435
}
```

Interpretation:

- `BERTScore F1 = 0.902` is high, so the model captures the general meaning of the reference summaries well.
- `ROUGE-1/2/L` remain only moderate, which suggests that lexical and structural match to the single-reference XSum targets is limited.
- `faithfulness_proxy = 0.623` is much lower than in GovReport, indicating weaker source-grounding.
- `coverage = 0.331` is comparable to GovReport, so the model captures some key reference content but still misses a substantial portion of it.
- This pattern is consistent with a more abstractive behavior on XSum: the summaries are semantically close to the target style, but not always tightly grounded in the source article.

Recommended interpretation for reporting:

> On XSum, the model achieves strong semantic similarity to the reference summaries but only moderate lexical overlap and weaker source-grounded faithfulness, suggesting fluent but highly abstractive summary generation.

## Cross-Dataset Takeaway

- GovReport is currently the more faithful setting.
- XSum is currently the more abstractive setting.
- If the project report emphasizes trustworthiness and factual grounding, GovReport results are easier to defend.
- If the report emphasizes semantic similarity to highly abstractive references, XSum results are acceptable but should be discussed with a faithfulness caveat.
