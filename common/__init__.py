"""
Shared, dependency-light helpers used by the training scripts in this repository.

Everything in this package is *pure* Python (plus NumPy / scikit-learn / jiwer
for a couple of metrics): no GPU, no model downloads, no network access. That
keeps the logic that decides every reported number - SQL execution matching,
reward functions, retrieval metrics, answer parsers, prompt builders - small,
readable and covered by the unit tests in `tests/`.

The heavy lifting (loading models, training, generation) stays in each
project's `train.py`.

Modules
-------
sql               text-to-SQL prompt, SQL clean-up and execution-based matching
reasoning         GSM8K answer parsing and the GRPO reward functions
function_calling  glaive data parser, tool-call parser and agent metrics
retrieval         IR metrics (nDCG@10, Recall@k, MRR@10) and the BM25 tokenizer
chartqa           ChartQA prompt and the official "relaxed accuracy" match
speech            word / character error rate
classification    binary-classifier metrics for the guardrail
lm_data           loss masking and padding for causal-LM fine-tuning
"""
