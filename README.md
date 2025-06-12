# eval_llms

**Single-file LLM evaluation tool**

This script provides a minimal, no-setup-needed way to evaluate multiple LLM predictions against reference answers using a variety of popular automatic metrics.

It auto-installs required dependencies on the first run, skips missing metrics gracefully, and outputs both a CSV file and plots summarizing performance.

Select mames for the models and the predictions according to you csv file.
---

## Metrics Implemented

| Metric                | Range     | Measures (↑ = better)                                      |
|-----------------------|-----------|-------------------------------------------------------------|
| **BLEU-1 / BLEU-4**   | 0–1       | Word-level and up-to-4-gram precision                      |
| **METEOR**            | 0–1       | Stem/synonym overlap (NLTK-based)                          |
| **ROUGE-L**           | 0–1       | Longest common subsequence (recall-focused)               |
| **CIDEr**             | 0–10      | TF-IDF weighted n-gram consensus (captioning benchmark)   |
| **CosineSim (Nomic)** | -1–1      | Embedding similarity using `nomic-ai/nomic-embed-text-v1` |

---

## Quick Start

```bash
python evaluate.py
