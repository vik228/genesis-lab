# 🌌 Genesis Lab: Core Generative AI Research

A unified, research-grade codebase tracing the full evolution from classical NLP to modern LLMs — implementing seminal papers from first principles. Covers n-gram models → neural embeddings (Word2Vec, GloVe) → subword tokenization (BPE) → sequence modeling (LSTM, Seq2Seq, Attention) → Transformers → the pretrain→finetune paradigm (GPT, ELMo, BERT) → scaling laws (Chinchilla) → efficiency (FlashAttention, MoE, LoRA) → alignment (RLHF, DPO) → reasoning (CoT, test-time compute). Each implementation uses Shape-Driven Development with explicit tensor dimension tracking.

## Structure
- `papers/<paper_name>/` – Self-contained implementations per paper
- `common/` – Shared utilities (tokenizers, schedulers, metrics, viz)
- `notebooks/` – Exploratory analyses and visualizations
- `experiments/` – Logs + configs + links to W&B runs
- `reports/` – Paper-style notes and results

## Quickstart
```bash
git init
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# first paper: Attention Is All You Need
python papers/attention_is_all_you_need/src/train_mt.py --config papers/attention_is_all_you_need/configs/transformer_base_iwslt14.yml
```
