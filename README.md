# 🌌 Genesis Lab: Core Generative AI Research

A unified, research-grade codebase to **implement, reproduce, and extend** seminal Generative‑AI papers.
Start here with *Attention Is All You Need* (2017), then grow toward GPT‑2, BERT, LoRA, QLoRA, MoE, and Diffusion.

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
