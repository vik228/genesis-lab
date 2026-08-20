# 🌌 Genesis Lab: Core Generative AI Research

A unified, research-grade codebase tracing the full evolution from classical NLP
to modern LLMs, implementing seminal papers from first principles. Each
implementation uses Shape-Driven Development with explicit tensor dimension
tracking.

The point is not to code the papers. It is to sit inside each era's constraints
long enough to feel why the next breakthrough had to happen.

**Currently on:** Word2Vec (Mikolov et al. 2013a).
Findings so far are in [`reports/`](reports/).

## The arc

n-gram models -> neural embeddings (Word2Vec, GloVe) -> subword tokenization
(BPE) -> sequence modeling (LSTM, Seq2Seq, Attention) -> Transformers -> the
pretrain/finetune paradigm (GPT, ELMo, BERT) -> scaling laws (Chinchilla) ->
efficiency (FlashAttention, MoE, LoRA) -> alignment (RLHF, DPO) -> reasoning
(CoT, test-time compute)

## Implemented so far

| paper | year | implementation |
|-------|------|----------------|
| A Neural Probabilistic Language Model (Bengio et al.) | 2003 | [`notebooks/bengio_2003_neural_probabilistic_nl_model.ipynb`](notebooks/bengio_2003_neural_probabilistic_nl_model.ipynb) |
| A Unified Architecture for NLP (Collobert & Weston) | 2008 | [`notebooks/collobert_and_weston_2008.ipynb`](notebooks/collobert_and_weston_2008.ipynb) |
| Efficient Estimation of Word Representations (Mikolov et al.) | 2013 | [`notebooks/mikolov-2013a.ipynb`](notebooks/mikolov-2013a.ipynb) + [`scripts/`](scripts/) |

## Structure

- `notebooks/` - the working implementation for each paper
- `scripts/` - training and evaluation entry points, lifted out of the notebooks
  so long runs do not depend on a live kernel
- `reports/` - findings that go past the papers' own claims
- `papers/<paper_name>/` - scaffolds for papers not started yet
- `common/` - shared utilities (tokenizers, schedulers, metrics, viz)
- `experiments/` - logs, configs, links to W&B runs

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# current paper: Word2Vec
python scripts/train_word2vec.py --kind skipgram --lr 0.025 --out outputs/run_skipgram
python scripts/train_word2vec.py --kind cbow     --lr 0.05  --out outputs/run_cbow

# analogy eval, the paper's own metric
python scripts/eval_word2vec.py outputs/run_skipgram outputs/run_cbow
```
