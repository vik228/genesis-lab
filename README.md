# 🌌 Genesis Lab: Core Generative AI Research

A unified, research-grade codebase tracing the full evolution from classical NLP
to modern LLMs, implementing seminal papers from first principles. Each
implementation uses Shape-Driven Development with explicit tensor dimension
tracking.

The point is not to code the papers. It is to sit inside each era's constraints
long enough to feel why the next breakthrough had to happen.

**Currently on:** Attention Is All You Need (Vaswani et al. 2017). Bahdanau attention, BPE and LayerNorm are done. On WMT14 En-Fr newstest2014, our RNNsearch-50 gets 18.76 BLEU with beam 12 (paper: 26.75, on a much larger corpus), and the BPE version beats the word-level one by +0.9 BLEU. Deeper analysis of both (rare-word F1, BLEU by sentence length) is paused, to be continued later.
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
| GloVe: Global Vectors for Word Representation (Pennington et al.) | 2014 | [`notebooks/GloVe.ipynb`](notebooks/GloVe.ipynb) |
| Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau et al.) | 2014 | [`notebooks/bahdanau-2014.ipynb`](notebooks/bahdanau-2014.ipynb) + [`scripts/train_bahdanau.py`](scripts/train_bahdanau.py) |
| Neural Machine Translation of Rare Words with Subword Units (Sennrich et al.) | 2016 | [`notebooks/sennrich-2016-bpe.ipynb`](notebooks/sennrich-2016-bpe.ipynb) + [`scripts/train_bpe_nmt.py`](scripts/train_bpe_nmt.py) |
| Layer Normalization (Ba, Kiros & Hinton) | 2016 | [`notebooks/ba-2016-layernorm.ipynb`](notebooks/ba-2016-layernorm.ipynb) |

## Structure

- `notebooks/` - the working implementation for each paper
- `scripts/` - training and evaluation entry points, lifted out of the notebooks
  so long runs do not depend on a live kernel
- `reports/` - findings that go past the papers' own claims
- `artifacts/` - source HTML of the per-paper revision pages
- `papers/<paper_name>/` - scaffolds for papers not started yet
- `common/` - shared utilities (tokenizers, schedulers, metrics, viz)
- `experiments/` - logs, configs, links to W&B runs

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# example: Word2Vec
python scripts/train_word2vec.py --kind skipgram --lr 0.025 --out outputs/run_skipgram
python scripts/train_word2vec.py --kind cbow     --lr 0.05  --out outputs/run_cbow

# analogy eval, the paper's own metric
python scripts/eval_word2vec.py outputs/run_skipgram outputs/run_cbow
```
