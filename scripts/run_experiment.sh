#!/usr/bin/env bash
# Current paper: Word2Vec (Mikolov et al. 2013a).
# Trains both models and runs the paper's own analogy eval on the pair.
set -euo pipefail

cd "$(dirname "$0")/.."

python scripts/train_word2vec.py --kind skipgram --lr 0.025 --out outputs/run_skipgram
python scripts/train_word2vec.py --kind cbow     --lr 0.05  --out outputs/run_cbow

python scripts/eval_word2vec.py outputs/run_skipgram outputs/run_cbow
