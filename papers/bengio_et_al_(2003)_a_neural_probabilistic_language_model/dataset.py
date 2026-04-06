"""
papers/bengio_et_al_(2003)_a_neural_probabilistic_language_model/dataset.py

Expectation:
Implement the data preprocessing logic specific to this paper/experiment.

    - Extraction: Load the Wikitext raw data (e.g. from huggingface `datasets`).
    - Transformation: Clean text, loop through characters/lines, filter out headers.
    - Vocabulary mapping: Build `word_to_idx`, `idx_to_word`, set up frequency threshold to map rare words to `<UNK>`, append `<START>` and `<END>`.
    - Dataset builder: Slide a window over tokenized sequences to construct `X` (context sequences of length n) and `Y` (target labels). Return them as raw `torch.Tensor`s.
"""
