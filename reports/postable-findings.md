# Postable findings

Running list of results worth writing up and eventually sharing. A finding lands
here the moment it turns up, not when it feels good enough. The point of the list
is to have something to compare against later.

Status values: `candidate` (noted, no writeup yet), `written-up` (has its own
file in this directory), `shared` (posted, with a link back).

---

## Word2Vec, Mikolov et al. 2013a

Implementation: [`notebooks/mikolov-2013a.ipynb`](../notebooks/mikolov-2013a.ipynb),
[`scripts/train_word2vec.py`](../scripts/train_word2vec.py),
[`scripts/eval_word2vec.py`](../scripts/eval_word2vec.py)

### F1 - The CBOW/skip-gram gap is almost entirely a named-entity gap

**Status:** candidate

Skip-gram beats CBOW by **+11.8 on the named-entity half** of the analogy set and
by **+0.0 on everything else**. The aggregate number the paper reports hides
this: it reads as a general advantage when it is one category.

Writeup needs: the per-category breakdown, and the training setup both models
shared.

### F2 - Raw neighbour entropy measures English, not the word

**Status:** candidate

The obvious way to measure a word's context diversity - pooled neighbour entropy
over its windows - does not work on raw text. The top-20 neighbours came back
roughly **19 out of 20 identical to the corpus top-20** for any word, because
stopwords dominate every window. The statistic was measuring the language.

After dropping the **top-100 words** and re-windowing, it separates cleanly:
`athens` (topical) sits low, `walking` (generic) is near-flat.

This one generalises past Word2Vec. Any pooled-count statistic over natural text
has the same failure mode.

Writeup needs: the before/after neighbour lists, and the entropy numbers for both
words in both conditions.

### F3 - Subsampling drops CBOW on the common half, as predicted

**Status:** candidate

Interventional test, not observational. Subsampling with
`keep_p = min(1, sqrt(t/f))` at `t=1e-5` leaves **27.9%** of the stream. Steps
matched across models (skip-gram 6,269,566 / CBOW 1,044,927 = 3.58 passes), with
the pass criterion **registered before the run**. Break-even update count works
out to about **2,146** from roughly `6 * count(w)` updates per word.

CBOW dropped on the common half exactly as the prediction said it would.

Writeup needs: the pre-registered criterion as written, and the per-half numbers.
Verdict analysis is still open by choice.

---

## Collobert & Weston 2008

Implementation: [`notebooks/collobert_and_weston_2008.ipynb`](../notebooks/collobert_and_weston_2008.ipynb)

No candidates recorded yet. The multitask-versus-single-task claim was left
untested when the paper was parked.

## Bengio et al. 2003

Implementation: [`notebooks/bengio_2003_neural_probabilistic_nl_model.ipynb`](../notebooks/bengio_2003_neural_probabilistic_nl_model.ipynb)

No candidates recorded yet.
