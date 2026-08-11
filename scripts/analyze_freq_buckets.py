"""Test whether the skip-gram / CBOW analogy gap is driven by word frequency.

Hypothesis (2026-08-11): skip-gram wins on rare words, CBOW on frequent ones,
and the apparent semantic-vs-syntactic story is a proxy for that. Every analogy
question is scored per-model, then bucketed by the training frequency of the
words it involves.

Bottleneck statistic: min(count) over the four words. A question can only be
answered if all four vectors are good, so the rarest word is the binding one.
Buckets are frequency quantiles, so every bucket holds the same number of
questions and their error bars are comparable.

Control: frequency and proper-noun-ness are correlated in this test set (the
capital/city/currency/nationality categories are both rare and proper nouns).
If the effect is real it must survive inside the purely grammatical categories,
so the same bucketing is repeated over syntactic-minus-gram6.

  python scripts/analyze_freq_buckets.py outputs/run_skipgram outputs/run_cbow
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
SEMANTIC = {
    "capital-common-countries",
    "capital-world",
    "currency",
    "city-in-state",
    "family",
}
# the one syntactic category built from proper nouns (Albania -> Albanian)
PROPER_NOUN_SYN = "gram6-nationality-adjective"
# categories whose words are named entities, regardless of the sem/syn label
NAMED_ENTITY = {
    "capital-common-countries",
    "capital-world",
    "city-in-state",
    "currency",
    PROPER_NOUN_SYN,
}
N_BUCKETS = 10


def load_questions(word_to_idx):
    """Flat list of in-vocab questions, keeping each one's category."""
    qs, cats, cur = [], [], None
    for line in (REPO / "data" / "questions-words.txt").read_text().split("\n"):
        if not line.strip():
            continue
        if line.startswith(":"):
            cur = line[1:].strip()
            continue
        w = line.lower().split()
        if len(w) != 4:
            continue
        if all(x in word_to_idx for x in w):
            qs.append([word_to_idx[x] for x in w])
            cats.append(cur)
    return np.array(qs, dtype=np.int64), np.array(cats)


def score(W, q, device, chunk=512):
    """Per-question correctness. W must already be L2-normalised rows."""
    t = torch.tensor(q, dtype=torch.long, device=device)
    hits = torch.empty(len(t), dtype=torch.bool, device=device)
    for i in range(0, len(t), chunk):
        b = t[i : i + chunk]
        v = W[b[:, 1]] - W[b[:, 0]] + W[b[:, 2]]
        v = v / (v.norm(dim=1, keepdim=True) + 1e-8)
        sims = v @ W.T
        sims.scatter_(1, b[:, :3], -2.0)
        hits[i : i + chunk] = sims.argmax(dim=1) == b[:, 3]
    return hits.cpu().numpy()


def quantile_buckets(stat, n_buckets):
    """Assign each question to an equal-count bucket by `stat` (ties kept together)."""
    order = np.argsort(stat, kind="stable")
    ranks = np.empty(len(stat), dtype=np.int64)
    ranks[order] = np.arange(len(stat))
    return np.minimum(ranks * n_buckets // len(stat), n_buckets - 1)


def se(c, n):
    """Binomial standard error of a proportion, in percentage points."""
    if n == 0:
        return 0.0
    p = c / n
    return 100 * np.sqrt(max(p * (1 - p), 1e-12) / n)


def report(title, stat, hits, names, mask=None):
    if mask is None:
        mask = np.ones(len(stat), dtype=bool)
    s, h = stat[mask], {k: v[mask] for k, v in hits.items()}
    n_total = mask.sum()
    if n_total < N_BUCKETS * 10:
        print(f"\n{title}: only {n_total} questions, skipping\n")
        return
    b = quantile_buckets(s, N_BUCKETS)

    print(f"\n{title}  ({n_total} questions, bucketed by min-count quantile)")
    head = (
        "  bucket  min-count range        n  "
        + "".join(f"{k:>18}" for k in names)
        + "        gap"
    )
    print(head)
    print("  " + "-" * (len(head) - 2))
    for i in range(N_BUCKETS):
        m = b == i
        n = int(m.sum())
        lo, hi = int(s[m].min()), int(s[m].max())
        cells, accs = "", []
        for k in names:
            c = int(h[k][m].sum())
            accs.append(100 * c / n)
            cells += f"{100*c/n:11.1f}% +-{se(c, n):3.1f}"
        gap = accs[0] - accs[1]
        print(f"  {i+1:>4}    {lo:>7} - {hi:<7} {n:>6}  {cells}  {gap:+8.1f}")

    print("  " + "-" * (len(head) - 2))
    cells = ""
    for k in names:
        c = int(h[k].sum())
        cells += f"{100*c/n_total:11.1f}% +-{se(c, n_total):3.1f}"
    accs = [100 * h[k].sum() / n_total for k in names]
    print(
        f"  all     {int(s.min()):>7} - {int(s.max()):<7} {n_total:>6}  "
        f"{cells}  {accs[0]-accs[1]:+8.1f}"
    )


def main():
    runs = sys.argv[1:] or ["outputs/run_skipgram", "outputs/run_cbow"]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    word_to_idx = json.loads((REPO / "data" / "w2v_cache.vocab.json").read_text())
    counts = np.load(REPO / "data" / "w2v_cache.npz")["counts"]

    q, cats = load_questions(word_to_idx)
    print(f"{len(q)} in-vocab questions, V={len(word_to_idx)}")

    hits, names = {}, []
    for run in runs:
        ck = torch.load(Path(run) / "latest.pt", map_location="cpu")
        W = ck["state_dict"]["word_emb.weight"].to(device)
        W = W / (W.norm(dim=1, keepdim=True) + 1e-8)
        name = ck["kind"]
        names.append(name)
        hits[name] = score(W, q, device)
        # cross-check against the known eval.txt totals
        print(
            f"  {name:<10} {hits[name].sum()}/{len(q)} "
            f"= {100*hits[name].mean():.1f}% total"
        )

    q_counts = counts[q]  # (Q, 4)
    min_count = q_counts.min(axis=1)  # bottleneck word
    tgt_count = q_counts[:, 3]  # the word that must be retrieved

    is_sem = np.isin(cats, list(SEMANTIC))
    is_ne = np.isin(cats, list(NAMED_ENTITY))

    # the split the bucket test actually points at: named entities vs the rest
    print("\nnamed-entity vs common-word split (cuts across sem/syn):")
    for label, m in [("named-entity cats", is_ne), ("everything else", ~is_ne)]:
        n = int(m.sum())
        cells = ""
        for k in names:
            c = int(hits[k][m].sum())
            cells += f"{100*c/n:11.1f}% +-{se(c, n):3.1f}"
        accs = [100 * hits[k][m].mean() for k in names]
        print(f"  {label:<20} n {n:>6}  {cells}  gap {accs[0]-accs[1]:+6.1f}")

    report("ALL QUESTIONS", min_count, hits, names)
    report("SEMANTIC ONLY", min_count, hits, names, mask=is_sem)
    report("SYNTACTIC ONLY", min_count, hits, names, mask=~is_sem)
    report(
        "SYNTACTIC, EXCLUDING gram6-nationality-adjective (the control)",
        min_count,
        hits,
        names,
        mask=~is_sem & (cats != PROPER_NOUN_SYN),
    )
    report(
        "ALL QUESTIONS, bucketed by TARGET-word count instead", tgt_count, hits, names
    )
    report(
        "NON-NAMED-ENTITY ONLY (frequency re-tested on the clean group)",
        min_count,
        hits,
        names,
        mask=~is_ne,
    )

    print("\nper-category median min-count (why the categories differ):")
    for c in sorted(set(cats.tolist()), key=lambda k: (k not in SEMANTIC, k)):
        m = cats == c
        tag = "sem" if c in SEMANTIC else "syn"
        cells = "".join(f"{100*hits[k][m].mean():8.1f}%" for k in names)
        print(
            f"  [{tag}] {c:<28} median {int(np.median(min_count[m])):>7}"
            f"   n {int(m.sum()):>5} {cells}"
        )


if __name__ == "__main__":
    main()
