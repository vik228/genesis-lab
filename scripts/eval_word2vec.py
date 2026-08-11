"""Evaluate trained word2vec checkpoints the way the paper does.

Semantic-Syntactic Word Relationship test set (questions-words.txt):
  vec(b) - vec(a) + vec(c) should land nearest to vec(d), cosine similarity,
  exact match only, with a/b/c excluded from the candidate set. A question is
  skipped if any of its four words is outside the vocabulary.

  python scripts/eval_word2vec.py outputs/run_skipgram outputs/run_cbow
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
PROBES = ["king", "france", "three", "computer", "war", "monday", "biggest", "cat"]


def load_questions(word_to_idx):
    cats, cur = {}, None
    total = 0
    for line in (REPO / "data" / "questions-words.txt").read_text().split("\n"):
        if not line.strip():
            continue
        if line.startswith(":"):
            cur = line[1:].strip()
            cats[cur] = {"qs": [], "skipped": 0}
            continue
        w = line.lower().split()
        if len(w) != 4:
            continue
        total += 1
        if all(x in word_to_idx for x in w):
            cats[cur]["qs"].append([word_to_idx[x] for x in w])
        else:
            cats[cur]["skipped"] += 1
    return cats, total


def analogy(W, cats, device, chunk=512):
    """W must already be L2-normalised rows."""
    out = {}
    for cat, d in cats.items():
        qs = d["qs"]
        if not qs:
            out[cat] = (0, 0, d["skipped"])
            continue
        q = torch.tensor(qs, dtype=torch.long, device=device)
        correct = 0
        for i in range(0, len(q), chunk):
            b = q[i : i + chunk]
            # vec(b) - vec(a) + vec(c), then renormalise so cosine == dot
            v = W[b[:, 1]] - W[b[:, 0]] + W[b[:, 2]]
            v = v / (v.norm(dim=1, keepdim=True) + 1e-8)
            sims = v @ W.T
            # the three input words can never be the answer
            sims.scatter_(1, b[:, :3], -2.0)
            correct += (sims.argmax(dim=1) == b[:, 3]).sum().item()
        out[cat] = (correct, len(qs), d["skipped"])
    return out


def main():
    runs = sys.argv[1:] or ["outputs/run_skipgram", "outputs/run_cbow"]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    word_to_idx = json.loads((REPO / "data" / "w2v_cache.vocab.json").read_text())
    idx_to_word = {i: w for w, i in word_to_idx.items()}
    cats, total_q = load_questions(word_to_idx)
    kept = sum(len(d["qs"]) for d in cats.values())
    print(
        f"analogy set: {total_q} questions, {kept} in-vocab "
        f"({100*kept/total_q:.1f}% coverage at V={len(word_to_idx)})\n"
    )

    results = {}
    for run in runs:
        p = Path(run)
        ck = torch.load(p / "latest.pt", map_location="cpu")
        W = ck["state_dict"]["word_emb.weight"].to(device)
        W = W / (W.norm(dim=1, keepdim=True) + 1e-8)
        name = ck["kind"]
        print(f"=== {name}  (step {ck['step']}, lr {ck['lr']}) ===")

        res = analogy(W, cats, device)
        sem_c = sum(c for k, (c, n, _) in res.items() if k in SEMANTIC)
        sem_n = sum(n for k, (c, n, _) in res.items() if k in SEMANTIC)
        syn_c = sum(c for k, (c, n, _) in res.items() if k not in SEMANTIC)
        syn_n = sum(n for k, (c, n, _) in res.items() if k not in SEMANTIC)
        for cat in sorted(res, key=lambda k: (k not in SEMANTIC, k)):
            c, n, sk = res[cat]
            tag = "sem" if cat in SEMANTIC else "syn"
            acc = f"{100*c/n:5.1f}%" if n else "    -"
            print(f"  [{tag}] {cat:<28} {acc}  ({c}/{n}, {sk} oov)")
        print(f"  {'SEMANTIC':<34} {100*sem_c/max(sem_n,1):5.1f}%  ({sem_c}/{sem_n})")
        print(f"  {'SYNTACTIC':<34} {100*syn_c/max(syn_n,1):5.1f}%  ({syn_c}/{syn_n})")
        print(
            f"  {'TOTAL':<34} {100*(sem_c+syn_c)/max(sem_n+syn_n,1):5.1f}%  "
            f"({sem_c+syn_c}/{sem_n+syn_n})\n"
        )
        results[name] = (sem_c, sem_n, syn_c, syn_n)

        print("  nearest neighbours:")
        for w in PROBES:
            if w not in word_to_idx:
                continue
            sims = W @ W[word_to_idx[w]]
            top = sims.topk(6).indices.tolist()[1:]
            print(f"    {w:<10} " + ", ".join(idx_to_word[i] for i in top))
        print()

        tr = np.load(p / "traces.npz")
        obs, pred = tr["observed"], tr["predicted"]
        print(f"  loss trace ({len(obs)} steps), mean diff per decile:")
        d = obs - pred
        for i in range(10):
            lo, hi = i * len(d) // 10, (i + 1) * len(d) // 10
            print(
                f"    {10*i:>3}-{10*(i+1):>3}%  diff {d[lo:hi].mean():+.4f}   "
                f"obs {obs[lo:hi].mean():.4f}"
            )
        print()

    if len(results) == 2:
        print("=== head to head (total analogy accuracy) ===")
        for name, (sc, sn, yc, yn) in results.items():
            print(
                f"  {name:<10} sem {100*sc/max(sn,1):5.1f}%   "
                f"syn {100*yc/max(yn,1):5.1f}%   "
                f"total {100*(sc+yc)/max(sn+yn,1):5.1f}%"
            )


if __name__ == "__main__":
    main()
