"""Standalone word2vec trainer - skip-gram or CBOW with a hierarchical-softmax head.

Lifted from notebooks/mikolov-2013a.ipynb so a full-epoch run does not depend on a
live Jupyter kernel. Same vocab, same Huffman tree, same modules, same masked BCE.

  python scripts/train_word2vec.py --kind skipgram --lr 4 --batch-size 16
  python scripts/train_word2vec.py --kind cbow --lr 8 --batch-size 16

Vocab, stream and tree are cached to data/w2v_cache.npz on the first run, so
later runs start training in seconds instead of rebuilding the tree.
"""

import argparse
import heapq
import itertools
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, IterableDataset

REPO = Path(__file__).resolve().parent.parent
LN2 = float(np.log(2))
MIN_COUNT = 5


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- vocab + tree
class Node:
    def __init__(self, freq, word=None, left=None, right=None, node_id=None):
        self.freq, self.word = freq, word
        self.left, self.right, self.node_id = left, right, node_id

    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman_tree(counts, vocab_size):
    heap = [Node(count, word=idx) for idx, count in enumerate(counts)]
    heapq.heapify(heap)
    node_id = 0
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        heapq.heappush(
            heap, Node(left.freq + right.freq, left=left, right=right, node_id=node_id)
        )
        node_id += 1
    assert heap[0].node_id == vocab_size - 2
    return heap[0]


def dfs(root, current_path, path, code, current_path_codes):
    """Iterative DFS - the notebook's recursive version can approach the
    recursion limit at depth 22 x V, and this is the only difference."""
    stack = [(root, 0)]
    cur_path, cur_code = [], []
    while stack:
        node, state = stack.pop()
        if state == 0:
            if node.left is None and node.right is None:
                path[node.word] = list(cur_path)
                code[node.word] = list(cur_code)
                continue
            cur_path.append(node.node_id)
            cur_code.append(0)
            stack.append((node, 1))
            stack.append((node.left, 0))
        elif state == 1:
            cur_code[-1] = 1
            stack.append((node, 2))
            stack.append((node.right, 0))
        else:
            cur_path.pop()
            cur_code.pop()


def build_cache(cache_path):
    log("building vocab + huffman tree (first run only)")
    raw_tokens = (REPO / "data" / "text8").read_text().split()
    log(f"total tokens {len(raw_tokens)}")
    tokens_dict = Counter(raw_tokens)
    tokens = [t for t, c in tokens_dict.items() if c >= MIN_COUNT]
    word_to_idx = {t: i for i, t in enumerate(tokens)}
    counts = [tokens_dict[t] for t in tokens]
    V = len(tokens)
    log(f"vocab {V}")

    stream = np.fromiter(
        (word_to_idx[t] for t in raw_tokens if t in word_to_idx), dtype=np.int32
    )
    del raw_tokens, tokens_dict
    log(f"stream {len(stream)}")

    root = build_huffman_tree(counts, V)
    paths, codes = [None] * V, [None] * V
    dfs(root, [], paths, codes, [])
    assert all(p is not None for p in paths), "some word got no huffman path"
    max_path_len = max(len(p) for p in paths)
    log(f"huffman depth min {min(len(p) for p in paths)} max {max_path_len}")

    masks = [None] * V
    for i, p in enumerate(paths):
        masks[i] = [1] * len(p) + [0] * (max_path_len - len(p))
        paths[i] = p + [0] * (max_path_len - len(p))
        codes[i] = codes[i] + [0] * (max_path_len - len(codes[i]))

    paths = np.array(paths, dtype=np.int32)
    codes = np.array(codes, dtype=np.float32)
    masks = np.array(masks, dtype=np.float32)

    # sanity: frequency-weighted average depth should be near the unigram entropy
    freq = np.array(counts, dtype=np.float64)
    avg_depth = float((masks.sum(axis=1) * freq).sum() / freq.sum())
    p = freq / freq.sum()
    entropy_bits = float(-(p * np.log2(p)).sum())
    log(
        f"freq-weighted avg depth {avg_depth:.4f}  unigram entropy {entropy_bits:.4f} bits"
    )
    assert avg_depth >= entropy_bits, "huffman beat the entropy bound - tree is wrong"
    log(f"expected init loss {avg_depth * LN2:.4f}")

    np.savez_compressed(
        cache_path,
        stream=stream,
        paths=paths,
        codes=codes,
        masks=masks,
        counts=np.array(counts, dtype=np.int64),
    )
    (cache_path.with_suffix(".vocab.json")).write_text(json.dumps(word_to_idx))
    log(f"cached to {cache_path}")


def load_cache():
    cache_path = REPO / "data" / "w2v_cache.npz"
    if not cache_path.exists():
        build_cache(cache_path)
    z = np.load(cache_path)
    word_to_idx = json.loads((cache_path.with_suffix(".vocab.json")).read_text())
    return (
        z["stream"],
        torch.from_numpy(z["paths"]),
        torch.from_numpy(z["codes"]),
        torch.from_numpy(z["masks"]),
        word_to_idx,
    )


# -------------------------------------------------------------------- datasets
class SkipGramDataset(IterableDataset):
    def __init__(self, stream, max_window_size):
        self.stream = stream
        self.max_window_size = max_window_size

    def __iter__(self):
        n = len(self.stream)
        for center_pos in range(n):
            radius = random.randint(1, self.max_window_size)
            left = max(0, center_pos - radius)
            right = min(n, center_pos + radius + 1)
            for context_pos in range(left, right):
                if center_pos == context_pos:
                    continue
                yield (self.stream[center_pos], self.stream[context_pos])


class CBOWDataset(IterableDataset):
    def __init__(self, stream, max_window_size):
        self.stream = stream
        self.max_window_size = max_window_size
        self.max_context = 2 * max_window_size

    def __iter__(self):
        n = len(self.stream)
        for center_pos in range(n):
            radius = random.randint(1, self.max_window_size)
            left = max(0, center_pos - radius)
            right = min(n, center_pos + radius + 1)
            # order does not matter for CBOW (the mean destroys it), so both
            # sides are concatenated and the row is padded only at the end,
            # which is what keeps context_mask aligned
            context = np.concatenate(
                [self.stream[left:center_pos], self.stream[center_pos + 1 : right]]
            )
            n_real = len(context)
            pad = self.max_context - n_real
            yield (
                np.concatenate([context, np.zeros(pad, dtype=self.stream.dtype)]),
                self.stream[center_pos],
                np.concatenate(
                    [np.ones(n_real, dtype=np.float32), np.zeros(pad, dtype=np.float32)]
                ),
            )


# ---------------------------------------------------------------------- models
class HSHead(nn.Module):
    """Shared hierarchical-softmax head. Both models reduce to: build an (B, D)
    projection h, then score it against the target word's huffman path."""

    def __init__(self, vocab_size, emb_dim, paths, codes, masks, max_exp=0.0):
        super().__init__()
        self.vocab_size, self.emb_dim = vocab_size, emb_dim
        # word2vec.c skips any node whose logit falls outside [-MAX_EXP, MAX_EXP]
        # (MAX_EXP = 6). torch.clamp reproduces that exactly: the value is
        # bounded AND the gradient is zero outside the range, so an out-of-range
        # node contributes no update to either word_emb or node_emb. 0 = off.
        self.max_exp = float(max_exp)
        self.word_emb = nn.Embedding(vocab_size, emb_dim)
        self.node_emb = nn.Embedding(vocab_size - 1, emb_dim)
        bound = 0.5 / emb_dim
        nn.init.uniform_(self.word_emb.weight, -bound, bound)
        nn.init.zeros_(self.node_emb.weight)
        self.register_buffer("paths", paths)
        self.register_buffer("codes", codes)
        self.register_buffer("masks", masks)

    def score(self, h, targets):
        node_vecs = self.node_emb(self.paths[targets])
        logits = (node_vecs * h.unsqueeze(1)).sum(dim=-1)
        code = self.codes[targets]
        mask = self.masks[targets]
        assert logits.shape == code.shape == mask.shape
        if self.max_exp:
            logits = logits.clamp(-self.max_exp, self.max_exp)
        # mask before BCE too: keeps a padded slot from ever producing inf,
        # since inf * 0.0 is nan, not 0.0
        logits = logits * mask
        per_node = F.binary_cross_entropy_with_logits(logits, code, reduction="none")
        return (per_node * mask).sum(dim=1).mean()


class SkipGram(HSHead):
    def forward(self, centers, targets):
        centers, targets = centers.long(), targets.long()
        h = self.word_emb(centers)
        assert h.shape == (centers.shape[0], self.emb_dim)
        return self.score(h, targets)


class CBOW(HSHead):
    def __init__(
        self, vocab_size, emb_dim, paths, codes, masks, max_window_size, max_exp=0.0
    ):
        super().__init__(vocab_size, emb_dim, paths, codes, masks, max_exp)
        self.input_len = 2 * max_window_size

    def forward(self, contexts, targets, context_mask):
        # `targets` here are the CENTER words - same head, opposite direction
        contexts, targets = contexts.long(), targets.long()
        batch = contexts.shape[0]
        context_emb = self.word_emb(contexts)
        assert context_emb.shape == (batch, self.input_len, self.emb_dim)
        # masked mean: numerator drops padded slots, denominator counts only
        # the real ones. keepdim=True so (batch, 1) broadcasts over D.
        h = (context_emb * context_mask.unsqueeze(-1)).sum(dim=1) / context_mask.sum(
            dim=1, keepdim=True
        )
        assert h.shape == (batch, self.emb_dim)
        return self.score(h, targets)


# -------------------------------------------------------------------- training
PROBES = ["king", "france", "three", "computer", "war"]


def neighbours(model, word_to_idx, idx_to_word, k=5):
    with torch.no_grad():
        W = model.word_emb.weight.detach()
        W = W / (W.norm(dim=1, keepdim=True) + 1e-8)
        out = []
        for w in PROBES:
            if w not in word_to_idx:
                continue
            sims = W @ W[word_to_idx[w]]
            top = sims.topk(k + 1).indices.tolist()[1:]
            out.append(f"{w}: " + ", ".join(idx_to_word[i] for i in top))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["skipgram", "cbow"], required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--emb-dim", type=int, default=100)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = one full epoch")
    ap.add_argument("--checkpoint-every", type=int, default=50_000)
    ap.add_argument("--log-every", type=int, default=5_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-decay", action="store_true")
    ap.add_argument(
        "--subsample",
        type=float,
        default=0.0,
        help="word2vec's frequent-word subsampling threshold t (1e-5 to match "
        "the paper, 0 = off). Applied to the training stream only - the vocab "
        "and the huffman tree stay on the original counts.",
    )
    ap.add_argument(
        "--max-exp",
        type=float,
        default=0.0,
        help="word2vec.c MAX_EXP logit cutoff (6.0 to match it, 0 = off)",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="pick up from --out's latest.pt: restores the weights, the step "
        "counter and the lr schedule, and keeps going to --max-steps. The "
        "remaining passes redraw their own subsample, so the token order "
        "differs from an uninterrupted run - the update count and the lr "
        "curve do not.",
    )
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    out = (
        Path(args.out)
        if args.out
        else REPO
        / "outputs"
        / (
            f"w2v_{args.kind}_lr{args.lr}_B{args.batch_size}_{time.strftime('%m%d-%H%M')}"
        )
    )
    out.mkdir(parents=True, exist_ok=True)
    log(f"device {device}  out {out}")
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))

    stream, paths, codes, masks, word_to_idx = load_cache()
    idx_to_word = {i: w for w, i in word_to_idx.items()}
    V = len(word_to_idx)
    n = len(stream)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.kind == "skipgram":
        model = SkipGram(V, args.emb_dim, paths, codes, masks, args.max_exp)
        # E[pairs per center] = E[2*radius] = 2*mean(1..window); an estimate,
        # used only to shape the lr decay
        examples = n * (args.window + 1)
    else:
        model = CBOW(V, args.emb_dim, paths, codes, masks, args.window, args.max_exp)
        examples = n
    model.to(device)

    # Subsampling keeps each occurrence with probability sqrt(t/f), so the
    # frequency that survives is sqrt(t*f) - frequent words are thinned hard,
    # rare ones untouched. Drawn per token per pass rather than once, because
    # word2vec subsamples on the fly while reading: every pass keeps a
    # different set of `the`s, so repeated passes are not repeated data.
    keep_p = None
    if args.subsample > 0:
        freq = np.bincount(stream, minlength=V) / n
        keep_p = np.minimum(1.0, np.sqrt(args.subsample / np.maximum(freq, 1e-12)))
        examples = int(examples * float((freq * keep_p).sum()))
    sub_rng = np.random.default_rng(args.seed)

    def epoch_loader():
        s = stream
        if keep_p is not None:
            s = stream[sub_rng.random(n, dtype=np.float32) < keep_p[stream]]
            log(f"subsampled stream {len(s)} of {n} ({100*len(s)/n:.1f}%)")
        cls = SkipGramDataset if args.kind == "skipgram" else CBOWDataset
        return DataLoader(cls(s, args.window), batch_size=args.batch_size)

    steps_per_epoch = examples // args.batch_size
    total_steps = args.max_steps or steps_per_epoch * args.epochs
    log(
        f"{args.kind} V={V} examples/epoch~{examples} B={args.batch_size} "
        f"total_steps~{total_steps}"
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    # word2vec's linear decay, floored so the tail still moves
    scheduler = (
        None
        if args.no_decay
        else LambdaLR(optimizer, lambda s: max(1e-4, 1 - s / total_steps))
    )

    observed, predicted = [], []
    step = 0
    win_obs = win_diff = 0.0
    win_n = 0

    if args.resume:
        ckpt = torch.load(out / "latest.pt", map_location=device)
        if ckpt["kind"] != args.kind:
            sys.exit(f"checkpoint is {ckpt['kind']}, not {args.kind}")
        want = {"word_emb.weight", "node_emb.weight"}
        if want - set(ckpt["state_dict"]):
            sys.exit(f"checkpoint is missing {want - set(ckpt['state_dict'])}")
        # strict=False because dump() saves only the two embedding tables;
        # the huffman path/code/mask buffers are rebuilt from the cache
        model.load_state_dict(ckpt["state_dict"], strict=False)
        step = int(ckpt["step"])
        if scheduler is not None:
            # LambdaLR reads last_epoch, so fast-forward it rather than
            # calling step() four million times
            scheduler.last_epoch = step - 1
            scheduler.step()
        tr = np.load(out / "traces.npz")
        observed, predicted = list(tr["observed"]), list(tr["predicted"])
        log(
            f"resumed from step {step} ({100*step/total_steps:.1f}%), "
            f"lr {optimizer.param_groups[0]['lr']:.4f}, "
            f"{len(observed)} trace points"
        )

    start_step = step
    t0 = time.perf_counter()

    def dump(tag):
        np.savez_compressed(
            out / "traces.npz",
            observed=np.array(observed, dtype=np.float32),
            predicted=np.array(predicted, dtype=np.float32),
        )
        torch.save(
            {
                "step": step,
                "kind": args.kind,
                "lr": args.lr,
                "state_dict": {
                    k: v.cpu()
                    for k, v in model.state_dict().items()
                    if k in ("word_emb.weight", "node_emb.weight")
                },
            },
            out / "latest.pt",
        )
        log(f"checkpoint {tag} step {step}")
        for line in neighbours(model, word_to_idx, idx_to_word):
            log(f"  nn  {line}")

    # a resumed run has already burnt some of its epochs, and how many is not
    # recoverable from the checkpoint, so let it draw fresh passes until the
    # step budget is spent - --max-steps is the real bound either way
    epochs = itertools.count() if args.resume else range(args.epochs)

    try:
        for epoch in epochs:
            if step >= total_steps:
                break
            for batch in epoch_loader():
                if step >= total_steps:
                    break
                if args.kind == "skipgram":
                    centers, contexts = batch
                    hs_target = contexts.long()
                    fwd = (centers.long().to(device), hs_target.to(device))
                else:
                    contexts, centers, cmask = batch
                    hs_target = centers.long()
                    fwd = (
                        contexts.long().to(device),
                        hs_target.to(device),
                        cmask.to(device),
                    )

                # zero-learning baseline: this batch's avg huffman depth * ln2
                pred = masks[hs_target].sum(dim=1).mean().item() * LN2
                loss = model(*fwd)
                lv = loss.item()

                # Waiting for inf is far too late - a diverging run sits at
                # finite-but-absurd losses (1e26) for tens of thousands of
                # steps first. Trip on any loss well above the zero-learning
                # baseline: real training only ever goes below it.
                if not math.isfinite(lv) or lv > 5 * pred:
                    log(
                        f"ABORT: loss {lv:.4g} vs baseline {pred:.4f} at step "
                        f"{step}. lr={args.lr} diverged - lower it, or enable "
                        f"--max-exp 6."
                    )
                    dump("pre-abort")
                    sys.exit(2)

                observed.append(lv)
                predicted.append(pred)
                win_obs += lv
                win_diff += lv - pred
                win_n += 1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                step += 1

                if step % args.log_every == 0:
                    el = time.perf_counter() - t0
                    rate = (step - start_step) / el
                    eta = (total_steps - step) / rate / 3600
                    md = win_diff / win_n
                    with torch.no_grad():
                        wn = model.word_emb.weight.norm(dim=1).mean().item()
                        nn_ = model.node_emb.weight.norm(dim=1).mean().item()
                    # a positive window diff means the model is now doing worse
                    # than predicting nothing - it precedes the actual overflow
                    # by tens of thousands of steps and is the signal to act on
                    warn = (
                        "  <<< DIFF POSITIVE, heading for divergence" if md > 0 else ""
                    )
                    log(
                        f"step {step}/{total_steps} ({100*step/total_steps:.1f}%)  "
                        f"obs {win_obs/win_n:.4f}  diff {md:+.4f}  "
                        f"lr {optimizer.param_groups[0]['lr']:.4f}  "
                        f"|w| {wn:.3g} |n| {nn_:.3g}  "
                        f"{rate:.0f} st/s  eta {eta:.2f}h{warn}"
                    )
                    win_obs = win_diff = 0.0
                    win_n = 0
                if step % args.checkpoint_every == 0:
                    dump("periodic")
    except KeyboardInterrupt:
        log("interrupted")

    dump("final")
    log(f"done in {(time.perf_counter()-t0)/3600:.2f}h")


if __name__ == "__main__":
    main()
