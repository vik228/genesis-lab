"""Bahdanau, Cho & Bengio (2014) trainer - RNNsearch / RNNencdec on WMT14 En-Fr.

Blocks 1-4 of notebooks/bahdanau-2014.ipynb (setup, data, model, training),
copied as-is, so a long run does not depend on a live Jupyter kernel. Run it on
the Colab VM with nohup; the Checkpointer resumes from Drive when the same
--experiment-name is reused. run_config is built the same way as in the
notebook's run_experiment, so a run started there can resume here.

  nohup python scripts/train_bahdanau.py --experiment-name bahdanau_wmt14_search50 --use-attention > search50.out 2>&1 &
  nohup python scripts/train_bahdanau.py --experiment-name bahdanau_wmt14_encdec50 > encdec50.out 2>&1 &
  python scripts/train_bahdanau.py --experiment-name bahdanau_wmt14_timing_attn --use-attention --n-train 17800 --epochs 1 --profile
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


# ============================================================================
# 1. Setup
# ============================================================================
import random
from collections import Counter
from typing import Any

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader

from common.utils.trainer import Trainer

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
seed = 42
random.seed(seed)
torch.manual_seed(seed)
print(f"Using device: {device}")

TOK_UNK = "<UNK>"
TOK_EOS = "<EOS>"
TOK_SOS = "<SOS>"
TOK_PAD = "<PAD>"
TOK_UNK_ID = 1
TOK_EOS_ID = 3
TOK_SOS_ID = 2
TOK_PAD_ID = 0


# ============================================================================
# 2. Data
# ============================================================================
# eos-data: HF WMT14 fr-en
# Pipeline: random sample -> Moses tokenize -> length filter -> cache.
# Colab: full download to VM disk (lost with the VM), tokenized cache on Drive
# (survives). Mac: streaming slice only, never the full download.
# Local DATA_DIR is relative to where nvim was launched; open nvim at the project root.
import importlib.util
import os
import subprocess
import sys

if importlib.util.find_spec("sacremoses") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "sacremoses"])

from datasets import load_dataset
from sacremoses import MosesTokenizer

try:
    import google.colab  # noqa: F401

    ON_COLAB = True
    HF_CACHE = "/root/colab-data/hf"
    CACHE_DIR = "/content/drive/MyDrive/genesis-lab/data/wmt14_fr_en"
except ImportError:
    ON_COLAB = False
    HF_CACHE = None
    CACHE_DIR = "data/wmt14_fr_en"

# pairs drawn before the length filter; default for load_hf_wmt14_data(n_sample=...)
N_SAMPLE = 2_000_000 if ON_COLAB else 2_000
MAX_LEN = 50  # RNN*-50: both sides at most 50 tokens
DATA_SEED = 42

if ON_COLAB:
    assert os.path.isdir(
        "/content/drive/MyDrive"
    ), "Drive not mounted: run :ColabDrive first"

_moses = {"en": MosesTokenizer(lang="en"), "fr": MosesTokenizer(lang="fr")}


def tokenize_pair(example):
    # escape=False keeps ' and & as they are instead of &apos; / &amp;
    tr = example["translation"]
    return {
        "src": _moses["en"].tokenize(tr["en"], escape=False),
        "tgt": _moses["fr"].tokenize(tr["fr"], escape=False),
    }


def to_lists(rows, max_len):
    src, tgt = [], []
    for row in rows:
        s, t = row["src"], row["tgt"]
        if not s or not t:
            continue
        if max_len is not None and (len(s) > max_len or len(t) > max_len):
            continue
        src.append(s)
        tgt.append(t)
    assert len(src) == len(tgt)
    return src, tgt


def load_split(split, n=None):
    if ON_COLAB:
        ds = load_dataset("wmt/wmt14", "fr-en", split=split, cache_dir=HF_CACHE)
        if n is not None:
            # train is the corpora concatenated one after another; taking the
            # first n would be mostly one corpus, so shuffle the whole split first
            ds = ds.shuffle(seed=DATA_SEED).select(range(n))
        return ds.map(
            tokenize_pair, remove_columns=["translation"], num_proc=os.cpu_count()
        )
    # Mac: streaming slice (not a random sample), just to exercise the code
    ds = load_dataset("wmt/wmt14", "fr-en", split=split, streaming=True)
    ds = ds.take(n if n is not None else 500)
    return [tokenize_pair(ex) for ex in ds]


# one cache file per n_sample, so the 100k timing cache and the 2M cache coexist
def load_hf_wmt14_data(n_sample=N_SAMPLE):
    cache_path = os.path.join(
        CACHE_DIR, f"tok_n{n_sample}_len{MAX_LEN}_seed{DATA_SEED}.pt"
    )
    if os.path.exists(cache_path):
        data = torch.load(cache_path)
        print("loaded cache:", cache_path)
    else:
        data = {}
        # train and val get the length filter; test stays complete, since the
        # paper evaluates on all of newstest2014 and Fig. 2 buckets it by length
        data["train_src"], data["train_tgt"] = to_lists(
            load_split("train", n_sample), MAX_LEN
        )
        data["val_src"], data["val_tgt"] = to_lists(
            load_split("validation"), MAX_LEN
        )  # newstest2013
        data["test_src"], data["test_tgt"] = to_lists(
            load_split("test"), None
        )  # newstest2014
        os.makedirs(CACHE_DIR, exist_ok=True)
        torch.save(data, cache_path + ".tmp")
        os.replace(cache_path + ".tmp", cache_path)
        print("saved cache:", cache_path)

    train_src, train_tgt = data["train_src"], data["train_tgt"]
    val_src, val_tgt = data["val_src"], data["val_tgt"]
    test_src, test_tgt = data["test_src"], data["test_tgt"]
    for name, s, t in [
        ("train", train_src, train_tgt),
        ("val", val_src, val_tgt),
        ("test", test_src, test_tgt),
    ]:
        assert len(s) == len(t) > 0, name
        print(
            f"{name}: {len(s)} pairs, max len src {max(map(len, s))} tgt {max(map(len, t))}"
        )
    assert max(map(len, train_src)) <= MAX_LEN and max(map(len, train_tgt)) <= MAX_LEN
    print("example:", train_src[0], "->", train_tgt[0])
    return train_src, train_tgt, val_src, val_tgt, test_src, test_tgt


# Data Loaders and pre-processors
def build_vocab(tokens, vocab_len=30000):
    tokens = [tok for tok_list in tokens for tok in tok_list]
    token_to_idx = {}
    for tok, idx in [
        (TOK_UNK, TOK_UNK_ID),
        (TOK_EOS, TOK_EOS_ID),
        (TOK_SOS, TOK_SOS_ID),
        (TOK_PAD, TOK_PAD_ID),
    ]:
        token_to_idx[tok] = idx
    tokens_freq = Counter(tokens)
    idx = len(token_to_idx)
    for item, count in tokens_freq.most_common(vocab_len - 4):
        token_to_idx[item] = idx
        idx += 1
    idx_to_token = {idx: token for token, idx in token_to_idx.items()}
    return token_to_idx, idx_to_token


def encode(sentence, token_to_idx):
    unk = token_to_idx[TOK_UNK]
    return [token_to_idx.get(tok, unk) for tok in sentence]


def prepare_dataset(src_sentences, target_sentences, src_vocab, tgt_vocab):
    assert len(src_sentences) == len(target_sentences)
    pairs = []
    for src_sentence, target_sentence in zip(src_sentences, target_sentences):
        pairs.append(
            (encode(src_sentence, src_vocab), encode(target_sentence, tgt_vocab))
        )
    return pairs


def collate(batch):
    src = []
    dec_in = []
    dec_tgt = []
    src_len = []
    for src_tokens, tgt_tokens in batch:
        src_tokens = src_tokens + [TOK_EOS_ID]
        src_len.append(len(src_tokens))
        src_tensor = torch.tensor(src_tokens, dtype=torch.long)
        src.append(src_tensor)
        dec_in_tokens = [TOK_SOS_ID] + tgt_tokens
        dec_tgt_tokens = tgt_tokens + [TOK_EOS_ID]
        dec_in.append(torch.tensor(dec_in_tokens, dtype=torch.long))
        dec_tgt.append(torch.tensor(dec_tgt_tokens, dtype=torch.long))

    def pad_seq(list_of_tokens):
        return torch.nn.utils.rnn.pad_sequence(
            list_of_tokens, batch_first=True, padding_value=TOK_PAD_ID
        )

    src_len = torch.tensor(src_len, dtype=torch.long)
    src = pad_seq(src)
    dec_in = pad_seq(dec_in)
    dec_tgt = pad_seq(dec_tgt)
    assert src.shape[0] == len(batch)
    assert src.shape[1] == src_len.max()
    assert dec_in.shape == dec_tgt.shape
    assert src_len.min() >= 1
    return src, src_len, dec_in, dec_tgt


def prepare_data_loaders(
    src_sentences, target_sentences, src_vocab, tgt_vocab, batch_size=32, shuffle=True
):
    pairs = prepare_dataset(src_sentences, target_sentences, src_vocab, tgt_vocab)
    return DataLoader(
        pairs, batch_size=batch_size, shuffle=shuffle, collate_fn=collate
    )  # pyright: ignore[reportArgumentType]  (a list is a valid map-style dataset)


# Vocab from train only, then train/val loaders. One path for the toy task and WMT14.
# n_train limits only the train loader (e.g. a timing run); the vocab is still
# built from all of train_src/train_tgt, so V matches the full run.
def prepare_data(
    train_src,
    train_tgt,
    val_src=None,
    val_tgt=None,
    vocab_size=30000,
    batch_size=32,
    n_train=None,
):
    assert len(train_src) == len(train_tgt)
    src_vocab, src_itos = build_vocab(train_src, vocab_len=vocab_size)
    tgt_vocab, tgt_itos = build_vocab(train_tgt, vocab_len=vocab_size)
    n = len(train_src) if n_train is None else min(n_train, len(train_src))
    train_loader = prepare_data_loaders(
        train_src[:n],
        train_tgt[:n],
        src_vocab,
        tgt_vocab,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = None
    if val_src is not None:
        assert val_tgt is not None and len(val_src) == len(val_tgt)
        val_loader = prepare_data_loaders(
            val_src, val_tgt, src_vocab, tgt_vocab, batch_size=batch_size, shuffle=False
        )
    print(
        f"vocab src {len(src_vocab)}, tgt {len(tgt_vocab)} | train {n} pairs, "
        f"{len(train_loader)} batches of {batch_size}"
        + (f" | val {len(val_src)} pairs" if val_src is not None else "")
    )
    return {
        "src_vocab": src_vocab,
        "tgt_vocab": tgt_vocab,
        "src_itos": src_itos,
        "tgt_itos": tgt_itos,
        "train_loader": train_loader,
        "val_loader": val_loader,
    }


# ============================================================================
# 3. Model
# ============================================================================
# Encoder
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, enc_hidden_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.enc_hidden_dim = enc_hidden_dim
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=TOK_PAD_ID)
        self.rnn = nn.GRU(
            input_size=emb_dim,
            hidden_size=enc_hidden_dim,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, src: torch.Tensor, src_len: torch.Tensor):
        """
        src shape : (batch_size, max_src_len)
        src_len : (batch_size,)
        """
        # 1. shape (batch_size, src_len, emb_dim)
        embedd = self.emb(src)
        packed_embedd = pack_padded_sequence(
            embedd, src_len.cpu(), batch_first=True, enforce_sorted=False
        )
        # packed_outputs shape (batch_size, max_src_len, 2*enc_hidden_dim)
        # hidden = (2, batch_size, enc_hidden_dim)
        packed_outputs, hidden = self.rnn(packed_embedd)
        encoder_outputs, _ = pad_packed_sequence(
            packed_outputs, batch_first=True, total_length=src.shape[1]
        )
        assert encoder_outputs.shape == (
            src.shape[0],
            src.shape[1],
            2 * self.enc_hidden_dim,
        )
        # hidden[0] = h_fwd_T (fwd state at each real last token, thanks to packing)
        # hidden[1] = h_bwd_1 (bwd state at position 1)
        assert hidden.shape == (2, src.shape[0], self.enc_hidden_dim)

        # decoder initial_hidden_state shape (batch_size, enc_hidden_dim)
        mask_src = src != TOK_PAD_ID
        assert mask_src.shape == (src.shape[0], src.shape[1])
        return encoder_outputs, hidden, mask_src


# Attention
class Attention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim, attn_dim):
        super().__init__()
        self.V = nn.Linear(attn_dim, 1, bias=True)
        self.W = nn.Linear(dec_hidden_dim, attn_dim, bias=True)
        self.U = nn.Linear(2 * enc_hidden_dim, attn_dim, bias=True)

    def precompute_keys(self, H_enc):
        return self.U(H_enc)

    def forward(self, s_prev, keys, H_enc, mask):
        s_prev_wa = self.W(s_prev)
        s_prev_wa = s_prev_wa.unsqueeze(1)
        e = self.V(torch.tanh(s_prev_wa + keys)).squeeze(-1)
        assert e.shape == (H_enc.shape[0], H_enc.shape[1])
        e = e.masked_fill(~mask, float("-inf"))
        attn_weights = torch.softmax(e, dim=1)
        assert attn_weights.shape == (H_enc.shape[0], H_enc.shape[1])
        context = torch.bmm(attn_weights.unsqueeze(1), H_enc)
        context = context.squeeze(1)
        assert context.shape == (H_enc.shape[0], H_enc.shape[2])
        return context, attn_weights


# Decoder: one step
class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim,
        dec_hidden_dim,
        enc_hidden_dim,
        attn_dim,
        max_out_dim,
        use_attention=True,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.dec_hidden_dim = dec_hidden_dim
        self.enc_hidden_dim = enc_hidden_dim
        self.attn_dim = attn_dim
        self.max_out_dim = max_out_dim
        self.vocab_size = vocab_size
        self.use_attention = use_attention
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=TOK_PAD_ID)
        # baseline (use_attention=False): no attention module at all
        self.attention = (
            Attention(
                enc_hidden_dim=enc_hidden_dim,
                dec_hidden_dim=dec_hidden_dim,
                attn_dim=attn_dim,
            )
            if use_attention
            else None
        )
        self.rnn = nn.GRUCell(
            input_size=emb_dim + (2 * enc_hidden_dim), hidden_size=dec_hidden_dim
        )
        self.bridge = nn.Linear(enc_hidden_dim, dec_hidden_dim)
        self.fc_out = nn.Linear(
            emb_dim + dec_hidden_dim + 2 * enc_hidden_dim, 2 * max_out_dim
        )
        self.W_o = nn.Linear(max_out_dim, vocab_size)

    def init_state(self, h_bwd_1):
        return torch.tanh(self.bridge(h_bwd_1))

    def forward(self, y_prev, s_prev, keys, H_enc, mask, c_fixed=None):
        """
        use_attention=True : c_i from attention, recomputed every step (keys, H_enc, mask used)
        use_attention=False: c_fixed = [h_fwd_T ; h_bwd_1], the same vector every step
        """
        # emb shape (B, emb_dim)
        emb = self.emb(y_prev)
        assert emb.shape == (y_prev.shape[0], self.emb_dim)
        # context shape (B, 2*enc_hidden_dim)
        # attn_weights (B, Tin), or None for the baseline
        if self.attention is not None:
            context, attn_weights = self.attention(s_prev, keys, H_enc, mask)
            assert attn_weights.shape == (y_prev.shape[0], H_enc.shape[1])
        else:
            assert c_fixed is not None, "baseline decoder needs c_fixed"
            context, attn_weights = c_fixed, None
        assert context.shape == (y_prev.shape[0], 2 * self.enc_hidden_dim)
        # rnn_input shape (B, emb_dim + 2*enc_hidden_dim)
        rnn_input = torch.cat((emb, context), dim=1)
        assert rnn_input.shape == (
            y_prev.shape[0],
            self.emb_dim + 2 * self.enc_hidden_dim,
        )
        # s_curr shape (B, dec_hidden_dim)
        s_curr = self.rnn(
            rnn_input,  # (B, emb_dim + 2*enc_hidden_dim)
            s_prev,  # (B, dec_hidden_dim)
        )
        assert s_curr.shape == (y_prev.shape[0], self.dec_hidden_dim)
        combined = torch.cat((emb, s_curr, context), dim=1)
        t_tilde = self.fc_out(combined)
        t = t_tilde.view(y_prev.shape[0], self.max_out_dim, 2)
        t = t.max(dim=2).values
        assert t.shape == (y_prev.shape[0], self.max_out_dim)
        logits = self.W_o(t)
        assert logits.shape == (y_prev.shape[0], self.vocab_size)
        return logits, s_curr, attn_weights


# Seq2Seq: teacher-forced forward and greedy predict
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def _init_values(self, src, src_len):
        H_enc, hidden, mask = self.encoder(src, src_len)
        h_fwd_T, h_bwd_1 = hidden[0], hidden[1]  # (B, H) each
        s = self.decoder.init_state(h_bwd_1)  # s_0 from h_bwd_1 in both models
        if self.decoder.use_attention:
            keys = self.decoder.attention.precompute_keys(H_enc)
            c_fixed = None
        else:
            keys = None
            c_fixed = torch.cat((h_fwd_T, h_bwd_1), dim=1)
            assert c_fixed.shape == (src.shape[0], 2 * self.encoder.enc_hidden_dim)
        return H_enc, s, keys, mask, c_fixed

    def forward(self, src, src_len, dec_in):
        H_enc, s, keys, mask, c_fixed = self._init_values(src, src_len)
        T_out = dec_in.shape[1]
        logits = []
        attn_weights = []
        for i in range(T_out):
            logits_i, s, a_i = self.decoder(
                dec_in[:, i], s, keys, H_enc, mask, c_fixed=c_fixed
            )
            logits.append(logits_i)
            attn_weights.append(a_i)
        logits = torch.stack(logits, dim=1)
        assert logits.shape == (src.shape[0], T_out, self.decoder.vocab_size)
        if not self.decoder.use_attention:
            return logits, None
        attn_weights = torch.stack(attn_weights, dim=1)
        assert attn_weights.shape == (src.shape[0], T_out, src.shape[1])
        return logits, attn_weights

    @torch.no_grad
    def predict(self, src, src_len, max_len):
        device = src.device
        batch_size, T_in = src.shape
        H_enc, s, keys, mask, c_fixed = self._init_values(src, src_len)
        y_prev = torch.full((batch_size,), TOK_SOS_ID, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        predictions = []
        attn_weights = []
        for _ in range(max_len):
            logits_i, s, a_i = self.decoder(
                y_prev, s, keys, H_enc, mask, c_fixed=c_fixed
            )
            y_pred = torch.argmax(logits_i, dim=1)
            # sentences that already emitted EOS get PAD from here on
            y_pred = y_pred.masked_fill(finished, TOK_PAD_ID)
            assert y_pred.shape == (batch_size,)
            predictions.append(y_pred)
            if a_i is not None:
                assert a_i.shape == (batch_size, T_in)
                attn_weights.append(a_i)
            finished |= y_pred == TOK_EOS_ID
            y_prev = y_pred
            if finished.all():
                break
        predictions = torch.stack(predictions, dim=1)
        T = predictions.shape[1]
        assert predictions.shape == (batch_size, T)
        if not self.decoder.use_attention:
            return predictions, None
        attn_weights = torch.stack(attn_weights, dim=1)
        assert attn_weights.shape == (batch_size, T, T_in)
        return predictions, attn_weights


# Build RNNsearch (use_attention=True) or the RNNencdec-style baseline (False)
def build_model(src_vocab_size, tgt_vocab_size, emb, H, A, max_out_dim, use_attention):
    encoder = Encoder(vocab_size=src_vocab_size, emb_dim=emb, enc_hidden_dim=H)
    decoder = Decoder(
        vocab_size=tgt_vocab_size,
        emb_dim=emb,
        enc_hidden_dim=H,
        dec_hidden_dim=H,
        attn_dim=A,
        max_out_dim=max_out_dim,
        use_attention=use_attention,
    )
    model = Seq2Seq(encoder=encoder, decoder=decoder)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: use_attention={use_attention}, {n_params:,} params")
    return model


# ============================================================================
# 4. Training
# ============================================================================
def make_batch_loss_fn(vocab_size: int):
    criterion = nn.CrossEntropyLoss(ignore_index=TOK_PAD_ID)

    def batch_loss_fn(
        model: nn.Module, batch: Any, dev: torch.device
    ) -> tuple[torch.Tensor, float]:
        src, src_len, dec_in, dec_tgt = [x.to(dev) for x in batch]
        logits, _ = model(src, src_len, dec_in)
        loss = criterion(logits.reshape(-1, vocab_size), dec_tgt.reshape(-1))
        n_tokens = (dec_tgt != TOK_PAD_ID).sum().item()
        return loss, float(n_tokens)

    return batch_loss_fn


def train(
    model: nn.Module,
    tgt_vocab_size: int,
    train_data_loader: DataLoader[Any],
    val_data_loader: DataLoader[Any] | None,
    epochs: int,
    rho: float = 0.95,
    eps: float = 1e-6,
    experiment_name: str | None = None,
    save_every_steps: int | None = None,
    run_config: dict[str, Any] | None = None,
    vocab: dict[str, Any] | None = None,
    profile: bool = False,
) -> Trainer:
    optimizer = torch.optim.Adadelta(model.parameters(), rho=rho, eps=eps)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        batch_loss_fn=make_batch_loss_fn(tgt_vocab_size),
        max_grad_norm=1.0,
        device=device,
        experiment_name=experiment_name,
        save_every_steps=save_every_steps,
        run_config=run_config,
        vocab=vocab,
        seed=seed,
        profile=profile,
    )
    trainer.fit(train_data_loader, val_data_loader, epochs=epochs)
    return trainer


# data from prepare_data, model_cfg = build_model kwargs (emb, H, A, max_out_dim, use_attention).
# run_config holds everything that defines the run, so the checkpointer refuses
# to resume a different setup; epochs is left out so a run can be extended.
def run_experiment(
    experiment_name, data, model_cfg, epochs, save_every_steps=None, profile=False
):
    model = build_model(len(data["src_vocab"]), len(data["tgt_vocab"]), **model_cfg)
    run_config = {
        **model_cfg,
        "src_vocab_size": len(data["src_vocab"]),
        "tgt_vocab_size": len(data["tgt_vocab"]),
        "n_train": len(data["train_loader"].dataset),
        "batch_size": data["train_loader"].batch_size,
    }
    trainer = train(
        model=model,
        tgt_vocab_size=len(data["tgt_vocab"]),
        train_data_loader=data["train_loader"],
        val_data_loader=data["val_loader"],
        epochs=epochs,
        experiment_name=experiment_name,
        save_every_steps=save_every_steps,
        run_config=run_config,
        vocab={"src": data["src_vocab"], "tgt": data["tgt_vocab"]},
        profile=profile,
    )
    return trainer


# ============================================================================
# Entry point
# ============================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-name", required=True)
    p.add_argument("--use-attention", action="store_true")
    p.add_argument("--vocab-size", type=int, default=30000)
    p.add_argument("--emb", type=int, default=620)
    p.add_argument("--hidden", type=int, default=1000)
    p.add_argument("--attn-dim", type=int, default=1000)
    p.add_argument("--max-out-dim", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=80)
    p.add_argument(
        "--n-sample",
        type=int,
        default=2_000_000,
        help="pairs sampled from WMT14 train before the length filter; one Drive cache per value",
    )
    p.add_argument(
        "--n-train",
        type=int,
        default=None,
        help="limit the train loader; the vocab still uses all of train",
    )
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--save-every-steps", type=int, default=1000)
    p.add_argument(
        "--profile", action="store_true", help="per-step timing and peak memory"
    )
    args = p.parse_args()

    train_src, train_tgt, val_src, val_tgt, _, _ = load_hf_wmt14_data(
        n_sample=args.n_sample
    )
    data = prepare_data(
        train_src,
        train_tgt,
        val_src,
        val_tgt,
        vocab_size=args.vocab_size,
        batch_size=args.batch_size,
        n_train=args.n_train,
    )
    model_cfg = {
        "emb": args.emb,
        "H": args.hidden,
        "A": args.attn_dim,
        "max_out_dim": args.max_out_dim,
        "use_attention": args.use_attention,
    }
    trainer = run_experiment(
        args.experiment_name,
        data,
        model_cfg,
        epochs=args.epochs,
        save_every_steps=args.save_every_steps,
        profile=args.profile,
    )
    if args.profile:
        print(trainer.profile_stats)


if __name__ == "__main__":
    main()
