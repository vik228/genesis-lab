import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        return torch.matmul(attn, V), attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.attn = ScaledDotProductAttention(dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        B, L, D = x.size()
        x = x.view(B, L, self.n_heads, self.d_k).transpose(1, 2)  # B, H, L, d_k
        return x

    def _combine_heads(self, x):
        B, H, L, d_k = x.size()
        x = x.transpose(1, 2).contiguous().view(B, L, H * d_k)
        return x

    def forward(self, Q, K, V, mask=None):
        Q = self._split_heads(self.W_q(Q))
        K = self._split_heads(self.W_k(K))
        V = self._split_heads(self.W_v(V))
        if mask is not None:
            mask = mask.unsqueeze(1)  # broadcast across heads
        context, attn = self.attn(Q, K, V, mask)
        context = self._combine_heads(context)
        return self.out(context), attn
