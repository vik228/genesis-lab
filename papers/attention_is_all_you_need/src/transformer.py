import torch
import torch.nn as nn
from .attention import MultiHeadAttention

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        import math
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # 1, L, D
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :].to(x.device)
        return self.dropout(x)

class FFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        attn_out, _ = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        attn1, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn1))
        attn2, _ = self.cross_attn(x, memory, memory, memory_mask)
        x = self.norm2(x + self.dropout(attn2))
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=512, n_heads=8, d_ff=2048, n_enc=6, n_dec=6, dropout=0.1, tie_weights=True):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model)
        self.pos = PositionalEncoding(d_model, dropout=dropout)
        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_enc)])
        self.dec_layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_dec)])
        self.proj = nn.Linear(d_model, tgt_vocab, bias=False)
        if tie_weights:
            self.proj.weight = self.tgt_embed.weight  # weight tying
        self.d_model = d_model

    def encode(self, src, src_mask=None):
        x = self.pos(self.src_embed(src))
        for layer in self.enc_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, memory, tgt_mask=None, memory_mask=None):
        x = self.pos(self.tgt_embed(tgt))
        for layer in self.dec_layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        return x

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        memory = self.encode(src, src_mask)
        out = self.decode(tgt, memory, tgt_mask, memory_mask)
        return self.proj(out)
