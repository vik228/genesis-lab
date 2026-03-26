# Attention Is All You Need (2017)

Paper: https://arxiv.org/abs/1706.03762

This folder contains a faithful-from-scratch implementation of the **encoder–decoder Transformer** with:
- Multi-Head Attention (self + cross)
- Sinusoidal position encodings
- Position-wise FFN
- Residual + LayerNorm (post-norm)
- Label smoothing (ε=0.1)
- Adam(β1=0.9, β2=0.98, ε=1e-9) + Noam LR
- Dropout=0.1, weight tying

## Run
```bash
python papers/attention_is_all_you_need/src/train_mt.py --config papers/attention_is_all_you_need/configs/transformer_base_iwslt14.yml
```
