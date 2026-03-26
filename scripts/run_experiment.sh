#!/usr/bin/env bash
set -euo pipefail
python papers/attention_is_all_you_need/src/train_mt.py --config papers/attention_is_all_you_need/configs/transformer_base_iwslt14.yml
