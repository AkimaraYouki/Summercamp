#!/bin/bash
# 모델5: g2-si-fastclear-s501 100ep 재평가 (noise=0, 고정 시드 9000~9099, speed shaping 제외)
cd "$(dirname "$0")"
python3 ../../src/evaluate_100ep.py --ckpt ep843 --set \
  run_tag=g2-si-fastclear-s501 hardcore=true frame_skip=2 fall_penalty=-10.0 \
  activation=gelu final_layer_init_scale=0.003 critic_output_scale=10.0
