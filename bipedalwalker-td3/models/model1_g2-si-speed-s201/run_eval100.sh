#!/bin/bash
# 모델1: g2-si-speed-s201 100ep 재평가 (noise=0, 고정 시드 9000~9099, speed shaping 제외)
cd "$(dirname "$0")"
python3 ../../src/evaluate_100ep.py --ckpt ep794 --set \
  run_tag=g2-si-speed-s201 hardcore=true frame_skip=2 fall_penalty=-10.0 \
  activation=gelu final_layer_init_scale=0.003 critic_output_scale=10.0
