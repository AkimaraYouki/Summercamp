#!/bin/bash
# 모델2: scratch-initscale 재현 학습 (from scratch, 약 3000ep+ 필요)
cd "$(dirname "$0")"
python3 ../../src/Bipedalwalker_TD3_live.py --set \
  run_tag=scratch-initscale hardcore=true frame_skip=2 fall_penalty=-10.0 \
  activation=gelu final_layer_init_scale=0.003 critic_output_scale=10.0 \
  num_episodes=3500
