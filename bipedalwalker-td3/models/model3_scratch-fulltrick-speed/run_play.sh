#!/bin/bash
# 모델3: scratch-fulltrick-speed (클리어 속도 1위)
cd "$(dirname "$0")"
python3 ../../src/Bipedalwalker_TD3_live.py --play 5 --render --ckpt ep2778 --set \
  run_tag=scratch-fulltrick-speed hardcore=true frame_skip=2 fall_penalty=-10.0 \
  activation=gelu use_layer_norm=true final_layer_init_scale=0.003 critic_output_scale=10.0
