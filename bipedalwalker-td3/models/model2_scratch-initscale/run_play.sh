#!/bin/bash
# 모델2: scratch-initscale (차선책 - 가장 단순한 레시피, 걸음 품질 1위)
cd "$(dirname "$0")"
python3 ../../src/Bipedalwalker_TD3_live.py --play 5 --render --ckpt ep3081 --set \
  run_tag=scratch-initscale hardcore=true frame_skip=2 fall_penalty=-10.0 \
  activation=gelu final_layer_init_scale=0.003 critic_output_scale=10.0
