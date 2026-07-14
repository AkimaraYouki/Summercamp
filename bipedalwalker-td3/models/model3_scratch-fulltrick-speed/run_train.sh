#!/bin/bash
# 모델3: scratch-fulltrick-speed 재현 학습 (from scratch, LayerNorm 포함, 약 2800ep+ 필요)
# 주의: 평가 시에는 speed shaping 보너스를 끄고 측정했지만(공정 비교),
#       학습 시에는 speed shaping을 켜서 진행했습니다 (아래 커맨드 그대로).
cd "$(dirname "$0")"
python3 ../../src/Bipedalwalker_TD3_live.py --set \
  run_tag=scratch-fulltrick-speed hardcore=true frame_skip=2 fall_penalty=-10.0 \
  activation=gelu use_layer_norm=true final_layer_init_scale=0.003 critic_output_scale=10.0 \
  speed_low_thresh=0.05 speed_high_thresh=0.3 speed_penalty=-0.02 speed_bonus=0.03 \
  num_episodes=3000
