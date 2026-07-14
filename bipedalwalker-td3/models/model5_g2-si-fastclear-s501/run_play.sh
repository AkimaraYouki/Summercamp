#!/bin/bash
# 모델5: g2-si-fastclear-s501 (참고용 - 부정적 결과, 배포 비권장)
cd "$(dirname "$0")"
python3 ../../src/Bipedalwalker_TD3_live.py --play 5 --render --ckpt ep843 --set \
  run_tag=g2-si-fastclear-s501 hardcore=true frame_skip=2 fall_penalty=-10.0 \
  activation=gelu final_layer_init_scale=0.003 critic_output_scale=10.0 \
  speed_low_thresh=0.05 speed_high_thresh=0.25 speed_penalty=-0.03 speed_bonus=0.05
