#!/bin/bash
# 모델1: g2-si-speed-s201 (최종 추천 모델) 로컬 재생
# 해결률 83% / 낙상률 12%(5개 중 최저) / 비낙상 평균 304.73
cd "$(dirname "$0")"
python3 ../../src/Bipedalwalker_TD3_live.py --play 5 --render --ckpt ep794 --set \
  run_tag=g2-si-speed-s201 hardcore=true frame_skip=2 fall_penalty=-10.0 \
  activation=gelu final_layer_init_scale=0.003 critic_output_scale=10.0 \
  speed_low_thresh=0.05 speed_high_thresh=0.3 speed_penalty=-0.02 speed_bonus=0.03
