#!/bin/bash
# 모델1: g2-si-speed-s201 재현 학습
# scratch-initscale(ep3081) 체크포인트에서 이어서 파인튜닝 -> 794ep 만에 solve(avg100>=300)
# 부모 체크포인트가 필요합니다: ../model2_scratch-initscale/checkpoints/.../ep3081_*.pt 를
# 이 폴더의 checkpoints/bipedalwalker-td3-hardcore-g2-si-speed-s201/ep3081_*.pt 로 먼저 복사한 뒤 실행하세요.
cd "$(dirname "$0")"
python3 ../../src/Bipedalwalker_TD3_live.py --resume --ckpt ep3081 --set \
  run_tag=g2-si-speed-s201 hardcore=true frame_skip=2 fall_penalty=-10.0 \
  activation=gelu final_layer_init_scale=0.003 critic_output_scale=10.0 \
  speed_low_thresh=0.05 speed_high_thresh=0.3 speed_penalty=-0.02 speed_bonus=0.03 \
  seed=201 num_episodes=2000
