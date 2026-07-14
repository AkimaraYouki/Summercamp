#!/bin/bash
# 모델5: g2-si-fastclear-s501 재현 학습
# scratch-initscale(ep3081)에서 이어서 파인튜닝, 공격적 speed shaping으로
# "빨리 클리어"를 노렸으나 역효과(가장 느리고 가장 불안정한 결과) - 재현 비권장,
# 참고/반면교사용으로만 보관.
# 부모 체크포인트를 checkpoints/bipedalwalker-td3-hardcore-g2-si-fastclear-s501/ep3081_*.pt 로
# 먼저 복사해야 --resume이 동작합니다.
cd "$(dirname "$0")"
python3 ../../src/Bipedalwalker_TD3_live.py --resume --ckpt ep3081 --set \
  run_tag=g2-si-fastclear-s501 hardcore=true frame_skip=2 fall_penalty=-10.0 \
  activation=gelu final_layer_init_scale=0.003 critic_output_scale=10.0 \
  speed_low_thresh=0.05 speed_high_thresh=0.25 speed_penalty=-0.03 speed_bonus=0.05 \
  seed=501 num_episodes=2000
