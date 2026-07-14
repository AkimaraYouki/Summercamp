#!/bin/bash
# 모델3: scratch-fulltrick-speed 100ep 재평가 (noise=0, 고정 시드 9000~9099, speed shaping 제외)
# 학습 시에는 speed shaping을 켜서 진행했지만, 평가는 다른 모델과 공정 비교를 위해 항상 제외합니다.
cd "$(dirname "$0")"
python3 ../../src/evaluate_100ep.py --ckpt ep2778 --set \
  run_tag=scratch-fulltrick-speed hardcore=true frame_skip=2 fall_penalty=-10.0 \
  activation=gelu use_layer_norm=true final_layer_init_scale=0.003 critic_output_scale=10.0
