#!/bin/bash
# 모델4: new-speed-s102 (학습 비용 최저 - 가장 저렴한 레시피)
cd "$(dirname "$0")"
python3 ../../src/Bipedalwalker_TD3_live.py --play 5 --render --ckpt ep1556 --set \
  run_tag=new-speed-s102 hardcore=true frame_skip=2 fall_penalty=-10.0 \
  activation=gelu speed_low_thresh=0.05 speed_high_thresh=0.3 speed_penalty=-0.02 speed_bonus=0.03
