#!/bin/bash
# 모델4: new-speed-s102 재현 학습 (from scratch, 트릭 없이 speed shaping만)
# 주의: 1556ep 만에 solve됐지만 재현 실험(g2-ns-replicate-s601, seed=601)은
#       2000ep 넘도록 수렴 실패(avg100=149.6) -- 시드 운이었을 가능성이 있어
#       재현 시 1556ep보다 훨씬 오래 걸리거나 실패할 수 있습니다. num_episodes를
#       넉넉히(4000) 잡아두었습니다.
cd "$(dirname "$0")"
python3 ../../src/Bipedalwalker_TD3_live.py --set \
  run_tag=new-speed-s102 hardcore=true frame_skip=2 fall_penalty=-10.0 \
  activation=gelu speed_low_thresh=0.05 speed_high_thresh=0.3 speed_penalty=-0.02 speed_bonus=0.03 \
  seed=102 num_episodes=4000
