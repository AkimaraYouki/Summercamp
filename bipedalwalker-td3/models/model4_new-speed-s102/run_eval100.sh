#!/bin/bash
# 모델4: new-speed-s102 100ep 재평가 (noise=0, 고정 시드 9000~9099, speed shaping 제외)
# 트릭 없이 speed shaping만으로 1556ep 만에 solve한 모델. 평가 시 셰이핑 보너스를 빼면
# 해결률이 82%->67%로 떨어지는 것이 바로 이 재평가로 확인된 결과입니다.
cd "$(dirname "$0")"
python3 ../../src/evaluate_100ep.py --ckpt ep1556 --set \
  run_tag=new-speed-s102 hardcore=true frame_skip=2 fall_penalty=-10.0 \
  activation=gelu
