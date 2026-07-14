모델5: g2-si-fastclear-s501  ⚠ 참고용, 배포 비권장 ⚠

계보: scratch-initscale(ep3081)에서 파인튜닝, speed shaping 공격적 조정(threshold 0.3→0.25, bonus 0.03→0.05, penalty -0.02→-0.03), 843ep 학습 (843ep는 파인튜닝 시작부터 재카운트한 값, 부모의 ep3081은 미포함)

체크포인트: checkpoints/bipedalwalker-td3-hardcore-g2-si-fastclear-s501/ep843_{actor,critic1,critic2}.pt

100ep 헤드리스 재평가 결과 (2026-07-10):
  - 해결률: 29.0%   — 5개 모델 중 최저
  - 낙상률: 20.0%   — 5개 모델 중 최고
  - 비낙상 평균 보상: 298.29 (300 미달)
  - 비낙상 평균 클리어 스텝: 532.5  — 5개 모델 중 최장

주장: threshold를 낮추고 bonus/penalty를 키우면 더 빨리, 안정적으로 클리어한다. 검증: 위 조정 적용. 결과: 해결률·낙상률·클리어속도 3개 지표 모두 5개 모델 중 최악 — 반증. 재현 학습 시 동일 결과 예상, 배포용 사용 금지.

실행법:
  ./run_play.sh     -> 로컬에서 5회 렌더링 재생 (noise=0, 성능 저하를 직접 확인 가능)
  ./run_train.sh    -> 동일 레시피로 재현 학습 (재현 비권장, 참고용)
  ./run_eval100.sh  -> 위 100ep 재평가 결과를 그대로 재현 (noise=0, 고정 시드 9000~9099, speed shaping 제외)

주의: --resume은 "이 run_tag 자신의 checkpoints 폴더"에서 ep3081을 찾습니다.
      재현 학습을 처음부터 하려면 모델2_scratch-initscale의 ep3081 체크포인트 3개 파일을
      checkpoints/bipedalwalker-td3-hardcore-g2-si-fastclear-s501/ 안에 먼저 복사하세요.

Colab 재현: (전제: /content/bipedalwalker-td3 압축 해제 + 의존성 설치 완료, ../../COLAB_실행_가이드.md 1~2단계)

```bash
%cd /content/bipedalwalker-td3/models/model5_g2-si-fastclear-s501

# 학습 재현 전 부모 체크포인트(ep3081) 복사 -- --resume이 이 폴더 안에서 ep3081을 찾음
!mkdir -p checkpoints/bipedalwalker-td3-hardcore-g2-si-fastclear-s501
!cp ../model2_scratch-initscale/checkpoints/bipedalwalker-td3-hardcore-scratch-initscale/ep3081_*.pt \
    checkpoints/bipedalwalker-td3-hardcore-g2-si-fastclear-s501/

!bash run_train.sh    # 학습 재현 (재현 비권장, 참고용)
!bash run_eval100.sh  # 100ep 재평가 (해결률 29% / 낙상률 20% 재현 확인 -- 반면교사 사례)
```

```bash
# 시연 + 녹화 (가상 디스플레이 먼저 시작 -- ../../COLAB_실행_가이드.md 3단계)
!python3 ../../src/Bipedalwalker_TD3_live.py --play 3 --ckpt ep843 --set \
  run_tag=g2-si-fastclear-s501 hardcore=true frame_skip=2 fall_penalty=-10.0 \
  activation=gelu final_layer_init_scale=0.003 critic_output_scale=10.0 \
  speed_low_thresh=0.05 speed_high_thresh=0.25 speed_penalty=-0.03 speed_bonus=0.05 \
  video_record_dir=/content/demo_videos
```

개별 보고서: ../../reports/model5_report.pdf
