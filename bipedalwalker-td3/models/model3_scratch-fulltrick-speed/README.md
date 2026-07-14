모델3: scratch-fulltrick-speed (클리어 속도 1위)

계보: from scratch, LayerNorm + init_scale=0.003 + critic_output_scale=10 + speed shaping, ep2778에 solve
체크포인트: checkpoints/bipedalwalker-td3-hardcore-scratch-fulltrick-speed/ep2778_{actor,critic1,critic2}.pt

100ep 헤드리스 재평가 결과 (2026-07-10, speed-shaping 보너스 제외한 순수 환경 보상 기준):
  - 해결률: 80.0%
  - 낙상률: 18.0%
  - 비낙상 평균 보상: 307.20
  - 비낙상 평균 클리어 스텝: 477.0 — 5개 모델 중 1위

주장: 안정화 트릭은 많을수록 낫다. 검증: 트릭 4개(본 모델) vs 트릭 2개(모델2) 비교. 결과: 해결률 80% vs 82%, 비낙상 평균 307.20 vs 310.96 — 반증. 클리어 속도만 5개 중 1위.

실행법:
  ./run_play.sh     -> 로컬에서 5회 렌더링 재생 (noise=0)
  ./run_train.sh    -> 동일 레시피로 재현 학습 (from scratch)
  ./run_eval100.sh  -> 위 100ep 재평가 결과를 그대로 재현 (noise=0, 고정 시드 9000~9099, speed shaping 제외)

Colab 재현: (전제: /content/bipedalwalker-td3 압축 해제 + 의존성 설치 완료, ../../COLAB_실행_가이드.md 1~2단계)

```bash
%cd /content/bipedalwalker-td3/models/model3_scratch-fulltrick-speed
!bash run_train.sh    # 학습 재현 (from scratch, LayerNorm 포함, 약 2800ep+ 필요 -- 오래 걸림)
```

```python
# 학습 중/후 TensorBoard로 곡선 확인 (같은 폴더에서, 별도 셀)
%load_ext tensorboard
%tensorboard --logdir tensorboard
```

```bash
!bash run_eval100.sh  # 100ep 재평가 (해결률 80% / 낙상률 18% 재현 확인, speed shaping 보너스 제외)
```

```bash
# 시연 + 녹화 (가상 디스플레이 먼저 시작 -- ../../COLAB_실행_가이드.md 3단계)
!python3 ../../src/Bipedalwalker_TD3_live.py --play 3 --ckpt ep2778 --set \
  run_tag=scratch-fulltrick-speed hardcore=true frame_skip=2 fall_penalty=-10.0 \
  activation=gelu use_layer_norm=true final_layer_init_scale=0.003 critic_output_scale=10.0 \
  video_record_dir=/content/demo_videos
```

개별 보고서: ../../reports/model3_report.pdf
