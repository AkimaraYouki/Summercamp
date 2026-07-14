모델2: scratch-initscale (차선책)

계보: from scratch, init_scale=0.003 + critic_output_scale=10, 셰이핑 없음, ep3081에 solve
체크포인트: checkpoints/bipedalwalker-td3-hardcore-scratch-initscale/ep3081_{actor,critic1,critic2}.pt

100ep 헤드리스 재평가 결과 (2026-07-10):
  - 해결률: 82.0% (모델1과 1episode 차이)
  - 낙상률: 18.0%
  - 비낙상 평균 보상: 310.96 — 5개 모델 중 1위 (표준편차 2.44, 최저)
  - 비낙상 평균 클리어 스텝: 482.3 — 5개 모델 중 2위

특징: 트릭 2개(init_scale+critic_scale)만 적용, LayerNorm·speed-shaping 없음. 해결률·걸음품질 모두 상위권 — 재학습 비용/복잡도를 아끼려면 1순위 대안.

실행법:
  ./run_play.sh     -> 로컬에서 5회 렌더링 재생 (noise=0)
  ./run_train.sh    -> 동일 레시피로 재현 학습 (from scratch)
  ./run_eval100.sh  -> 위 100ep 재평가 결과를 그대로 재현 (noise=0, 고정 시드 9000~9099, speed shaping 제외)

Colab 재현: (전제: /content/bipedalwalker-td3 압축 해제 + 의존성 설치 완료, ../../COLAB_실행_가이드.md 1~2단계)

```bash
%cd /content/bipedalwalker-td3/models/model2_scratch-initscale
!bash run_train.sh    # 학습 재현 (from scratch, 약 3000ep+ 필요 -- 오래 걸림)
!bash run_eval100.sh  # 100ep 재평가 (해결률 82% / 낙상률 18% 재현 확인)
```

```bash
# 시연 + 녹화 (가상 디스플레이 먼저 시작 -- ../../COLAB_실행_가이드.md 3단계)
!python3 ../../src/Bipedalwalker_TD3_live.py --play 3 --ckpt ep3081 --set \
  run_tag=scratch-initscale hardcore=true frame_skip=2 fall_penalty=-10.0 \
  activation=gelu final_layer_init_scale=0.003 critic_output_scale=10.0 \
  video_record_dir=/content/demo_videos
```

개별 보고서: ../../reports/model2_report.pdf
