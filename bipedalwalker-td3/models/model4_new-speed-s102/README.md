모델4: new-speed-s102 (학습 비용 최저)

계보: from scratch, 안정화 트릭 없이 speed shaping만, ep1556에 solve (학습 중 판정은 셰이핑 보상 포함 기준)
체크포인트: checkpoints/bipedalwalker-td3-hardcore-new-speed-s102/ep1556_{actor,critic1,critic2}.pt

100ep 헤드리스 재평가 결과 (2026-07-10, 셰이핑 보너스 제외한 공정 비교 기준):
  - 해결률: 67.0% — 셰이핑 포함 측정치(82%)와 15%p 차이, 5개 모델 중 최저
  - 낙상률: 19.0%
  - 비낙상 평균 보상: 303.21
  - 비낙상 평균 클리어 스텝: 492.9

재현성: 동일 레시피(시드만 변경) 재실험이 2000ep 내 미수렴. 1556ep solve는 시드에 의존한 결과일 가능성 — 재현성 미확정. 배포 비권장, 참고/실험용.

실행법:
  ./run_play.sh     -> 로컬에서 5회 렌더링 재생 (noise=0)
  ./run_train.sh    -> 동일 레시피로 재현 학습 (from scratch, 재현 안 될 수 있음)
  ./run_eval100.sh  -> 위 100ep 재평가 결과를 그대로 재현 (noise=0, 고정 시드 9000~9099, speed shaping 제외)

Colab 재현: (전제: /content/bipedalwalker-td3 압축 해제 + 의존성 설치 완료, ../../COLAB_실행_가이드.md 1~2단계)

```bash
%cd /content/bipedalwalker-td3/models/model4_new-speed-s102
!bash run_train.sh    # 학습 재현 (from scratch, num_episodes=4000으로 여유 있게 설정 -- 재현 실패 가능성 있음, 위 "재현성" 참고)
!bash run_eval100.sh  # 100ep 재평가 (해결률 67% / 낙상률 19% 재현 확인, speed shaping 보너스 제외)
```

```bash
# 시연 + 녹화 (가상 디스플레이 먼저 시작 -- ../../COLAB_실행_가이드.md 3단계)
!python3 ../../src/Bipedalwalker_TD3_live.py --play 3 --ckpt ep1556 --set \
  run_tag=new-speed-s102 hardcore=true frame_skip=2 fall_penalty=-10.0 \
  activation=gelu speed_low_thresh=0.05 speed_high_thresh=0.3 speed_penalty=-0.02 speed_bonus=0.03 \
  video_record_dir=/content/demo_videos
```

개별 보고서: ../../reports/model4_report.pdf
