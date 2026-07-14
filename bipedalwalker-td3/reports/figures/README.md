알고리즘 비교(TD3 vs SAC vs PPO vs DDPG) 슬라이드 "1M step 지점 값 — 통합 5개 데이터" 표의 원본 데이터와 검증 스크립트.

파일:
- `algo_comparison_raw_data.csv` — algo, seed_label, source(ours/teammate), env_step, reward 전체 시계열
- `algo_comparison_ours_raw.json` — 우리 팀 4시드(seed0~3) 원본 (csv와 동일 데이터, json 형식)
- `algo_comparison_teammate_raw.json` — 팀원(타 조) 원본
- `summarize_1M_step.py` — 위 csv에서 env_step=1,000,000에 가장 가까운 지점의 reward를 뽑아 슬라이드 표를 재현하는 스크립트 (외부 의존성 없음, 표준 라이브러리만 사용)

## 검증 결과 (2026-07-14)

`summarize_1M_step.py`로 재현한 값과 슬라이드에 인쇄된 값을 대조한 결과, **팀원(teammate) 열 4개 중 3개가 슬라이드 값과 다르다.**

| algo | 슬라이드 팀원 | 원본데이터 팀원(최근접 지점) | 차이 |
|---|---:|---:|---:|
| TD3 | 291.4 | 257.7 | 33.7 |
| SAC | 323.7 | 324.2 | 0.5 (반올림 수준, 일치로 간주) |
| PPO | 242.2 | 253.7 | 11.5 |
| DDPG | -125.4 | -135.6 | 10.2 |

우리 팀 4시드(seed0~3) 값은 전부 원본과 정확히 일치한다. 팀원 값만 슬라이드에 다른 출처(예: 이전 버전 스냅샷)가 들어간 것으로 보인다. 통합 MEAN/STD도 팀원 값에 연동되므로 같이 바뀐다 — 예: TD3 MEAN 267.3→260.6, STD 58.6→57.3.

슬라이드를 고칠 때는 이 스크립트 출력값을 그대로 쓰면 된다.

## Colab에서 검증

표준 라이브러리만 쓰므로 별도 설치가 필요 없다.

```bash
%cd /content/Summercamp/bipedalwalker-td3/reports/figures
!python3 summarize_1M_step.py
```

출력이 위 "검증 결과" 표(원본데이터 열)와 동일하게 나오면 재현된 것이다.
