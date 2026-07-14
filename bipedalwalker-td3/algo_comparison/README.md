# TD3 vs SAC vs PPO vs DDPG — 공정 비교 실험

사전에 정리한 실험 설계(프로토콜·스크립트 골격)를 실행하기 위한 디렉토리. 학습 미실행, 골격만 구성됨.

## 공정성 원칙 (반드시 지킬 것)

1. **환경: raw `BipedalWalker-v3` (flat, hardcore 아님), wrapper 전부 제거.**
   `fall_penalty` 완화, `speed_bonus`/`speed_penalty`, `frame_skip`은 전부 TD3 전용 커리큘럼
   트릭이라 다른 알고리즘에 붙이면 "알고리즘이 잘한 건지 셰이핑이 잘한 건지" 구분이 안 됨.
   4개 알고리즘 전부 셰이핑 없는 원본 reward로 학습·평가한다.
2. **x축은 episode가 아니라 environment step.** 에피소드 길이가 알고리즘 성능에 따라
   달라지므로(못 걸으면 금방 끝남) step 수 기준으로 통일해야 비교가 성립한다.
3. **하이퍼파라미터는 "전부 동일"이 아니라 "각자 검증된 기본값".** PPO에 SAC용 하이퍼파라미터를
   강제하면 오히려 불공정해진다. TD3는 `../src/Bipedalwalker_TD3_live.py`의 기존 튜닝값을
   그대로 쓰고, SAC/PPO/DDPG는 Stable-Baselines3 zoo의 검증된 기본값을 사용한다.
4. **시드 최소 3~5개.** 특히 DDPG는 시드 편차가 큰 걸로 알려져 있음(TD3 논문이 지적한 문제
   그 자체). 평균±표준편차(shaded band)로 제시, 단일 시드 곡선은 쓰지 않는다.
5. **평가는 반드시 `eval_common.py` 하나로 통일.** 각 라이브러리의 내부 로깅을 그대로 비교하면
   안 됨 — 체크포인트를 로드해서 동일 조건(같은 eval seed, deterministic action)으로
   외부에서 재평가한다.
6. **구현 출처를 명시한다.** TD3는 우리가 직접 짠 구현, SAC/PPO/DDPG는 Stable-Baselines3
   참조 구현 — 발표 방법론 슬라이드에 이 사실을 그대로 밝힌다. 직접 구현한 버그를
   "알고리즘 성능"으로 오인할 위험을 원천 차단하기 위함.

## Colab에서 실행

로컬 GPU 없이 Colab에서 그대로 돌릴 수 있다. 다만 `config.TOTAL_STEPS = 2,000,000`이라 알고리즘 1개·시드 1개 학습에도 수 시간이 걸린다 — Colab 세션 하나로 4알고리즘×4시드(16개 런)를 전부 끝내려고 하지 말고, 아래 커맨드로 하나씩 백그라운드 실행하고 세션이 끊기지 않게 관리하는 걸 전제로 한다.

### 1) 저장소 준비 + 의존성 설치

```bash
%cd /content/Summercamp
!pip install -q -r bipedalwalker-td3/requirements.txt
!pip install -q -r bipedalwalker-td3/algo_comparison/requirements.txt
%cd bipedalwalker-td3/algo_comparison
```

렌더링을 안 하므로(학습·평가 전부 `render_mode=None`) 가상 디스플레이(xvfb)는 필요 없다.

### 2) TD3 학습 (우리 구현)

```bash
!python3 run_td3.py --seed 0
```

**주의**: 이 스크립트는 아직 `config.TOTAL_STEPS` 도달 시 자동으로 멈추는 워처가 구현돼 있지 않다(스크립트 docstring에 "TODO"로 명시됨). `num_episodes=5000`이 상한이라, 2,000,000 step 근처에서 수동으로 중단(Colab 셀 중지)해야 한다. seed는 `config.SEEDS = [0, 1, 2, 3]` 전부 돌리려면 `--seed`만 바꿔 반복 실행한다.

### 3) SAC / PPO / DDPG 학습 (Stable-Baselines3)

```bash
!python3 run_baselines.py --algo sac --seed 0
!python3 run_baselines.py --algo ppo --seed 0
!python3 run_baselines.py --algo ddpg --seed 0
```

이쪽은 SB3의 `model.learn(total_timesteps=...)`이 `TOTAL_STEPS`에서 정확히 멈추므로 수동 중단이 필요 없다. seed 0~3 반복.

### 4) 공통 재평가

체크포인트는 `results/checkpoints/bipedalwalker-td3-algocmp-td3-s{seed}/`(TD3, 메인 파이프라인의 `env_type_tag()` 규칙을 그대로 따름) 또는 `results/checkpoints/algocmp-{algo}-s{seed}/`(SAC/PPO/DDPG)에 저장된다.

```bash
# TD3: --ckpt에는 파일 경로가 아니라 태그만 넘긴다 (예: ep1200, 확장자·경로 없이)
!python3 eval_common.py --algo td3 --ckpt ep1200

# SAC/PPO/DDPG: SB3 저장 파일(.zip) 경로를 그대로 넘긴다
!python3 eval_common.py --algo sac --ckpt results/checkpoints/algocmp-sac-s0/best_model.zip
```

`--algo td3`일 때 내부적으로 메인 파이프라인의 `load_ckpt(agent, ckpt_path)`를 그대로 호출하는데, 이 함수는 `checkpoint_dir/env_type_tag()/{태그}_actor.pt` 형식을 기대한다 — 이 README 상단 docstring 예시(`--ckpt results/checkpoints/algocmp-td3-s0/ep1200.pt`)처럼 전체 경로를 넘기면 안 맞는다.

## 디렉토리 구조

```
algo_comparison/
├── README.md              (이 파일)
├── requirements.txt        SB3 등 추가 의존성
├── config.py               공유 프로토콜 상수 (step budget, seeds, eval 설정)
├── run_td3.py               우리 TD3를 raw 환경/공유 프로토콜로 실행하는 래퍼
├── run_baselines.py          SB3로 SAC/PPO/DDPG 실행
├── eval_common.py            4개 알고리즘 공통 외부 평가 스크립트
└── results/                   체크포인트/커브/평가 결과 (실행 후 생성)
```
