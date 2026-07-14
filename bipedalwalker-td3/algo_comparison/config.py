"""TD3/SAC/PPO/DDPG 공정 비교 실험의 공유 프로토콜 상수.

여기 값을 바꾸면 run_td3.py / run_baselines.py / eval_common.py 전부에 동일하게
반영된다 -- 4개 알고리즘이 같은 예산/시드/평가 조건을 쓰도록 강제하기 위함.
"""

ENV_ID = "BipedalWalker-v3"   # hardcore=False, wrapper 없는 raw 환경
TOTAL_STEPS = 2_000_000        # 알고리즘별 학습 예산 (episode가 아니라 env step 기준)
SEEDS = [0, 1, 2, 3]            # 알고리즘당 최소 4개 시드

# 학습 중 주기적으로 정지시켜 eval_common.py로 재평가할 step 간격
EVAL_EVERY_STEPS = 20_000
EVAL_EPISODES = 20               # 매 평가 시점마다 도는 held-out 에피소드 수
EVAL_SEED_BASE = 9000             # 학습 seed와 절대 겹치지 않는 평가 전용 시드 범위

# 네트워크 용량은 가능한 만큼 통일 (TD3 쪽 GELU ablation에서 검증된 값)
HIDDEN_SIZE = 256
HIDDEN_LAYERS = 2
ACTIVATION = "gelu"

ALGOS = ["td3", "sac", "ppo", "ddpg"]

RESULTS_DIR = "results"
