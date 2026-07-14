"""4개 알고리즘(TD3/SAC/PPO/DDPG) 공통 외부 평가 스크립트.

각 라이브러리가 내부적으로 로깅하는 값을 그대로 비교하면 구현마다 로깅 방식이 달라
공정하지 않다. 체크포인트를 로드해서 전부 동일 조건(같은 eval seed, deterministic
action, raw 환경)으로 여기서 재평가한 숫자만 최종 비교에 쓴다.

사용:
    # td3는 load_ckpt()가 checkpoint_dir/env_type_tag()/{태그}_actor.pt를 직접 조립하므로
    # 경로가 아니라 태그만 넘긴다 (예: ep1200)
    python3 eval_common.py --algo td3 --ckpt ep1200
    # sac/ppo/ddpg는 SB3 저장 파일(.zip) 경로를 그대로 넘긴다
    python3 eval_common.py --algo sac --ckpt results/checkpoints/algocmp-sac-s0/best_model.zip
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import config as C


def eval_td3(ckpt_path: str):
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    import Bipedalwalker_TD3_live as m

    m.cfg.hardcore = False
    m.cfg.frame_skip = 1
    m.cfg.fall_penalty = -100.0
    m.cfg.speed_penalty = 0.0
    m.cfg.speed_bonus = 0.0
    m.cfg.activation = C.ACTIVATION

    probe_env = m.build_env()
    state_dim = probe_env.observation_space.shape[0]
    action_dim = probe_env.action_space.shape[0]
    probe_env.close()

    agent = m.TD3Agent(state_dim, action_dim, m.cfg)
    m.load_ckpt(agent, ckpt_path)

    env = m.build_env(render_mode=None)
    rewards = []
    for ep in range(C.EVAL_EPISODES):
        state, _ = env.reset(seed=C.EVAL_SEED_BASE + ep)
        total, steps, done = 0.0, 0, False
        while not done and steps < m.cfg.eval_max_steps:
            action = agent.select_action(state, noise_std=0.0)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += reward
            steps += 1
        rewards.append(total)
    env.close()
    return rewards


def eval_sb3(algo: str, ckpt_path: str):
    from stable_baselines3 import SAC, PPO, DDPG
    import gymnasium as gym

    algo_cls = {"sac": SAC, "ppo": PPO, "ddpg": DDPG}[algo]
    model = algo_cls.load(ckpt_path)
    env = gym.make(C.ENV_ID, hardcore=False)

    rewards = []
    for ep in range(C.EVAL_EPISODES):
        obs, _ = env.reset(seed=C.EVAL_SEED_BASE + ep)
        total, done = 0.0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += reward
        rewards.append(total)
    env.close()
    return rewards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True, choices=C.ALGOS)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.algo == "td3":
        rewards = eval_td3(args.ckpt)
    else:
        rewards = eval_sb3(args.algo, args.ckpt)

    rewards = np.array(rewards)
    summary = {
        "algo": args.algo,
        "ckpt": args.ckpt,
        "n_episodes": C.EVAL_EPISODES,
        "mean": float(rewards.mean()),
        "std": float(rewards.std()),
        "min": float(rewards.min()),
        "max": float(rewards.max()),
        "solved_rate": float((rewards >= 300).sum() / len(rewards)),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    out = args.out or f"{C.RESULTS_DIR}/eval_{args.algo}_{Path(args.ckpt).stem}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
