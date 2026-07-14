"""체크포인트 하나를 노이즈 없이·고정 시드 100개 에피소드로 재평가합니다.

학습 중 avg_reward_100은 탐험 노이즈가 섞여 있고 speed shaping 보너스가 켜진 채로
측정될 수 있어서, 모델 간 공정 비교에는 쓸 수 없습니다. 이 스크립트는
  - noise_std=0 (결정론적 행동)
  - 고정 시드 9000~9099 (매번 같은 100개 지형)
  - speed shaping 보너스/페널티는 학습 때 값과 무관하게 항상 0으로 강제
으로 재평가해서, README/발표 자료에 쓰인 해결률·낙상률·비낙상 평균·클리어 스텝을
그대로 재현합니다.

사용 예 (모델1):
  python3 evaluate_100ep.py --ckpt ep794 --set \\
      run_tag=g2-si-speed-s201 hardcore=true frame_skip=2 fall_penalty=-10.0 \\
      activation=gelu final_layer_init_scale=0.003 critic_output_scale=10.0
"""
import argparse
import json
from pathlib import Path

import numpy as np

import Bipedalwalker_TD3_live as m

EVAL_SEED_START = 9000
NUM_EPISODES = 100
SOLVE_THRESHOLD = 300.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=str, default="latest",
                    help="불러올 체크포인트 태그, 예: 'ep794'. 기본 'latest'는 가장 최근 기록 체크포인트")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="run_train.sh와 동일한 형식의 Config 오버라이드 (트릭/구조 설정만 넣으면 됩니다)")
    p.add_argument("--out-dir", type=str, default="../../results/eval100",
                    help="결과 JSON을 저장할 디렉터리")
    args = p.parse_args()

    for kv in args.set:
        key, sep, value = kv.partition("=")
        if not sep:
            raise ValueError(f"--set은 KEY=VALUE 형식이어야 합니다: {kv!r}")
        if not hasattr(m.cfg, key):
            raise ValueError(f"알 수 없는 Config 필드: {key!r}")
        current = getattr(m.cfg, key)
        setattr(m.cfg, key, m._coerce(value, current))

    return args


def main():
    args = parse_args()

    # 공식 재평가는 speed shaping 보너스/페널티를 항상 뺍니다 (학습 때 켰든 안 켰든 무관).
    m.cfg.speed_penalty = 0.0
    m.cfg.speed_bonus = 0.0

    ckpt_tag = args.ckpt
    if ckpt_tag == "latest":
        resolved = m.resolve_latest_ckpt()
        if resolved is None:
            raise FileNotFoundError(f"'{Path(m.cfg.checkpoint_dir) / m.env_type_tag()}'에 체크포인트가 없습니다.")
        ckpt_tag = resolved

    env = m.build_env()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = m.TD3Agent(state_dim, action_dim, m.cfg)
    m.load_ckpt(agent, ckpt_tag)

    rewards, steps, fell = [], [], []
    for i in range(NUM_EPISODES):
        seed = EVAL_SEED_START + i
        state, _ = env.reset(seed=seed)
        total_reward, step_count, done = 0.0, 0, False
        while not done and step_count < m.cfg.eval_max_steps:
            action = agent.select_action(state, noise_std=0.0)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step_count += 1
        rewards.append(total_reward)
        steps.append(step_count)
        fell.append(bool(env.unwrapped.game_over))
        if (i + 1) % 10 == 0:
            print(f"[100ep 재평가] ... {i + 1}/{NUM_EPISODES} 완료")

    env.close()

    rewards_arr = np.array(rewards)
    fell_arr = np.array(fell)
    solved = rewards_arr >= SOLVE_THRESHOLD

    solve_rate = float(solved.mean() * 100)
    fall_rate = float(fell_arr.mean() * 100)
    non_fall_mean_reward = float(rewards_arr[~fell_arr].mean()) if (~fell_arr).any() else float("nan")
    solved_steps = [s for s, ok in zip(steps, solved) if ok]
    mean_clear_steps = float(np.mean(solved_steps)) if solved_steps else float("nan")

    summary = {
        "run_tag": m.cfg.run_tag,
        "ckpt": ckpt_tag,
        "num_episodes": NUM_EPISODES,
        "eval_seed_range": [EVAL_SEED_START, EVAL_SEED_START + NUM_EPISODES - 1],
        "solve_rate_pct": round(solve_rate, 2),
        "fall_rate_pct": round(fall_rate, 2),
        "non_fall_mean_reward": round(non_fall_mean_reward, 2),
        "mean_clear_steps": round(mean_clear_steps, 2),
        "episode_rewards": [float(r) for r in rewards],
        "episode_steps": [int(s) for s in steps],
        "episode_fell": fell,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{m.cfg.run_tag or 'model'}_{ckpt_tag}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[100ep 재평가] run_tag={m.cfg.run_tag} ckpt={ckpt_tag}")
    print(f"  해결률(reward>=300): {solve_rate:.1f}%")
    print(f"  낙상률: {fall_rate:.1f}%")
    print(f"  비낙상 평균 reward: {non_fall_mean_reward:.2f}")
    print(f"  평균 클리어 step: {mean_clear_steps:.1f}")
    print(f"  저장 완료 -> {out_path}")


if __name__ == "__main__":
    main()
