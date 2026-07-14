"""공정 비교용 SAC/PPO/DDPG 실행 (Stable-Baselines3 참조 구현).

TD3는 우리가 직접 짠 구현(run_td3.py)을 쓰고, 나머지 세 알고리즘은 여기서 SB3의
검증된 참조 구현을 그대로 사용한다 -- 직접 구현한 버그를 "알고리즘 성능"으로 오인하는
걸 막기 위함(발표 방법론 슬라이드에 명시할 것).

하이퍼파라미터는 SB3 zoo의 BipedalWalker-v3 기본값을 따르고(config.py의 네트워크
크기/활성화만 통일), 환경은 raw BipedalWalker-v3(hardcore=False, wrapper 없음)로 고정한다.

사용:
    pip install -r requirements.txt
    python3 run_baselines.py --algo sac --seed 0
    python3 run_baselines.py --algo ppo --seed 0
    python3 run_baselines.py --algo ddpg --seed 0
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config as C


def build_env():
    import gymnasium as gym
    return gym.make(C.ENV_ID, hardcore=False)


def make_policy_kwargs():
    import torch.nn as nn
    net_arch = [C.HIDDEN_SIZE] * C.HIDDEN_LAYERS
    return dict(net_arch=net_arch, activation_fn=nn.GELU)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True, choices=["sac", "ppo", "ddpg"])
    ap.add_argument("--seed", type=int, required=True, choices=C.SEEDS)
    args = ap.parse_args()

    from stable_baselines3 import SAC, PPO, DDPG
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor

    run_tag = f"algocmp-{args.algo}-s{args.seed}"
    results_dir = Path(C.RESULTS_DIR)
    tb_dir = results_dir / "tensorboard" / run_tag
    ckpt_dir = results_dir / "checkpoints" / run_tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_env = Monitor(build_env())
    eval_env = Monitor(build_env())

    policy_kwargs = make_policy_kwargs()

    algo_cls = {"sac": SAC, "ppo": PPO, "ddpg": DDPG}[args.algo]
    # zoo 기본값 위주. PPO는 on-policy라 net_arch를 pi/vf로 분리해야 하므로 별도 처리.
    if args.algo == "ppo":
        policy_kwargs["net_arch"] = dict(pi=[C.HIDDEN_SIZE] * C.HIDDEN_LAYERS,
                                          vf=[C.HIDDEN_SIZE] * C.HIDDEN_LAYERS)
        model = algo_cls("MlpPolicy", train_env, seed=args.seed, verbose=1,
                          tensorboard_log=str(tb_dir), policy_kwargs=policy_kwargs)
    else:
        model = algo_cls("MlpPolicy", train_env, seed=args.seed, verbose=1,
                          tensorboard_log=str(tb_dir), policy_kwargs=policy_kwargs)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(ckpt_dir),
        log_path=str(ckpt_dir),
        eval_freq=C.EVAL_EVERY_STEPS,
        n_eval_episodes=C.EVAL_EPISODES,
        deterministic=True,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=C.EVAL_EVERY_STEPS,
        save_path=str(ckpt_dir),
        name_prefix=run_tag,
    )

    print(f"[run_baselines] {run_tag} 학습 시작, total_steps={C.TOTAL_STEPS}")
    model.learn(total_timesteps=C.TOTAL_STEPS, callback=[eval_cb, ckpt_cb],
                tb_log_name=run_tag)
    model.save(str(ckpt_dir / f"{run_tag}_final"))
    print(f"[run_baselines] {run_tag} 완료")


if __name__ == "__main__":
    main()
