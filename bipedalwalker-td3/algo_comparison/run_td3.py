"""공정 비교용 TD3 실행 래퍼.

우리 기존 파이프라인(../src/Bipedalwalker_TD3_live.py)은 episode 수 기준으로만 멈추고
env step 총량 기준의 조기 종료 로직이 없다. 학습 루프 자체를 건드리면 다른 실험에서
쓰는 공용 파이프라인이 흔들리므로, 여기서는 셰이핑을 전부 끄고 raw 환경으로 충분히 긴
num_episodes로 띄운 뒤 -- config.TOTAL_STEPS 도달 시 auto_stop_watcher.sh와 같은 방식의
외부 워처로 강제 종료하는 걸 전제로 한다 (watcher 스크립트는 아직 미작성, TODO).

사용:
    python3 run_td3.py --seed 0
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config as C

PIPELINE = Path(__file__).parent.parent / "src" / "Bipedalwalker_TD3_live.py"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, choices=C.SEEDS)
    args = ap.parse_args()

    run_tag = f"algocmp-td3-s{args.seed}"
    cmd = [
        sys.executable, str(PIPELINE),
        "--set",
        f"run_tag={run_tag}",
        "hardcore=false",
        "frame_skip=1",          # raw 환경 -- 다른 알고리즘과 동일 조건
        "fall_penalty=-100.0",   # gym 기본값, 완화 없음
        "speed_penalty=0.0",
        "speed_bonus=0.0",
        f"activation={C.ACTIVATION}",
        f"seed={args.seed}",
        "num_episodes=5000",      # 상한선. 실제 종료는 TOTAL_STEPS 기준 외부 워처가 담당 (TODO)
        f"tb_dir={C.RESULTS_DIR}/tensorboard",
        f"checkpoint_dir={C.RESULTS_DIR}/checkpoints",
    ]
    print("[run_td3]", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
