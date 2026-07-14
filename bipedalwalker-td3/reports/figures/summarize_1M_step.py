"""1M-step 지점 알고리즘별 reward를 원본 CSV(algo_comparison_raw_data.csv)에서
직접 산출해 "1M step 지점 값 - 통합 5개 데이터" 표를 재현한다.

각 (algo, seed/teammate) 시계열에서 env_step이 1,000,000에 가장 가까운 지점의
reward를 사용한다. teammate 데이터는 로깅 주기가 달라 정확히 1,000,000 지점이
없는 경우가 있어 최근접 지점을 쓴다.

실행:
    python3 summarize_1M_step.py
"""
import csv
import statistics
from collections import defaultdict

TARGET_STEP = 1_000_000
ALGOS = ["td3", "sac", "ppo", "ddpg"]
SEEDS = ["seed0", "seed1", "seed2", "seed3"]


def load(path="algo_comparison_raw_data.csv"):
    rows = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            rows[(row["algo"], row["source"], row["seed_label"])].append(
                (int(row["env_step"]), float(row["reward"]))
            )
    return rows


def nearest(points, target=TARGET_STEP):
    return min(points, key=lambda p: abs(p[0] - target))[1]


def main():
    rows = load()
    header = f"{'algo':6}{'seed0':>10}{'seed1':>10}{'seed2':>10}{'seed3':>10}{'teammate':>10}{'mean':>10}{'std':>10}"
    print(header)
    print("-" * len(header))
    for algo in ALGOS:
        seed_vals = [nearest(rows[(algo, "ours", s)]) for s in SEEDS]
        teammate_key = next(k for k in rows if k[0] == algo and k[1] == "teammate")
        teammate_val = nearest(rows[teammate_key])
        all5 = seed_vals + [teammate_val]
        mean = statistics.mean(all5)
        std = statistics.pstdev(all5)
        line = f"{algo:6}" + "".join(f"{v:10.1f}" for v in seed_vals) + f"{teammate_val:10.1f}{mean:10.1f}{std:10.1f}"
        print(line)


if __name__ == "__main__":
    main()
