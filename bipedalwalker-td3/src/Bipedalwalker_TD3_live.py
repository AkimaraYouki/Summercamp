"""
BipedalWalker-v3 TD3 구현 -- LIVE-PLAY 변형 (Bipedalwalker_TD3.py를 복제한 버전)
- 환경: Gymnasium BipedalWalker-v3 (Box2D 필요)
- 알고리즘: TD3 (Twin Delayed Deep Deterministic Policy Gradient)
  -> 행동이 연속값(4차원, [-1, 1])이라 DQN을 쓸 수 없어 Actor-Critic 구조를 사용합니다.
  -> DDPG보다 Critic 과대평가 문제를 줄이기 위해 Critic 2개, 타겟 정책 노이즈, 지연된 Actor 업데이트를 사용합니다.

이 파일은 원본 Bipedalwalker_TD3.py를 안전하게 그대로 두기 위해 복제한 버전으로,
원본에는 없는 두 기능이 추가돼 있습니다:
  1) --play 로 render_mode="human" 창을 띄우면 pygame 화면에 실시간 점수/스텝/관측벡터
     (hull 각도·각속도, 전후진 속도, 힙/무릎 각도·각속도, 다리 접지 여부)를 오버레이합니다.
  2) --play --live 로 재생하면서 동시에 그 경험(버퍼 적재+업데이트)으로 이어서 학습할 수
     있습니다. 이때 전진 속도(vel_x) 3구간 보상 셰이핑(speed_low_thresh/high_thresh/
     penalty/bonus)도 함께 켤 수 있어, 매 스텝 꾸준히 빨리 걸어야만 가점을 받고 "한 번
     반짝 빨리 갔다가 느려지는" 전략은 통하지 않게 만들 수 있습니다.
  두 기능 다 관련 Config 값이 기본(off)이면 원본과 완전히 동일하게 동작합니다.

설치 (Box2D 필요! Python 3.12 가상환경 권장):
    py -3.12 -m venv venv
    venv\\Scripts\\activate
    pip install swig "gymnasium[box2d]" torch numpy matplotlib

실행 (하이퍼파라미터는 Config를 고치거나 --set으로, 실행 모드만 CLI로 고른다):
    python Bipedalwalker_TD3_live.py                                     # 학습
    python Bipedalwalker_TD3_live.py --play 3 --ckpt ep400               # 특정 체크포인트로 재생(HUD 오버레이 자동 적용)
    python Bipedalwalker_TD3_live.py --play 10 --live --ckpt ep2585 \\
        --set frame_skip=2 fall_penalty=-10.0 activation=gelu run_tag=hc-tricks-gelu \\
              speed_low_thresh=0.05 speed_high_thresh=0.3 speed_penalty=-0.02 speed_bonus=0.03 live_noise_std=0.05
        # ep2585 재생하면서 속도 3구간 보상 셰이핑 켜고 이어서 학습, live-epN으로 신기록 저장
    python Bipedalwalker_TD3_live.py --set frame_skip=2 activation=gelu run_tag=hc-anneal \\
        fall_penalty_anneal=true fall_penalty_start=-10.0 fall_penalty_end=-100.0 fall_penalty_anneal_episodes=1500
        # 초반엔 관대(-10)하다가 1500ep에 걸쳐 원래 gym 수준(-100)까지 점점 엄격해지는 추락 페널티
        # (--set은 한 번만 쓰고 안에 값들을 공백으로 나열 -- 두 번 쓰면 앞에 쓴 --set은 무시됨)
    python Bipedalwalker_TD3_live.py --play 3 --set hud_overlay=false   # 오버레이 끄고 순수 재생만

체크포인트: 학습 중 최근 100개 평균 리워드가 기록을 갱신할 때마다
checkpoints/bipedalwalker-td3-<hardcore|run_tag>/ep{episode}_*.pt 로 자동 저장됩니다.
--live 로 이어서 학습할 때 새로 갱신되는 기록은 live-ep{episode}_*.pt 로 따로 저장되어
원본 epN 체크포인트를 덮어쓰지 않습니다.
"""

import argparse
import re
import time
import random
import collections
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import gymnasium as gym

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 0. 하이퍼파라미터
# ============================================================
@dataclass
class Config:
    env_id: str = "BipedalWalker-v3"
    seed: int = 42

    num_episodes: int = 5000
    max_steps_per_episode: int = 1600   # 공식 문서 기준 해결 조건

    gamma: float = 0.99
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    tau: float = 0.005                  # 타겟 네트워크 소프트 업데이트 비율

    # TD3 전용 설정
    # policy_noise: 타겟 행동에 더하는 작은 노이즈입니다.
    # noise_clip: 타겟 행동 노이즈가 너무 커지지 않도록 자르는 범위입니다.
    # policy_delay: Critic은 매번 업데이트하고 Actor는 몇 번에 한 번만 업데이트합니다.
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2

    batch_size: int = 128
    buffer_size: int = 300_000
    min_buffer_size: int = 2000

    # GPU 사용 시에도 환경 시뮬레이션은 CPU에서 돌기 때문에,
    # 너무 자주 업데이트하면 CPU-GPU 데이터 이동 때문에 오히려 느려질 수 있습니다.
    # 1이면 매 step마다 학습, 2이면 두 step마다 한 번 학습합니다.
    update_every: int = 1

    # 탐험용 가우시안 노이즈 (연속 행동이라 epsilon-greedy 대신 노이즈를 더함)
    noise_std_start: float = 0.3
    noise_std_end: float = 0.05
    noise_decay_episodes: int = 800

    solved_threshold: float = 300.0     # 공식 문서 기준 (300점 이상)

    # hardcore=True: 매 에피소드 장애물(둔덕/구덩이/계단)이 무작위로 섞여 나오는
    # 공식 hardcore 난이도. False면 일반 BipedalWalker-v3(장애물 없음, 훨씬 쉬움).
    hardcore: bool = True

    # ---- hardcore 수렴을 돕는 트릭 (ugurcanozalp/td3-sac-bipedal-walker-hardcore-v3 참고) ----
    # frame_skip: 같은 행동을 이 횟수만큼 반복 적용하고 보상을 합산합니다(제어 주파수를
    # 낮춰 크레딧 할당을 쉽게 함). 1이면 기존과 동일(매 프레임 새 행동).
    frame_skip: int = 1
    # fall_penalty: 추락(game_over)으로 에피소드가 끝난 스텝의 보상을 이 값으로 덮어씁니다.
    # gym 기본값 -100은 hardcore에서 너무 가혹해 에이전트가 탐험을 꺼리게 만들 수 있습니다.
    # -100.0이면 기존과 동일(덮어쓰지 않음). hardcore에서 잘 안 풀리면 -10.0 근처를 시도.
    # fall_penalty_anneal=True면 아래 값 대신 fall_penalty_start/end로 에피소드에 따라
    # 점점 세지는(-10 -> -100) 어닐링을 사용합니다.
    fall_penalty: float = -100.0
    # ---- fall_penalty 어닐링 ----
    # 문제의식: fall_penalty를 -10으로 고정하면 학습 초반(에피소드가 짧아 못 벌어놓은 보상이
    # 적을 때)엔 "과감한 탐험"을 잘 유도하지만, 학습이 진행돼 한 에피소드에서 200~300점을
    # 벌 수 있게 된 뒤에도 여전히 -10이면 그 페널티가 상대적으로 너무 작아져서 "가끔 넘어져도
    # 괜찮다"고 잘못 학습할 수 있습니다. noise_std_start/end와 같은 패턴으로, 에피소드가
    # 진행될수록 fall_penalty_start(관대함) -> fall_penalty_end(원래 gym 수준의 엄격함)로
    # 선형 보간해 "초반엔 관대하게 탐험 유도, 후반엔 넘어짐을 진짜로 나쁘게" 만듭니다.
    fall_penalty_anneal: bool = False
    fall_penalty_start: float = -10.0
    fall_penalty_end: float = -100.0
    fall_penalty_anneal_episodes: int = 1500

    # ---- 신경망 구조 (Actor/Critic 공용) ----
    hidden_dim: int = 256                # 은닉층 하나당 유닛(뉴런) 수
    num_hidden_layers: int = 2           # 은닉층 개수 (원본 구조와 동일하게 기본 2)
    # "relu" / "tanh" / "leaky_relu" / "elu" / "gelu" 중 하나
    activation: str = "relu"

    # ---- 학습 안정화 트릭 (ugurcanozalp/td3-sac-bipedal-walker-hardcore-v3 참고) ----
    # use_layer_norm: 각 은닉층(Linear) 뒤, 활성함수 전에 LayerNorm을 추가합니다.
    # False면 기존과 동일(정규화 없음).
    use_layer_norm: bool = False
    # final_layer_init_scale: Actor/Critic 마지막 층 가중치를 uniform(-s, +s)로 작게
    # 초기화하고 bias를 없앱니다(DDPG 논문 관례 - 학습 초반 정책/Q값이 요동치는 것을 억제).
    # 0.0이면 기존과 동일(PyTorch 기본 초기화, bias 포함). 참고 repo는 0.003을 사용.
    final_layer_init_scale: float = 0.0
    # critic_output_scale: Critic이 출력하는 Q값에 곱하는 배율. 1.0이면 기존과 동일.
    # 참고 repo는 마지막 층을 작게 초기화한 만큼 초기 Q값이 너무 작아지지 않도록 10을 곱함.
    critic_output_scale: float = 1.0

    # ---- Live-play 전용 (이 파일에서만 사용) ----
    # hud_overlay: --play로 render_mode="human" 창을 띄울 때, 매 스텝 점수/스텝수/관측벡터를
    # 화면에 텍스트로 오버레이합니다. render_mode가 human이 아니면 자동으로 아무 효과 없음.
    hud_overlay: bool = True
    # hud_model_label: 비어있지 않으면 HUD 상단에 "어느 모델을 재생 중인지" 배너로 표시합니다.
    # 시연 영상 녹화 시 여러 모델을 구분하기 위한 용도. 기본값 ""이면 기존과 동일(배너 없음).
    hud_model_label: str = ""
    # video_record_dir: 비어있지 않으면 --play 재생 중 각 에피소드를 mp4로 저장합니다.
    # 기본값 ""이면 기존과 완전히 동일(녹화 없음).
    video_record_dir: str = ""
    # 전진 속도(obs[2], 정규화된 vel_x) 기준 3구간 보상 셰이핑.
    # speed_low_thresh 미만: speed_penalty 적용 (너무 느림/제자리)
    # speed_low_thresh ~ speed_high_thresh: 보상 변화 없음 (중립 구간)
    # speed_high_thresh 이상: speed_bonus 적용 (충분히 빠름)
    # 매 스텝 즉시 판정하므로 "한 번 반짝 빨리 갔다가 느려지는" 전략은 계속 가점을 못 받고,
    # speed_penalty/speed_bonus가 둘 다 0.0이면(기본값) 기존과 완전히 동일하게 동작.
    speed_low_thresh: float = 0.05
    speed_high_thresh: float = 0.3
    speed_penalty: float = 0.0
    speed_bonus: float = 0.0
    # live_learn: --play --live 조합일 때, 화면으로 보면서 동시에 버퍼 적재+업데이트를
    # 계속 진행합니다(재생하는 에피소드 경험으로 이어서 학습).
    live_learn: bool = False
    live_noise_std: float = 0.05   # live_learn 중 사용할 탐험 노이즈 크기

    # 중간 시각화 설정
    # render_mode="human"은 학습 속도를 크게 떨어뜨리므로 기본값은 False로 둡니다.
    enable_visualization: bool = False  # 학습 중간중간 렌더링 창을 띄울지 여부
    visualize_every: int = 250          # 몇 에피소드마다 한 번씩 시각화할지

    checkpoint_dir: str = "checkpoints"  # 체크포인트 저장 위치

    # 같은 설정으로 하이퍼파라미터만 바꿔서 여러 개를 동시에 돌릴 때 체크포인트/
    # 텐서보드/리워드기록이 서로 안 섞이게 프로세스마다 다르게 지정.
    # 예: --set run_tag=wide400
    run_tag: str = ""

    print_every: int = 1                # 몇 에피소드마다 콘솔에 리워드를 찍을지 (1=매 에피소드)
    tensorboard: bool = True            # TensorBoard 로깅 여부
    tb_dir: str = "tensorboard"         # TensorBoard 로그 저장 위치

    eval_max_steps: int = 2000          # evaluate()에서 에피소드당 최대 스텝 (무한루프 방지용 상한)

    # "auto": CUDA 가능하면 GPU, 아니면 CPU / "cuda": CUDA 강제(없으면 경고 후 CPU) / "cpu": CPU 강제
    device: str = "auto"
    cpu_threads: int = 1                # CPU일 때 스레드 수 (보통 1이 더 빠름)


def setup_device(device_setting: str) -> torch.device:
    """PyTorch 학습 장치를 선택하고 CUDA 정보를 출력합니다."""
    device_setting = str(device_setting).lower().strip()

    if device_setting not in {"auto", "cuda", "cpu"}:
        raise ValueError('Config.device는 "auto", "cuda", "cpu" 중 하나여야 합니다.')

    if device_setting == "cpu":
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        if device_setting == "cuda":
            print("[경고] Config.device='cuda'로 설정했지만 CUDA GPU를 찾지 못했습니다. CPU로 실행합니다.")
        device = torch.device("cpu")

    if device.type == "cuda":
        print(f"[GPU 사용] {torch.cuda.get_device_name(0)}")
        print(f"[CUDA 버전] torch.version.cuda = {torch.version.cuda}")
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    else:
        print("[CPU 사용] torch.cuda.is_available() = False")

    return device


cfg = Config()
cfg.device = setup_device(cfg.device)

random.seed(cfg.seed)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
if cfg.device.type == "cuda":
    torch.cuda.manual_seed_all(cfg.seed)
else:
    torch.set_num_threads(cfg.cpu_threads)


class FrameSkipRewardShapeWrapper(gym.Wrapper):
    """hardcore 수렴에 도움이 됐다고 알려진 두 트릭을 하나로 묶은 래퍼
    (참고: ugurcanozalp/td3-sac-bipedal-walker-hardcore-v3 의 MyWalkerWrapper).
    1) frame_skip: 같은 행동을 skip번 반복하고 보상을 합산해 제어 주파수를 낮춥니다.
    2) fall_penalty: 추락(game_over)으로 끝난 스텝의 보상을 이 값으로 덮어씁니다.
       gym 기본값은 -100인데, 이 페널티가 너무 크면 에이전트가 넘어지는 것 자체를
       극도로 피하려다 hardcore의 장애물을 넘는 탐험을 포기해버립니다. -10 근처로
       완화하면 "넘어져도 괜찮으니 과감하게 탐험"하도록 유도할 수 있습니다.
    frame_skip=1, fall_penalty=-100.0 이면 원래 환경과 동일하게 동작합니다."""

    def __init__(self, env: gym.Env, frame_skip: int = 1, fall_penalty: float = -100.0):
        super().__init__(env)
        self.frame_skip = max(1, frame_skip)
        self.fall_penalty = fall_penalty

    def step(self, action):
        total_reward = 0.0
        obs, reward, terminated, truncated, info = None, 0.0, False, False, {}
        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if self.env.unwrapped.game_over:
                reward = self.fall_penalty
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class SpeedShapingWrapper(gym.Wrapper):
    """전진 속도(obs[2], 정규화된 vel_x)를 3구간으로 나눠 보상을 더하는 래퍼.
    speed_low_thresh 미만이면 penalty, speed_high_thresh 이상이면 bonus, 그 사이는 그대로.
    매 스텝 즉시 판정하기 때문에 "한 번 반짝 빨리 갔다가 느려지는" 전략은 계속 가점을
    못 받고, 꾸준히 빠르게 걸어야만 누적 보상이 늘어납니다.
    penalty=0.0, bonus=0.0 이면 원래 환경과 동일하게 동작합니다."""

    def __init__(self, env: gym.Env, low_thresh: float = 0.05, high_thresh: float = 0.3,
                 penalty: float = 0.0, bonus: float = 0.0):
        super().__init__(env)
        self.low_thresh = low_thresh
        self.high_thresh = high_thresh
        self.penalty = penalty
        self.bonus = bonus

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        vel_x = obs[2]
        if vel_x < self.low_thresh:
            reward += self.penalty
        elif vel_x >= self.high_thresh:
            reward += self.bonus
        return obs, reward, terminated, truncated, info


_OBS_HUD_FONT_CACHE: dict = {}


class HudOverlayWrapper(gym.Wrapper):
    """render_mode="human"일 때 pygame 창에 실시간 점수/스텝/관측벡터를 오버레이합니다.
    렌더링에만 관여하고 obs/reward/done은 그대로 통과시키므로 학습에는 영향이 없습니다."""

    def __init__(self, env: gym.Env, model_label: str = ""):
        super().__init__(env)
        self.total_reward = 0.0
        self.step_count = 0
        self.model_label = model_label
        # 비어있지 않은 리스트를 넣어두면(run_play가 에피소드마다 새로 할당) 매 프레임을
        # 캡처해 담습니다 -- 시연 영상 녹화용. None이면 기존과 동일하게 아무 캡처도 안 함.
        self.frame_sink: list | None = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.total_reward = 0.0
        self.step_count = 0
        self._draw(obs, 0.0)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.total_reward += reward
        self.step_count += 1
        self._draw(obs, reward)
        return obs, reward, terminated, truncated, info

    def _draw(self, obs, step_reward: float):
        base = self.unwrapped
        if getattr(base, "render_mode", None) != "human" or getattr(base, "screen", None) is None:
            return
        import pygame

        if "big" not in _OBS_HUD_FONT_CACHE:
            pygame.font.init()
            _OBS_HUD_FONT_CACHE["big"] = pygame.font.SysFont("Menlo", 16)
            _OBS_HUD_FONT_CACHE["small"] = pygame.font.SysFont("Menlo", 13)
            _OBS_HUD_FONT_CACHE["label"] = pygame.font.SysFont("Menlo", 20, bold=True)
        font_big = _OBS_HUD_FONT_CACHE["big"]
        font_small = _OBS_HUD_FONT_CACHE["small"]
        font_label = _OBS_HUD_FONT_CACHE["label"]

        y = 8
        if self.model_label:
            text = f"▶ {self.model_label}"
            surf = font_label.render(text, True, (255, 255, 255))
            banner = pygame.Surface((surf.get_width() + 20, surf.get_height() + 10))
            banner.fill((20, 90, 200))
            banner.blit(surf, (10, 5))
            base.screen.blit(banner, (10, y))
            y += banner.get_height() + 8

        lines = [
            (f"step {self.step_count:4d}   step_r {step_reward:+6.2f}   total {self.total_reward:8.2f}", font_big),
            (f"hull_angle {obs[0]:+.2f}  ang_vel {obs[1]:+.2f}  vel_x {obs[2]:+.2f}  vel_y {obs[3]:+.2f}", font_small),
            (f"hip1 {obs[4]:+.2f}/{obs[5]:+.2f}  knee1 {obs[6]:+.2f}/{obs[7]:+.2f}  contact1 {obs[8]:.0f}", font_small),
            (f"hip2 {obs[9]:+.2f}/{obs[10]:+.2f}  knee2 {obs[11]:+.2f}/{obs[12]:+.2f}  contact2 {obs[13]:.0f}", font_small),
        ]
        for text, font in lines:
            shadow = font.render(text, True, (0, 0, 0))
            surf = font.render(text, True, (255, 230, 0))
            base.screen.blit(shadow, (11, y + 1))
            base.screen.blit(surf, (10, y))
            y += surf.get_height() + 2
        pygame.display.flip()

        if self.frame_sink is not None:
            frame = pygame.surfarray.array3d(base.screen)
            self.frame_sink.append(frame.transpose(1, 0, 2))


def find_wrapper(env: gym.Env, wrapper_cls: type) -> gym.Wrapper | None:
    """래퍼 체인을 따라가며 wrapper_cls의 인스턴스를 찾습니다 (없으면 None)."""
    while isinstance(env, gym.Wrapper):
        if isinstance(env, wrapper_cls):
            return env
        env = env.env
    return None


def build_env(render_mode: str | None = None):
    env = gym.make(cfg.env_id, hardcore=cfg.hardcore, render_mode=render_mode)
    fall_penalty_active = cfg.fall_penalty_anneal or cfg.fall_penalty != -100.0
    if cfg.frame_skip != 1 or fall_penalty_active:
        initial_fall_penalty = cfg.fall_penalty_start if cfg.fall_penalty_anneal else cfg.fall_penalty
        env = FrameSkipRewardShapeWrapper(env, frame_skip=cfg.frame_skip, fall_penalty=initial_fall_penalty)
    if cfg.speed_penalty != 0.0 or cfg.speed_bonus != 0.0:
        env = SpeedShapingWrapper(env, low_thresh=cfg.speed_low_thresh, high_thresh=cfg.speed_high_thresh,
                                   penalty=cfg.speed_penalty, bonus=cfg.speed_bonus)
    if cfg.hud_overlay and render_mode == "human":
        env = HudOverlayWrapper(env, model_label=cfg.hud_model_label)
    return env


ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
}


def get_activation(name: str) -> type[nn.Module]:
    key = str(name).lower().strip()
    if key not in ACTIVATIONS:
        raise ValueError(f"알 수 없는 activation: {name!r} (선택지: {list(ACTIVATIONS)})")
    return ACTIVATIONS[key]


# ============================================================
# 0-1. CLI + 체크포인트
#      하이퍼파라미터는 위 Config를 직접 고치는 게 기본, CLI는 그때그때
#      다르게 주고 싶은 것만: 학습할지/재생만 할지, 어느 체크포인트를 쓸지,
#      화면에 렌더링할지. Config 필드를 한 번만 다르게 쓰고 싶으면
#      --set key=value로 덮어쓸 수 있음 (파일을 고치지 않고).
# ============================================================
def _coerce(value: str, like):
    """--set KEY=VALUE의 문자열 값을, 기존 Config 필드 값(like)과 같은
    타입(bool/int/float/None/str)으로 바꿔준다."""
    if isinstance(like, bool):
        return value.strip().lower() in ("1", "true", "yes", "on")
    if value == "None":
        return None
    if isinstance(like, int):
        return int(value)
    if isinstance(like, float):
        return float(value)
    return value


def env_type_tag() -> str:
    """체크포인트를 저장/검색할 하위 폴더 이름."""
    parts = ["bipedalwalker", "td3"]
    if cfg.hardcore:
        parts.append("hardcore")
    if cfg.run_tag:
        parts.append(cfg.run_tag)
    return "-".join(parts)


def save_ckpt(agent: "TD3Agent", tag: str) -> None:
    ckpt_dir = Path(cfg.checkpoint_dir) / env_type_tag()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(agent.actor.state_dict(), ckpt_dir / f"{tag}_actor.pt")
    torch.save(agent.critic1.state_dict(), ckpt_dir / f"{tag}_critic1.pt")
    torch.save(agent.critic2.state_dict(), ckpt_dir / f"{tag}_critic2.pt")


def load_ckpt(agent: "TD3Agent", tag: str) -> None:
    ckpt_dir = Path(cfg.checkpoint_dir) / env_type_tag()
    agent.actor.load_state_dict(torch.load(ckpt_dir / f"{tag}_actor.pt", map_location=cfg.device))
    agent.critic1.load_state_dict(torch.load(ckpt_dir / f"{tag}_critic1.pt", map_location=cfg.device))
    agent.critic2.load_state_dict(torch.load(ckpt_dir / f"{tag}_critic2.pt", map_location=cfg.device))


def resync_targets(agent: "TD3Agent") -> None:
    """이어서 학습(--resume)할 때만 필요: 타겟 네트워크는 에이전트 생성 시점에
    (그때의) 온라인 네트워크를 그대로 복사해서 만들어지는데, load_ckpt()로 온라인
    네트워크 가중치를 나중에 덮어써도 타겟은 자동으로 안 따라온다. 그대로 두면
    학습 초반 타겟이 여전히 "생성 당시의 랜덤 초기화값"이라 막 불러온 좋은
    가중치가 아니라 그 랜덤 타겟 쪽으로 다시 끌려간다 -- 그래서 이어서 학습을
    시작하기 전에 타겟도 강제로 다시 맞춰준다."""
    agent.actor_target.load_state_dict(agent.actor.state_dict())
    agent.critic1_target.load_state_dict(agent.critic1.state_dict())
    agent.critic2_target.load_state_dict(agent.critic2.state_dict())


def resolve_latest_ckpt() -> str | None:
    """가장 에피소드 번호가 큰(=가장 나중에 기록을 갱신했던) 체크포인트 태그를 찾는다."""
    ckpt_dir = Path(cfg.checkpoint_dir) / env_type_tag()
    best = None
    if ckpt_dir.exists():
        for p in ckpt_dir.glob("ep*_actor.pt"):
            m = re.search(r"ep(\d+)_", p.name)
            if m:
                ep = int(m.group(1))
                if best is None or ep > best:
                    best = ep
    return f"ep{best}" if best is not None else None


def parse_args():
    p = argparse.ArgumentParser(
        description="BipedalWalker TD3 -- 하이퍼파라미터는 Config를 고치거나 --set으로, "
                     "실행 방식만 아래 옵션으로 고른다."
    )
    p.add_argument("--play", type=int, default=0, metavar="N",
                    help="학습 없이 체크포인트를 불러와 N 에피소드 화면에 렌더링하고 종료")
    p.add_argument("--live", action="store_true",
                    help="--play와 함께 사용: 화면을 보면서 동시에 버퍼 적재+업데이트를 계속 진행 "
                         "(재생한 에피소드 경험으로 이어서 학습, 새 기록마다 live-ep{N} 체크포인트 저장)")
    p.add_argument("--skip-train", action="store_true",
                    help="학습을 건너뛰고 기존 체크포인트로 시연만 실행")
    p.add_argument("--ckpt", type=str, default="latest",
                    help="불러올 체크포인트 태그, 예: 'ep400'. 기본 'latest'는 가장 최근 기록 체크포인트")
    p.add_argument("--render", action="store_true",
                    help="(일반 학습 모드) 학습 후 시연 에피소드를 화면에 렌더링")
    p.add_argument("--plot", action="store_true",
                    help="학습 없이, 저장된 리워드 기록으로 학습 곡선 그래프만 띄우고 종료")
    p.add_argument("--resume", action="store_true",
                    help="학습을 처음부터 새로 하지 않고, --ckpt 체크포인트 가중치를 불러와서 이어서 학습 "
                         "(가중치만 이어받음; 리플레이 버퍼/에피소드 카운터는 새로 시작)")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="Config 필드를 이번 실행에서만 덮어쓰기, 예: --set run_tag=wide hidden_dim=400 num_hidden_layers=3")
    args = p.parse_args()

    for kv in args.set:
        key, sep, value = kv.partition("=")
        if not sep:
            raise ValueError(f"--set은 KEY=VALUE 형식이어야 합니다: {kv!r}")
        if not hasattr(cfg, key):
            raise ValueError(f"알 수 없는 Config 필드: {key!r} (이 파일 상단 Config 클래스 참고)")
        current = getattr(cfg, key)
        setattr(cfg, key, _coerce(value, current))

    return args


def run_play(n_episodes: int, ckpt_tag: str) -> None:
    """학습 없이 저장된 체크포인트만 불러와서 화면에 재생."""
    probe_env = build_env()
    state_dim = probe_env.observation_space.shape[0]
    action_dim = probe_env.action_space.shape[0]
    probe_env.close()

    agent = TD3Agent(state_dim, action_dim, cfg)
    ckpt = ckpt_tag if ckpt_tag != "latest" else resolve_latest_ckpt()
    if ckpt is None:
        raise FileNotFoundError(
            f"'{Path(cfg.checkpoint_dir) / env_type_tag()}'에 체크포인트가 없습니다. "
            f"먼저 인자 없이 실행해서 학습부터 해주세요."
        )
    load_ckpt(agent, ckpt)
    print(f"[play] 체크포인트 '{ckpt}' 로드 완료, {n_episodes}에피소드 재생")
    evaluate(agent, num_episodes=n_episodes, render=True, video_dir=cfg.video_record_dir or None)


def run_play_live(n_episodes: int, ckpt_tag: str) -> None:
    """--play --live 조합: 체크포인트를 불러와 화면으로 재생하면서 동시에 그 경험으로
    버퍼 적재+업데이트를 계속 진행합니다(리플레이 버퍼/옵티마이저 상태는 새로 시작,
    가중치만 이어받음 -- --resume과 동일한 원칙). 원본 체크포인트(epN)를 덮어쓰지
    않도록 새 기록은 항상 'live-epN' 태그로 따로 저장합니다."""
    probe_env = build_env()
    state_dim = probe_env.observation_space.shape[0]
    action_dim = probe_env.action_space.shape[0]
    probe_env.close()

    agent = TD3Agent(state_dim, action_dim, cfg)
    ckpt = ckpt_tag if ckpt_tag != "latest" else resolve_latest_ckpt()
    if ckpt is None:
        raise FileNotFoundError(
            f"'{Path(cfg.checkpoint_dir) / env_type_tag()}'에 체크포인트가 없습니다. "
            f"먼저 인자 없이 실행해서 학습부터 해주세요."
        )
    load_ckpt(agent, ckpt)
    resync_targets(agent)
    print(f"[live-play] 체크포인트 '{ckpt}' 로드 완료, {n_episodes}에피소드 재생하며 이어서 학습합니다 "
          f"(신기록마다 live-epN 으로 별도 저장, 원본 체크포인트는 보존)")

    env = build_env(render_mode="human")
    recent = collections.deque(maxlen=max(10, n_episodes))
    best_avg = -float("inf")

    for ep in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0.0
        step = 0
        done = False
        while not done and step < cfg.max_steps_per_episode:
            action = agent.select_action(state, noise_std=cfg.live_noise_std)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.buffer.push(state, action, reward, next_state, float(done))
            agent.update()

            state = next_state
            total_reward += reward
            step += 1

        recent.append(total_reward)
        avg_recent = sum(recent) / len(recent)
        print(f"[live-play] Episode {ep + 1}/{n_episodes}: Reward={total_reward:8.2f} "
              f"(최근 {len(recent)}개 평균 {avg_recent:8.2f})")

        if avg_recent > best_avg:
            best_avg = avg_recent
            tag = f"live-ep{ep + 1}"
            save_ckpt(agent, tag)
            print(f"  >> live 학습 중 신기록! -> {tag} 체크포인트 저장")

    env.close()
    print(f"[live-play] 완료. 체크포인트: {cfg.checkpoint_dir}/{env_type_tag()}/live-ep*.pt")


# ============================================================
# 1. Actor 네트워크
#    상태(24차원) -> 행동(4차원, [-1, 1] 범위)
#    마지막에 tanh를 씌워 행동 범위를 [-1, 1]로 강제합니다.
#    은닉층 개수/크기/활성함수는 Config(hidden_dim/num_hidden_layers/activation)로 조절.
# ============================================================
def _make_hidden_stack(in_dim: int, hidden_dim: int, num_hidden_layers: int,
                        act_cls: type[nn.Module], use_layer_norm: bool) -> list[nn.Module]:
    """은닉층 Linear(-> LayerNorm) -> 활성함수를 num_hidden_layers번 쌓습니다."""
    layers: list[nn.Module] = []
    dim = in_dim
    for _ in range(num_hidden_layers):
        layers.append(nn.Linear(dim, hidden_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(act_cls())
        dim = hidden_dim
    return layers


def _make_final_layer(hidden_dim: int, out_dim: int, init_scale: float) -> nn.Linear:
    """init_scale > 0 이면 DDPG 관례대로 uniform(-s, +s)로 작게 초기화하고 bias를 없앱니다."""
    if init_scale > 0:
        layer = nn.Linear(hidden_dim, out_dim, bias=False)
        nn.init.uniform_(layer.weight, -init_scale, init_scale)
    else:
        layer = nn.Linear(hidden_dim, out_dim)
    return layer


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dim: int = 256, num_hidden_layers: int = 2, activation: str = "relu",
                 use_layer_norm: bool = False, final_layer_init_scale: float = 0.0):
        super().__init__()
        act_cls = get_activation(activation)
        layers = _make_hidden_stack(state_dim, hidden_dim, num_hidden_layers, act_cls, use_layer_norm)
        layers += [_make_final_layer(hidden_dim, action_dim, final_layer_init_scale), nn.Tanh()]  # 출력을 [-1, 1]로 제한
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


# ============================================================
# 2. Critic 네트워크
#    (상태, 행동) 쌍 -> Q값 (그 상태에서 그 행동을 했을 때 기대되는 누적 보상)
# ============================================================
class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dim: int = 256, num_hidden_layers: int = 2, activation: str = "relu",
                 use_layer_norm: bool = False, final_layer_init_scale: float = 0.0,
                 output_scale: float = 1.0):
        super().__init__()
        act_cls = get_activation(activation)
        layers = _make_hidden_stack(state_dim + action_dim, hidden_dim, num_hidden_layers, act_cls, use_layer_norm)
        layers.append(_make_final_layer(hidden_dim, 1, final_layer_init_scale))
        self.net = nn.Sequential(*layers)
        self.output_scale = output_scale

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=1)
        return self.net(x) * self.output_scale


# ============================================================
# 3. Replay Buffer
# ============================================================
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ============================================================
# 4. TD3 에이전트
# ============================================================
class TD3Agent:
    def __init__(self, state_dim: int, action_dim: int, cfg: Config):
        self.cfg = cfg
        self.action_dim = action_dim
        self.device = cfg.device
        self.total_updates = 0

        net_kwargs = dict(hidden_dim=cfg.hidden_dim, num_hidden_layers=cfg.num_hidden_layers,
                           activation=cfg.activation, use_layer_norm=cfg.use_layer_norm,
                           final_layer_init_scale=cfg.final_layer_init_scale)
        critic_kwargs = dict(net_kwargs, output_scale=cfg.critic_output_scale)

        # Actor: 온라인 / 타겟
        self.actor = Actor(state_dim, action_dim, **net_kwargs).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, **net_kwargs).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # TD3는 Critic을 2개 사용합니다.
        # 두 Critic 중 더 작은 Q값을 사용하면, 행동의 가치를 지나치게 높게 평가하는 문제를 줄일 수 있습니다.
        self.critic1 = Critic(state_dim, action_dim, **critic_kwargs).to(self.device)
        self.critic2 = Critic(state_dim, action_dim, **critic_kwargs).to(self.device)
        self.critic1_target = Critic(state_dim, action_dim, **critic_kwargs).to(self.device)
        self.critic2_target = Critic(state_dim, action_dim, **critic_kwargs).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=cfg.critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=cfg.critic_lr)

        self.buffer = ReplayBuffer(cfg.buffer_size)

    def select_action(self, state: np.ndarray, noise_std: float = 0.0) -> np.ndarray:
        """Actor가 결정론적으로 행동을 출력하고, 탐험을 위해 가우시안 노이즈를 더합니다."""
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy()[0]

        if noise_std > 0:
            action = action + np.random.normal(0, noise_std, size=self.action_dim)

        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def soft_update(self, target_net: nn.Module, source_net: nn.Module):
        """타겟 네트워크를 온라인 네트워크 쪽으로 아주 조금씩(tau만큼) 이동시킵니다."""
        for target_param, param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(
                self.cfg.tau * param.data + (1.0 - self.cfg.tau) * target_param.data
            )

    def update(self):
        if len(self.buffer) < max(self.cfg.batch_size, self.cfg.min_buffer_size):
            return None, None

        self.total_updates += 1

        states, actions, rewards, next_states, dones = self.buffer.sample(self.cfg.batch_size)

        states = torch.from_numpy(states).to(self.device, dtype=torch.float32)
        actions = torch.from_numpy(actions).to(self.device, dtype=torch.float32)
        rewards = torch.from_numpy(rewards).to(self.device, dtype=torch.float32).unsqueeze(1)
        next_states = torch.from_numpy(next_states).to(self.device, dtype=torch.float32)
        dones = torch.from_numpy(dones).to(self.device, dtype=torch.float32).unsqueeze(1)

        # --- Critic 업데이트 ---
        # TD3 타겟:
        # 1) actor_target(s')로 다음 행동을 구합니다.
        # 2) 그 행동에 작은 노이즈를 더해 Critic이 특정 행동값에 과하게 맞춰지는 것을 줄입니다.
        # 3) critic1_target, critic2_target 중 더 작은 Q값을 사용합니다.
        with torch.no_grad():
            noise = torch.randn_like(actions) * self.cfg.policy_noise
            noise = noise.clamp(-self.cfg.noise_clip, self.cfg.noise_clip)

            next_actions = self.actor_target(next_states) + noise
            next_actions = next_actions.clamp(-1.0, 1.0)

            target_q1 = self.critic1_target(next_states, next_actions)
            target_q2 = self.critic2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target = rewards + self.cfg.gamma * target_q * (1 - dones)

        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        critic1_loss = nn.functional.mse_loss(current_q1, target)
        critic2_loss = nn.functional.mse_loss(current_q2, target)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        actor_loss = None
        if self.total_updates % self.cfg.policy_delay == 0:
            # --- Actor 업데이트 ---
            # TD3는 Critic이 어느 정도 더 자주 학습된 뒤 Actor를 업데이트합니다.
            # 이렇게 하면 부정확한 Critic을 보고 Actor가 너무 빨리 흔들리는 문제를 줄일 수 있습니다.
            actor_loss = -self.critic1(states, self.actor(states)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # --- 타겟 네트워크 소프트 업데이트 ---
            self.soft_update(self.actor_target, self.actor)
            self.soft_update(self.critic1_target, self.critic1)
            self.soft_update(self.critic2_target, self.critic2)

        critic_loss = critic1_loss + critic2_loss
        return critic_loss.item(), None if actor_loss is None else actor_loss.item()


# ============================================================
# 5. 노이즈 스케줄링 (탐험 강도를 점점 줄임)
# ============================================================
def get_noise_std(episode: int, cfg: Config) -> float:
    progress = min(episode / cfg.noise_decay_episodes, 1.0)
    return cfg.noise_std_start + progress * (cfg.noise_std_end - cfg.noise_std_start)


def get_fall_penalty(episode: int, cfg: Config) -> float:
    """fall_penalty_anneal=True일 때, 에피소드가 진행될수록 fall_penalty_start(관대함)에서
    fall_penalty_end(엄격함)로 선형 보간합니다. 학습 초반엔 넘어져도 크게 안 깎여서 과감한
    탐험을 유도하고, 학습이 진행돼 한 에피소드에서 벌 수 있는 보상이 커진 뒤에는 넘어지는
    것도 그만큼 진짜로 나쁜 일이 되게 만듭니다."""
    progress = min(episode / cfg.fall_penalty_anneal_episodes, 1.0)
    return cfg.fall_penalty_start + progress * (cfg.fall_penalty_end - cfg.fall_penalty_start)


# ============================================================
# 5-1. 중간 시각화 함수
#      학습용 환경과는 별개로, 렌더링 전용 환경을 하나 만들어서
#      현재까지 학습된 정책(노이즈 없이)이 실제로 어떻게 걷는지 보여줍니다.
#      이 에피소드의 경험은 학습에 사용하지 않습니다 (순수 관찰용).
# ============================================================
def visualize_current_policy(agent: TD3Agent, vis_env, episode_num: int):
    state, _ = vis_env.reset()
    total_reward = 0.0
    done = False
    step_count = 0

    print(f"\n  >> [중간 시각화] Episode {episode_num} 시점 정책으로 렌더링 중...")
    while not done and step_count < cfg.max_steps_per_episode:
        action = agent.select_action(state, noise_std=0.0)  # 노이즈 없이 현재 실력 그대로
        state, reward, terminated, truncated, _ = vis_env.step(action)
        done = terminated or truncated
        total_reward += reward
        step_count += 1

    print(f"  >> [중간 시각화] Episode {episode_num} 결과: Reward = {total_reward:.2f}, Steps = {step_count}\n")


# ============================================================
# 6. 학습 루프
# ============================================================
def train(resume_ckpt: str | None = None):
    """resume_ckpt를 주면, 새로 만든 에이전트에 그 체크포인트 가중치를 먼저
    불러온 뒤 학습을 시작한다 (이어서 학습). 단, 가중치만 이어받을 뿐 -- 리플레이
    버퍼와 에피소드 카운터(best_avg_reward 등)는 새로 시작한다."""
    env = build_env()
    state_dim = env.observation_space.shape[0]   # BipedalWalker: 24
    action_dim = env.action_space.shape[0]         # BipedalWalker: 4 (연속값)
    fall_wrapper = find_wrapper(env, FrameSkipRewardShapeWrapper) if cfg.fall_penalty_anneal else None

    agent = TD3Agent(state_dim, action_dim, cfg)

    if resume_ckpt is not None:
        ckpt = resume_ckpt if resume_ckpt != "latest" else resolve_latest_ckpt()
        if ckpt is None:
            raise FileNotFoundError(
                f"'{Path(cfg.checkpoint_dir) / env_type_tag()}'에 체크포인트가 없습니다. "
                f"--resume 없이 먼저 학습부터 해주세요."
            )
        load_ckpt(agent, ckpt)
        resync_targets(agent)
        print(f"[resume] 체크포인트 '{ckpt}' 가중치를 불러와서 이어서 학습합니다 "
              f"(리플레이 버퍼/에피소드 카운터는 새로 시작)")

    # 시각화 전용 환경은 한 번만 만들어서 재사용 (매번 새로 만들면 창이 계속 뜨고 닫혀서 느려짐)
    vis_env = None
    if cfg.enable_visualization:
        vis_env = build_env(render_mode="human")

    episode_rewards = []
    recent_rewards = collections.deque(maxlen=100)
    best_avg_reward = -float("inf")
    best_ckpt_tag = None

    writer = None
    if cfg.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        run_id = f"td3_{env_type_tag()}_{time.strftime('%Y%m%d-%H%M%S')}"
        tb_path = Path(cfg.tb_dir) / run_id
        writer = SummaryWriter(log_dir=str(tb_path))
        print(f"TensorBoard: {tb_path}  (tensorboard --logdir {cfg.tb_dir})")

    print(f"TD3 학습 시작! device={cfg.device}, state_dim={state_dim}, action_dim={action_dim}, "
          f"hardcore={cfg.hardcore}, hidden_dim={cfg.hidden_dim}, num_hidden_layers={cfg.num_hidden_layers}, "
          f"activation={cfg.activation}")
    if cfg.frame_skip != 1 or cfg.fall_penalty != -100.0:
        print(f"[hardcore 트릭] frame_skip={cfg.frame_skip}, fall_penalty={cfg.fall_penalty}")
    if cfg.fall_penalty_anneal:
        print(f"[fall_penalty 어닐링] {cfg.fall_penalty_start} -> {cfg.fall_penalty_end} "
              f"({cfg.fall_penalty_anneal_episodes}에피소드에 걸쳐 선형 증가)")
    if cfg.use_layer_norm or cfg.final_layer_init_scale > 0 or cfg.critic_output_scale != 1.0:
        print(f"[안정화 트릭] use_layer_norm={cfg.use_layer_norm}, "
              f"final_layer_init_scale={cfg.final_layer_init_scale}, critic_output_scale={cfg.critic_output_scale}")
    print(f"체크포인트: 최근 100개 평균이 기록을 갱신할 때마다 {cfg.checkpoint_dir}/{env_type_tag()}/ 에 저장")
    print("-" * 60)

    for episode in range(cfg.num_episodes):
        state, _ = env.reset(seed=cfg.seed + episode)
        noise_std = get_noise_std(episode, cfg)
        if fall_wrapper is not None:
            fall_wrapper.fall_penalty = get_fall_penalty(episode, cfg)

        episode_reward = 0.0
        for step in range(cfg.max_steps_per_episode):
            action = agent.select_action(state, noise_std)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.buffer.push(state, action, reward, next_state, float(done))
            if step % cfg.update_every == 0:
                agent.update()

            state = next_state
            episode_reward += reward

            if done:
                break

        episode_rewards.append(episode_reward)
        recent_rewards.append(episode_reward)
        avg_recent = np.mean(recent_rewards)

        episode_steps = step + 1

        if writer is not None:
            writer.add_scalar("train/episode_reward", episode_reward, episode + 1)
            writer.add_scalar("train/avg_reward_100", avg_recent, episode + 1)
            writer.add_scalars(
                "train/steps",
                {"episode_steps": episode_steps, "max_steps": cfg.max_steps_per_episode},
                episode + 1,
            )
            if fall_wrapper is not None:
                writer.add_scalar("train/fall_penalty", fall_wrapper.fall_penalty, episode + 1)

        # --- 새 기록(최근 100개 평균 기준) 갱신 시 체크포인트 저장 ---
        # len(recent_rewards)가 maxlen(100)을 채우기 전에는 avg_recent가 소수 에피소드만의
        # 평균이라 우연히 높게 나올 수 있음(특히 이미 좋은 정책을 --resume한 경우 1번째
        # 에피소드가 요행히 고득점일 수 있음). 그런 값이 "기록"으로 고정되면 이후 진짜
        # 100개 평균이 이를 넘지 못해 정작 학습이 잘 된 뒤의 체크포인트가 저장되지 않는
        # 문제가 있어, 롤링 윈도우가 다 찰 때까지는 신기록 갱신 대상에서 제외한다.
        if len(recent_rewards) >= recent_rewards.maxlen and avg_recent > best_avg_reward:
            best_avg_reward = avg_recent
            best_ckpt_tag = f"ep{episode + 1}"
            save_ckpt(agent, best_ckpt_tag)
            print(f"  >> 새 기록! 최근 100개 평균 {avg_recent:.2f} -> {best_ckpt_tag} 체크포인트 저장")

        if (episode + 1) % cfg.print_every == 0:
            print(
                f"Episode [{episode + 1}/{cfg.num_episodes}] "
                f"Reward: {episode_reward:8.2f} | "
                f"최근 100개 평균: {avg_recent:8.2f} | "
                f"noise_std: {noise_std:.3f}"
            )

        if avg_recent >= cfg.solved_threshold and len(recent_rewards) == 100:
            print(f"\n환경을 풀었습니다! (Episode {episode + 1}, 평균 보상 {avg_recent:.2f})")
            if vis_env is not None:
                visualize_current_policy(agent, vis_env, episode + 1)
            break

        # 일정 에피소드마다 현재 정책이 어떻게 걷는지 렌더링으로 확인
        if cfg.enable_visualization and (episode + 1) % cfg.visualize_every == 0:
            visualize_current_policy(agent, vis_env, episode + 1)

    if writer is not None:
        writer.close()
    if vis_env is not None:
        vis_env.close()
    env.close()
    return agent, episode_rewards, best_ckpt_tag


# ============================================================
# 7. 학습 곡선 시각화
#    학습이 끝날 때 자동으로 창을 띄우지 않고, PNG로만 저장 + 원본 데이터를
#    남겨둔다. 실제로 보고 싶을 때는 `--plot`으로 원하는 시점에 따로 띄운다.
# ============================================================
def plot_rewards(episode_rewards, show: bool = False):
    rewards = np.array(episode_rewards)
    window = 20
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
    else:
        moving_avg = rewards

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3, label="Episode Reward")
    plt.plot(range(window - 1, window - 1 + len(moving_avg)), moving_avg, label=f"{window}-episode 이동평균")
    plt.axhline(y=cfg.solved_threshold, color="red", linestyle="--", label=f"해결 기준 ({cfg.solved_threshold})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("BipedalWalker-v3 TD3 학습 곡선")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("bipedalwalker_td3_training_curve.png", dpi=150)
    print("학습 곡선을 bipedalwalker_td3_training_curve.png 로 저장했습니다.")
    if show:
        plt.show()
    else:
        plt.close()


def save_reward_history(episode_rewards) -> None:
    ckpt_dir = Path(cfg.checkpoint_dir) / env_type_tag()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    np.save(ckpt_dir / "reward_history.npy", np.array(episode_rewards))


def load_reward_history():
    path = Path(cfg.checkpoint_dir) / env_type_tag() / "reward_history.npy"
    if not path.exists():
        raise FileNotFoundError(f"'{path}'가 없습니다. 먼저 인자 없이 실행해서 학습부터 해주세요.")
    return np.load(path)


# ============================================================
# 8. 학습된 에이전트 시연 (렌더링)
# ============================================================
def evaluate(agent: TD3Agent, num_episodes: int = 3, render: bool = True, video_dir: str | None = None):
    env = build_env(render_mode="human" if render else None)
    hud = find_wrapper(env, HudOverlayWrapper) if video_dir else None
    if video_dir:
        import os
        os.makedirs(video_dir, exist_ok=True)

    for ep in range(num_episodes):
        frames: list | None = [] if video_dir else None
        if hud is not None:
            hud.frame_sink = frames

        state, _ = env.reset()
        total_reward = 0.0
        done = False
        step_count = 0
        while not done and step_count < cfg.eval_max_steps:
            action = agent.select_action(state, noise_std=0.0)  # 평가 시엔 노이즈 없이
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step_count += 1
        print(f"[평가] Episode {ep + 1}: Reward = {total_reward:.2f}")

        if video_dir and frames:
            import imageio
            path = f"{video_dir}/{cfg.run_tag or 'play'}_ep{ep + 1}.mp4"
            imageio.mimsave(path, frames, fps=30, macro_block_size=None)
            print(f"[video] 저장 완료 -> {path} ({len(frames)}프레임)")

    env.close()


if __name__ == "__main__":
    args = parse_args()

    if args.plot:
        # 학습 없이, 저장된 리워드 기록으로 학습 곡선 그래프만 띄우기
        plot_rewards(load_reward_history(), show=True)

    elif args.play:
        if args.live:
            # 화면으로 재생하면서 동시에 그 경험으로 이어서 학습
            run_play_live(args.play, args.ckpt)
        else:
            # 학습 없이 체크포인트만 불러와서 화면으로 재생
            run_play(args.play, args.ckpt)

    elif args.skip_train:
        # 학습 없이 기존 체크포인트를 시연만
        probe_env = build_env()
        state_dim = probe_env.observation_space.shape[0]
        action_dim = probe_env.action_space.shape[0]
        probe_env.close()

        agent = TD3Agent(state_dim, action_dim, cfg)
        ckpt = args.ckpt if args.ckpt != "latest" else resolve_latest_ckpt()
        if ckpt is None:
            raise FileNotFoundError(
                f"'{Path(cfg.checkpoint_dir) / env_type_tag()}'에 체크포인트가 없습니다. "
                f"먼저 인자 없이 실행해서 학습부터 해주세요."
            )
        load_ckpt(agent, ckpt)
        print(f"[skip-train] 체크포인트 '{ckpt}' 로 시연")
        evaluate(agent, num_episodes=3, render=args.render)

    else:
        # 기본: 학습 -> 학습곡선/리워드 기록 저장(창은 안 띄움) -> 시연
        trained_agent, rewards_history, best_ckpt_tag = train(resume_ckpt=args.ckpt if args.resume else None)
        save_reward_history(rewards_history)
        plot_rewards(rewards_history, show=False)

        evaluate(trained_agent, num_episodes=3, render=args.render)

        if best_ckpt_tag is not None:
            print(f"\n최고 기록 체크포인트: {best_ckpt_tag} "
                  f"(나중에 'python \"{__file__}\" --play 3' 로 바로 재생 가능)")
            print(f"학습 곡선: python \"{__file__}\" --plot")
