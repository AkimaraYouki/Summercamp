# Colab에서 파이프라인 실행하기 (재현·검증 가이드)

이 문서는 압축 파일(zip) 하나만 받은 상태에서, Colab에서 코드를 그대로 실행해서 README에 적힌 수치(해결률·낙상률 등)를 재현하고 검증

## 1) 압축 파일을 Colab에 업로드 + 해제

**작은 파일(수백 MB 이하)**: 팝업으로 직접 업로드.

```python
from google.colab import files
uploaded = files.upload()   # 받은 zip 파일 선택
```

```bash
!unzip -q bipedalwalker-td3.zip -d /content
%cd /content
!ls bipedalwalker-td3
```

**기준**: `src/`, `models/`, `results/`, `reports/` 4개 폴더가 보이면 정상 해제된 것.

**큰 파일(체크포인트+영상 다 포함해서 GB 단위)**: Google Drive에 올려두고 마운트하는 게 낫다.

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
!unzip -q "/content/drive/MyDrive/bipedalwalker-td3.zip" -d /content
%cd /content
```

## 2) 의존성 설치

```bash
!apt-get update -qq && apt-get install -y xvfb > /dev/null
!pip install -q pyvirtualdisplay
!pip install -q -r bipedalwalker-td3/requirements.txt
```

`--play`(시연/녹화)는 pygame `render_mode="human"` 창을 띄우는 방식이라 디스플레이 없는 Colab에는 가상 디스플레이(xvfb)가 필요하다. 학습(`run_train.sh`)·100ep 재평가(`run_eval100.sh`)는 렌더링을 안 해서 이 단계 없이도 된다.

**기준**: 에러 없이 설치 로그가 끝까지 출력되면 정상.

## 3) 가상 디스플레이 시작 (세션당 한 번, `--play` 쓸 때만)

```python
from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400, 900))
display.start()
```

**기준**: `<pyvirtualdisplay.display.Display at 0x...>`가 출력되면 정상 시작된 것(에러 아님).

## 4) 시연 + 녹화

`run_play.sh` 기본값에는 `video_record_dir`이 없어서 mp4로 안 남는다. python 커맨드를 직접 호출하며 `video_record_dir`을 추가로 지정한다.

```bash
%cd /content/bipedalwalker-td3/models/model1_g2-si-speed-s201
!python3 ../../src/Bipedalwalker_TD3_live.py --play 3 --ckpt ep794 --set \
  run_tag=g2-si-speed-s201 hardcore=true frame_skip=2 fall_penalty=-10.0 \
  activation=gelu final_layer_init_scale=0.003 critic_output_scale=10.0 \
  video_record_dir=/content/demo_videos
```

```python
from IPython.display import Video
Video("/content/demo_videos/g2-si-speed-s201_ep1.mp4", embed=True)
```

**기준**: 3개 에피소드가 콘솔에 `[평가] Episode N: Reward = ...`로 찍히고, `/content/demo_videos/`에 mp4 3개가 생기면 정상. 셀에서 영상이 재생되면 시연이 재현된 것 — README의 `results/videos/model1_.../demo.gif`와 같은 보행을 보여야 한다.

다른 모델은 `--ckpt`·`--set`의 `run_tag`와 트릭 값만 해당 모델 폴더 `run_train.sh`에 있는 값으로 바꾸면 된다. (예: 모델2는 `--ckpt ep3081 --set run_tag=scratch-initscale ...`)

## 5) 학습 재현

```bash
%cd /content/bipedalwalker-td3/models/model1_g2-si-speed-s201
!bash run_train.sh   # 부모 체크포인트가 필요하면 폴더 안 README 참고
```

**기준**: 매 에피소드 `Episode [N/2000] Reward: ... | 최근 100개 평균: ...`가 출력되고, 최근 100개 평균이 300을 넘으면 `환경을 풀었습니다!`가 뜨고 종료된다. 시간이 오래 걸리는 게 정상이라 끝까지 안 돌려도, 로그의 "최근 100개 평균"이 점점 올라가는 추세만 확인해도 재현 검증은 충분하다.

### TensorBoard로 학습 곡선 실시간 확인

학습 스크립트가 콘솔에 찍어주는 `TensorBoard: tensorboard/... (tensorboard --logdir tensorboard)` 힌트 그대로, 같은 폴더(`models/model1_g2-si-speed-s201`)에서 Colab 매직 커맨드로 띄울 수 있다. 학습 셀을 실행한 뒤 별도 셀에서:

```python
%load_ext tensorboard
%tensorboard --logdir tensorboard
```

**기준**: TensorBoard 대시보드가 셀 출력에 인라인으로 뜨고, `reward`·`loss` 등의 스칼라 그래프가 학습이 진행될수록 갱신되면 정상이다.

## 6) 100ep 고정시드 재평가 (해결률·낙상률 재현)

```bash
%cd /content/bipedalwalker-td3/models/model1_g2-si-speed-s201
!bash run_eval100.sh
```

**기준**: 아래처럼 출력되면, 이 문서와 같은 폴더의 `README.md` "결과" 표(모델1 행: 해결률 83% / 낙상률 12% / 비낙상 평균 304.73 / 클리어 502.6)와 값이 일치해야 재현된 것이다.

```
[100ep 재평가] run_tag=g2-si-speed-s201 ckpt=ep794
  해결률(reward>=300): 83.0%
  낙상률: 12.0%
  비낙상 평균 reward: 304.73
  평균 클리어 step: 502.6
```

## 7) 알고리즘 비교 파이프라인 (TD3 vs SAC/PPO/DDPG)

`bipedalwalker-td3/algo_comparison/`에 별도 스캐폴드가 있다. 2,000,000 step 학습이라 시간이 오래 걸리므로 세션 하나로 다 끝내려 하지 말 것. 자세한 건 `algo_comparison/README.md` 참고.

```bash
%cd /content/bipedalwalker-td3/algo_comparison
!pip install -q -r requirements.txt
!python3 run_td3.py --seed 0          # 자동 종료 워처 미구현, 2,000,000 step 근처서 수동 중단 필요
!python3 run_baselines.py --algo sac --seed 0   # SB3라 total_timesteps에서 자동 종료
```

**기준**: 이 스크립트들은 실행이 오래 걸려서 "끝까지 도는지"보다 "정상적으로 학습 로그가 찍히기 시작하는지"만 확인해도 코드 자체는 검증된다. 실제 비교 데이터는 아래 8번에서 완결된 원본으로 확인한다.

## 8) 알고리즘 비교 원본 데이터 검증

슬라이드 "1M step 지점 값" 표를 원본 CSV에서 재계산해서 대조한다. 외부 의존성 없음(표준 라이브러리만 사용).

```bash
%cd /content/bipedalwalker-td3/reports/figures
!python3 summarize_1M_step.py
```

**기준**: `reports/figures/README.md`의 "검증 결과" 표(원본데이터 열)와 출력이 같아야 한다. 참고로 슬라이드에 인쇄된 값과는 팀원(teammate) 열 3개(TD3/PPO/DDPG)가 다르게 나온다 — 이건 알려진 불일치이며 `figures/README.md`에 원인과 함께 기록돼 있다.

## 결과물 다운로드

작업 끝나면 결과 폴더를 다시 압축해서 로컬로 받는다.

```bash
%cd /content
!zip -r results_from_colab.zip bipedalwalker-td3/models/*/checkpoints bipedalwalker-td3/results
```

```python
from google.colab import files
files.download("/content/results_from_colab.zip")
```

---
