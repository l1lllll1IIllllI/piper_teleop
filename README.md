# 양팔 실시간 교시 (RealSense 사람 양손 → PiPER + Milly)

RealSense D435i 앞에서 사람이 양손으로 동작하면 두 로봇팔이 실시간으로 따라 움직입니다.

| 사람 | 로봇 | 연결 |
|---|---|---|
| 왼손 | 왼팔 **PiPER** (AgileX) | candleLight USB-CAN (`piper_sdk`) |
| 오른손 | 오른팔 **Milly** (Motomind) | usctocan USB-CAN (`motomind_milly` SDK) |

| 파일 | 내용 |
|---|---|
| `teleop.py` | 메인 실행 파일, 상단 `SETTINGS` 에 조정값, `ROBOTS` 에 로봇별 READY 자세·그리퍼 길이 |
| `human_tracker.py` | RealSense(color+depth) + MediaPipe Pose/Hands → 양손 위치·방향·그립 (손마다 필터 따로) |
| `piper_kin.py` / `milly_kin.py` | 기구학 FK/IK (PiPER: SDK DH, Milly: SDK URDF) — 둘 다 같은 base 좌표 (X 앞, Y 왼쪽, Z 위) |
| `arm_base.py` | 공통 제어 스레드 (관절 속도 제한, 모든 관절 같은 비율로 감속) |
| `piper_arm.py` / `milly_arm.py` | 로봇별 연결/enable/명령 전송 |
| `MILLY.urdf` | Milly URDF 사본 (MIT, Motomind) — SDK 가 설치되어 있으면 SDK 쪽 URDF 를 씀 |

## 환경

`~/teleop_env` (Python 3.10, mediapipe 0.10.14, numpy 1.26, opencv 4.10, pyrealsense2, piper_sdk 0.6.2).

### Milly SDK 설치 (1회)

[motomind_milly_sdk_release](https://github.com/MotomindKR/motomind_milly_sdk_release) 의 Python 3.10 wheel 을 teleop_env 에 설치합니다.
`motomind_robot_model` (pinocchio 필요) 은 설치하지 않습니다 — teleop 은 실시간 관절 명령(`move_mit`)만 쓰고 IK 는 `milly_kin.py` 가 계산합니다.
`--no-deps` 는 mediapipe 가 쓰는 numpy 1.26 을 건드리지 않게 하기 위함입니다.

```bash
cd ~ && git clone https://github.com/MotomindKR/motomind_milly_sdk_release.git
~/teleop_env/bin/python -m pip install --no-deps ~/motomind_milly_sdk_release/dist/linux_x86_64/python3.10/motomind_milly_sdk-1.1.0-cp310-cp310-linux_x86_64.whl
~/teleop_env/bin/python -m pip install "pyyaml>=6.0"
~/teleop_env/bin/python -c "import motomind_milly as mm; print(mm.__version__)"
```

## 실행 순서

```bash
# 0) CAN 두 개 활성화 (sudo 필요, 재부팅/절전 후마다)
#    can0/can1 이름은 꽂힌 순서에 따라 바뀜 → teleop 이 USB 제품 이름으로 자동 선택 (PiPER = candleLight, Milly = usctocan)
#    PiPER 의 can_activate.sh 는 어댑터가 하나일 때만 동작하므로 직접 켬
for c in can0 can1; do sudo ip link set $c down; sudo ip link set $c type can bitrate 1000000; sudo ip link set $c up; done

# (1회) 카메라 IMU 권한 — 카메라 기울기 자동 보정에 필요. 설치 후 카메라를 뽑았다 다시 꽂기
sudo curl -fsSL -o /etc/udev/rules.d/99-realsense-libusb.rules https://raw.githubusercontent.com/IntelRealSense/librealsense/master/config/99-realsense-libusb.rules
sudo udevadm control --reload-rules && sudo udevadm trigger

# 1) Milly 제품 ID 확인 (팔에 각인, 예: MILLY_6TG5) — 모터를 움직이지 않음
~/teleop_env/bin/motomind-milly-robots --interface can0 --interface can1

# 2) 로봇 없이 확인 — 화면에서 양손 좌표축/그립/목표 관절값이 자연스러운지
~/teleop_env/bin/python teleop.py --dry-run

# 3) 실제 로봇 — 처음엔 느리게, 한 팔씩
~/teleop_env/bin/python teleop.py --arms left  --max-vel 30                        # PiPER 만
~/teleop_env/bin/python teleop.py --arms right --max-vel 30 --milly-id MILLY_XXXX  # Milly 만
~/teleop_env/bin/python teleop.py --max-vel 40 --table-z 0.01 --milly-id MILLY_XXXX   # 양팔
```

`teleop.py` 의 `MILLY_PRODUCT_ID` 에 ID 를 적어 두면 `--milly-id` 는 생략 가능.

실제 로봇 첫 실행 절차: 실행 → `r` (양팔 READY 자세로 천천히 이동) → 카메라를 정면으로 보고 양손을 편한 작업 자세로 → `k` (보정) → `SPACE` (양팔 추종 시작) → 천천히 움직여 방향 확인.
한 팔만 시작/정지는 `1` (왼팔 PiPER) / `2` (오른팔 Milly).

## 키

| 키 | 동작 |
|---|---|
| `SPACE` | 양팔 추종 시작/정지 (클러치). 하나라도 추종 중이면 모두 정지. 시작 순간을 기준으로 **상대 이동** → 시작 시 로봇이 튀지 않음 |
| `1` / `2` | 왼팔(PiPER) / 오른팔(Milly) 만 추종 시작/정지 |
| `r` / `z` | READY / ZERO(접힘) 자세로 천천히 이동 (추종 정지 상태인 팔만) |
| `g` | 그리퍼 추종 ON/OFF (부품을 잡은 채로 손가락을 자유롭게 하고 싶을 때 OFF) |
| `o` | 손 회전 추종 모드 전환: `twist`(기본, 비틀기만) → `yaw` → `full` → `off` → `wrist`(비틀기+꺾기+좌우) |
| `t` | 비틀기 boost ON/OFF (기본 OFF): ×1.5 + 50° 이상 비튼 채로 있으면 툴이 계속 회전 (다리 나사 돌리기 등) |
| `m` | 위치 매핑 전환: `camera`(기본) → `shoulder` → `absolute` (추종 중에도 가능, 튀지 않음) |
| `k` | 보정 (추종 정지 상태에서): 양 어깨 선으로 사람 정면 방향 보정 + [`absolute`] 팔 길이로 scale, 지금 손 자세 = 지금 툴 자세. 카메라를 정면으로 보고 팔·손이 다 보이게 |
| `b` | 그리퍼 `binary`(기본, 열림/닫힘) ↔ `proportional`(벌린 만큼) |
| `a` | 속도 적응 스케일 ON(기본)/OFF |
| `d` | 손바닥 depth 방식 `region`(기본, 영역 평균) → `plane`(영역 평면 ∩ 기준점 시선) → `patch`(중심 13x13) — 화면 `jitter` 비교용 |
| `[` `]` | 위치 스케일 ×0.8 / ×1.25 (정밀 조립은 작게) |
| `c` | CSV 기록 (`teleop_logs/teleop_<시각>_left_piper.csv`, `..._right_milly.csv`): 명령/실측 관절, 그리퍼, 손 위치, 손 속도/gain, 목표 TCP, 추적 메시지/경고 |
| `x` | 비상정지 (PiPER EmergencyStop + Milly 전자식 정지 — 팔이 천천히 내려감) |
| `q` / `ESC` | 종료. PiPER 는 enable 유지 (마지막 위치), **Milly 는 SDK shutdown (10초 동안 감쇠 → 팔이 내려감)** → Milly 가 켜져 있으면 `q` 를 한 번 더 눌러야 종료. 먼저 `z` 로 접어 두는 것을 권장 |

클러치는 마우스를 들어 옮기는 것과 같습니다. 팔이 불편한 위치에 오면 `SPACE` 로 놓고, 손을 편한 곳으로 옮긴 뒤 다시 `SPACE`.

화면: 가운데 카메라 영상 (거울상 아님 → 사람 왼손이 화면 오른쪽), 오른쪽 패널 = PiPER(왼손), 왼쪽 패널 = Milly(오른손). 손 옆에 담당 로봇 이름 표시.

## 매핑

손바닥 위치 변화 → 그리퍼 끝(TCP) 위치 변화, 손 회전 → 툴 회전, IK 로 관절 계산 (양팔 같은 방식).
두 로봇 모두 base 좌표를 X 앞 / Y 왼쪽 / Z 위로 맞춰서 같은 손 동작에 같은 방향으로 움직입니다
(Milly URDF 는 팔이 -Y 쪽으로 뻗는 좌표라 `milly_kin.py` 에서 90° 돌려 맞춤, 툴 축도 `tool_rz_deg=-90` 로 PiPER 와 맞춤).
엄지-검지 벌림 → 그리퍼 열림. 이전의 `--mode joint` (사람 관절각 → PiPER 관절) 는 양팔 버전에서 뺐습니다.

### 위치 매핑 (`--map`, `m` 키)

| 모드 | 손 위치 | 로봇 위치 | 장점 | 단점 |
|---|---|---|---|---|
| `camera` (기본) | 손바닥 (카메라 기준) | 클러치 시점 기준 이동량 × gain (속도 적응) | 떨림 가장 적음, 정밀 작업용 gain | 몸이 기울거나 흔들려도 로봇이 움직임, 클러치할 때마다 대응이 바뀜 |
| `shoulder` | 손바닥 − 어깨 | 위와 같음 | 몸을 기울여도 로봇은 그대로 | 어깨 추정 떨림이 더해짐 (어깨는 강하게 필터), 어깨가 보여야 함 |
| `absolute` | 손바닥 − 어깨 | **사람 어깨 = 로봇 base 원점** → TCP = 원점 + (손바닥 − 어깨) × scale | 같은 손 위치·자세 = 항상 같은 로봇 위치·자세, 클러치해도 안 바뀜 | 속도 적응 없음. 시작할 때 차이를 3초 동안 천천히 맞춤 → 로봇이 저절로 움직임 |

**`absolute` 좌표**: 손이 어깨보다 위면 z > 0, 아래면 z < 0 / 어깨보다 앞이면 x > 0 / 왼쪽이면 y > 0 (각 팔마다 그 쪽 어깨 기준).
- scale = 로봇 도달 거리 (J2 축 → 그리퍼 끝: PiPER 0.76 m, Milly 0.78 m) / 사람 팔 길이 (어깨 → 손바닥, `k` 때 depth 로 측정,
  안 되면 Pose 모델 추정) → 보통 1.1~1.2. 팔을 끝까지 뻗으면 로봇도 끝까지. `--pos-scale`, `[` `]` 가 여기에 곱해짐.
- 방향: `k` (또는 첫 시작) 때의 손 자세 = 그때의 툴 자세. 비틀기 속도 모드는 꺼짐 (같은 손 자세 = 같은 툴 자세 유지).
- **원점 높이 (`--abs-origin-z`, 기본 0.3 m)**: 사람 어깨 높이 = 로봇 z 0.3 m. 사람은 보통 어깨보다 20~30 cm 아래에서 작업하므로
  그때 로봇은 작업대 근처 (z 0~0.1 m). 부호 의미는 같음 (어깨 높이 기준 위 +, 아래 −). `0` 이면 원점 = 로봇 base 바닥 →
  손이 어깨보다 낮으면 목표가 작업대 아래라 `--table-z` 높이에서 멈춤.
- 손 방향 추정이 뒤집혀 기준이 다시 잡힌 경우엔 그 차이를 유지 (뒤집힌 추정 쪽으로 툴이 돌아가지 않게) → 다시 맞추려면 정지 후 `k`.

### 기본값 (11/5 성능 저하 보고 후 이전 동작으로 되돌림)

새 기능은 모두 켜고 끌 수 있고, 기본은 이전과 같습니다: 회전 `twist` (비틀기 1:1), boost OFF, 손바닥 depth `region` (평균), 매핑 `camera`.
바뀐 점 중 기본으로 켜진 것은 카메라 기울기 IMU 보정 (축 정렬 보정)뿐 — 비교하려면 `--camera-pitch 0`.
하나씩 켜 보며 비교: `o` (wrist), `t` (boost), `d` (plane), `m` (shoulder/absolute). 각각 `c` 로 기록해 두면 분석 가능.

### 손 회전 (`--rot`, `o` 키)

- **비틀기 (손가락 축 회전 → 그리퍼 축 J6)**: × `TWIST_GAIN` (1.5). 사람 손목은 ±60~80° 정도라 부족 →
  기준에서 50° 이상 비튼 채로 있으면 툴이 계속 돌아감 (80° 에서 최대 60°/s). 손을 가운데로 돌리면 멈추고 돌아간 만큼은 유지
  (나사 돌리기처럼: 비틀어 유지 → 가운데로 → 다시). J6 한계 3° 전에서 멈춤. 클러치하면 초기화. 화면 `twist+`.
- **꺾기 (손바닥 쪽/손등 쪽 → 툴 위아래)**, **좌우 (엄지 쪽/새끼 쪽 → 툴 좌우)**: 손 방향을 그대로 쓰지 않고
  **아래팔(Pose 팔꿈치→손목) 기준 손 각도**로 계산 → 팔을 옮길 때 손과 아래팔이 같이 돌아가는 부분은 빠지고 손목 관절을 꺾은 만큼만 반영.
  데드밴드 5°, 손이 0.25 m/s 이상으로 이동 중엔 갱신 안 함, 변화 속도 90°/s 제한. 좌우는 사람 범위가 좁아 × 1.5.
  아래팔이 안 보이면 마지막 값 유지 (화면 `wrist flex/dev: --`).
  base 가까이에서 툴을 위로 드는 쪽은 자기충돌 방지로 제한됨 (READY 위치에서 약 10°), PiPER 는 J5 한계 70° 도 있음.
- 이전 동작 (비틀기만, 배율 1, 속도 모드 없음에 가까움) 은 `twist`.

## Milly 관련

- 제어: SDK `move_mit` (관절별 위치 + kp/kd) 로 50 Hz 목표 갱신, SDK supervisor 가 100 Hz 로 전송 (SDK `milly_teleop.py` 예제와 같은 방식).
- PiPER 도 같은 방식으로 맞춤 (`--piper-ctrl follow`, 기본): ModeCtrl MIT 모드 (`0xAD`, piper_sdk 의 "가장 빠른 응답" 모드) 로
  펌웨어가 경로를 다시 계획하지 않고 100 Hz 명령 위치를 바로 따라감 → 두 팔 모두 `arm_base` 의 속도 제한 경로 그대로 움직임.
  이전 방식 (MOVE J, 명령마다 펌웨어가 속도 비율로 다시 계획 → 지연·뭉개짐) 은 `--piper-ctrl movej`.
  **follow 는 처음 쓰는 모드이므로 `--arms left --max-vel 30` 으로 먼저 확인.**
  게인은 SDK 출하 기본값 (kp 30/30/30/15/15/5, kd 1.5/1.5/1.5/1/1/0.5, 그리퍼 kp 5) — `milly_arm.py` 상단.
- 중력 보상 토크를 보내지 않으므로 팔이 목표보다 조금 처질 수 있음 → 화면 `cmd` / `fb` 차이, CSV `fb*_deg` 로 확인.
- READY = `[0, -52, -45, 0, 37, 0]°` (TCP ≈ (0.30, 0, 0.25) m, 툴 45° 아래, PiPER READY 와 같은 위치).
- 그리퍼 길이 `tcp_offset=0.14` m 는 URDF 손가락 관절 위치(flange 에서 0.105 m)로 추정한 값 → **실제 flange~손가락 끝 길이를 재서 `ROBOTS["milly"]`에 입력** (바닥 높이 `--table-z` 가 이 값 기준).
- J6 (그리퍼 축 회전) 범위가 ±86° 로 PiPER(±120°) 보다 좁음.
- 소프트웨어 관절 한계 = SDK MIT 명령 범위에서 3° 안쪽.
- 종료는 항상 SDK `shutdown()` (SDK 권장). 10초 동안 감쇠가 0 이 되므로 팔을 받치거나 `z` 로 접은 뒤 종료.

## 집기 정밀도 (손 추정 오차 줄이기, 양팔 공통)

MediaPipe 손 방향 추정은 위치보다 훨씬 떨리고, 손가락을 오므리면 손 위치·방향 추정 자체가 같이 변합니다.
그래서 기본값을 다음과 같이 설정:

- **회전 `twist`/`wrist` 모드**: 손 자기 축 기준 회전 → 툴 자기 축 기준 회전 (위 "손 회전" 참고).
  예전엔 손 방향을 그대로 써서 팔을 옮길 때 저절로 꺾이고 돌아가는 손목이 툴에 다 들어갔음 (기록: 툴 축이 평균 15°/s) → 비틀기만 썼음.
  지금은 꺾기/좌우를 아래팔 기준으로 계산해서 같이 켬.
- **툴 방향이 팔 방위각을 따라감 (`FOLLOW_AZIMUTH`)**: 좌우로 이동하면 J1 만 돌고 툴은 바깥쪽을 향함 (이전엔 좌우 15 cm 에 J4 -37°, J6 59° 비틀림).
- **명령 목표는 로봇 현재 위치에서 3 cm 이내 (`MAX_LEAD_M`)**: 로봇이 늦을 때 관절 보간 경로가 바닥 아래로 지나가지 않게. 관절은 모두 같은 비율로 감속 (직선 보간).
- **닿지 않는 목표 → 경계까지 이동**: 이전엔 목표를 무시하고 멈췄음 (기록: J5 70° 한계에서 자주 발생).
- **IK 가 J4 를 0 쪽으로 약하게 당김 (`IK_REST_WEIGHT`)**: 아래로 내려가 J5≈0 (손목 특이점) 일 때 J4/J6 가 서로 반대로 크게 도는 것 방지.
- **회전 데드밴드 3°**: 작은 손 떨림은 무시.
- **손가락 움직이는 동안 팔 고정**: 집으려고 손가락을 오므리는 동안 팔은 멈추고, 손가락이 멈추면 그 순간을 새 기준으로 다시 추종.
  손이 0.10 m/s 이상으로 이동 중일 땐 고정 안 함 (팔 이동 중 grip ratio 흔들림으로 이동량이 버려지는 것 방지).
  화면에 `[fingers moving: arm hold]` 표시.
- **손 방향 뒤집힘 제거**: 한 프레임에 35° 이상 바뀌면 무시, 4 프레임 이상 계속되면 로봇은 그대로 두고 회전 기준만 다시 잡음 (기록에서 30~180° 튐 → J6 이 휙 돌던 원인).
- **binary 그리퍼 + 히스테리시스**: grip ratio < 0.45 닫힘, > 0.70 열림. 사이 값은 상태 유지 → 잡은 물건을 손가락 힘을 풀어도 놓치지 않음.

- **속도 적응 스케일 (기본 ON)**: 손이 빠르면(0.2 m/s 이상) 1:1, 느리면(0.03 m/s 이하) 1:2.5 (`ADAPTIVE_G_MIN=0.4`) 로 자동 축소.
  멀리는 빠르게, 물체 근처에선 천천히 움직이면 정밀해짐.
  대신 손을 원래 위치로 되돌려도 로봇이 원래 위치로 정확히 오지 않음 → 팔이 불편해지면 `SPACE` 로 클러치.
  `--pos-scale` 은 이 gain 전체에 곱해짐 (기본 1.0 권장).
- **손바닥 영역 depth (`region`)**: 손목·MCP 다각형 안의 depth 점 전체(수천 개)에서 배경/손가락 끝을 걸러내고 평면을 맞춘 뒤,
  기준점 (손목 + 검지~새끼 MCP 평균 픽셀, 엄지 제외) 의 시선과 평면이 만나는 점. 평균 depth 를 쓰면 손바닥이 기울었을 때
  유효 점 분포가 바뀔 때마다 거리가 흔들림 (합성 시험: 기울어진 손바닥 z 떨림 3.3 mm → 0.2 mm). 평면이 비스듬하면 평균 depth.
  손바닥이 카메라 쪽을 향해 평평하게 보일 때만 depth 평면 법선을 손 방향에 섞음 (60° 이상 비스듬하면 안 씀).
- **depth preset**: Medium Density + 레이저 최대 (`--depth-preset`). 2 m 정적 물체 실측 떨림 14.8 mm → 7.0 mm.

사용 팁:
- 멀리 갈 땐 빠르게, 물체 근처에선 천천히. 그래도 크면 `[` 로 전체 스케일 축소.
- 물체 위에서 잠깐 멈췄다가 내려가서 집기. 팔이 불편하면 `SPACE` 로 클러치 놓고 손 위치 다시 잡기.
- 손바닥이 카메라를 향하고 손이 몸통 앞을 가리지 않게 (depth 가 몸통에 걸리면 튐).

## 자세/카메라 배치

- 사람은 카메라를 **정면으로 마주 보고** 섭니다 (1~1.5 m). 상체와 양손이 화면에 들어와야 함.
  손 ↔ 왼/오른쪽은 Pose 손목 위치로 짝짓기 때문에 양 어깨~손목이 보여야 안정적 (안 보이면 MediaPipe 왼손/오른손 라벨 사용).
- 기본은 로봇이 사람과 **같은 방향**을 본다고 가정 (사람 왼쪽 = 로봇 +Y). 로봇이 사람을 마주 보고 있으면 `--mirror` (양팔 공통).
- 두 로봇 base 를 같은 방향(앞 = 작업대 쪽)으로 나란히, 팔이 서로 닿지 않을 만큼 띄워서 설치. 두 팔 사이 충돌 검사는 없음.
- 카메라 기울기는 시작할 때 **D435i IMU 중력 방향으로 자동 보정** (위아래 + 옆 기울기). 화면 `tilt 아래/옆 (IMU)`.
  IMU 권한이 없으면 (`(manual)` 표시) 실행 순서의 udev 규칙 설치, 또는 `--camera-pitch 20` 처럼 직접 입력 (안 하면 앞/뒤 이동에 위/아래가 섞임). 시작할 때 카메라를 건드리지 않기.
- 손바닥이 카메라 쪽으로 정면일 때 depth 가 가장 안정적.
- 해상도: USB 3 연결이면 848x480 (D435 depth 최적 해상도), USB 2 면 640x480 (`--res`). 시작 메시지/화면 `cam` 에 표시.
  **카메라를 허브가 아닌 PC 의 USB 3 (파란색) 포트에 USB 3 케이블로 연결.**
- depth 떨림은 거리 제곱에 비례 → 카메라를 손에서 0.6~0.8 m 에 두면 1 m 일 때의 절반 이하. 상체(어깨)는 화면에 들어와야 함.
- 양손 처리로 fps 가 약 20.

## 안전 장치

- 관절 최대 속도 제한 (`MAX_JOINT_VEL_DEG`, 기본 60°/s, 양팔), 소프트웨어 관절 한계
- TCP 작업 공간 박스 (`WORKSPACE_MIN/MAX`, 각 로봇 base 기준), 바닥(z) 아래로 가는 명령은 정지
- 손 추적 끊김 → 즉시 정지, 1 s 이상이면 그 팔만 추종 해제
- depth 가 한 프레임에 25 cm 이상 튀면 무시
- 명령-실제 관절 오차 15° 이상 0.5 s 지속 (충돌/걸림) → 그 팔 추종 해제
- PiPER 모터 드라이버 오류 / Milly fault 또는 명령 거부 → 그 팔 추종 해제
- 두 팔끼리 충돌 검사는 없음 → 설치 간격으로 확보
- **물리 비상정지 버튼을 항상 손 닿는 곳에.**

## 조정 포인트 (`teleop.py`)

- `ROBOTS[...]`: 로봇별 `tcp_offset` (flange → 그리퍼 끝, PiPER 0.13 / Milly 0.14 m), `ready_deg`.
- `WORKSPACE_MIN/MAX`: 조립 작업대 범위에 맞게 (각 로봇 base 기준). **z 최소값(기본 0.05 m) = 그리퍼 끝 최저 높이**, z=0 은 로봇 base 바닥면.
  부품이 base 와 같은 면에 있으면 `--table-z 0.01` (단 `tcp_offset` 이 실제 그리퍼 길이와 맞을 때만). 화면 `LIMIT: Z floor` 로 걸림 표시.
- `GRIPPER_MAX_UM` (70 mm), `GRIPPER_EFFORT` (1 N·m, 최대 5): PiPER 그리퍼. Milly 그리퍼 세기는 `milly_arm.py` 의 `GRIPPER_KP`.
- `GRIP_CLOSE_THRESH / GRIP_OPEN_THRESH`: 화면의 `grip ratio` 값을 보고 손에 맞게.
- `PIPER_CAN_USB` / `MILLY_CAN_USB`: CAN 자동 선택에 쓰는 USB 제품 이름. 직접 지정은 `--piper-can can1` 등.
