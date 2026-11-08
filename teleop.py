"""
사람 양손 동작 → 양팔 로봇 실시간 교시 (RealSense D435i + MediaPipe)
    왼손  → 왼팔  PiPER  (piper_sdk, candleLight CAN)
    오른손 → 오른팔 Milly (Motomind Milly SDK)

실행 예)
    # 1) 로봇 없이 화면/매핑만 확인
    ~/teleop_env/bin/python teleop.py --dry-run

    # 2) 실제 로봇 (CAN 두 개 모두 1 Mbps 로 켠 뒤)
    ~/teleop_env/bin/python teleop.py --milly-id MILLY_XXXX

    # 한쪽만: --arms left (PiPER 만) / --arms right (Milly 만)

키
    SPACE : 양팔 추종 시작/정지 (클러치) — 시작 순간의 손 ↔ 로봇 자세를 기준으로 "상대 이동"
    1 / 2 : 왼팔(PiPER) / 오른팔(Milly) 만 추종 시작/정지
    r     : READY 자세로 천천히 이동 (추종 정지 상태인 팔만)
    z     : ZERO(접힘) 자세로 천천히 이동 (추종 정지 상태인 팔만)
    g     : 그리퍼 추종 ON/OFF (OFF 이면 현재 그리퍼 유지)
    o     : 손 회전 추종 모드 전환 twist(비틀기만, 기본) → yaw → full → off → wrist(비틀기+꺾기+좌우)
    t     : 비틀기 boost ON/OFF (x1.5 + 50° 이상 비튼 채로 있으면 계속 회전 — 나사 돌리기용, 기본 OFF)
    m     : 위치 매핑 전환 camera(카메라 기준 상대) → shoulder(어깨 기준 상대) → absolute(어깨 기준 절대)
    k     : 보정 (추종 정지 상태에서) — 양 어깨 선으로 정면 방향 보정 + [absolute] 팔 길이로 scale, 지금 손 자세 = 지금 툴 자세
    b     : 그리퍼 binary(열림/닫힘) ↔ proportional 전환
    a     : 속도 적응 스케일 ON/OFF (느리게 움직이면 정밀)
    d     : 손바닥 depth 방식 region(기본) → plane(평면 교점) → patch (화면 jitter 비교용)
    [ / ] : 위치 스케일 감소/증가 (정밀 조립 시 작게)
    c     : CSV 기록 시작/정지 (teleop_logs/, 팔마다 파일 하나)
    x     : 비상정지 (양팔)
    q/ESC : 종료 — PiPER 는 마지막 위치 유지, Milly 는 SDK shutdown (10초 동안 감쇠 → 팔을 받치거나 먼저 z)
"""

import argparse
import collections
import csv
import os
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

import milly_kin
import piper_kin
from human_tracker import HumanTracker
from milly_arm import MillyArm
from piper_arm import PiperArm, find_can_by_usb_product


# ============================================================
# SETTINGS
# ============================================================

PIPER_CAN = "auto"                # auto: USB 제품 이름이 candleLight 인 CAN (재부팅/절전 후 can0/can1 이름이 바뀌어도 찾음)
PIPER_CAN_USB = "candleLight"
MILLY_PRODUCT_ID = "MILLY_Z4D7"   # 팔에 각인된 ID (motomind-milly-robots 로 확인) — 또는 --milly-id
MILLY_CAN_USB = "usctocan"

# 사람 손 → 로봇
SIDE_ROBOT = {"left": "piper", "right": "milly"}

# 로봇 속도
ROBOT_SPEED_PERCENT = 50          # PiPER 펌웨어 속도 비율 (ModeCtrl, movej 방식일 때)
PIPER_CTRL = "follow"             # follow: 펌웨어 MIT 고속 추종 (Milly 처럼 명령 위치를 바로 따라감)
                                  # movej: MOVE J (명령마다 펌웨어가 경로를 다시 계획 → 지연, 이전 방식)
MAX_JOINT_VEL_DEG = 60.0          # 추종 중 관절 최대 속도 (deg/s)
SLOW_MOVE_VEL_DEG = 20.0          # r / z 자세 이동 속도 (deg/s)

# Cartesian 매핑
POS_SCALE = 1.0                   # 사람 손 이동량 × POS_SCALE = 로봇 TCP 이동량
MAPPING = "camera"                # camera:   카메라 기준 손바닥 이동량 (클러치 상대 이동)
                                  # shoulder: 어깨 기준 손 위치 (손바닥 - 어깨) 이동량 → 몸이 흔들리거나 기울어도 로봇은 그대로
                                  # absolute: 사람 어깨 = 로봇 base 원점 (ABS_ORIGIN), 로봇 TCP = 원점 + (손바닥 - 어깨) × scale
                                  #           손이 어깨보다 위면 z > 0, 아래면 z < 0 / 앞이면 x > 0 / 왼쪽이면 y > 0
                                  #           같은 손 위치 = 항상 같은 로봇 위치 (클러치해도 안 바뀜, 속도 적응 스케일 없음)
                                  #           손 방향도 k (또는 첫 시작) 때의 손 자세 ↔ 툴 자세로 고정 대응
MAPPINGS = ["camera", "shoulder", "absolute"]
ABS_ORIGIN = np.array([0.0, 0.0, 0.30])  # [absolute] 사람 어깨에 대응하는 로봇 base 좌표 (m) — --abs-origin-z 로 높이 변경
                                         #   z 0.3: 사람은 보통 어깨보다 20~30 cm 아래에서 작업 → 로봇 작업 높이 0~0.1 m
ABS_AUTO_SCALE = True             # [absolute] scale = 로봇 도달 거리 (J2 축 → 그리퍼 끝) / 사람 팔 길이 (어깨 → 손바닥), × --pos-scale
ABS_SCALE_RANGE = (0.5, 1.5)
ABS_BLEND_SEC = 3.0               # [absolute] 시작/기준 재설정 때 로봇 ↔ 대응 자세 차이를 이 시간 동안 천천히 없앰
ROT_SCALE = 1.0

# 속도 적응 스케일 (PRISM 방식): 손이 느리면 작게, 빠르면 크게 → 멀리는 빠르게, 가까이선 정밀하게
#   로봇 이동 = Σ (프레임별 손 이동 × gain),  gain = POS_SCALE × interp(손 속도, [V_SLOW, V_FAST], [G_MIN, 1])
ADAPTIVE_GAIN = True
ADAPTIVE_V_SLOW = 0.03            # m/s 이하 → 최소 gain (정밀)
ADAPTIVE_V_FAST = 0.20            # m/s 이상 → gain 1 (그대로)
ADAPTIVE_G_MIN = 0.4               # 기록 분석: 0.25 면 추종 시간 절반 이상이 최소 gain → 손을 너무 크게 움직여야 함
ADAPTIVE_SPEED_WINDOW = 0.15      # 손 속도 계산 구간 (s) — 프레임 단위 떨림이 속도로 잡히지 않게
# 손 회전 추종
#   wrist: 손목 비틀기 → 툴 비틀기 (J6), 손목 꺾기 → 툴 위아래, 손목 좌우 → 툴 좌우
#          꺾기/좌우는 아래팔 기준 손 각도 → 팔을 옮길 때 저절로 같이 돌아가는 손 방향은 빠짐
#   twist: 비틀기만 (이전 wrist 기본) | yaw: base 수직축 회전만 | full: base 기준 3축 | off: 툴 방향 고정
ROT_MODE = "twist"                # 기본 = 이전과 같은 비틀기만 (꺾기/좌우는 o 로 wrist)
ROT_DEADBAND_DEG = 3.0            # 이보다 작은 손 회전은 무시 (손 추정 떨림 → 손목 관절 비틀림 방지)
# 비틀기: 사람 손목은 ±60~80° 정도라 부족 → 배율 + 한계 근처에서 계속 회전 (위치 + 속도 혼합)
TWIST_BOOST = False               # t 키: 비틀기 배율 TWIST_BOOST_GAIN + 한계 근처 계속 회전 (다리 나사 돌리기 등). 끄면 1:1 (이전과 같음)
TWIST_BOOST_GAIN = 1.5            # 사람 비틀기 × 배율 = 툴 비틀기 (boost 일 때)
TWIST_RATE_START_DEG = 50.0       # 기준에서 이 이상 비튼 채로 있으면 툴이 계속 돌아감
TWIST_RATE_FULL_DEG = 80.0        # 이만큼 비틀면 최대 속도
TWIST_RATE_MAX_DEG_S = 60.0       # 계속 회전 최대 속도 (툴 deg/s) — 손을 가운데로 돌리면 멈춤, 돌아간 만큼은 유지
# 꺾기 / 좌우 (아래팔 기준)
FLEX_GAIN = 1.0                   # 꺾기 (손바닥 쪽/손등 쪽, 사람 ±60°)
DEV_GAIN = 1.5                    # 좌우 (엄지 쪽/새끼 쪽, 사람 범위가 -25 ~ +35° 로 좁음)
WRIST_ANG_DEADBAND_DEG = 5.0      # 아래팔 방향 추정 떨림 (Pose) 이 손보다 커서 비틀기보다 크게
WRIST_ANG_HOLD_SPEED = 0.25       # 손이 이보다 빠르게 이동 중이면 꺾기/좌우 갱신 안 함 (이동 중 Pose 지연)
WRIST_ANG_MAX_RATE_DEG_S = 90.0   # 꺾기/좌우 명령 변화 속도 한계 (추정이 튀거나 갱신이 재개될 때 툴이 휙 돌지 않게)
FOLLOW_AZIMUTH = True             # TCP 가 좌우로 돌아가면 툴 방향도 같이 돌림 (J1 만 돌고 J4/J6 비틀림 없음)
IK_REST_WEIGHT = np.array([0.0, 0.0, 0.0, 0.05, 0.0, 0.0])   # J4 를 0 쪽으로 약하게 (손목 특이점에서 J4/J6 꼬임 방지)
WORKSPACE_MIN = np.array([0.10, -0.40, 0.05])   # TCP 허용 범위 (각 로봇 base frame, m). z=0 은 로봇 base 바닥면
#   z 최소값 = 그리퍼 끝이 내려갈 수 있는 최저 높이 (--table-z 로 변경). 부품이 base 와 같은 면에 있으면 0.01 정도,
#   단 로봇별 tcp_offset 이 실제 그리퍼 길이와 맞아야 함 (길이가 더 길면 그만큼 바닥에 부딪힘)
WORKSPACE_MAX = np.array([0.55, 0.40, 0.55])
MIN_RADIUS = 0.12                 # base 축에서 TCP 최소 수평 거리 (자기 충돌 방지)
IK_REACH_WARN_M = 0.02
IK_REJECT_M = 0.04                # IK 위치 오차가 이 이상이면 (팔이 못 닿는 곳) 해를 버리고 직전 목표 유지
IK_MAX_ROT_GIVE_DEG = 30.0        # 위치 우선 IK 에서 툴 방향이 목표와 이 이상 달라지면 해를 버림
MAX_LEAD_M = 0.03                 # 명령 목표를 현재 로봇 위치에서 이 거리 이내로 (로봇이 늦을 때 관절 보간 경로가 바닥 아래로 지나가는 것 방지)

# 그리퍼
GRIPPER_MAX_UM = 70000            # PiPER 최대 열림 (0.001 mm)
GRIPPER_EFFORT = 1000             # PiPER 0.001 N·m (최대 5000)
GRIP_MODE = "binary"              # binary: 열림/닫힘 2단계 (잡은 물건 놓치지 않음) | proportional: 손가락 벌린 만큼
GRIP_CLOSE_THRESH = 0.45          # [binary] grip ratio 가 이 아래로 내려가면 닫힘
GRIP_OPEN_THRESH = 0.70           # [binary] 이 위로 올라가면 열림 (사이 구간은 상태 유지 = 히스테리시스)
GRIP_CLOSE_RATIO = 0.30           # [proportional] 이 이하이면 완전히 닫힘
GRIP_OPEN_RATIO = 1.00            # [proportional] 이 이상이면 완전히 열림

# 손가락을 움직이는 동안 팔 고정 → 손가락이 멈추면 기준 재설정
# (손가락을 오므리면 손 위치/방향 추정이 같이 변해서 잡는 순간 로봇이 밀리는 것 방지)
FINGER_MOTION_RATE = 1.2          # grip ratio 변화 속도 (1/s) 가 이보다 크면 "손가락 움직이는 중"
FINGER_SETTLE_SEC = 0.15          # 이 시간 동안 조용하면 다시 추종
FINGER_FREEZE_MAX_SPEED = 0.10    # 손이 이보다 빠르게 이동 중이면 고정 안 함 (팔을 움직일 때 grip ratio 가 흔들려 이동이 버려지는 것 방지)

# 안전
LOST_HOLD_SEC = 0.3               # 추적 끊김이 이 시간 이상이면 정지 (짧은 끊김은 마지막 목표로 계속)
LOST_DISENGAGE_SEC = 1.0          # 이 시간 이상 끊기면 추종 자동 해제 (다시 SPACE 필요)
TRACKING_ERR_DEG = 15.0           # 명령-실제 관절 오차가 이 이상 지속되면 (충돌/걸림) 추종 해제
TRACKING_ERR_SEC = 0.5
FAULT_CHECK_SEC = 0.2             # 모터 드라이버 오류(충돌/스톨/과전류) 확인 주기
QUIT_CONFIRM_SEC = 3.0            # Milly 가 켜져 있으면 q 를 이 시간 안에 한 번 더 눌러야 종료

# 자기 충돌 방지 (base 기준 m)
WRIST_MIN_RADIUS = 0.06           # 손목 중심(J5)이 base 축에서 이보다 가까우면 거부
BASE_KEEPOUT_RADIUS = 0.10        # 손목~그리퍼 끝 선분이 이 원기둥 안으로 들어오면 거부
BASE_KEEPOUT_HEIGHT = 0.30

# 카메라
CAMERA_PITCH_DEG = None           # None: D435i IMU 로 자동 (권한 없으면 0) | 숫자: 카메라가 아래를 내려다보는 각도 (+)


# ============================================================
# ROBOTS
# ============================================================

@dataclass
class RobotSpec:

    name: str
    K: object                       # 기구학 모듈 (fk, fk_frames, ik, JOINT_MIN/MAX, WRIST_FRAME, WRIST_TO_FLANGE)
    ready_deg: np.ndarray
    zero_deg: np.ndarray
    tcp_offset: float               # flange → 그리퍼 끝 (flange z 방향, m)
    engage_ok: object               # q(rad) → 접힌 자세가 아니면 True
    tool_rz_deg: float = 0.0        # flange 축 → 공통 툴 축 (x: 그리퍼 아래쪽, y: 왼쪽) 회전 [wrist 모드 꺾기/돌리기용]
    ws_min: np.ndarray = field(default_factory=lambda: WORKSPACE_MIN.copy())
    ws_max: np.ndarray = field(default_factory=lambda: WORKSPACE_MAX.copy())


ROBOTS = {
    # READY: TCP ≈ (0.30, 0, 0.25) m, 툴 45° 아래 (두 로봇 같음)
    "piper": RobotSpec(
        name="PiPER", K=piper_kin,
        ready_deg=np.array([0.0, 60.0, -55.0, 0.0, 45.0, 0.0]),
        zero_deg=np.zeros(6),
        tcp_offset=0.13,
        engage_ok=lambda q: np.rad2deg(q[1]) >= 20.0,      # J2 ≈ 0 은 접힌 자세 (관절 한계에 걸려 손목만 움직임)
    ),
    "milly": RobotSpec(
        name="Milly", K=milly_kin,
        ready_deg=np.array([0.0, -52.0, -45.0, 0.0, 37.0, 0.0]),
        zero_deg=np.zeros(6),
        tcp_offset=0.14,            # 손가락 관절이 flange 에서 0.105 m + 손가락 → 실제 그리퍼 끝까지 재서 맞추기
        engage_ok=lambda q: np.rad2deg(q[1]) <= -20.0,     # J2 ≈ 0 (전부 0) 은 접힌 자세
        tool_rz_deg=-90.0,          # Milly flange x = 왼쪽, y = 툴 위쪽 → PiPER 와 같은 툴 축으로
    ),
}


# ============================================================
# SAFETY CHECK
# ============================================================

def pose_problem(spec, q):
    """바닥/로봇 몸체와 부딪힐 자세면 이유 문자열, 괜찮으면 "" """

    K = spec.K
    wrist = K.fk_frames(q)[K.WRIST_FRAME][:3, 3]
    tcp = K.fk(q, spec.tcp_offset)[:3, 3]

    if np.hypot(wrist[0], wrist[1]) < WRIST_MIN_RADIUS:
        return "wrist too close to base"

    for s in np.linspace(0.0, 1.0, 6):
        p = wrist + s * (tcp - wrist)
        if p[2] < spec.ws_min[2]:
            return f"below table z={p[2]:.2f}"
        if np.hypot(p[0], p[1]) < BASE_KEEPOUT_RADIUS and p[2] < BASE_KEEPOUT_HEIGHT:
            return "gripper near base"

    return ""


def adapt_tool_pitch(spec, R, p):
    """
    TCP 가 base 쪽으로 가까이 올수록 툴을 필요한 만큼만 더 수직으로 세움.
    툴이 45° 로 고정이면 손목 중심이 TCP 보다 (J5→TCP 길이)·cos(45°) 만큼 뒤에 있어서
    조금만 뒤로 와도 손목/그리퍼가 base 금지 영역에 들어감 → 손목이 TCP 위로 오도록.
    """

    reach = spec.K.WRIST_TO_FLANGE + spec.tcp_offset
    r = np.hypot(p[0], p[1])

    # 손목 수평 거리 = r - reach·cos(pitch) ≥ BASE_KEEPOUT_RADIUS + 여유 → 필요한 최소 아래 각
    c = np.clip((r - BASE_KEEPOUT_RADIUS - 0.01) / reach, 0.0, 1.0)
    need = np.arccos(c) + np.deg2rad(3.0)

    tz = R[:, 2]
    pitch_down = np.arcsin(np.clip(-tz[2], -1.0, 1.0))

    if pitch_down >= need:
        return R

    axis = np.cross(tz, [0.0, 0.0, -1.0])
    if np.linalg.norm(axis) < 1e-6:
        return R

    return piper_kin.matrix_from_rotvec(axis / np.linalg.norm(axis) * (need - pitch_down)) @ R


def robot_reach(spec, n=2000):
    """J2 축(로봇 어깨) → 그리퍼 끝 최대 거리 (m), 관절 한계 안에서 무작위 자세로 계산"""

    K = spec.K
    rng = np.random.default_rng(0)
    q = np.zeros(6)
    sh = K.fk_frames(q)[1][:3, 3]
    best = 0.0
    for _ in range(n):
        q[1:] = rng.uniform(K.JOINT_MIN[1:], K.JOINT_MAX[1:])
        best = max(best, np.linalg.norm(K.fk(q, spec.tcp_offset)[:3, 3] - sh))
    return best


def rot_z(a):

    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def soft_deadband_vec(v, deadband_deg):
    """크기가 deadband 이하이면 0, 넘으면 넘은 만큼만 (연속, 튀지 않음)"""

    n = np.linalg.norm(v)
    db = np.deg2rad(deadband_deg)

    if n <= db:
        return np.zeros_like(v)

    return v * (n - db) / n


# ============================================================
# ONE ARM (사람 손 하나 → 로봇 하나)
# ============================================================

class ArmTeleop:

    def __init__(self, side, spec, arm, opts):

        self.side = side
        self.spec = spec
        self.K = spec.K
        self.arm = arm
        self.o = opts                   # 양팔 공통 옵션 (DualTeleop: pos_scale, rot_mode, adaptive, grip_mode, grip_follow, mirror, dry_run)

        self.gain = 0.0
        self.grip_closed = False
        self.grip_hist = collections.deque(maxlen=4)     # (t, grip_ratio)
        self.finger_moving = False
        self.finger_quiet_since = None

        self.engaged = False
        self.lost_since = None
        self.held = False
        self.err_since = None
        self.fault_check_t = 0.0
        self.track_hist = collections.deque(maxlen=60)   # 최근 약 2초 추적 결과
        self.palm_hist = collections.deque(maxlen=30)    # 최근 약 1초 손바닥 위치 (떨림 표시용)
        self.warn = ""

        self.ref = None
        self.q_ik = None
        self.speed_hist = collections.deque(maxlen=30)   # (t, palm) — 손 이동 속도
        self.hand_speed = 0.0
        self.last_target = None
        self.last_good_p = None
        self.tool_rz = rot_z(np.deg2rad(spec.tool_rz_deg))

        self.cal = None                 # [absolute] dict(R=손 자세, T=툴 자세, ang=손목 각도, scale, arm_len) — k 로 저장
        self.arm_len_fn = None          # side → (사람 팔 길이 m, 출처) — DualTeleop 이 tracker 로 연결
        self.reach = robot_reach(spec)
        self.blend_off = np.zeros(3)
        self.blend_rot = np.zeros(3)
        self.blend_rot_decay = True
        self.blend_t0 = 0.0
        self.twist_extra = 0.0          # 비틀기 속도 모드로 더 돌아간 양 (rad)
        self.twist_step = 0.0
        self.t_rot = None
        self.t_ang = None
        self.wrist_d = np.zeros(2)      # 아래팔 기준 [꺾기, 좌우] 기준 대비 변화 (rad, 배율 전)

        self.rec_file = None
        self.rec_writer = None

    @property
    def label(self):

        return f"{'L' if self.side == 'left' else 'R'}:{self.spec.name}"

    def pos(self, st):
        """매핑에 쓰는 손 위치 — camera: 손바닥 (카메라 기준), shoulder/absolute: 손바닥 - 어깨. 모르면 None"""

        if not st.hand_valid:
            return None
        return st.palm if self.o.mapping == "camera" else st.hand_vec

    def lost_msg(self, st):

        return st.msg or ("shoulder not found" if st.hand_valid else "hand not found")

    def calibrate(self, st):
        """
        [absolute] 위치 scale (로봇 도달 거리 / 사람 팔 길이) + 지금 손 자세 ↔ 로봇 현재 목표 툴 자세 (방향 절대 대응 기준)
        위치 원점은 고정: 사람 어깨 = ABS_ORIGIN (로봇 base)
        """

        if not st.hand_valid:
            return False

        L, src = self.arm_len_fn(self.side) if self.arm_len_fn else (None, "")
        scale = 1.0
        if ABS_AUTO_SCALE and L:
            scale = float(np.clip(self.reach / L, *ABS_SCALE_RANGE))

        self.cal = dict(
            R=st.R_hand.copy(),
            T=self.K.fk(self.arm.q_target, self.spec.tcp_offset),
            ang=None if st.wrist_ang is None else st.wrist_ang.copy(),
            scale=scale,
            arm_len=L,
            src=src,
        )
        print(f"[{self.label}] absolute 보정: 팔 길이 {f'{L:.2f} m ({src})' if L else '측정 못 함'}, "
              f"로봇 도달 {self.reach:.2f} m → scale {scale:.2f} (× pos-scale {self.o.pos_scale:.2f}), 지금 손 자세 = 지금 툴 자세")
        return True

    def abs_point(self, h):
        """사람 어깨 = ABS_ORIGIN, (손바닥 - 어깨) × scale"""

        return self.o.abs_origin + self.o.mirror @ h * (self.cal["scale"] * self.o.pos_scale)

    # --------------------------------------------------------

    def engage(self, st):

        if not st.hand_valid:
            self.warn = "추적이 안 되어 시작할 수 없음"
            return

        if self.pos(st) is None:
            self.warn = "어깨가 안 보여 시작할 수 없음 (shoulder/absolute 매핑)"
            return

        q0 = self.arm.q_cmd

        # 접힌 자세는 관절 한계에 걸려 손목만 움직임 → READY 에서 시작해야 함
        if not self.spec.engage_ok(q0) or pose_problem(self.spec, q0):
            self.warn = "접힌 자세에서는 시작 불가 → r (READY) 이동 후 SPACE"
            return

        if not self.arm.at_target(tol_deg=1.0):
            self.warn = "자세 이동 중 → 끝난 뒤 SPACE"
            return

        if self.o.mapping == "absolute" and self.cal is None:
            self.calibrate(st)
            print(f"[{self.label}] (k 없이 시작 → 지금 저장, 다시 하려면 정지 후 k)")

        self.reanchor(st, q0)
        self.engaged = True
        self.lost_since = None
        self.finger_moving = False
        self.warn = ""

    def reanchor(self, st, q=None, keep_rot_offset=False):
        """
        지금 사람 손 ↔ 로봇 목표 자세를 새 기준으로 (로봇은 움직이지 않음)
        [absolute] 기준은 고정 → 로봇 ↔ 대응 자세 차이를 blend 로 (keep_rot_offset: 회전 차이는 줄이지 않고 유지)
        """

        q = self.arm.q_target if q is None else q
        h = self.pos(st)

        self.ref = dict(
            q=q,
            T=self.K.fk(q, self.spec.tcp_offset),
            palm=h.copy(),
            R=st.R_hand.copy(),
            ang=None if st.wrist_ang is None else st.wrist_ang.copy(),
        )
        self.q_ik = q.copy()
        self.last_good_p = self.ref["T"][:3, 3].copy()

        # 회전: 기준 툴 방향에 지금까지 돌아간 양이 들어 있음 → 누적값 초기화
        self.twist_extra = 0.0
        self.twist_step = 0.0
        self.t_rot = None
        self.t_ang = None
        self.wrist_d = np.zeros(2)

        # 속도 적응 스케일 누적 상태
        self.offset = np.zeros(3)
        self.palm_prev = h.copy()
        self.speed_hist.clear()
        self.speed_hist.append((st.t, h.copy()))
        self.hand_speed = 0.0
        self.gain = ADAPTIVE_G_MIN * self.o.pos_scale

        # [absolute] 로봇 ↔ 대응 자세 차이 → ABS_BLEND_SEC 동안 줄여 나감 (튀지 않게)
        if self.o.mapping == "absolute":
            if self.cal is None:          # 추종 중 m 으로 바꾼 경우
                self.calibrate(st)
            c = self.cal
            if st.wrist_ang is not None and c["ang"] is not None:
                self.wrist_d = st.wrist_ang - c["ang"]
            p_abs = np.clip(self.abs_point(h), self.spec.ws_min, self.spec.ws_max)
            R_abs = self.orient(st, p_abs, c, twist_rate=False)
            self.blend_off = self.ref["T"][:3, 3] - self.abs_point(h)
            self.blend_rot = self.K.rotvec_from_matrix(self.ref["T"][:3, :3] @ R_abs.T)
            self.blend_rot_decay = not keep_rot_offset
            self.blend_t0 = st.t

    def update_speed(self, st):
        """최근 ADAPTIVE_SPEED_WINDOW 동안의 손바닥 이동 속도 (m/s)"""

        h = self.pos(st)
        self.speed_hist.append((st.t, h.copy()))
        t_old, p_old = self.speed_hist[0]
        for t_h, p_h in self.speed_hist:
            if st.t - t_h <= ADAPTIVE_SPEED_WINDOW:
                t_old, p_old = t_h, p_h
                break
        self.hand_speed = np.linalg.norm(h - p_old) / max(st.t - t_old, 1e-3)

    def disengage(self, reason=""):

        self.engaged = False
        self.arm.hold()
        if reason:
            self.warn = reason

    # --------------------------------------------------------

    def target_cartesian(self, st):

        r = self.ref
        K = self.K
        spec = self.spec
        mirror = self.o.mirror
        h = self.pos(st)
        relative_adaptive = self.o.adaptive and self.o.mapping != "absolute"

        absolute = self.o.mapping == "absolute"
        k_blend = max(0.0, 1.0 - (st.t - self.blend_t0) / ABS_BLEND_SEC)

        if absolute:
            # 어깨 = 로봇 원점 고정 대응 + 시작 때 차이 (천천히 0 으로)
            self.gain = self.cal["scale"] * self.o.pos_scale
            p = self.abs_point(h) + k_blend * self.blend_off
        elif relative_adaptive:
            g = np.interp(self.hand_speed, [ADAPTIVE_V_SLOW, ADAPTIVE_V_FAST], [ADAPTIVE_G_MIN, 1.0])
            self.gain = g * self.o.pos_scale
            self.offset = self.offset + self.gain * (h - self.palm_prev)
            self.palm_prev = h.copy()
            p = r["T"][:3, 3] + mirror @ self.offset
        else:
            self.gain = self.o.pos_scale
            p = r["T"][:3, 3] + mirror @ (h - r["palm"]) * self.o.pos_scale

        p = np.clip(p, spec.ws_min, spec.ws_max)
        rad = np.hypot(p[0], p[1])
        if rad < MIN_RADIUS:
            p[:2] *= MIN_RADIUS / max(rad, 1e-6)

        # 범위 끝에서 손을 더 밀어도 누적되지 않게 (되돌아올 때 바로 반응)
        if relative_adaptive:
            self.offset = mirror @ (p - r["T"][:3, 3])

        # 로봇이 따라오는 중이면 목표를 로봇 근처로 (원래 목표는 유지 → 다음 프레임에 계속 따라감)
        p_cmd = K.fk(self.arm.q_cmd, spec.tcp_offset)[:3, 3]
        lead = np.linalg.norm(p - p_cmd)
        if lead > MAX_LEAD_M:
            p = p_cmd + (p - p_cmd) * (MAX_LEAD_M / lead)

        T = np.eye(4)
        T[:3, 3] = p

        # [absolute] 손 방향도 보정 자세 기준 (고정 대응), 시작 때 차이는 천천히 0 으로
        R = self.orient(st, p, self.cal if absolute else r, twist_rate=not absolute)
        if absolute:
            k_rot = k_blend if self.blend_rot_decay else 1.0
            R = K.matrix_from_rotvec(k_rot * self.blend_rot) @ R
        T[:3, :3] = R

        T[:3, :3] = adapt_tool_pitch(spec, T[:3, :3], p)
        self.last_target = T

        q, pe, re = self.solve_ik(T)

        # 닿지 않는 곳 → 마지막으로 닿았던 위치와 목표 사이에서 닿는 가장 먼 점으로 (멈추지 않고 경계를 따라 미끄러짐)
        if pe > IK_REJECT_M and self.last_good_p is not None:
            lo, hi = self.last_good_p.copy(), p.copy()
            best = None
            for _ in range(4):
                T[:3, 3] = 0.5 * (lo + hi)
                qm, pem, rem = self.solve_ik(T)
                if pem <= IK_REACH_WARN_M:
                    lo, best = T[:3, 3].copy(), (qm, pem, rem)
                else:
                    hi = T[:3, 3].copy()
            if best is not None:
                q, pe, re = best
                p = lo
                T[:3, 3] = p

        if pe > IK_REJECT_M:
            self.warn = f"도달 불가 (위치 오차 {pe * 1000:.0f} mm) → 목표 무시"
            self.q_ik = self.arm.q_target.copy()
            return None

        # 비틀기 속도 모드: J6 이 한계에 닿으면 더 돌리지 않음 (한계에서 벗어나는 쪽은 허용)
        if self.twist_step != 0.0:
            j6 = np.rad2deg(q[5])
            lo, hi = self.K.JOINT_MIN_DEG[5] + 3.0, self.K.JOINT_MAX_DEG[5] - 3.0
            if (j6 > hi and q[5] >= self.q_ik[5] - 1e-4) or (j6 < lo and q[5] <= self.q_ik[5] + 1e-4):
                self.twist_extra -= self.twist_step
            self.twist_step = 0.0

        self.q_ik = q
        self.last_good_p = p.copy()

        if pe > IK_REACH_WARN_M:
            self.warn = f"도달 한계 (위치 오차 {pe * 1000:.0f} mm)"
        else:
            self.warn = ""

        return q

    def solve_ik(self, T):
        """방향까지 맞추고, 위치가 안 닿으면 (J5 한계 등) 위치 우선으로 다시 (툴 방향이 조금 양보)"""

        tcp = self.spec.tcp_offset
        q, pe, re = self.K.ik(T, self.q_ik, tcp, rest_weight=IK_REST_WEIGHT)

        if pe > IK_REACH_WARN_M:
            q2, pe2, re2 = self.K.ik(T, self.q_ik, tcp, rot_weight=0.02, rest_weight=IK_REST_WEIGHT)
            if pe2 < pe and np.rad2deg(re2) < IK_MAX_ROT_GIVE_DEG:
                q, pe, re = q2, pe2, re2

        return q, pe, re

    def orient(self, st, p, rr, twist_rate=True):
        """
        목표 툴 방향 (base 기준 3x3). rr: 기준 dict (T=기준 툴 자세, R=기준 손 자세, ang=기준 손목 각도)
        twist_rate: 비틀기 속도 모드 (absolute 에서는 끔 — 같은 손 자세 = 같은 툴 자세 유지)
        """

        K = self.K
        mirror = self.o.mirror

        # 기준 툴 방향: TCP 가 base 축 둘레로 돈 만큼 같이 돌림 (팔이 옆으로 가도 툴은 바깥쪽을 향함)
        R_ref = rr["T"][:3, :3]
        if FOLLOW_AZIMUTH:
            p0 = rr["T"][:3, 3]
            R_ref = rot_z(np.arctan2(p[1], p[0]) - np.arctan2(p0[1], p0[0])) @ R_ref

        dR = mirror @ (st.R_hand @ rr["R"].T) @ mirror

        if self.o.rot_mode in ("wrist", "twist"):
            # base 기준 회전벡터 → 기준 손 좌표축 성분 [손가락 축 (비틀기), 손 옆 축 (꺾기), 손바닥 법선 (좌우)]
            H = mirror @ rr["R"]
            rv_h = H.T @ K.rotvec_from_matrix(dR)
            tw = soft_deadband_vec(rv_h[:1], ROT_DEADBAND_DEG)[0]
            if not self.o.twist_boost:
                twist = tw
            elif twist_rate:
                twist = self.twist_cmd(st, tw)
            else:
                twist = TWIST_BOOST_GAIN * tw
            flex, dev = self.flex_dev(st, rr) if self.o.rot_mode == "wrist" else (0.0, 0.0)
            rv_h = np.array([twist, flex, dev]) * ROT_SCALE
            # 손 축 → 공통 툴 축: 손가락 = 툴 z(접근), 손바닥 법선 = 툴 x, 손 옆(오른쪽) = -툴 y → 로봇 flange 축
            rv_t = self.tool_rz @ np.array([rv_h[2], -rv_h[1], rv_h[0]])
            return R_ref @ K.matrix_from_rotvec(rv_t)

        if self.o.rot_mode == "full":
            rv = K.rotvec_from_matrix(dR)
            return K.matrix_from_rotvec(soft_deadband_vec(rv, ROT_DEADBAND_DEG) * ROT_SCALE) @ R_ref

        if self.o.rot_mode == "yaw":
            # 손 회전 중 base Z(수직) 축 성분만 (swing-twist 분해의 twist)
            yaw = 2.0 * np.arctan2(dR[1, 0] - dR[0, 1], 1.0 + np.trace(dR))
            yaw = soft_deadband_vec(np.array([yaw]), ROT_DEADBAND_DEG)[0] * ROT_SCALE
            return rot_z(yaw) @ R_ref

        return R_ref

    def twist_cmd(self, st, tw):
        """
        손목 비틀기 (rad, 기준 대비) → 툴 비틀기: TWIST_GAIN 배 + 한계 근처에서 비튼 채로 있으면 계속 회전
        (손을 가운데로 돌리면 멈추고, 그동안 돌아간 양은 유지 → 사람 손목 범위보다 크게 돌릴 수 있음)
        """

        dt = 0.0 if self.t_rot is None else min(max(st.t - self.t_rot, 0.0), 0.1)
        self.t_rot = st.t

        rate = np.interp(np.degrees(abs(tw)), [TWIST_RATE_START_DEG, TWIST_RATE_FULL_DEG], [0.0, TWIST_RATE_MAX_DEG_S])
        self.twist_step = np.sign(tw) * np.deg2rad(rate) * dt
        self.twist_extra += self.twist_step

        return TWIST_BOOST_GAIN * tw + self.twist_extra

    def flex_dev(self, st, rr):
        """아래팔 기준 손목 [꺾기, 좌우] 기준 대비 변화 × 배율 (rad). 아래팔을 모르거나 손이 빠르게 이동 중이면 마지막 값"""

        dt = 0.0 if self.t_ang is None else min(max(st.t - self.t_ang, 0.0), 0.1)
        self.t_ang = st.t

        if st.wrist_ang is not None and self.hand_speed < WRIST_ANG_HOLD_SPEED:
            if rr["ang"] is None:                 # 기준 잡을 때 아래팔을 몰랐으면 지금 값을 (마지막 명령 유지) 기준으로
                rr["ang"] = st.wrist_ang - self.wrist_d
            d = st.wrist_ang - rr["ang"]
            max_step = np.deg2rad(WRIST_ANG_MAX_RATE_DEG_S) * dt
            self.wrist_d = self.wrist_d + np.clip(d - self.wrist_d, -max_step, max_step)

        db = [soft_deadband_vec(self.wrist_d[i:i + 1], WRIST_ANG_DEADBAND_DEG)[0] for i in range(2)]
        sgn = np.linalg.det(self.o.mirror)        # 좌우 반전이면 손 기준 회전 방향도 반대
        return sgn * FLEX_GAIN * db[0], sgn * DEV_GAIN * db[1]

    # --------------------------------------------------------

    def step(self, st):

        now = st.t
        h = self.pos(st)
        self.track_hist.append("" if h is not None else self.lost_msg(st))
        if h is not None:
            self.palm_hist.append(h.copy())

        # ---------------- 그리퍼 ----------------
        if self.engaged and self.o.grip_follow and st.hand_valid:
            if self.o.grip_mode == "binary":
                if self.grip_closed and st.grip_ratio > GRIP_OPEN_THRESH:
                    self.grip_closed = False
                elif not self.grip_closed and st.grip_ratio < GRIP_CLOSE_THRESH:
                    self.grip_closed = True
                self.arm.set_gripper(0.0 if self.grip_closed else 1.0)
            else:
                g = (st.grip_ratio - GRIP_CLOSE_RATIO) / (GRIP_OPEN_RATIO - GRIP_CLOSE_RATIO)
                self.arm.set_gripper(g)

        if not self.engaged:
            return

        # ---------------- 추적 끊김 ----------------
        if h is None:
            if self.lost_since is None:
                self.lost_since = now
                self.held = False
            lost = now - self.lost_since
            if lost > LOST_HOLD_SEC and not self.held:
                self.arm.hold()
                self.held = True
            if lost > LOST_DISENGAGE_SEC:
                self.disengage(f"추적 끊김 → 추종 해제 ({self.lost_msg(st)})")
            return

        self.lost_since = None
        self.update_speed(st)

        # ---------------- 손가락 움직이는 중 → 팔 고정, 멈추면 기준 재설정 ----------------
        # (손이 빠르게 이동 중일 땐 grip ratio 흔들림으로 보고 고정 안 함)
        self.grip_hist.append((now, st.grip_ratio))
        t_old, r_old = self.grip_hist[0]
        rate = abs(st.grip_ratio - r_old) / max(now - t_old, 1e-3)

        if rate > FINGER_MOTION_RATE and self.hand_speed < FINGER_FREEZE_MAX_SPEED:
            self.finger_moving = True
            self.finger_quiet_since = None
            return

        if self.finger_moving:
            self.finger_quiet_since = self.finger_quiet_since or now
            if now - self.finger_quiet_since < FINGER_SETTLE_SEC:
                return
            self.finger_moving = False
            self.reanchor(st)
            return

        # 손 방향 추정이 크게 바뀐 채 유지됨 (뒤집힘 등) → 로봇을 돌리지 않고 기준만 새로
        if st.extra.get("rot_reset"):
            self.reanchor(st, keep_rot_offset=True)     # [absolute] 뒤집힌 추정 쪽으로 툴이 천천히 돌아가지 않게
            return

        # ---------------- 목표 관절 ----------------
        q = self.target_cartesian(st)
        if q is None:
            return
        q = np.clip(q, self.K.JOINT_MIN, self.K.JOINT_MAX)

        # 바닥/자기 충돌 체크
        # 범위를 벗어난 목표는 버리고 직전 목표 유지 (멈췄다 움직였다 반복하지 않게)
        problem = pose_problem(self.spec, q)
        if problem:
            self.warn = f"작업 공간 밖 ({problem}) → 목표 무시"
            self.q_ik = self.arm.q_target.copy()     # IK 시작점을 마지막 정상 목표로 되돌림
            return

        self.arm.set_target(q)

        # ---------------- 모터 오류 / 명령-실제 오차 (실제 로봇만) ----------------
        if not self.o.dry_run:
            if now - self.fault_check_t > FAULT_CHECK_SEC:
                self.fault_check_t = now
                faults = self.arm.motor_faults()
                if faults:
                    detail = "; ".join(f"{j} {'/'.join(f)}" for j, f in faults.items())
                    self.disengage(f"모터 오류 {detail} → 추종 해제 (재실행 시 해제 가능)")
                    return

            err = np.max(np.abs(np.rad2deg(self.arm.read_joints() - self.arm.q_cmd)))
            if err > TRACKING_ERR_DEG:
                self.err_since = self.err_since or now
                if now - self.err_since > TRACKING_ERR_SEC:
                    self.disengage(f"관절 오차 {err:.0f}° (충돌/걸림?) → 추종 해제")
            else:
                self.err_since = None

    # --------------------------------------------------------

    def start_record(self, stamp):

        os.makedirs("teleop_logs", exist_ok=True)
        path = f"teleop_logs/teleop_{stamp}_{self.side}_{self.spec.name.lower()}.csv"
        self.rec_file = open(path, "w", newline="")
        self.rec_writer = csv.writer(self.rec_file)
        self.rec_writer.writerow(
            ["t", "engaged"] + [f"q{i + 1}_deg" for i in range(6)] + [self.arm.gripper_col]
            + ["palm_x", "palm_y", "palm_z", "grip_ratio"]
            + ["hand_valid", "hand_speed", "gain", "finger_moving", "tgt_x", "tgt_y", "tgt_z",
               "n_pts", "depth_normal", "msg", "warn", "hand_rx", "hand_ry", "hand_rz"]
            + [f"fb{i + 1}_deg" for i in range(6)]
            + ["mapping", "hand_vec_x", "hand_vec_y", "hand_vec_z", "flex_deg", "dev_deg",
               "cmd_flex_deg", "cmd_dev_deg", "twist_extra_deg"]
        )
        print("기록 시작:", path)

    def stop_record(self):

        if self.rec_file is not None:
            self.rec_file.close()
        self.rec_file = self.rec_writer = None

    def record(self, st):

        if self.rec_writer is None:
            return

        palm = st.palm if st.hand_valid else [np.nan] * 3
        self.rec_writer.writerow(
            [f"{time.time():.3f}", int(self.engaged)]
            + [f"{v:.3f}" for v in np.rad2deg(self.arm.q_cmd)]
            + [self.arm.gripper_value()]
            + [f"{v:.4f}" for v in palm]
            + [f"{st.grip_ratio:.3f}" if st.grip_ratio is not None else ""]
            + [int(st.hand_valid), f"{self.hand_speed:.3f}", f"{self.gain:.2f}", int(self.finger_moving)]
            + ([f"{v:.4f}" for v in self.last_target[:3, 3]] if self.last_target is not None else [""] * 3)
            + [st.extra.get("n_pts", ""), int(bool(st.extra.get("depth_normal"))), st.msg, self.warn]
            + ([f"{v:.4f}" for v in self.K.rotvec_from_matrix(st.R_hand)] if st.hand_valid else [""] * 3)
            + [f"{v:.3f}" for v in np.rad2deg(self.arm.read_joints())]
            + [self.o.mapping]
            + ([f"{v:.4f}" for v in st.hand_vec] if st.hand_vec is not None else [""] * 3)
            + ([f"{v:.2f}" for v in np.rad2deg(st.wrist_ang)] if st.wrist_ang is not None else [""] * 2)
            + [f"{v:.2f}" for v in np.rad2deg(self.wrist_d)]
            + [f"{np.rad2deg(self.twist_extra):.2f}"]
        )

    # --------------------------------------------------------

    def panel(self, st, H, width=300):

        panel = np.zeros((H, width, 3), dtype=np.uint8)

        q_cmd = np.rad2deg(self.arm.q_cmd)
        q_fb = np.rad2deg(self.arm.read_joints())
        T = self.K.fk(self.arm.q_cmd, self.spec.tcp_offset)

        hand = "LEFT hand" if self.side == "left" else "RIGHT hand"
        lines = [
            (f"{self.spec.name}  <- {hand}", (0, 255, 255)),
            (">> ENGAGED" if self.engaged else "   IDLE (SPACE / " + ("1)" if self.side == "left" else "2)"),
             (0, 255, 0) if self.engaged else (0, 200, 255)),
            (f"hand {'OK' if st.hand_valid else '--'}  {st.msg}", (0, 255, 0) if st.hand_valid else (0, 0, 255)),
            (f"gain x{self.gain:.2f}  speed {self.hand_speed:.2f} m/s", (255, 255, 255)),
            self.rot_line(st),
            (f"depth N={st.extra['n_pts']}" if "n_pts" in st.extra else "", (200, 200, 200)),
            (f"grip {self.o.grip_mode} {'ON' if self.o.grip_follow else 'OFF'}"
             f"{'  CLOSED' if self.grip_closed and self.o.grip_mode == 'binary' else ''}", (255, 255, 255)),
            ("[fingers moving: arm hold]" if self.finger_moving else "", (0, 200, 255)),
            self.track_line(),
            self.jitter_line(),
            self.limit_line(T, q_cmd),
            ("REC" if self.rec_writer else "", (0, 0, 255)),
            ("     cmd     fb", (200, 200, 200)),
        ]
        for i in range(6):
            lines.append((f"J{i + 1} {q_cmd[i]:7.1f} {q_fb[i]:7.1f}", (255, 255, 255)))

        lines += [
            (f"TCP [m] {T[0, 3]:.3f} {T[1, 3]:.3f} {T[2, 3]:.3f}", (255, 255, 255)),
            (self.arm.gripper_text(), (255, 255, 255)),
        ]
        if st.grip_ratio is not None:
            lines.append((f"grip ratio {st.grip_ratio:.2f}", (200, 200, 200)))
        if self.warn:
            lines.append((self.warn_ascii()[:40], (0, 0, 255)))

        y = 22
        for text, col in lines:
            if text:
                cv2.putText(panel, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.47, col, 1, cv2.LINE_AA)
            y += 20

        if self.engaged:
            cv2.rectangle(panel, (0, 0), (width - 1, H - 1), (0, 255, 0), 3)

        return panel

    def rot_line(self, st):
        """손목 각도: 아래팔 기준 꺾기/좌우 (사람), 비틀기 속도 모드로 더 돌아간 양"""

        if st.wrist_ang is None:
            text = "flex/dev --"
        else:
            f, d = np.rad2deg(st.wrist_ang)
            text = f"flex {f:+3.0f} dev {d:+3.0f}"
        if abs(self.twist_extra) > 1e-3:
            text += f" tw{np.rad2deg(self.twist_extra):+.0f}"
        if self.o.mapping == "absolute":
            text += f" CAL x{self.cal['scale']:.2f}" if self.cal is not None else " no cal"
        return (text, (200, 200, 200))

    def limit_line(self, T, q_deg):
        """한계에 걸려 더 못 가는 것 표시 (손을 더 움직여도 로봇이 안 움직이는 이유)"""

        hit = []
        if T[2, 3] < self.spec.ws_min[2] + 0.005:
            hit.append(f"Z floor {self.spec.ws_min[2]:.2f}")
        for j in range(6):
            if q_deg[j] <= self.K.JOINT_MIN_DEG[j] + 1 or q_deg[j] >= self.K.JOINT_MAX_DEG[j] - 1:
                hit.append(f"J{j + 1}")
        return (("LIMIT: " + " ".join(hit)) if hit else "", (0, 0, 255))

    def jitter_line(self):
        """손을 가만히 들고 있을 때 축별 손바닥 떨림 (body frame: 앞뒤=카메라 depth 축)"""

        if len(self.palm_hist) < 20:
            return ("", None)

        sd = np.std(np.array(self.palm_hist), axis=0) * 1000
        return (f"jitter mm F/B {sd[0]:.1f} L/R {sd[1]:.1f} U/D {sd[2]:.1f}", (200, 200, 200))

    def track_line(self):

        n = len(self.track_hist)
        if n == 0:
            return ("", None)

        fails = [m for m in self.track_hist if m]
        ok = 100.0 * (n - len(fails)) / n
        text = f"track {ok:3.0f}%"
        if fails:
            top = collections.Counter(fails).most_common(1)[0][0]
            text += f"  ({top})"

        col = (0, 255, 0) if ok > 90 else (0, 200, 255) if ok > 60 else (0, 0, 255)
        return (text, col)

    def warn_ascii(self):
        """OpenCV putText 는 한글 미지원 → 화면엔 영문 요약, 터미널엔 원문"""

        table = {
            "추적이 안": "Cannot engage: hand not tracked",
            "어깨가 안": "Cannot engage: shoulder not visible",
            "접힌 자세": "Folded -> press r (READY), then SPACE",
            "자세 이동": "Moving -> wait, then SPACE",
            "추적 끊김": "Tracking lost -> disengaged",
            "도달 한계": "Reach limit",
            "도달 불가": "Unreachable -> target ignored",
            "depth 방식": "Depth method changed -> SPACE",
            "모터 오류": "Motor fault -> disengaged (terminal)",
            "관절 오차": "Joint error too large -> disengaged",
        }
        if self.warn.startswith("작업 공간"):
            return "Rejected: " + self.warn[self.warn.find("(") + 1:self.warn.find(")")]

        for k, v in table.items():
            if self.warn.startswith(k):
                return v
        return self.warn


# ============================================================
# DUAL TELEOP (카메라 하나, 팔 여러 개)
# ============================================================

class Options:
    """양팔 공통 옵션 (키로 바꾸면 양팔 같이)"""

    def __init__(self, args):

        self.pos_scale = args.pos_scale
        self.rot_mode = args.rot
        self.adaptive = not args.no_adaptive
        self.grip_mode = args.grip
        self.grip_follow = True
        self.mirror = np.diag([1.0, -1.0, 1.0]) if args.mirror else np.eye(3)
        self.mapping = args.map
        self.twist_boost = TWIST_BOOST
        self.abs_origin = ABS_ORIGIN.copy()
        self.abs_origin[2] = args.abs_origin_z
        self.dry_run = args.dry_run


def make_arm(robot, args):

    if robot == "piper":
        can = args.piper_can
        if can == "auto":
            can = find_can_by_usb_product(PIPER_CAN_USB) or "can0"
        return PiperArm(
            can_name=can,
            dry_run=args.dry_run,
            speed_percent=args.speed,
            follow=args.piper_ctrl == "follow",
            max_joint_vel_deg=args.max_vel,
            gripper_effort=GRIPPER_EFFORT,
            gripper_max_um=GRIPPER_MAX_UM,
        )

    can = args.milly_can
    if can == "auto":
        can = find_can_by_usb_product(MILLY_CAN_USB)
    return MillyArm(
        product_id=args.milly_id,
        can_name=can,
        dry_run=args.dry_run,
        max_joint_vel_deg=args.max_vel,
    )


class DualTeleop:

    def __init__(self, args, tracker=None, arms=None):

        self.args = args
        self.o = Options(args)
        sides = ["left", "right"] if args.arms == "both" else [args.arms]

        self.arms = []
        for side in sides:
            robot = SIDE_ROBOT[side]
            arm = arms[side] if arms else make_arm(robot, args)
            self.arms.append(ArmTeleop(side, ROBOTS[robot], arm, self.o))

        self.tracker = tracker or HumanTracker(
            sides=sides,
            resolution=None if args.res == "auto" else tuple(int(v) for v in args.res.split("x")),
            camera_pitch_deg=args.camera_pitch,
            depth_preset=args.depth_preset,
            palm_method=args.palm,
        )
        for a in self.arms:
            a.arm_len_fn = self.tracker.arm_length
        self.recording = False
        self.cal_pending = 0            # k: 어깨 선 보정 뒤 필터가 다시 안정될 때까지 기다렸다가 손 위치 저장
        self.quit_armed_t = None
        self.msg = ""

    # --------------------------------------------------------

    def connect(self):
        """PiPER 먼저 (종료해도 자세 유지), Milly 나중. 하나라도 실패하면 이미 연결된 팔 정리 후 종료"""

        done = []
        try:
            for a in sorted(self.arms, key=lambda a: a.spec.name != "PiPER"):
                a.arm.connect()
                a.arm.start(initial_q=np.deg2rad(a.spec.ready_deg) if self.args.dry_run else None)
                done.append(a)
        except BaseException:
            for a in done:
                a.arm.stop()
            raise

    def toggle_record(self):

        if self.recording:
            for a in self.arms:
                a.stop_record()
            self.recording = False
            print("기록 종료")
            return

        stamp = time.strftime("%Y%m%d_%H%M%S")
        for a in self.arms:
            a.start_record(stamp)
        self.recording = True

    def milly_live(self):

        return any(isinstance(a.arm, MillyArm) and a.arm.enabled for a in self.arms)

    def draw(self, img, states, fps):

        self.tracker.draw(img, states, labels={a.side: a.spec.name for a in self.arms})

        H, W = img.shape[:2]

        tr = self.tracker
        head = (f"{'DRY-RUN  ' if self.args.dry_run else ''}fps {fps:4.1f}  scale {self.o.pos_scale:.2f}  "
                f"map {self.o.mapping.upper()}  rot {self.o.rot_mode.upper()}{'+BOOST' if self.o.twist_boost else ''}  "
                f"{'adaptive' if self.o.adaptive and self.o.mapping != 'absolute' else 'fixed'}"
                f"{'  MIRROR' if self.args.mirror else ''}")
        head2 = (f"cam {tr.width}x{tr.height} tilt {tr.camera_pitch_deg:.0f}/{tr.camera_roll_deg:.0f} ({tr.tilt_source})"
                 f"  body yaw {tr.body_yaw_deg:+.0f}  depth {tr.palm_method}")
        cv2.putText(img, head, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, head2, (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        if self.msg:
            cv2.putText(img, self.msg, (10, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

        # 카메라 영상은 거울상이 아님: 사람 왼손이 화면 오른쪽 → 왼손(PiPER) 패널을 오른쪽에
        panels = {a.side: a.panel(states[a.side], H) for a in self.arms}
        return np.hstack([p for p in [panels.get("right"), img, panels.get("left")] if p is not None])

    # --------------------------------------------------------

    def run(self):

        self.connect()

        win = "Dual Teleop"
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

        t_prev = time.perf_counter()
        fps = 0.0
        last_warn = {a.side: "" for a in self.arms}

        try:
            while True:

                img, states = self.tracker.read()
                for a in self.arms:
                    a.step(states[a.side])
                    a.record(states[a.side])
                    if a.warn and a.warn != last_warn[a.side]:
                        print(f"[!] {a.label}: {a.warn}")
                    last_warn[a.side] = a.warn

                now = time.perf_counter()
                fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-3)
                t_prev = now

                if self.cal_pending:
                    self.cal_pending -= 1
                    if self.cal_pending == 0:
                        self.finish_calibration(states)

                if self.quit_armed_t is not None and now - self.quit_armed_t > QUIT_CONFIRM_SEC:
                    self.quit_armed_t = None
                    self.msg = ""

                cv2.imshow(win, self.draw(img, states, fps))
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    if not self.milly_live() or self.quit_armed_t is not None:
                        break
                    # Milly 는 종료 시 SDK shutdown (감쇠 → 팔이 내려감) → 한 번 더 확인
                    self.quit_armed_t = now
                    self.msg = "Milly will power down (damping 10 s): support it or press z first. q again = quit"
                    print("[!] Milly 종료 시 10초 동안 감쇠 → 팔이 천천히 내려갑니다. 팔을 받치거나 z 로 접은 뒤 q 를 한 번 더")

                elif key == ord(" "):
                    if any(a.engaged for a in self.arms):
                        for a in self.arms:
                            a.disengage()
                    else:
                        for a in self.arms:
                            a.engage(states[a.side])

                elif key in (ord("1"), ord("2")):
                    side = "left" if key == ord("1") else "right"
                    for a in self.arms:
                        if a.side == side:
                            if a.engaged:
                                a.disengage()
                            else:
                                a.engage(states[a.side])

                elif key in (ord("r"), ord("z")):
                    for a in self.arms:
                        if not a.engaged:
                            pose = a.spec.ready_deg if key == ord("r") else a.spec.zero_deg
                            a.arm.set_target(np.deg2rad(pose), max_vel_deg=SLOW_MOVE_VEL_DEG)

                elif key == ord("g"):
                    self.o.grip_follow = not self.o.grip_follow

                elif key == ord("o"):
                    modes = ["twist", "yaw", "full", "off", "wrist"]
                    self.o.rot_mode = modes[(modes.index(self.o.rot_mode) + 1) % len(modes)]
                    self.reanchor_engaged(states)    # 기준 재설정 (튀지 않게)

                elif key == ord("m"):
                    self.o.mapping = MAPPINGS[(MAPPINGS.index(self.o.mapping) + 1) % len(MAPPINGS)]
                    for a in self.arms:
                        a.cal = None             # 손 위치 기준이 바뀌었으므로 absolute 보정 다시
                    self.reanchor_engaged(states)
                    print(f"위치 매핑: {self.o.mapping}")

                elif key == ord("t"):
                    self.o.twist_boost = not self.o.twist_boost
                    self.reanchor_engaged(states)
                    print(f"비틀기 boost: {'ON (x' + str(TWIST_BOOST_GAIN) + ' + 계속 회전)' if self.o.twist_boost else 'OFF (1:1)'}")

                elif key == ord("k"):
                    self.start_calibration()

                elif key == ord("a"):
                    self.o.adaptive = not self.o.adaptive
                    self.reanchor_engaged(states)

                elif key == ord("d"):
                    methods = ["region", "plane", "patch"]
                    self.tracker.palm_method = methods[(methods.index(self.tracker.palm_method) + 1) % len(methods)]
                    print(f"손바닥 depth: {self.tracker.palm_method}")
                    self.tracker.reset_filters()
                    for a in self.arms:
                        a.palm_hist.clear()
                        if a.engaged:
                            a.disengage("depth 방식 변경 → SPACE 로 다시 시작")

                elif key == ord("b"):
                    self.o.grip_mode = "proportional" if self.o.grip_mode == "binary" else "binary"

                elif key in (ord("["), ord("]")):
                    f = 0.8 if key == ord("[") else 1.25
                    self.o.pos_scale = float(np.clip(self.o.pos_scale * f, 0.1, 3.0))
                    self.reanchor_engaged(states)

                elif key == ord("c"):
                    self.toggle_record()

                elif key == ord("x"):
                    for a in self.arms:
                        a.engaged = False
                        a.arm.emergency_stop()

        finally:
            for a in self.arms:
                a.engaged = False
                a.arm.hold()
            time.sleep(0.1)
            for a in self.arms:
                a.arm.stop()
                a.stop_record()
            self.tracker.close()
            cv2.destroyAllWindows()
            print("종료 (PiPER 는 마지막 위치 유지)")

    def reanchor_engaged(self, states):

        for a in self.arms:
            if a.engaged and a.pos(states[a.side]) is not None:
                a.reanchor(states[a.side])
            elif a.engaged:
                a.disengage("추적 끊김 → 추종 해제 (모드 변경)")

    def start_calibration(self):
        """k: 1) 양 어깨 선 → 정면 방향 보정 2) 잠시 뒤 [absolute] 팔 길이 scale + 지금 손 자세 = 지금 툴 자세"""

        if any(a.engaged for a in self.arms):
            self.msg = "Calibrate: stop following first (SPACE), then k"
            return

        yaw = self.tracker.calibrate_body(self.tracker.shoulders())
        if yaw is None:
            print("[!] 보정: 양 어깨가 보이지 않거나 너무 비스듬함 → 정면 방향 보정 생략 (카메라를 마주 보고 다시 k)")
        else:
            print(f"보정: 사람 정면 방향 {yaw:+.1f}° (카메라 기준)")
        for a in self.arms:
            a.palm_hist.clear()
        self.cal_pending = 10
        self.msg = "Calibrating: hold hands still in the neutral working pose"

    def finish_calibration(self, states):

        for a in self.arms:
            if not a.calibrate(states[a.side]):
                print(f"[!] 보정: {a.label} 손이 안 보여 저장 못 함")
        self.msg = ""


# ============================================================
# MAIN
# ============================================================

def main():

    ap = argparse.ArgumentParser(description="RealSense 사람 양손 → PiPER(왼팔) + Milly(오른팔) 실시간 교시")
    ap.add_argument("--arms", choices=["both", "left", "right"], default="both",
                    help="both: 양팔 / left: 왼손→PiPER 만 / right: 오른손→Milly 만")
    ap.add_argument("--dry-run", action="store_true", help="로봇 연결 없이 실행")
    ap.add_argument("--mirror", action="store_true", help="로봇이 사람을 마주 보고 있을 때 (좌우 반전)")
    ap.add_argument("--piper-can", default=PIPER_CAN, help="PiPER CAN 이름 (auto: candleLight 어댑터 자동 선택)")
    ap.add_argument("--milly-id", default=MILLY_PRODUCT_ID, help="Milly 제품 ID (팔에 각인, 예: MILLY_6TG5)")
    ap.add_argument("--milly-can", default="auto", help="Milly CAN 이름 (auto: usctocan 어댑터, SDK 가 지원할 때만 사용)")
    ap.add_argument("--speed", type=int, default=ROBOT_SPEED_PERCENT, help="PiPER 펌웨어 속도 비율")
    ap.add_argument("--max-vel", type=float, default=MAX_JOINT_VEL_DEG, help="관절 최대 속도 deg/s (양팔)")
    ap.add_argument("--pos-scale", type=float, default=POS_SCALE)
    ap.add_argument("--camera-pitch", type=float, default=CAMERA_PITCH_DEG,
                    help="카메라가 아래를 내려다보는 각도 (deg). 생략하면 IMU 로 자동")
    ap.add_argument("--res", default="auto", choices=["auto", "848x480", "640x480"],
                    help="카메라 해상도 (auto: 848x480, USB 2 면 640x480)")
    ap.add_argument("--map", choices=MAPPINGS, default=MAPPING,
                    help="위치 매핑: camera(카메라 기준 상대) / shoulder(어깨 기준 상대) / absolute(어깨 기준 절대, k 로 보정)")
    ap.add_argument("--abs-origin-z", type=float, default=ABS_ORIGIN[2],
                    help="[absolute] 사람 어깨 높이에 대응하는 로봇 z (m, base 바닥 = 0, 기본 0.3). 0 이면 손이 어깨보다 아래 = 작업대 아래")
    ap.add_argument("--piper-ctrl", choices=["follow", "movej"], default=PIPER_CTRL,
                    help="PiPER 제어: follow(MIT 고속 추종, Milly 와 같은 방식) / movej(이전 방식)")
    ap.add_argument("--rot", choices=["wrist", "twist", "yaw", "full", "off"], default=ROT_MODE,
                    help="손 회전 추종: twist(비틀기만, 기본) / wrist(비틀기+꺾기+좌우) / yaw(수직축만) / full(base 3축) / off(고정)")
    ap.add_argument("--grip", choices=["binary", "proportional"], default=GRIP_MODE)
    ap.add_argument("--no-adaptive", action="store_true", help="속도 적응 스케일 끄기 (고정 pos-scale)")
    ap.add_argument("--palm", choices=["region", "plane", "patch"], default="region",
                    help="손바닥 depth: region(손바닥 영역 평균, 기본) / plane(영역 평면과 기준점 시선의 교점) / patch(중심 13x13)")
    ap.add_argument("--table-z", type=float, default=WORKSPACE_MIN[2],
                    help="그리퍼 끝 최저 높이 (m, 각 로봇 base 바닥면 = 0). 부품이 같은 면에 있으면 0.01 정도")
    ap.add_argument("--depth-preset", default="medium_density",
                    choices=["medium_density", "high_accuracy", "default", "hand", "high_density"])
    args = ap.parse_args()

    for spec in ROBOTS.values():
        spec.ws_min[2] = args.table_z

    DualTeleop(args).run()


if __name__ == "__main__":
    main()
