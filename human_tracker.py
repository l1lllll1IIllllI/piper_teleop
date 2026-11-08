"""
사람 팔/손 추적 (RealSense D435i + MediaPipe Pose/Hands)

출력 좌표계 = "body frame" (로봇 base 와 같은 축 정의)
    X: 사람 정면 (카메라 쪽)
    Y: 사람 왼쪽
    Z: 위
→ 사람이 카메라를 마주 보고 서 있다고 가정

- 위치(손바닥, 어깨, 팔꿈치, 손목): RealSense depth 로 deprojection (단위 m)
- 손 방향(회전)·그립: MediaPipe Hands world landmark
- 손목 꺾기/좌우 (wrist_ang): 아래팔(Pose 팔꿈치→손목) 기준 손 각도 → 팔을 옮길 때 같이 돌아가는 손 방향은 빠짐
- 어깨 기준 손 위치 (hand_vec = 손바닥 - 어깨): 몸이 흔들려도 손-몸 상대 위치
- 카메라 기울기: D435i IMU 중력 방향으로 자동 보정 (권한 없으면 camera_pitch_deg 사용)
  사람 정면 방향: calibrate_body() — 양 어깨 선으로 카메라 ↔ 사람 좌우 회전 보정
- 양손 동시 추적: 손마다 필터/튐 제거 상태 따로, 손 ↔ 왼/오른쪽은 Pose 손목 위치로 짝지음
"""

import collections
import time
from dataclasses import dataclass, field

import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs


mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

POSE_IDX = {
    "right": dict(shoulder=12, elbow=14, wrist=16),
    "left": dict(shoulder=11, elbow=13, wrist=15),
}

# 손바닥 기준점 (wrist, index/middle/ring/pinky MCP) — 엄지(1)는 엄지만 움직여도 위치가 바뀌어 제외
PALM_IDX = [0, 5, 9, 13, 17]

# 손바닥 영역 다각형 (손목, 엄지 CMC, 검지~새끼 MCP) — 손가락을 오므려도 거의 안 변하는 부분
PALM_POLY_IDX = [0, 1, 5, 9, 13, 17]

# RealSense depth preset (정적 물체 2 m 실측: Default 14.8 mm → Medium Density + 레이저 최대 7.0 mm, 유효 픽셀 94%)
DEPTH_PRESETS = {"default": 1, "hand": 2, "high_accuracy": 3, "high_density": 4, "medium_density": 5}

# 손바닥 평면 법선 사용 조건 (PCA: 가장 얇은 방향 분산 / 두 번째 분산)
NORMAL_MAX_FLATNESS = 0.2
NORMAL_MIN_POINTS = 80
NORMAL_MIN_SPREAD = 0.012        # 두 번째 방향 표준편차가 이보다 작으면 (손 옆면만 보임) 법선 안 씀 (m)
NORMAL_MIN_FACING = 0.5           # 법선과 카메라 시선의 |cos| 이 이보다 작으면 (60° 이상 비스듬) depth 법선 안 씀
NORMAL_BLEND = 0.7               # depth 법선 비중 (나머지는 MediaPipe 추정)

# 손바닥 거리: 영역 점들로 평면을 맞춘 뒤 기준점 픽셀의 시선과 만나는 점 (평균 depth 는 손바닥이 기울면
# 유효 점 분포가 바뀔 때마다 기준점과 다른 곳의 거리가 되어 흔들림)
PLANE_MAX_FLATNESS = 0.35
PLANE_MIN_FACING = 0.35           # 평면과 시선이 이보다 비스듬하면 (|cos|) 평균 depth 사용

# 카메라 해상도: D435 depth 최적은 848x480 (USB 3 필요). 안 되면 640x480
CAMERA_RES = [(848, 480), (640, 480)]
IMU_SAMPLE_SEC = 0.5              # 시작 시 중력 방향 측정 시간

# 어깨 / 아래팔
FILTER_SHOULDER = (0.8, 0.3)      # 어깨는 천천히 움직임 → 손보다 강하게 (손 위치에서 빼므로 떨림이 손에 더해짐)
SHOULDER_HOLD_SEC = 1.0           # 어깨가 잠깐 안 보여도 마지막 값 사용
FOREARM_MIN_LEN = 0.12            # Pose world 팔꿈치→손목 길이가 이보다 짧으면 (추정 실패) 아래팔 방향 안 씀 (m)
FILTER_WRIST_ANG = (1.0, 0.3)
BODY_YAW_MAX_DEG = 45.0           # 어깨 선 보정 한계 (이보다 크면 사람이 옆으로 선 것 → 거부)

# 팔 길이 (어깨 → 손바닥): absolute 매핑 scale = 로봇 도달 거리 / 사람 팔 길이
WRIST_TO_PALM = 0.07              # 손목 → 손바닥 중심 (m)
ARM_LEN_RANGE = (0.45, 0.90)      # 이 밖의 측정값은 버림 (팔꿈치가 가려져 depth 가 몸통에 걸린 경우 등)
ARM_LEN_MIN_SAMPLES = 15

# One Euro 필터 (min_cutoff [Hz], beta): cutoff 가 낮을수록 정지 시 떨림↓, beta 가 클수록 빠른 동작 지연↓
FILTER_POS = (1.5, 0.4)
FILTER_ROT = (1.5, 0.3)
ROT_JUMP_DEG = 35.0               # 손 방향이 한 프레임에 이보다 크게 바뀌면 추정 오류(뒤집힘)로 보고 무시
ROT_JUMP_ACCEPT = 4               # 연속 이만큼 계속되면 실제 변화로 받아들이고 회전 기준 재설정 (st.extra["rot_reset"])
FILTER_GRIP = (2.0, 0.5)


# ============================================================
# ONE EURO FILTER (지연은 작게, 떨림은 줄이기)
# ============================================================

class OneEuro:

    def __init__(self, min_cutoff=1.0, beta=0.05, d_cutoff=1.0):

        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):

        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):

        self.x_prev = None

    def __call__(self, x, t):

        x = np.asarray(x, dtype=float)

        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            return x

        dt = max(t - self.t_prev, 1e-3)
        self.t_prev = t

        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * np.linalg.norm(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat

        return x_hat


def orthonormalize(M):

    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt

    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    return R


def unit(v):

    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


# ============================================================
# RESULT
# ============================================================

@dataclass
class HumanState:

    t: float = 0.0

    hand_valid: bool = False
    palm: np.ndarray = None          # body frame (m)
    R_hand: np.ndarray = None        # body frame, columns = [손가락 방향, 손 옆방향, 손바닥 법선]
    grip_ratio: float = None         # 엄지-검지 거리 / 손바닥 크기

    arm_valid: bool = False
    shoulder: np.ndarray = None
    elbow: np.ndarray = None
    wrist: np.ndarray = None
    hand_vec: np.ndarray = None      # 손바닥 - 어깨 (body frame, m), 어깨를 모르면 None
    wrist_ang: np.ndarray = None     # 아래팔 기준 손 [꺾기 (손 옆축 회전), 좌우 (손바닥 법선축 회전)] rad, 모르면 None

    # 화면 표시용
    pose_landmarks: object = None
    hand_landmarks: object = None
    palm_px: tuple = None
    palm_cam: np.ndarray = None
    R_hand_cam: np.ndarray = None
    msg: str = ""
    extra: dict = field(default_factory=dict)


# ============================================================
# PALM FROM DEPTH REGION
# ============================================================

def palm_from_region(depth_m, poly_px, intr, depth_min=0.25, depth_max=2.5, band=0.03, center_px=None, use_plane=False):
    """
    손바닥 다각형 안의 depth 점 전체로 손바닥 위치·법선 추정 (카메라 좌표, m)

    - 위치: 기준점 픽셀 center_px (2D 는 MediaPipe 가 정확, 없으면 다각형 중심) + 영역 depth 의 trimmed mean
            use_plane: 거리 = 영역 점들로 맞춘 평면과 기준점 시선이 만나는 점 (비스듬하거나 점이 부족하면 평균)
            → 구멍/가장자리 배경/몸통 depth 에 덜 흔들림
    - 법선: 영역 점들의 PCA 최소 분산 방향 (평평하게 보일 때만)

    return: (point (3,), normal (3,) or None, n_points) — 점이 부족하면 (None, None, n)
    """

    H, W = depth_m.shape
    poly = np.asarray(poly_px, dtype=np.float64)

    x0, y0 = np.maximum(np.floor(poly.min(axis=0)).astype(int), 0)
    x1, y1 = np.minimum(np.ceil(poly.max(axis=0)).astype(int), [W - 1, H - 1])
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None, None, 0

    mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(poly - [x0, y0]).astype(np.int32), 1)

    vs, us = np.nonzero(mask)
    z = depth_m[y0 + vs, x0 + us]

    ok = (z > depth_min) & (z < depth_max)
    if ok.sum() < 30:
        return None, None, int(ok.sum())

    z_med = np.median(z[ok])
    ok &= np.abs(z - z_med) < band          # 손 뒤 몸통/배경, 손 앞 손가락 끝 제거
    n = int(ok.sum())
    if n < 30:
        return None, None, n

    us, vs, z = us[ok] + x0, vs[ok] + y0, z[ok]

    c = poly.mean(axis=0) if center_px is None else np.asarray(center_px, dtype=np.float64)
    ray = np.array([(c[0] - intr.ppx) / intr.fx, (c[1] - intr.ppy) / intr.fy, 1.0])
    z_mean = float(z.mean())
    point = ray * z_mean

    normal = None
    if n >= NORMAL_MIN_POINTS:
        pts = np.column_stack([(us - intr.ppx) / intr.fx * z, (vs - intr.ppy) / intr.fy * z, z])
        m = pts.mean(axis=0)
        evals, evecs = np.linalg.eigh(np.cov((pts - m).T))
        if evals[1] > NORMAL_MIN_SPREAD ** 2:
            nrm = evecs[:, 0]
            if use_plane and evals[0] / evals[1] < PLANE_MAX_FLATNESS:
                denom = np.dot(ray, nrm)
                if abs(denom) / np.linalg.norm(ray) > PLANE_MIN_FACING:
                    zc = np.dot(m, nrm) / denom
                    if abs(zc - z_mean) < band:
                        point = ray * zc
            if evals[0] / evals[1] < NORMAL_MAX_FLATNESS:
                normal = nrm

    return point, normal, n


# ============================================================
# CAMERA SETUP
# ============================================================

def measure_up_vector(sample_sec=IMU_SAMPLE_SEC):
    """
    D435i 가속도계로 "위" 방향 (카메라 좌표 단위벡터). 실패하면 None
    정지 상태의 가속도계 값 = 중력 반대 방향 (카메라가 수평이면 약 (0, -9.8, 0))
    """

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.accel)

    try:
        pipe.start(cfg)
    except RuntimeError as exc:
        print(f"[!] IMU 를 읽을 수 없음 ({str(exc).split(' Last Error')[0][-60:]})")
        print("    RealSense udev 규칙 설치 후 카메라를 다시 꽂으면 자동 보정됩니다 (README). 지금은 --camera-pitch 값 사용")
        return None

    acc = []
    try:
        t0 = time.time()
        while time.time() - t0 < sample_sec:
            for f in pipe.wait_for_frames(1000):
                if f.is_motion_frame():
                    m = f.as_motion_frame().get_motion_data()
                    acc.append([m.x, m.y, m.z])
    except RuntimeError:
        pass
    finally:
        pipe.stop()

    if len(acc) < 10:
        print("[!] IMU 데이터 부족 → --camera-pitch 값 사용")
        return None

    acc = np.array(acc)
    g = np.linalg.norm(acc.mean(axis=0))
    if not 8.5 < g < 11.0 or np.linalg.norm(acc.std(axis=0)) > 0.5:
        print(f"[!] IMU 값 이상 (|a|={g:.2f}, 카메라가 흔들리는 중?) → --camera-pitch 값 사용")
        return None

    return unit(acc.mean(axis=0))


def body_rotation(up_cam):
    """카메라 좌표 → body (X: 사람 정면 = 카메라 쪽, Y: 사람 왼쪽, Z: 위) 회전 행렬. 정면은 카메라 시선의 반대를 수평으로"""

    Z = unit(np.asarray(up_cam, dtype=float))
    f = np.array([0.0, 0.0, -1.0])
    X = unit(f - np.dot(f, Z) * Z)
    Y = np.cross(Z, X)
    return np.vstack([X, Y, Z])


def camera_tilt_deg(up_cam):
    """(아래로 내려다본 각, 옆으로 기운 각) deg"""

    up = unit(np.asarray(up_cam, dtype=float))
    pitch = np.degrees(np.arctan2(-up[2], -up[1]))
    roll = np.degrees(np.arctan2(up[0], np.hypot(up[1], up[2])))
    return pitch, roll


def pick_resolution(fps=30):
    """CAMERA_RES 중 컬러·depth 가 모두 지원하는 첫 해상도 (USB 2 연결이면 848x480 없음)"""

    try:
        dev = rs.context().query_devices()[0]
    except (RuntimeError, IndexError):
        return CAMERA_RES[-1]

    usb = dev.get_info(rs.camera_info.usb_type_descriptor) if dev.supports(rs.camera_info.usb_type_descriptor) else "?"
    sizes = []
    for s in dev.query_sensors():
        sizes.append({(p.as_video_stream_profile().width(), p.as_video_stream_profile().height())
                      for p in s.get_stream_profiles() if p.is_video_stream_profile() and p.fps() == fps})
    sizes = [s for s in sizes if s]

    for res in CAMERA_RES:
        if all(res in s for s in sizes):
            if res != CAMERA_RES[0]:
                print(f"[!] 카메라가 USB {usb} 로 연결됨 → {CAMERA_RES[0][0]}x{CAMERA_RES[0][1]} 불가, "
                      f"{res[0]}x{res[1]} 사용 (USB 3 포트 + USB 3 케이블 권장)")
            return res

    return CAMERA_RES[-1]


def rot_z(a):

    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# ============================================================
# TRACKER
# ============================================================

HAND_MATCH_MAX = 0.15             # Pose 손목 ↔ 손 손목 픽셀 거리 한계 (화면 폭 비율)


class _Side:
    """한쪽 손의 필터·튐 제거 상태"""

    def __init__(self, side):

        self.side = side
        self.idx = POSE_IDX[side]
        mk = lambda: OneEuro(*FILTER_POS)
        self.f_palm = mk()
        self.f_R = OneEuro(*FILTER_ROT)
        self.f_grip = OneEuro(*FILTER_GRIP)
        self.f_sh = OneEuro(*FILTER_SHOULDER)
        self.f_el, self.f_wr = mk(), mk()
        self.f_ang = OneEuro(*FILTER_WRIST_ANG)
        self.arm_len_depth = collections.deque(maxlen=90)   # 최근 약 3초 팔 길이 측정 (depth 3D)
        self.arm_len_world = collections.deque(maxlen=90)   # (MediaPipe Pose world, 모델 추정)
        self.reset()

    def reset(self):

        for f in (self.f_palm, self.f_R, self.f_grip, self.f_sh, self.f_el, self.f_wr, self.f_ang):
            f.reset()
        self.last_palm_cam = None
        self.jump_count = 0
        self.last_R_cam = None
        self.rot_jump_count = 0
        self.shoulder = None             # 필터된 어깨 (body frame), 마지막으로 본 시각
        self.shoulder_t = -1e9
        self.forearm_cam = None          # 아래팔 방향 (팔꿈치→손목, MediaPipe 카메라 축 단위벡터)


class HumanTracker:
    """
    sides: 추적할 사람 손 ("left", "right") — read() 는 {side: HumanState} 반환
    resolution: (w, h) 또는 None (CAMERA_RES 중 카메라가 지원하는 첫 번째)
    camera_pitch_deg: None 이면 IMU 로 자동 (실패하면 0), 숫자면 그 값 (아래를 내려다보면 +)
    """

    def __init__(
        self,
        sides=("left", "right"),
        resolution=None,
        fps=30,
        camera_pitch_deg=None,
        depth_min=0.25,
        depth_max=2.5,
        max_jump_m=0.25,
        depth_preset="medium_density",
        palm_method="region",
    ):

        self.sides = tuple(sides)
        # 어깨 선 보정 / 손 짝짓기는 양쪽 모두 필요 → 추적은 항상 양쪽, 결과는 sides 만
        self.s = {side: _Side(side) for side in ("left", "right")}
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.max_jump_m = max_jump_m
        self.palm_method = palm_method      # region: 손바닥 영역 평균 depth | plane: 영역 평면 ∩ 기준점 시선 | patch: 중심 13x13

        # ---------------- 카메라 → body 회전 ----------------
        # camera(x 오른쪽, y 아래, z 앞) → body(X 앞, Y 왼쪽, Z 위)
        self.tilt_source = "manual"
        up = None
        if camera_pitch_deg is None:
            up = measure_up_vector()
            if up is not None:
                self.tilt_source = "IMU"
            camera_pitch_deg = 0.0
        if up is None:
            phi = np.deg2rad(camera_pitch_deg)   # 카메라가 아래를 내려다보면 +
            up = np.array([0.0, -np.cos(phi), -np.sin(phi)])
        self.C = body_rotation(up)
        self.body_yaw_deg = 0.0
        self.camera_pitch_deg, self.camera_roll_deg = camera_tilt_deg(up)
        print(f"카메라 기울기 ({self.tilt_source}): 아래로 {self.camera_pitch_deg:.1f}°, 옆으로 {self.camera_roll_deg:.1f}°")

        # ---------------- RealSense ----------------
        if resolution is None:
            resolution = pick_resolution(fps)
        self.width, self.height = resolution
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, fps)
        profile = self.pipeline.start(cfg)
        print(f"카메라 {self.width}x{self.height} @ {fps} fps")

        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        if depth_preset in DEPTH_PRESETS:
            depth_sensor.set_option(rs.option.visual_preset, DEPTH_PRESETS[depth_preset])
            if depth_sensor.supports(rs.option.laser_power):
                depth_sensor.set_option(rs.option.laser_power, depth_sensor.get_option_range(rs.option.laser_power).max)
        self.align = rs.align(rs.stream.color)
        self.intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

        # ---------------- MediaPipe ----------------
        self.pose = mp_pose.Pose(
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.hands = mp_hands.Hands(
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )

    # --------------------------------------------------------

    def close(self):

        self.pipeline.stop()
        self.pose.close()
        self.hands.close()

    def reset_filters(self):

        for s in self.s.values():
            s.reset()

    def calibrate_body(self, shoulders):
        """
        양 어깨 선이 body Y 축(왼쪽)이 되도록 수직축 회전 보정 → 사람이 카메라를 비스듬히 마주 봐도 앞/옆이 섞이지 않음
        shoulders: {"left": 왼쪽 어깨, "right": 오른쪽 어깨} (현재 body frame). return: 보정 각 (deg) 또는 None
        """

        if shoulders.get("left") is None or shoulders.get("right") is None:
            return None

        d = shoulders["left"] - shoulders["right"]
        if np.hypot(d[0], d[1]) < 0.2:
            return None
        yaw = np.arctan2(d[1], d[0]) - np.pi / 2       # 정상이면 d ≈ +Y → 0
        yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
        if abs(np.degrees(yaw + np.deg2rad(self.body_yaw_deg))) > BODY_YAW_MAX_DEG:
            return None

        self.C = rot_z(-yaw) @ self.C
        self.body_yaw_deg += np.degrees(yaw)
        self.reset_filters()
        return self.body_yaw_deg

    def arm_length(self, side):
        """사람 팔 길이 (어깨 → 손바닥, m) 와 출처. depth 측정 우선, 부족하면 Pose 모델 추정, 둘 다 없으면 (None, "")"""

        s = self.s[side]
        if len(s.arm_len_depth) >= ARM_LEN_MIN_SAMPLES:
            return float(np.median(s.arm_len_depth)), "depth"
        if len(s.arm_len_world) >= ARM_LEN_MIN_SAMPLES:
            return float(np.median(s.arm_len_world)), "pose"
        return None, ""

    def shoulders(self):
        """{"left": 어깨, "right": 어깨} — 추적 중인 쪽만 (calibrate_body 용)"""

        return {side: s.shoulder for side, s in self.s.items()}

    def _reject_rot_jump(self, s, R_cam, st):
        """손 방향 뒤집힘 제거 (기록: 한 프레임에 30~180° 튐 → J6 이 휙 돌거나 한계에 걸림)"""

        if s.last_R_cam is not None:
            jump = np.degrees(np.arccos(np.clip((np.trace(R_cam @ s.last_R_cam.T) - 1.0) / 2.0, -1.0, 1.0)))
            if jump > ROT_JUMP_DEG:
                s.rot_jump_count += 1
                if s.rot_jump_count < ROT_JUMP_ACCEPT:
                    st.extra["rot_jump"] = True
                    return s.last_R_cam
                s.f_R.reset()                # 필터가 뒤집힌 두 방향을 섞지 않게
                st.extra["rot_reset"] = True
            s.rot_jump_count = 0

        s.last_R_cam = R_cam
        return R_cam

    # --------------------------------------------------------
    # DEPTH → 3D
    # --------------------------------------------------------

    def _point3d(self, depth_m, u, v, r=4):

        h, w = depth_m.shape
        u, v = int(round(u)), int(round(v))

        if not (0 <= u < w and 0 <= v < h):
            return None

        patch = depth_m[max(v - r, 0):v + r + 1, max(u - r, 0):u + r + 1]
        vals = patch[(patch > self.depth_min) & (patch < self.depth_max)]

        if vals.size < 5:
            return None

        z = float(np.median(vals))
        return np.array(rs.rs2_deproject_pixel_to_point(self.intr, [float(u), float(v)], z))

    def to_body(self, p_cam):

        return self.C @ p_cam

    # --------------------------------------------------------
    # MAIN
    # --------------------------------------------------------

    def read(self):
        """한 프레임 처리 → (color_bgr, {side: HumanState})"""

        frames = self.align.process(self.pipeline.wait_for_frames())
        color = np.asanyarray(frames.get_color_frame().get_data())
        depth_m = np.asanyarray(frames.get_depth_frame().get_data()).astype(np.float32) * self.depth_scale

        t = time.perf_counter()
        states = {side: HumanState(t=t) for side in self.sides}

        rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        pose_res = self.pose.process(rgb)
        hand_res = self.hands.process(rgb)

        H, W = color.shape[:2]
        wrist_px = {}

        for side, s in self.s.items():
            st = states.get(side) or HumanState(t=t)
            st.pose_landmarks = pose_res.pose_landmarks
            wrist_px[side] = self._arm(pose_res, depth_m, s, st, W, H, t)

        assign = self._assign_hands(hand_res, wrist_px, W, H)

        for side, st in states.items():
            if side not in assign:
                st.msg = "hand not found"
                continue
            i = assign[side]
            self._hand(hand_res.multi_hand_landmarks[i], hand_res.multi_hand_world_landmarks[i],
                       depth_m, self.s[side], st, W, H, t)

        return color, states

    def _arm(self, pose_res, depth_m, s, st, W, H, t):
        """어깨/팔꿈치/손목 3D, 아래팔 방향 → s, st. return: Pose 손목 픽셀 (손 선택용) 또는 None"""

        s.forearm_cam = None
        if pose_res.pose_landmarks is None:
            return None

        lm = pose_res.pose_landmarks.landmark
        w = lm[s.idx["wrist"]]
        wrist_px = (w.x * W, w.y * H) if w.visibility > 0.5 else None

        # 아래팔 방향: Pose world landmark (모델 추정 3D) — 팔이 카메라 쪽을 향해 팔꿈치가 가려져도 나옴
        wl = pose_res.pose_world_landmarks
        if wl is not None and lm[s.idx["elbow"]].visibility > 0.5 and w.visibility > 0.5:
            e, r = wl.landmark[s.idx["elbow"]], wl.landmark[s.idx["wrist"]]
            f = np.array([r.x - e.x, r.y - e.y, r.z - e.z])
            if np.linalg.norm(f) > FOREARM_MIN_LEN:
                s.forearm_cam = unit(f)
            sh = wl.landmark[s.idx["shoulder"]]
            if lm[s.idx["shoulder"]].visibility > 0.5:
                L = np.linalg.norm([e.x - sh.x, e.y - sh.y, e.z - sh.z]) + np.linalg.norm(f) + WRIST_TO_PALM
                if ARM_LEN_RANGE[0] < L < ARM_LEN_RANGE[1]:
                    s.arm_len_world.append(L)

        pts = {}
        for name, i in s.idx.items():
            pts[name] = None
            if lm[i].visibility > 0.5:
                p = self._point3d(depth_m, lm[i].x * W, lm[i].y * H)
                if p is not None:
                    pts[name] = self.to_body(p)

        if pts["shoulder"] is not None:
            s.shoulder = s.f_sh(pts["shoulder"], t)
            s.shoulder_t = t
        elif t - s.shoulder_t > SHOULDER_HOLD_SEC:
            s.shoulder = None
            s.f_sh.reset()

        if all(v is not None for v in pts.values()):
            L = (np.linalg.norm(pts["elbow"] - pts["shoulder"]) + np.linalg.norm(pts["wrist"] - pts["elbow"])
                 + WRIST_TO_PALM)
            if ARM_LEN_RANGE[0] < L < ARM_LEN_RANGE[1]:
                s.arm_len_depth.append(L)
            st.shoulder = s.shoulder
            st.elbow = s.f_el(pts["elbow"], t)
            st.wrist = s.f_wr(pts["wrist"], t)
            st.arm_valid = True

        return wrist_px

    def _assign_hands(self, hand_res, wrist_px, W, H):
        """{side: 손 index} — 1) Pose 손목과 가까운 손 (양손이면 거리 합이 최소인 짝), 2) handedness 라벨"""

        if not hand_res.multi_hand_landmarks:
            return {}

        n = len(hand_res.multi_hand_landmarks)
        lim = HAND_MATCH_MAX * W
        dist = {}
        for side, px in wrist_px.items():
            if px is None:
                continue
            for i, hl in enumerate(hand_res.multi_hand_landmarks):
                d = np.hypot(hl.landmark[0].x * W - px[0], hl.landmark[0].y * H - px[1])
                if d < lim:
                    dist[side, i] = d

        # 가능한 짝 중 (짝 수 최대, 거리 합 최소)
        sides = [sd for sd in self.s if wrist_px.get(sd) is not None]
        best, best_key = {}, (0, 0.0)
        for i0 in [None] + list(range(n)):
            for i1 in [None] + list(range(n)):
                pick = dict(zip(sides, [i0, i1][:len(sides)]))
                used = [i for i in pick.values() if i is not None]
                if len(used) != len(set(used)) or any(i is not None and (sd, i) not in dist for sd, i in pick.items()):
                    continue
                key = (len(used), -sum(dist[sd, i] for sd, i in pick.items() if i is not None))
                if key > best_key:
                    best, best_key = {sd: i for sd, i in pick.items() if i is not None}, key

        # Pose 로 못 정한 쪽은 handedness 라벨 (MediaPipe 는 거울 영상 기준이라 반대로 나옴)
        for side in self.s:
            if side in best:
                continue
            want = "Left" if side == "right" else "Right"
            for i, hd in enumerate(hand_res.multi_handedness):
                if hd.classification[0].label == want and i not in best.values():
                    best[side] = i
                    break

        return best

    def _hand(self, img_lm, world_lm, depth_m, s, st, W, H, t):

        st.hand_landmarks = img_lm

        # ---------------- 손바닥 위치 (depth) ----------------
        palm_px = np.mean([[img_lm.landmark[i].x * W, img_lm.landmark[i].y * H] for i in PALM_IDX], axis=0)
        st.palm_px = (int(palm_px[0]), int(palm_px[1]))

        n_depth = None
        p_cam = None
        if self.palm_method in ("region", "plane"):
            poly = [[img_lm.landmark[i].x * W, img_lm.landmark[i].y * H] for i in PALM_POLY_IDX]
            if self.palm_method == "plane":
                p_cam, n_depth, n_pts = palm_from_region(depth_m, poly, self.intr, self.depth_min, self.depth_max,
                                                         center_px=palm_px, use_plane=True)
            else:
                p_cam, n_depth, n_pts = palm_from_region(depth_m, poly, self.intr, self.depth_min, self.depth_max)
            st.extra["n_pts"] = n_pts
        if p_cam is None:
            p_cam = self._point3d(depth_m, *palm_px, r=6)
            st.extra["patch_fallback"] = self.palm_method != "patch"

        if p_cam is None:
            st.msg = "no depth at palm"
            return

        # 한 프레임에 너무 크게 튀면 (depth 가 배경/몸통에 걸린 경우) 버림
        # 마지막 정상값 기준으로 비교, 연속으로 계속 튀면 실제 이동으로 보고 받아들임
        if s.last_palm_cam is not None and np.linalg.norm(p_cam - s.last_palm_cam) > self.max_jump_m:
            s.jump_count += 1
            if s.jump_count < 5:
                st.msg = "depth jump rejected"
                return

        s.jump_count = 0
        s.last_palm_cam = p_cam

        # ---------------- 손 방향 (world landmark, 카메라 축 기준) ----------------
        P = np.array([[l.x, l.y, l.z] for l in world_lm.landmark])

        x_h = unit(P[9] - P[0])                               # 손목 → 중지 MCP (손가락 방향)
        n = np.cross(P[5] - P[0], P[17] - P[0])               # 오른손 기준 손바닥 바깥쪽
        if s.side == "left":
            n = -n
        n = unit(n)

        # 손바닥이 카메라 쪽을 향해 평평하게 잘 보일 때만 depth 평면 법선을 섞음 (부호는 MediaPipe 쪽에 맞춤)
        # 비스듬히 보이면 (예: 손바닥이 아래를 향함) depth 점이 띠 모양이라 법선이 틀어짐 → 섞지 않음
        if n_depth is not None and abs(np.dot(n_depth, unit(p_cam))) < NORMAL_MIN_FACING:
            n_depth = None
        if n_depth is not None:
            if np.dot(n_depth, n) < 0:
                n_depth = -n_depth
            n = unit(NORMAL_BLEND * n_depth + (1.0 - NORMAL_BLEND) * n)
            st.extra["depth_normal"] = True

        z_h = unit(n - np.dot(n, x_h) * x_h)
        y_h = np.cross(z_h, x_h)
        R_cam = np.column_stack([x_h, y_h, z_h])

        R_cam = self._reject_rot_jump(s, R_cam, st)

        # ---------------- 아래팔 기준 손목 꺾기 / 좌우 ----------------
        # 아래팔 축 x_f, 손바닥 법선을 아래팔에 수직으로 투영한 z_f → 손가락 방향이 x_f 에서 얼마나 기울었나
        #   꺾기 = 손 옆축(y) 회전, 좌우 = 손바닥 법선축(z) 회전 (R_cam 열 순서와 같은 오른손 좌표)
        if s.forearm_cam is not None:
            xh, zh = R_cam[:, 0], R_cam[:, 2]
            xf = s.forearm_cam
            zf = unit(zh - np.dot(zh, xf) * xf)
            yf = np.cross(zf, xf)
            ang = np.array([-np.arctan2(np.dot(xh, zf), np.dot(xh, xf)), np.arctan2(np.dot(xh, yf), np.dot(xh, xf))])
            st.wrist_ang = s.f_ang(ang, t)
        else:
            s.f_ang.reset()

        R_body = orthonormalize(s.f_R((self.C @ R_cam).ravel(), t).reshape(3, 3))

        palm_size = np.linalg.norm(P[9] - P[0])
        grip = np.linalg.norm(P[4] - P[8]) / max(palm_size, 1e-6)

        st.palm_cam = p_cam
        st.R_hand_cam = R_cam
        st.palm = s.f_palm(self.to_body(p_cam), t)
        if s.shoulder is not None:
            st.hand_vec = st.palm - s.shoulder
        st.R_hand = R_body
        st.grip_ratio = float(s.f_grip(np.array([grip]), t)[0])
        st.hand_valid = True

    # --------------------------------------------------------
    # DRAW
    # --------------------------------------------------------

    def draw(self, img, states, labels=None):
        """labels: {side: 화면에 손 옆에 쓸 이름 (예: 로봇 이름)}"""

        st0 = next(iter(states.values()))
        if st0.pose_landmarks is not None:
            mp_draw.draw_landmarks(img, st0.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        for side, st in states.items():

            if st.hand_landmarks is not None:
                mp_draw.draw_landmarks(img, st.hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if st.hand_valid:
                # 손 좌표축 (빨강: 손가락 방향, 초록: 옆, 파랑: 손바닥 법선)
                o = st.palm_cam
                for k, col in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
                    tip = o + 0.08 * st.R_hand_cam[:, k]
                    a = rs.rs2_project_point_to_pixel(self.intr, o.tolist())
                    b = rs.rs2_project_point_to_pixel(self.intr, tip.tolist())
                    cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), col, 3)

            if st.palm_px is not None:
                text = (labels or {}).get(side, side)
                x = min(st.palm_px[0] + 12, img.shape[1] - 12 * len(text))
                cv2.putText(img, text, (x, max(st.palm_px[1] - 12, 40)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
