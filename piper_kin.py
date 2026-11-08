"""
PiPER 기구학 (FK / IK)

- DH 파라미터는 piper_sdk 의 C_PiperForwardKinematics(dh_is_offset=0x01) 와 동일
  (Modified DH, Craig 방식) → 펌웨어가 보고하는 EndPose 와 같은 기준
- 단위: 내부 계산은 m / rad
- IK: Damped Least Squares (현재 해에서 출발 → 연속적인 해)
"""

import numpy as np


# ============================================================
# DH PARAMETERS (piper_sdk, dh_is_offset = 0x01)
# ============================================================

DH_A = np.array([0.0, 0.0, 0.28503, -0.02198, 0.0, 0.0])
DH_ALPHA = np.array([0.0, -np.pi / 2, 0.0, np.pi / 2, -np.pi / 2, np.pi / 2])
DH_THETA = np.deg2rad([0.0, -172.22, -102.78, 0.0, 0.0, 0.0])
DH_D = np.array([0.123, 0.0, 0.0, 0.25075, 0.0, 0.091])


# ============================================================
# SOFTWARE JOINT LIMITS (deg)
# ============================================================

JOINT_MIN_DEG = np.array([-150.0, 0.0, -170.0, -100.0, -70.0, -120.0])
JOINT_MAX_DEG = np.array([150.0, 180.0, 0.0, 100.0, 70.0, 120.0])

JOINT_MIN = np.deg2rad(JOINT_MIN_DEG)
JOINT_MAX = np.deg2rad(JOINT_MAX_DEG)

# 손목 중심 = fk_frames(q)[WRIST_FRAME] 원점 (J5), 손목 중심 → flange 거리 (DH d6)
WRIST_FRAME = 4
WRIST_TO_FLANGE = 0.091


# ============================================================
# BASIC TRANSFORMS
# ============================================================

def _link_transform(alpha, a, theta, d):

    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)

    return np.array([
        [ct, -st, 0.0, a],
        [st * ca, ct * ca, -sa, -sa * d],
        [st * sa, ct * sa, ca, ca * d],
        [0.0, 0.0, 0.0, 1.0],
    ])


def tool_transform(tcp_offset):

    T = np.eye(4)
    T[2, 3] = tcp_offset

    return T


def fk_frames(q):
    """
    q: (6,) rad
    return: 각 joint frame 의 base 기준 4x4 리스트 (frame1 ~ frame6)
    """

    frames = []
    T = np.eye(4)

    for i in range(6):
        T = T @ _link_transform(DH_ALPHA[i], DH_A[i], q[i] + DH_THETA[i], DH_D[i])
        frames.append(T.copy())

    return frames


def fk(q, tcp_offset=0.0):
    """flange(또는 TCP) 의 base 기준 4x4"""

    return fk_frames(q)[-1] @ tool_transform(tcp_offset)


# ============================================================
# ROTATION UTILS
# ============================================================

def rotvec_from_matrix(R):

    cos_angle = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos_angle)

    if angle < 1e-6:
        return np.zeros(3)

    if np.pi - angle < 1e-4:
        # 180도 근처: 대각 성분으로 축 계산
        axis = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
        axis[1] = np.copysign(axis[1], R[0, 1] + R[1, 0])
        axis[2] = np.copysign(axis[2], R[0, 2] + R[2, 0])
        return axis / np.linalg.norm(axis) * angle

    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ]) / (2.0 * np.sin(angle))

    return axis * angle


def matrix_from_rotvec(v):

    angle = np.linalg.norm(v)

    if angle < 1e-9:
        return np.eye(3)

    k = v / angle
    K = np.array([
        [0.0, -k[2], k[1]],
        [k[2], 0.0, -k[0]],
        [-k[1], k[0], 0.0],
    ])

    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def matrix_to_rpy_deg(R):
    """PiPER EndPose 와 같은 rpy (R = Rz(yaw) Ry(pitch) Rx(roll))"""

    pitch = np.arctan2(-R[2, 0], np.hypot(R[0, 0], R[1, 0]))
    yaw = np.arctan2(R[1, 0], R[0, 0])
    roll = np.arctan2(R[2, 1], R[2, 2])

    return np.rad2deg([roll, pitch, yaw])


# ============================================================
# JACOBIAN (geometric, base frame)
# ============================================================

def jacobian(q, tcp_offset=0.0):

    frames = fk_frames(q)
    T_end = frames[-1] @ tool_transform(tcp_offset)
    p_end = T_end[:3, 3]

    J = np.zeros((6, 6))

    for i, T in enumerate(frames):
        z = T[:3, 2]
        o = T[:3, 3]
        J[:3, i] = np.cross(z, p_end - o)
        J[3:, i] = z

    return J, T_end


# ============================================================
# IK (Damped Least Squares)
# ============================================================

def ik(T_target, q_seed, tcp_offset=0.0, **kw):
    """
    T_target: 원하는 TCP 4x4 (base 기준)
    q_seed:   시작 joint (rad), 보통 직전 명령값 → 해가 튀지 않음
    rest, rest_weight: (6,) 관절별로 rest 자세 쪽으로 약하게 당김 (예: J4 → 0)
                       손목 특이점(J5≈0, J4·J6 축 일치)에서 J4/J6 가 서로 반대로 크게 도는 것 방지

    return: (q_solution, pos_err_m, rot_err_rad)
    """

    return dls_ik(
        T_target, q_seed,
        lambda q: jacobian(q, tcp_offset),
        lambda q: fk(q, tcp_offset),
        JOINT_MIN, JOINT_MAX, **kw,
    )


def dls_ik(
    T_target,
    q_seed,
    jacobian_fn,
    fk_fn,
    q_min,
    q_max,
    use_orientation=True,
    rot_weight=0.25,
    damping=0.02,
    iterations=20,
    max_step=np.deg2rad(10.0),
    rest=None,
    rest_weight=None,
):
    """로봇 공통 DLS IK. jacobian_fn(q) → (J 6x6, T), fk_fn(q) → T"""

    JOINT_MIN, JOINT_MAX = q_min, q_max
    q = np.clip(np.array(q_seed, dtype=float), JOINT_MIN, JOINT_MAX)

    W = np.diag([1.0, 1.0, 1.0] + [rot_weight if use_orientation else 0.0] * 3)
    pw2 = np.zeros(6) if rest_weight is None else np.asarray(rest_weight, dtype=float) ** 2
    rest = np.zeros(6) if rest is None else np.asarray(rest, dtype=float)

    def solve(Jw, e, free):
        # min |Jw dq - e|² + damping²|dq|² + Σ w²(q + dq - rest)²   (free 관절만)
        Jf = Jw[:, free]
        H = Jf.T @ Jf + np.diag(damping ** 2 + pw2[free])
        g = Jf.T @ e + pw2[free] * (rest - q)[free]
        dq = np.zeros(6)
        dq[free] = np.linalg.solve(H, g)
        return dq

    for _ in range(iterations):

        J, T = jacobian_fn(q)

        e_pos = T_target[:3, 3] - T[:3, 3]
        e_rot = rotvec_from_matrix(T_target[:3, :3] @ T[:3, :3].T)

        if np.linalg.norm(e_pos) < 1e-4 and (not use_orientation or np.linalg.norm(e_rot) < 1e-3):
            break

        e = W @ np.concatenate([e_pos, e_rot])
        Jw = W @ J

        dq = solve(Jw, e, np.ones(6, dtype=bool))

        # 한계에 붙은 관절이 더 밖으로 가려고 하면 그 관절을 빼고 다시 계산
        # (안 빼면 clip 에 막혀 그 자리에서 멈춤 → 다른 관절이 대신 움직이지 못함)
        blocked = ((q <= JOINT_MIN + 1e-6) & (dq < 0)) | ((q >= JOINT_MAX - 1e-6) & (dq > 0))
        if np.any(blocked):
            dq = solve(Jw, e, ~blocked)

        step = np.max(np.abs(dq))
        if step > max_step:
            dq *= max_step / step

        q = np.clip(q + dq, JOINT_MIN, JOINT_MAX)

    T = fk_fn(q)
    pos_err = np.linalg.norm(T_target[:3, 3] - T[:3, 3])
    rot_err = np.linalg.norm(rotvec_from_matrix(T_target[:3, :3] @ T[:3, :3].T))

    return q, pos_err, rot_err
