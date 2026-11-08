"""
Motomind Milly 기구학 (FK / IK)

- Milly SDK 의 URDF (RAW 좌표: URDF 관절값 = 모터 raw rad) 에서 관절 체인을 읽음
  설치된 motomind_milly 패키지의 URDF 우선, 없으면 이 폴더의 MILLY.urdf (MIT, Motomind)
- 좌표계: teleop 공통 base frame (PiPER 와 같은 정의)
    X: 팔이 뻗는 앞쪽 (URDF base_link 의 -Y)
    Y: 왼쪽          (URDF base_link 의 +X)
    Z: 위, 원점 = J1 회전축이 base 바닥면과 만나는 점
- IK: piper_kin 과 같은 DLS (현재 해에서 출발 → 연속적인 해)
"""

import importlib.util
import os
import xml.etree.ElementTree as ET

import numpy as np

from piper_kin import dls_ik, matrix_from_rotvec, matrix_to_rpy_deg, rotvec_from_matrix  # noqa: F401


# ============================================================
# URDF
# ============================================================

def urdf_path():

    spec = importlib.util.find_spec("motomind_milly")
    if spec is not None and spec.submodule_search_locations:
        p = os.path.join(list(spec.submodule_search_locations)[0], "robot", "MILLY.urdf")
        if os.path.isfile(p):
            return p

    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "MILLY.urdf")


def _rpy(r, p, y):

    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)

    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def _load_chain(path):
    """joint_1 ~ joint_6: (부모 기준 원점 4x4, 회전축 단위벡터, 하한, 상한)"""

    root = ET.parse(path).getroot()
    chain = []

    for i in range(1, 7):
        j = root.find(f"joint[@name='joint_{i}']")
        o = j.find("origin")
        T = np.eye(4)
        T[:3, :3] = _rpy(*map(float, o.get("rpy").split()))
        T[:3, 3] = list(map(float, o.get("xyz").split()))
        axis = np.array(list(map(float, j.find("axis").get("xyz").split())))
        lim = j.find("limit")
        chain.append((T, axis / np.linalg.norm(axis), float(lim.get("lower")), float(lim.get("upper"))))

    return chain


URDF = urdf_path()
_CHAIN = _load_chain(URDF)


# ============================================================
# BASE FRAME / LIMITS
# ============================================================

# URDF base_link → teleop base frame (X 앞 = URDF -Y, Y 왼쪽 = URDF +X)
_J1_ORIGIN = _CHAIN[0][0][:3, 3]
T_BASE = np.eye(4)
T_BASE[:3, :3] = np.array([
    [0.0, -1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
])
T_BASE[:3, 3] = -T_BASE[:3, :3] @ np.array([_J1_ORIGIN[0], _J1_ORIGIN[1], 0.0])

# 소프트웨어 관절 한계 = URDF(=SDK MIT 명령 범위) 에서 양쪽 3° 여유 (SDK 실측 안전 범위를 넘으면 fault)
_MARGIN = np.deg2rad(3.0)
JOINT_MIN = np.array([c[2] for c in _CHAIN]) + _MARGIN
JOINT_MAX = np.array([c[3] for c in _CHAIN]) - _MARGIN
JOINT_MIN_DEG = np.rad2deg(JOINT_MIN)
JOINT_MAX_DEG = np.rad2deg(JOINT_MAX)

# 손목 중심 = fk_frames(q)[WRIST_FRAME] 원점 (J5), 손목 중심 → flange 거리
WRIST_FRAME = 4


# ============================================================
# FK / JACOBIAN
# ============================================================

def _axis_rot(k, a):

    K = np.array([
        [0.0, -k[2], k[1]],
        [k[2], 0.0, -k[0]],
        [-k[1], k[0], 0.0],
    ])

    return np.eye(3) + np.sin(a) * K + (1.0 - np.cos(a)) * (K @ K)


def tool_transform(tcp_offset):

    T = np.eye(4)
    T[2, 3] = tcp_offset

    return T


def _joints(q):
    """각 관절의 (회전 전 frame 4x4, 회전축 base 기준) 리스트 + flange 4x4"""

    T = T_BASE.copy()
    out = []

    for (T_o, axis, _, _), qi in zip(_CHAIN, q):
        T = T @ T_o
        out.append((T.copy(), T[:3, :3] @ axis))
        R = np.eye(4)
        R[:3, :3] = _axis_rot(axis, qi)
        T = T @ R

    return out, T


def fk_frames(q):
    """관절 frame 의 base 기준 4x4 리스트 (frames[i] 원점 = joint_(i+1) 축 위, piper_kin 과 같은 의미)"""

    return [T for T, _ in _joints(q)[0]]


def fk(q, tcp_offset=0.0):
    """flange(또는 TCP) 의 base 기준 4x4. flange z = 그리퍼 접근 방향"""

    return _joints(q)[1] @ tool_transform(tcp_offset)


def jacobian(q, tcp_offset=0.0):

    out, T_end = _joints(q)
    T_end = T_end @ tool_transform(tcp_offset)
    p_end = T_end[:3, 3]

    J = np.zeros((6, 6))

    for i, (T, z) in enumerate(out):
        J[:3, i] = np.cross(z, p_end - T[:3, 3])
        J[3:, i] = z

    return J, T_end


WRIST_TO_FLANGE = float(np.linalg.norm(fk(np.zeros(6))[:3, 3] - fk_frames(np.zeros(6))[WRIST_FRAME][:3, 3]))


# ============================================================
# IK
# ============================================================

def ik(T_target, q_seed, tcp_offset=0.0, **kw):
    """piper_kin.ik 와 같은 인자/반환 (q, pos_err_m, rot_err_rad)"""

    return dls_ik(
        T_target, q_seed,
        lambda q: jacobian(q, tcp_offset),
        lambda q: fk(q, tcp_offset),
        JOINT_MIN, JOINT_MAX, **kw,
    )
