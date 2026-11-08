"""
로봇팔 공통 스트리밍 제어 (PiPER / Milly)

- 모든 명령은 별도 제어 스레드 한 곳에서만 전송 (서브클래스의 _send)
- 제어 스레드는 목표 joint 를 향해 "관절 속도 제한" 을 걸어 조금씩 이동
  → 사람 동작 추정이 튀어도 로봇은 max_vel 이상으로 움직이지 않음
- dry_run=True 이면 아무것도 보내지 않음 (명령값 = 가상 로봇 상태)
"""

import threading
import time

import numpy as np


class StreamingArm:

    name = "arm"
    gripper_col = "gripper"

    def __init__(self, joint_min, joint_max, dry_run=False, max_joint_vel_deg=60.0, command_dt=0.01):

        self.joint_min = np.asarray(joint_min, dtype=float)
        self.joint_max = np.asarray(joint_max, dtype=float)
        self.dry_run = dry_run
        self.default_max_vel = np.deg2rad(max_joint_vel_deg)
        self.command_dt = command_dt

        self._lock = threading.Lock()
        self._running = False
        self._thread = None

        self._q_cmd = np.zeros(6)
        self._q_target = np.zeros(6)
        self._max_vel = self.default_max_vel
        self._gripper_ratio = None       # 0(닫힘) ~ 1(열림), None = 그리퍼 명령 안 보냄

    # ========================================================
    # 서브클래스 구현
    # ========================================================

    def connect(self):

        raise NotImplementedError

    def _read_joints_hw(self):

        raise NotImplementedError

    def _send(self, q, gripper_ratio):

        raise NotImplementedError

    def motor_faults(self):
        """{"J1": [오류 이름, ...]} — 오류 없으면 빈 dict"""

        return {}

    def gripper_text(self):

        return ""

    def gripper_value(self):
        """CSV 기록용 그리퍼 값"""

        return ""

    def emergency_stop(self):

        self._running = False

    def stop(self):
        """명령 전송만 중단"""

        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    # ========================================================
    # FEEDBACK
    # ========================================================

    def read_joints(self):
        """실제 로봇 joint (rad). dry-run 이면 명령값"""

        if self.dry_run:
            with self._lock:
                return self._q_cmd.copy()

        return self._read_joints_hw()

    # ========================================================
    # COMMAND
    # ========================================================

    def set_target(self, q_rad, max_vel_deg=None):

        q = np.clip(np.asarray(q_rad, dtype=float), self.joint_min, self.joint_max)

        with self._lock:
            self._q_target = q
            self._max_vel = self.default_max_vel if max_vel_deg is None else np.deg2rad(max_vel_deg)

    def set_gripper(self, opening_ratio):
        """opening_ratio: 0(닫힘) ~ 1(열림), None 이면 그리퍼 명령 중단"""

        with self._lock:
            self._gripper_ratio = None if opening_ratio is None else float(np.clip(opening_ratio, 0.0, 1.0))

    def hold(self):
        """현재 명령 위치에서 정지"""

        with self._lock:
            self._q_target = self._q_cmd.copy()

    @property
    def q_cmd(self):

        with self._lock:
            return self._q_cmd.copy()

    @property
    def q_target(self):

        with self._lock:
            return self._q_target.copy()

    def at_target(self, tol_deg=0.5):

        with self._lock:
            return np.max(np.abs(self._q_target - self._q_cmd)) < np.deg2rad(tol_deg)

    # ========================================================
    # CONTROL THREAD
    # ========================================================

    def start(self, initial_q=None):

        q0 = self.read_joints() if initial_q is None else np.asarray(initial_q, dtype=float)

        with self._lock:
            self._q_cmd = q0.copy()
            self._q_target = q0.copy()

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):

        next_t = time.perf_counter()
        last_t = next_t
        self.max_gap = 0.0              # 명령 간격 최대값 (s) — Milly SDK 는 100 ms 넘으면 fault

        while self._running:

            now = time.perf_counter()
            gap = now - last_t
            last_t = now
            if gap > 0.06 and not self.dry_run:
                print(f"[!] {self.name} 명령 간격 {gap * 1000:.0f} ms (목표 {self.command_dt * 1000:.0f} ms) — 메인 루프가 CPU 를 오래 점유?")
            self.max_gap = max(self.max_gap, gap)

            with self._lock:
                # 모든 관절을 같은 비율로 줄임 (관절별로 자르면 관절마다 도착 시간이 달라 경로가 휘어짐)
                step = self._max_vel * self.command_dt
                dq = self._q_target - self._q_cmd
                m = np.max(np.abs(dq))
                if m > step:
                    dq *= step / m
                self._q_cmd = self._q_cmd + dq
                q = self._q_cmd.copy()
                grip = self._gripper_ratio

            if not self.dry_run:
                self._send(q, grip)

            next_t += self.command_dt
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.perf_counter()
