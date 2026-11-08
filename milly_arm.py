"""
Motomind Milly 로봇팔 인터페이스 (실시간 교시용)

- Milly SDK (motomind_milly) 의 move_mit 로 관절 위치 목표를 스트리밍 (SDK milly_teleop 예제와 같은 방식)
  SDK supervisor 가 100 Hz 로 마지막 목표를 계속 전송, 이 스레드는 속도 제한된 목표를 50 Hz 로 갱신
- 종료(stop) = SDK shutdown: 10초 동안 damping 을 0 까지 낮춤 → 팔이 중력으로 천천히 내려감 (받치거나 z 로 접은 뒤 종료)
- 단위: rad. 그리퍼 모터 7: 닫힘 0.0, 열림 2.11 rad
"""

import inspect
import time

import numpy as np

import milly_kin as K
from arm_base import StreamingArm


# SDK 출하 기본 게인 (profiles/MILLY_DEFAULT.yaml, milly_teleop 예제와 같음): joint_1 ~ joint_6
MIT_KP = (30.0, 30.0, 30.0, 15.0, 15.0, 5.0)
MIT_KD = (1.5, 1.5, 1.5, 1.0, 1.0, 0.5)
GRIPPER_ID = 7
GRIPPER_OPEN_RAD = 2.11
GRIPPER_KP = 5.0                  # 낮을수록 부드럽게 잡음 (0 ~ 50)
GRIPPER_KD = 0.5

# SDK MIT 명령 범위 (URDF 한계와 같음) — 이 밖으로는 보내지 않음
CMD_MIN = K.JOINT_MIN - np.deg2rad(3.0)
CMD_MAX = K.JOINT_MAX + np.deg2rad(3.0)


class MillyArm(StreamingArm):

    name = "Milly"
    gripper_col = "gripper_rad"

    def __init__(
        self,
        product_id="",
        can_name=None,
        dry_run=False,
        max_joint_vel_deg=60.0,
        command_dt=0.02,
        kp=MIT_KP,
        kd=MIT_KD,
        gripper_kp=GRIPPER_KP,
        gripper_kd=GRIPPER_KD,
    ):

        super().__init__(K.JOINT_MIN, K.JOINT_MAX, dry_run, max_joint_vel_deg, command_dt)

        self.product_id = product_id
        self.can_name = can_name
        self.kp = tuple(kp)
        self.kd = tuple(kd)
        self.gripper_kp = gripper_kp
        self.gripper_kd = gripper_kd

        self.arm = None
        self.enabled = False
        self.rejected = ""              # move_mit 거부 (스레드에서 기록 → motor_faults 로 보고)

    # ========================================================
    # CONNECT / ENABLE
    # ========================================================

    def connect(self):

        if self.dry_run:
            print("[DRY-RUN] Milly 에 연결하지 않습니다.")
            return

        if not self.product_id:
            raise RuntimeError(
                "Milly 제품 ID 가 필요합니다 (--milly-id MILLY_XXXX).\n"
                "    확인: ~/teleop_env/bin/motomind-milly-robots   (팔에 각인된 ID)")

        try:
            import motomind_milly as mm
        except ImportError as exc:
            raise RuntimeError(
                "Milly SDK (motomind_milly) 가 ~/teleop_env 에 설치되어 있지 않습니다 — README 의 Milly 설치 참고") from exc

        kwargs = {}
        if self.can_name:
            try:
                if "channel" in inspect.signature(mm.create_arm).parameters:
                    kwargs["channel"] = self.can_name
            except (TypeError, ValueError):
                pass

        print(f"Milly: {self.product_id} 연결{' (' + self.can_name + ')' if kwargs else ''}")
        self.arm = mm.create_arm(self.product_id, **kwargs)

        q = self.arm.get_joint_angles()
        if q is None or len(q) < 6 or any(v is None for v in list(q)[:6]):
            self.arm.disconnect()
            self.arm = None
            raise RuntimeError(f"Milly 관절 피드백 없음 ({q}) — 팔 전원 / CAN 선 확인")
        print("Milly 현재 관절 (deg):", np.round(np.rad2deg(np.asarray(q[:6], dtype=float)), 1))

        result = self.arm.enable()          # 현재 자세 HOLD
        if not result.ok:
            reason = self._fault_reason()
            self.arm.shutdown()
            self.arm = None
            raise RuntimeError(f"Milly enable 실패: {result.message} {reason}")

        self.enabled = True
        print("Milly ready (enabled, 현재 자세 HOLD)")

    def _fault_reason(self):

        try:
            return str(self.arm.manager.fault_reason() or "")
        except Exception:
            return ""

    # ========================================================
    # FEEDBACK
    # ========================================================

    def _read_joints_hw(self):

        if self.arm is None:
            return self.q_cmd

        q = self.arm.get_joint_angles()
        q_cmd = self.q_cmd
        return np.array([q_cmd[i] if v is None else float(v) for i, v in enumerate(list(q)[:6])])

    def motor_faults(self):

        if self.dry_run or self.arm is None:
            return {}

        faults = {}
        if self.rejected:
            faults["Milly"] = [self.rejected]
        if not self.arm.is_ok():
            faults.setdefault("Milly", []).append(f"{self.arm.state()} {self._fault_reason()}".strip())

        return faults

    def read_gripper_rad(self):

        if self.dry_run or self.arm is None:
            with self._lock:
                return (self._gripper_ratio or 0.0) * GRIPPER_OPEN_RAD

        for s in self.arm.states():
            if s.id == GRIPPER_ID and s.position is not None:
                return float(s.position)

        return float("nan")

    def gripper_text(self):

        g = self.read_gripper_rad()
        return f"gripper {g:.2f} rad ({100.0 * g / GRIPPER_OPEN_RAD:.0f}% open)"

    def gripper_value(self):

        return f"{self.read_gripper_rad():.3f}"

    # ========================================================
    # COMMAND
    # ========================================================

    def _send(self, q, gripper_ratio):

        if self.arm is None:
            return

        q = np.clip(q, CMD_MIN, CMD_MAX)

        for i in range(6):
            if not self.arm.move_mit(i + 1, position=float(q[i]), velocity=0.0, kp=self.kp[i], kd=self.kd[i]):
                self.rejected = f"move_mit J{i + 1} rejected"

        if gripper_ratio is not None:
            if not self.arm.move_mit(GRIPPER_ID, position=float(gripper_ratio * GRIPPER_OPEN_RAD), velocity=0.0,
                                     kp=self.gripper_kp, kd=self.gripper_kd):
                self.rejected = "move_mit gripper rejected"

    def emergency_stop(self):
        """Milly 전자식 정지 (damping, fault). 재시작 시 프로그램 재실행 필요"""

        self._running = False
        if not self.dry_run and self.arm is not None:
            self.arm.electronic_emergency_stop()
        print("!!! Milly ELECTRONIC EMERGENCY STOP")

    def stop(self):
        """명령 전송 중단 → SDK shutdown (10초 동안 damping 0 까지 → 팔을 받치세요)"""

        super().stop()

        if self.dry_run or self.arm is None:
            return

        print("Milly shutdown: 10초 동안 damping 을 낮춥니다 — 팔을 받치거나 거치하세요")
        try:
            if self.enabled:
                self.arm.hold_mode()
                time.sleep(0.1)
                result = self.arm.shutdown()
                if not result.ok:
                    print("Milly shutdown 실패:", result.message)
            else:
                self.arm.disconnect()
        finally:
            self.arm = None
            self.enabled = False
        print("Milly shutdown 완료")
