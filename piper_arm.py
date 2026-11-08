"""
PiPER 로봇팔 인터페이스 (실시간 교시용)

- 100 Hz 제어 스레드, 관절 속도 제한 (arm_base.StreamingArm)
- follow=True (기본): ModeCtrl 의 MIT 모드 (0xAD) — 펌웨어가 경로를 다시 계획하지 않고 명령 위치를 바로 따라감
  (Milly 의 move_mit 스트리밍과 같은 방식: 이 스레드의 속도 제한 경로가 그대로 로봇 경로)
  follow=False: MOVE J (명령마다 펌웨어가 속도 비율로 경로를 다시 계획 → 지연·뭉개짐, 이전 방식)
- 종료(stop) 시 모터 enable 유지 → 마지막 위치에서 정지 (팔이 떨어지지 않음)
"""

import glob
import os
import time

import numpy as np

import piper_kin as K
from arm_base import StreamingArm


def find_can_by_usb_product(keyword):
    """USB 제품 이름에 keyword 가 들어간 CAN 인터페이스 이름 (예: candleLight → can1), 없으면 None"""

    for net in sorted(glob.glob("/sys/class/net/can*")):
        dev = os.path.realpath(os.path.join(net, "device"))
        for d in (dev, os.path.dirname(dev)):
            try:
                with open(os.path.join(d, "product")) as f:
                    if keyword.lower() in f.read().lower():
                        return os.path.basename(net)
            except OSError:
                pass

    return None


class PiperArm(StreamingArm):

    name = "PiPER"
    gripper_col = "gripper_um"

    def __init__(
        self,
        can_name="can0",
        dry_run=False,
        speed_percent=50,
        follow=True,
        max_joint_vel_deg=60.0,
        command_dt=0.01,
        gripper_effort=1000,
        gripper_max_um=70000,
    ):

        super().__init__(K.JOINT_MIN, K.JOINT_MAX, dry_run, max_joint_vel_deg, command_dt)

        self.can_name = can_name
        self.follow = follow
        self.speed_percent = 100 if follow else int(speed_percent)
        self.mit_mode = 0xAD if follow else 0x00
        self.gripper_effort = int(gripper_effort)
        self.gripper_max_um = int(gripper_max_um)

        self.piper = None

    # ========================================================
    # CONNECT / ENABLE (test_piper_connect.py 와 같은 절차)
    # ========================================================

    def connect(self):

        if self.dry_run:
            print("[DRY-RUN] PiPER 에 연결하지 않습니다.")
            return

        from piper_sdk import C_PiperInterface_V2

        print(f"PiPER: {self.can_name} 연결")
        self.piper = C_PiperInterface_V2(self.can_name)
        self.piper.ConnectPort()
        time.sleep(1.0)

        status = self.piper.GetArmStatus()

        if status.arm_status.ctrl_mode == 0x02:
            print("Teaching mode detected → 해제")
            self.piper.EmergencyStop(0x01)
            time.sleep(1.0)
            self.piper.DisableArm(7)
            time.sleep(1.0)
            self.piper.EmergencyStop(0x02)
            time.sleep(1.0)

        if self.piper.GetArmStatus().arm_status.arm_status == 0x01:
            print("Emergency stop detected → 해제")
            self.piper.EmergencyStop(0x02)
            time.sleep(1.0)

        for _ in range(20):
            self.piper.ModeCtrl(0x01, 0x01, self.speed_percent, self.mit_mode)
            time.sleep(0.1)
            if self.piper.GetArmStatus().arm_status.ctrl_mode == 0x01:
                break
        else:
            raise RuntimeError(f"PiPER CAN control mode 진입 실패 ({self.can_name}: 팔 전원 / CAN 선 확인)")

        # 모터 드라이버 오류 (충돌/과전류/스톨 등) → 확인 후 해제
        faults = self.motor_faults()
        if faults:
            print("\n!!! PiPER 모터 드라이버 오류:")
            for j, flags in faults.items():
                print(f"    {j}: {', '.join(flags)}")
            print("    그리퍼/손목이 로봇 몸체나 작업대에 걸려 있지 않은지 먼저 확인하세요.")
            print("    해제하면 해당 모터가 현재 위치에서 다시 힘을 받습니다 (움직이지 않음).")
            if input("    오류를 해제하고 enable 할까요? [y/N] ").strip().lower() != "y":
                raise RuntimeError("사용자가 오류 해제를 취소함")
            for j in faults:
                self.piper.JointConfig(int(j[1:]), 0x00, 0x00, 500, 0xAE)
                time.sleep(0.1)
            time.sleep(0.5)

        t0 = time.time()
        while not self.piper.EnablePiper():
            if time.time() - t0 > 5.0:
                faults = self.motor_faults()
                detail = "; ".join(f"{j}: {', '.join(f)}" for j, f in faults.items()) or "오류 플래그 없음"
                raise RuntimeError(f"PiPER 모터 enable 실패 (5초 초과) — {detail}")
            time.sleep(0.05)

        # 그리퍼 enable
        self.piper.GripperCtrl(0, self.gripper_effort, 0x01, 0)
        time.sleep(0.2)

        status = self.piper.GetArmStatus()
        if status.arm_status.arm_status != 0x00:
            raise RuntimeError(f"PiPER arm status 가 NORMAL 이 아님: {status.arm_status}")

        print(f"PiPER ready (CAN control, enabled, {'MIT follow' if self.follow else 'MOVE J ' + str(self.speed_percent) + '%'})")

    # ========================================================
    # FEEDBACK
    # ========================================================

    def _read_joints_hw(self):

        if self.piper is None:
            return self.q_cmd

        j = self.piper.GetArmJointMsgs().joint_state

        return np.deg2rad([
            j.joint_1 / 1000.0,
            j.joint_2 / 1000.0,
            j.joint_3 / 1000.0,
            j.joint_4 / 1000.0,
            j.joint_5 / 1000.0,
            j.joint_6 / 1000.0,
        ])

    FAULT_FLAGS = [
        ("collision_status", "collision"),
        ("stall_status", "stall"),
        ("driver_overcurrent", "overcurrent"),
        ("driver_error_status", "driver error"),
        ("driver_overheating", "driver overheat"),
        ("motor_overheating", "motor overheat"),
        ("voltage_too_low", "low voltage"),
    ]

    def motor_faults(self):

        if self.dry_run or self.piper is None:
            return {}

        ls = self.piper.GetArmLowSpdInfoMsgs()
        faults = {}

        for j in range(1, 7):
            foc = getattr(ls, f"motor_{j}").foc_status
            flags = [name for attr, name in self.FAULT_FLAGS if getattr(foc, attr, False)]
            if flags:
                faults[f"J{j}"] = flags

        return faults

    def read_gripper_um(self):

        if self.dry_run or self.piper is None:
            with self._lock:
                return int((self._gripper_ratio or 0.0) * self.gripper_max_um)

        return self.piper.GetArmGripperMsgs().gripper_state.grippers_angle

    def gripper_text(self):

        return f"gripper {self.read_gripper_um() / 1000:.1f} mm"

    def gripper_value(self):

        return self.read_gripper_um()

    # ========================================================
    # COMMAND
    # ========================================================

    def _send(self, q, gripper_ratio):

        cmd = np.round(np.rad2deg(q) * 1000.0).astype(int)
        self.piper.ModeCtrl(0x01, 0x01, self.speed_percent, self.mit_mode)
        self.piper.JointCtrl(*[int(c) for c in cmd])
        if gripper_ratio is not None:
            self.piper.GripperCtrl(int(gripper_ratio * self.gripper_max_um), self.gripper_effort, 0x01, 0)

    def emergency_stop(self):
        """PiPER 비상정지 (모터가 감쇠하며 천천히 내려감). 재시작 시 프로그램 재실행 필요"""

        self._running = False
        if not self.dry_run and self.piper is not None:
            self.piper.EmergencyStop(0x01)
        print("!!! PiPER EMERGENCY STOP 전송")

    def stop(self):
        """명령 전송만 중단. 모터는 enable 상태로 마지막 위치 유지 (팔이 떨어지지 않음)"""

        super().stop()
