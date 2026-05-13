#
#  Copyright (C) 2026 Parth Maradia
#  Licensed under the Apache License, Version 2.0
#
"""Terminal-only 1x crop collector for the MgACT terminal servo.

This collector uses ground-truth TF like CheatCode/DataCollect, but records only
the final alignment/insertion window. It stores compact 1x crops and residual
labels instead of full-frame full-episode demonstrations.
"""
from __future__ import annotations

import math
import os
import time
import uuid
from pathlib import Path

# DataCollect snapshots these defaults at import time.
os.environ.setdefault("AIC_IMAGE_SCALE", "1.0")
os.environ.setdefault("AIC_DATA_DIR", "/home/ubuntu/ws_aic/data/episodes_terminal_1x")

import h5py
import numpy as np

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion
from rclpy.time import Time
from tf2_ros import TransformException

from .DataCollect import DataCollect, _impedance_schedule
from .terminal_servo_model import (
    base_to_local_delta,
    center_crop_or_pad,
    local_to_base_delta,
    quat_wxyz_to_matrix,
    ros_image_to_np,
)


PHASE_NAMES = ("ALIGN", "INSERT", "HOLD")
PHASE_TO_ID = {name: i for i, name in enumerate(PHASE_NAMES)}


def _pose_to_xyz_quat(pose: Pose) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float32)
    quat = np.array(
        [pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z],
        dtype=np.float32,
    )
    return xyz, quat


def _copy_pose(pose: Pose) -> Pose:
    return Pose(
        position=Point(x=pose.position.x, y=pose.position.y, z=pose.position.z),
        orientation=Quaternion(
            w=pose.orientation.w,
            x=pose.orientation.x,
            y=pose.orientation.y,
            z=pose.orientation.z,
        ),
    )


def _offset_pose_local(pose: Pose, offset_local: np.ndarray) -> Pose:
    out = _copy_pose(pose)
    _xyz, quat = _pose_to_xyz_quat(out)
    delta = local_to_base_delta(quat, offset_local)
    out.position.x += float(delta[0])
    out.position.y += float(delta[1])
    out.position.z += float(delta[2])
    return out


class DataCollectTerminal1x(DataCollect):
    """Ground-truth terminal collector with randomized local offsets."""

    def __init__(self, parent_node):
        self._crop_size = int(os.environ.get("AIC_TERMINAL_CROP", "640"))
        self._z_start_min = float(os.environ.get("AIC_TERMINAL_Z_START_MIN", "0.030"))
        self._z_start_max = float(os.environ.get("AIC_TERMINAL_Z_START_MAX", "0.050"))
        self._z_min = float(os.environ.get("AIC_TERMINAL_Z_MIN", "-0.015"))
        self._descent_step = float(os.environ.get("AIC_TERMINAL_DESCENT_STEP", "0.0005"))
        self._dt = float(os.environ.get("AIC_TERMINAL_DT", "0.05"))
        self._align_steps = int(os.environ.get("AIC_TERMINAL_ALIGN_STEPS", "36"))
        self._hold_seconds = float(os.environ.get("AIC_TERMINAL_HOLD_SECONDS", "5.0"))
        self._max_offset = float(os.environ.get("AIC_TERMINAL_MAX_OFFSET", "0.025"))
        self._approach_z = float(os.environ.get("AIC_TERMINAL_APPROACH_Z", "0.12"))
        seed = int(os.environ.get("AIC_TERMINAL_SEED", str(time.time_ns() % (2**32))))
        self._rng = np.random.default_rng(seed)

        self._buf_crop_center = []
        self._buf_crop_right = []
        self._buf_tcp_translation = []
        self._buf_tcp_quaternion = []
        self._buf_target_translation = []
        self._buf_target_quaternion = []
        self._buf_residual_base = []
        self._buf_residual_local = []
        self._buf_offset_local = []
        self._buf_confidence = []
        self._buf_hold = []
        self._buf_phase_id = []
        self._buf_phase_name = []

        super().__init__(parent_node)
        self.get_logger().info(
            "DataCollectTerminal1x: "
            f"dir={self._out_dir} crop={self._crop_size} "
            f"z_start=[{self._z_start_min:.3f},{self._z_start_max:.3f}] "
            f"max_offset={self._max_offset:.3f}"
        )

    def _reset_buffers(self) -> None:
        super()._reset_buffers()
        self._buf_crop_center.clear()
        self._buf_crop_right.clear()
        self._buf_tcp_translation.clear()
        self._buf_tcp_quaternion.clear()
        self._buf_target_translation.clear()
        self._buf_target_quaternion.clear()
        self._buf_residual_base.clear()
        self._buf_residual_local.clear()
        self._buf_offset_local.clear()
        self._buf_confidence.clear()
        self._buf_hold.clear()
        self._buf_phase_id.clear()
        self._buf_phase_name.clear()

    def _sample_offset_local(self) -> np.ndarray:
        radius = self._max_offset * math.sqrt(float(self._rng.random()))
        theta = 2.0 * math.pi * float(self._rng.random())
        return np.array([radius * math.cos(theta), radius * math.sin(theta), 0.0], dtype=np.float32)

    def _record_terminal_step(
        self,
        obs: Observation,
        current_pose: Pose,
        target_pose: Pose,
        stiffness_diag: list[float],
        damping_diag: list[float],
        z_offset: float,
        phase_name: str,
        offset_local: np.ndarray,
    ) -> None:
        try:
            center = center_crop_or_pad(ros_image_to_np(obs.center_image), self._crop_size)
            right = center_crop_or_pad(ros_image_to_np(obs.right_image), self._crop_size)
            self._buf_crop_center.append(center)
            self._buf_crop_right.append(right)
            if self._image_encoding is None:
                self._image_encoding = obs.center_image.encoding

            w = obs.wrist_wrench.wrench
            self._buf_wrench.append(
                np.array(
                    [w.force.x, w.force.y, w.force.z, w.torque.x, w.torque.y, w.torque.z],
                    dtype=np.float32,
                )
            )

            js = obs.joint_states
            if self._joint_names is None and js.name:
                self._joint_names = list(js.name)
            self._buf_joint_pos.append(np.asarray(js.position, dtype=np.float32))
            self._buf_joint_vel.append(np.asarray(js.velocity, dtype=np.float32))
            self._buf_joint_eff.append(np.asarray(js.effort, dtype=np.float32))

            cur_xyz, cur_quat = _pose_to_xyz_quat(current_pose)
            tgt_xyz, tgt_quat = _pose_to_xyz_quat(target_pose)
            residual_base = (tgt_xyz - cur_xyz).astype(np.float32)
            residual_local = base_to_local_delta(cur_quat, residual_base).astype(np.float32)

            try:
                tcp = obs.controller_state.tcp_pose
                tcp_xyz = np.array([tcp.position.x, tcp.position.y, tcp.position.z], dtype=np.float32)
                tcp_quat = np.array([tcp.orientation.w, tcp.orientation.x, tcp.orientation.y, tcp.orientation.z], dtype=np.float32)
            except Exception:
                tcp_xyz = cur_xyz.copy()
                tcp_quat = cur_quat.copy()

            self._buf_act_translation.append(cur_xyz)
            self._buf_act_quaternion.append(cur_quat)
            self._buf_act_stiffness_diag.append(np.asarray(stiffness_diag, dtype=np.float32))
            self._buf_act_damping_diag.append(np.asarray(damping_diag, dtype=np.float32))
            self._buf_z_offset.append(np.float32(z_offset))
            self._buf_tcp_translation.append(tcp_xyz)
            self._buf_tcp_quaternion.append(tcp_quat)
            self._buf_target_translation.append(tgt_xyz)
            self._buf_target_quaternion.append(tgt_quat)
            self._buf_residual_base.append(residual_base)
            self._buf_residual_local.append(residual_local)
            self._buf_offset_local.append(np.asarray(offset_local, dtype=np.float32))
            self._buf_confidence.append(np.float32(math.exp(-np.linalg.norm(residual_local[:2]) / 0.012)))
            self._buf_hold.append(np.float32(1.0 if phase_name == "HOLD" else 0.0))
            self._buf_phase_id.append(np.int64(PHASE_TO_ID[phase_name]))
            self._buf_phase_name.append(phase_name)

            t = self.time_now()
            self._buf_timestamp.append(t.nanoseconds * 1e-9)
        except Exception as ex:
            self.get_logger().warn(f"DataCollectTerminal1x: failed to record step: {ex}")

    def _move_record_terminal(
        self,
        move_robot: MoveRobotCallback,
        current_pose: Pose,
        target_pose: Pose,
        z_offset: float,
        phase_name: str,
        offset_local: np.ndarray,
    ) -> None:
        K_diag, D_diag = _impedance_schedule(z_offset)
        if phase_name == "HOLD":
            K_diag = [90.0, 90.0, 30.0, 40.0, 40.0, 40.0]
            D_diag = [54.0, 54.0, 18.0, 24.0, 24.0, 24.0]

        self.set_pose_target(move_robot=move_robot, pose=current_pose, stiffness=K_diag, damping=D_diag)
        self.sleep_for(self._dt)
        if self._get_observation is not None:
            obs = self._get_observation()
            if obs is not None:
                self._record_terminal_step(
                    obs,
                    current_pose=current_pose,
                    target_pose=target_pose,
                    stiffness_diag=K_diag,
                    damping_diag=D_diag,
                    z_offset=z_offset,
                    phase_name=phase_name,
                    offset_local=offset_local,
                )

    def _move_no_record(self, move_robot: MoveRobotCallback, pose: Pose, z_offset: float) -> None:
        K_diag, D_diag = _impedance_schedule(z_offset)
        self.set_pose_target(move_robot=move_robot, pose=pose, stiffness=K_diag, damping=D_diag)
        self.sleep_for(self._dt)

    def _save_episode(self, task: Task, success: bool) -> None:
        n = len(self._buf_timestamp)
        if n == 0:
            self.get_logger().warn("DataCollectTerminal1x: empty episode, nothing to save")
            return

        ts_ms = int(time.time() * 1000)
        fname = self._out_dir / f"episode_{ts_ms}_{uuid.uuid4().hex[:8]}.h5"
        try:
            ts = np.asarray(self._buf_timestamp, dtype=np.float64)
            duration = float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0
            with h5py.File(fname, "w") as f:
                f.attrs["success"] = bool(success)
                f.attrs["collector_version"] = "DataCollectTerminal1x"
                f.attrs["num_steps"] = n
                f.attrs["duration_s"] = duration
                f.attrs["control_hz"] = float(n / duration) if duration > 0 else 0.0
                f.attrs["image_scale"] = 1.0
                f.attrs["crop_size"] = self._crop_size
                f.attrs["camera_names"] = np.asarray(["center", "right"], dtype="S")
                f.attrs["cable_name"] = str(task.cable_name)
                f.attrs["plug_name"] = str(task.plug_name)
                f.attrs["target_module_name"] = str(task.target_module_name)
                f.attrs["port_name"] = str(task.port_name)
                f.attrs["image_encoding"] = self._image_encoding or ""
                f.attrs["phase_layout"] = "0=ALIGN, 1=INSERT, 2=HOLD"
                if self._joint_names:
                    f.attrs["joint_names"] = np.array(self._joint_names, dtype="S")

                obs_g = f.create_group("observations")
                crops_g = obs_g.create_group("crops")
                crops_g.create_dataset(
                    "center",
                    data=np.stack(self._buf_crop_center),
                    compression="gzip",
                    compression_opts=4,
                )
                crops_g.create_dataset(
                    "right",
                    data=np.stack(self._buf_crop_right),
                    compression="gzip",
                    compression_opts=4,
                )
                obs_g.create_dataset("wrench", data=np.stack(self._buf_wrench))
                obs_g.create_dataset("joint_position", data=np.stack(self._buf_joint_pos))
                obs_g.create_dataset("joint_velocity", data=np.stack(self._buf_joint_vel))
                obs_g.create_dataset("joint_effort", data=np.stack(self._buf_joint_eff))
                obs_g.create_dataset("tcp_translation", data=np.stack(self._buf_tcp_translation))
                obs_g.create_dataset("tcp_quaternion_wxyz", data=np.stack(self._buf_tcp_quaternion))

                act_g = f.create_group("actions")
                act_g.create_dataset("translation", data=np.stack(self._buf_act_translation))
                act_g.create_dataset("quaternion_wxyz", data=np.stack(self._buf_act_quaternion))
                act_g.create_dataset("stiffness_diag", data=np.stack(self._buf_act_stiffness_diag))
                act_g.create_dataset("damping_diag", data=np.stack(self._buf_act_damping_diag))
                act_g.create_dataset("z_offset", data=np.asarray(self._buf_z_offset, dtype=np.float32))
                act_g.create_dataset("phase_id", data=np.asarray(self._buf_phase_id, dtype=np.int64))
                act_g.create_dataset("phase_name", data=np.asarray(self._buf_phase_name, dtype="S"))

                label_g = f.create_group("labels")
                label_g.create_dataset("target_translation", data=np.stack(self._buf_target_translation))
                label_g.create_dataset("target_quaternion_wxyz", data=np.stack(self._buf_target_quaternion))
                label_g.create_dataset("residual_base", data=np.stack(self._buf_residual_base))
                label_g.create_dataset("residual_local", data=np.stack(self._buf_residual_local))
                label_g.create_dataset("offset_local", data=np.stack(self._buf_offset_local))
                label_g.create_dataset("confidence", data=np.asarray(self._buf_confidence, dtype=np.float32))
                label_g.create_dataset("hold", data=np.asarray(self._buf_hold, dtype=np.float32))
                f.create_dataset("timestamps", data=ts)

            self.get_logger().info(f"DataCollectTerminal1x: wrote {n} terminal steps -> {fname}")
        except Exception as ex:
            self.get_logger().error(f"DataCollectTerminal1x: failed to write {fname}: {ex}")

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        self.get_logger().info(f"DataCollectTerminal1x.insert_cable() task: {task}")
        self._task = task
        self._get_observation = get_observation
        self._reset_buffers()
        success = False

        try:
            port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
            cable_tip_frame = f"{task.cable_name}/{task.plug_name}_link"
            for frame in [port_frame, cable_tip_frame]:
                if not self._wait_for_tf("base_link", frame):
                    return False

            try:
                port_tf_stamped = self._parent_node._tf_buffer.lookup_transform("base_link", port_frame, Time())
            except TransformException as ex:
                self.get_logger().error(f"Could not look up port transform: {ex}")
                return False
            port_transform = port_tf_stamped.transform

            z_start = float(self._rng.uniform(self._z_start_min, self._z_start_max))
            offset0 = self._sample_offset_local()
            self.get_logger().info(
                f"Terminal episode z_start={z_start:.4f} "
                f"offset_local=({offset0[0]:+.4f},{offset0[1]:+.4f})"
            )

            # Move high above the selected port without recording.
            for t in range(80):
                frac = t / 79.0
                try:
                    pose = self.calc_gripper_pose(
                        port_transform,
                        slerp_fraction=frac,
                        position_fraction=frac,
                        z_offset=self._approach_z,
                        reset_xy_integrator=True,
                    )
                    self._move_no_record(move_robot, pose, self._approach_z)
                except TransformException as ex:
                    self.get_logger().warn(f"TF lookup failed during approach: {ex}")

            # Move to randomized terminal start without recording.
            for t in range(30):
                frac = (t + 1) / 30.0
                try:
                    target_pose = self.calc_gripper_pose(
                        port_transform,
                        z_offset=z_start,
                        reset_xy_integrator=True,
                    )
                    pose = _offset_pose_local(target_pose, offset0 * frac)
                    self._move_no_record(move_robot, pose, z_start)
                except TransformException as ex:
                    self.get_logger().warn(f"TF lookup failed during terminal start: {ex}")

            # Constant-height alignment with decaying lateral offset.
            for i in range(self._align_steps):
                keep = 1.0 - 0.85 * ((i + 1) / max(1, self._align_steps))
                offset = offset0 * keep
                try:
                    target_pose = self.calc_gripper_pose(port_transform, z_offset=z_start, reset_xy_integrator=True)
                    pose = _offset_pose_local(target_pose, offset)
                    self._move_record_terminal(move_robot, pose, target_pose, z_start, "ALIGN", offset)
                except TransformException as ex:
                    self.get_logger().warn(f"TF lookup failed during terminal align: {ex}")

            # Descend while removing the remaining offset.
            z_offset = z_start
            total_drop = max(1e-6, z_start - self._z_min)
            while z_offset >= self._z_min:
                z_offset -= self._descent_step
                progress = min(1.0, max(0.0, (z_start - z_offset) / total_drop))
                offset = offset0 * 0.15 * (1.0 - progress) ** 2
                phase = "INSERT" if z_offset <= 0.0 else "ALIGN"
                try:
                    target_pose = self.calc_gripper_pose(port_transform, z_offset=z_offset, reset_xy_integrator=True)
                    pose = _offset_pose_local(target_pose, offset)
                    self._move_record_terminal(move_robot, pose, target_pose, z_offset, phase, offset)
                except TransformException as ex:
                    self.get_logger().warn(f"TF lookup failed during terminal insert: {ex}")

            hold_steps = max(1, int(round(self._hold_seconds / max(1e-3, self._dt))))
            for _ in range(hold_steps):
                try:
                    target_pose = self.calc_gripper_pose(port_transform, z_offset=self._z_min, reset_xy_integrator=True)
                    self._move_record_terminal(
                        move_robot,
                        target_pose,
                        target_pose,
                        self._z_min,
                        "HOLD",
                        np.zeros(3, dtype=np.float32),
                    )
                except TransformException as ex:
                    self.get_logger().warn(f"TF lookup failed during terminal hold: {ex}")

            success = True
            return True
        finally:
            try:
                self._save_episode(task, success=success)
            except Exception as ex:
                self.get_logger().error(f"DataCollectTerminal1x: _save_episode failed: {ex}")
            self._get_observation = None

