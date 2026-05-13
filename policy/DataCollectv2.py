#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#

"""Balanced high-quality data collector for MgACT fine-tuning.

This is intentionally close to DataCollect, but adds the bits the first dataset
was missing:

* a separate default output directory;
* a higher default camera scale (0.5 rather than 0.25);
* explicit phase labels: APPROACH, ALIGN, INSERT, HOLD;
* recorded terminal hold actions instead of only sleeping after insertion.

It still uses the CheatCode-style ground-truth TF controller, so only run it for
data collection with ground_truth:=true.
"""

import os

# Must be set before importing DataCollect: that module snapshots this env var
# into its IMAGE_SCALE constant at import time.
os.environ.setdefault("AIC_IMAGE_SCALE", "0.5")
os.environ.setdefault(
    "AIC_DATA_DIR",
    "/home/ubuntu/ws_aic/data/episodes_v2_balanced_0p5",
)

import h5py
import numpy as np

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_ros import TransformException

from .DataCollect import DataCollect, _impedance_schedule


PHASE_NAMES = ("APPROACH", "ALIGN", "INSERT", "HOLD")
PHASE_TO_ID = {name: i for i, name in enumerate(PHASE_NAMES)}


class DataCollectv2(DataCollect):
    """DataCollect with balanced-dataset metadata and terminal HOLD recording."""

    def __init__(self, parent_node):
        self._buf_phase_id = []
        self._buf_phase_name = []
        self._hold_seconds = float(os.environ.get("AIC_DC2_HOLD_SECONDS", "5.0"))
        self._hold_dt = float(os.environ.get("AIC_DC2_HOLD_DT", "0.05"))
        self._descent_step = float(os.environ.get("AIC_DC2_DESCENT_STEP", "0.0005"))
        self._z_min = float(os.environ.get("AIC_DC2_Z_MIN", "-0.015"))
        self._align_z = float(os.environ.get("AIC_DC2_ALIGN_Z", "0.12"))
        super().__init__(parent_node)
        self.get_logger().info(
            "DataCollectv2: phase labels enabled "
            f"hold={self._hold_seconds:.1f}s image_scale={os.environ.get('AIC_IMAGE_SCALE')}"
        )

    def _reset_buffers(self) -> None:
        super()._reset_buffers()
        self._buf_phase_id.clear()
        self._buf_phase_name.clear()

    def _phase_from_z(self, z_offset: float) -> str:
        if z_offset <= 0.0:
            return "INSERT"
        if z_offset <= self._align_z:
            return "ALIGN"
        return "APPROACH"

    def _record_step(
        self,
        obs,
        pose,
        stiffness_diag: list[float],
        damping_diag: list[float],
        z_offset: float,
        phase_name: str | None = None,
    ) -> None:
        before = len(self._buf_timestamp)
        super()._record_step(obs, pose, stiffness_diag, damping_diag, z_offset)
        if len(self._buf_timestamp) <= before:
            return
        name = phase_name or self._phase_from_z(z_offset)
        if name not in PHASE_TO_ID:
            name = "APPROACH"
        self._buf_phase_id.append(np.int64(PHASE_TO_ID[name]))
        self._buf_phase_name.append(name)

    def _move_and_record(
        self,
        move_robot: MoveRobotCallback,
        pose,
        z_offset: float,
        phase_name: str | None = None,
    ) -> None:
        K_diag, D_diag = _impedance_schedule(z_offset)
        if phase_name == "HOLD":
            # Keep the seated connector compliant along insertion, but resist
            # lateral drift. This mirrors the insertion schedule and makes HOLD
            # a learnable low-force terminal action.
            K_diag = [90.0, 90.0, 30.0, 40.0, 40.0, 40.0]
            D_diag = [54.0, 54.0, 18.0, 24.0, 24.0, 24.0]

        if self._get_observation is not None:
            try:
                obs = self._get_observation()
                if obs is not None:
                    self._record_step(
                        obs,
                        pose,
                        K_diag,
                        D_diag,
                        z_offset,
                        phase_name=phase_name,
                    )
            except Exception as ex:
                self.get_logger().warn(f"DataCollectv2: get_observation failed: {ex}")

        self.set_pose_target(
            move_robot=move_robot,
            pose=pose,
            stiffness=K_diag,
            damping=D_diag,
        )

    def _save_episode(self, task: Task, success: bool) -> None:
        before = set(self._out_dir.glob("episode_*.h5"))
        super()._save_episode(task, success=success)
        after = set(self._out_dir.glob("episode_*.h5"))
        new_files = list(after - before)
        if not new_files:
            # Fallback for rare cases where mtime resolution or interrupted
            # writes make set-diff unhelpful.
            new_files = list(after)
        if not new_files:
            return

        fname = max(new_files, key=lambda p: p.stat().st_mtime)
        try:
            with h5py.File(fname, "a") as f:
                n = int(f.attrs.get("num_steps", len(self._buf_phase_id)))
                phase_id = np.asarray(self._buf_phase_id, dtype=np.int64)
                if len(phase_id) < n:
                    pad = np.full((n - len(phase_id),), PHASE_TO_ID["HOLD"], dtype=np.int64)
                    phase_id = np.concatenate([phase_id, pad], axis=0)
                phase_id = phase_id[:n]

                phase_names = np.asarray(
                    [PHASE_NAMES[int(i)] for i in phase_id],
                    dtype="S",
                )

                f.attrs["collector_version"] = "DataCollectv2"
                f.attrs["phase_layout"] = "0=APPROACH, 1=ALIGN, 2=INSERT, 3=HOLD"
                f.attrs["phase_names"] = np.asarray(PHASE_NAMES, dtype="S")
                f.attrs["hold_seconds"] = self._hold_seconds
                f.attrs["descent_step"] = self._descent_step
                f.attrs["z_min"] = self._z_min

                act_g = f["actions"]
                for key in ("phase_id", "phase_name"):
                    if key in act_g:
                        del act_g[key]
                act_g.create_dataset("phase_id", data=phase_id)
                act_g.create_dataset("phase_name", data=phase_names)

            self.get_logger().info(f"DataCollectv2: annotated phases -> {fname}")
        except Exception as ex:
            self.get_logger().error(f"DataCollectv2: failed to annotate {fname}: {ex}")

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        self.get_logger().info(f"DataCollectv2.insert_cable() task: {task}")
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
                port_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
                    "base_link",
                    port_frame,
                    Time(),
                )
            except TransformException as ex:
                self.get_logger().error(f"Could not look up port transform: {ex}")
                return False
            port_transform = port_tf_stamped.transform

            z_offset = 0.2
            last_pose = None

            # Smoothly move above the selected port.
            for t in range(0, 100):
                interp_fraction = t / 100.0
                try:
                    last_pose = self.calc_gripper_pose(
                        port_transform,
                        slerp_fraction=interp_fraction,
                        position_fraction=interp_fraction,
                        z_offset=z_offset,
                        reset_xy_integrator=True,
                    )
                    self._move_and_record(
                        move_robot=move_robot,
                        pose=last_pose,
                        z_offset=z_offset,
                        phase_name="APPROACH",
                    )
                except TransformException as ex:
                    self.get_logger().warn(f"TF lookup failed during interpolation: {ex}")
                self.sleep_for(0.05)

            # Descend through explicit ALIGN and INSERT labels.
            while z_offset >= self._z_min:
                z_offset -= self._descent_step
                phase_name = self._phase_from_z(z_offset)
                if int(round((0.2 - z_offset) / max(self._descent_step, 1e-9))) % 20 == 0:
                    self.get_logger().info(f"z_offset: {z_offset:0.5} phase={phase_name}")
                try:
                    last_pose = self.calc_gripper_pose(port_transform, z_offset=z_offset)
                    self._move_and_record(
                        move_robot=move_robot,
                        pose=last_pose,
                        z_offset=z_offset,
                        phase_name=phase_name,
                    )
                except TransformException as ex:
                    self.get_logger().warn(f"TF lookup failed during insertion: {ex}")
                self.sleep_for(0.05)

            # Record terminal HOLD actions. The original collector only slept
            # here, so ACT never saw the "stay inserted" behavior.
            hold_steps = max(1, int(round(self._hold_seconds / max(self._hold_dt, 1e-3))))
            self.get_logger().info(f"Recording HOLD for {hold_steps} steps")
            for _ in range(hold_steps):
                try:
                    last_pose = self.calc_gripper_pose(port_transform, z_offset=z_offset)
                except TransformException:
                    pass
                if last_pose is not None:
                    self._move_and_record(
                        move_robot=move_robot,
                        pose=last_pose,
                        z_offset=z_offset,
                        phase_name="HOLD",
                    )
                self.sleep_for(self._hold_dt)

            self.get_logger().info("DataCollectv2.insert_cable() exiting...")
            success = True
            return True
        finally:
            try:
                self._save_episode(task, success=success)
            except Exception as ex:
                self.get_logger().error(f"DataCollectv2: _save_episode failed: {ex}")
            self._get_observation = None
