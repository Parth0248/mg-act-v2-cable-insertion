#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import os
import time
import uuid
from pathlib import Path

import cv2

import cv2
import h5py
import numpy as np

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion, Transform
from rclpy.duration import Duration
from rclpy.time import Time
from std_msgs.msg import Header
from tf2_ros import TransformException
from transforms3d._gohlketransforms import quaternion_multiply, quaternion_slerp

from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from geometry_msgs.msg import Vector3, Wrench

QuaternionTuple = tuple[float, float, float, float]

# Fallback impedance (matches Policy.set_pose_target defaults). Used only when
# variable-impedance scheduling is disabled (env var AIC_VARIABLE_IMPEDANCE=0).
_DEFAULT_STIFFNESS = [90.0, 90.0, 90.0, 50.0, 50.0, 50.0]
_DEFAULT_DAMPING = [50.0, 50.0, 50.0, 20.0, 20.0, 20.0]
_DEFAULT_WRENCH_GAINS = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]

# Image downsample factor — must match the eval policy's preprocessing.
# Stored as an HDF5 file attribute so the eval policy reads it back.
IMAGE_SCALE = float(os.environ.get("AIC_IMAGE_SCALE", "0.25"))


def _impedance_schedule(z_offset: float) -> tuple[list[float], list[float]]:
    """Return (stiffness_diag, damping_diag) as 6-element lists for a given z_offset.

    Phase 1 — Approach (z_offset > 0.10 m, free space):
        High K everywhere. The arm should reject perturbations and move fast.
    Phase 2 — Descent (0 < z_offset <= 0.10 m, anticipating contact):
        Linearly ramp down K_z (from 120 -> 50) so the demo teaches the model
        to soften before contact. K_xy held at 120 to stay aligned over port.
    Phase 3 — Insertion (z_offset <= 0, in contact):
        K_z low (30) for compliance along approach axis; K_xy moderate (90)
        to resist lateral drift off the port. Damping bumped to 0.6*K for stability.

    The schedule depends only on z_offset (observable state), so every episode
    produces a clean, learnable mapping from state -> impedance.
    """
    if z_offset > 0.10:
        K_xy, K_z, K_rot, damp_ratio = 120.0, 120.0, 60.0, 0.5
    elif z_offset > 0.0:
        # Linear interp from approach (z=0.10) -> insertion-edge (z=0.0)
        t = z_offset / 0.10  # 1.0 at z=0.10, 0.0 at z=0.0
        K_xy = 120.0
        K_z = 50.0 + t * (120.0 - 50.0)
        K_rot = 40.0 + t * (60.0 - 40.0)
        damp_ratio = 0.5
    else:
        # In contact: low z-stiffness for compliance, moderate xy to resist drift.
        K_xy, K_z, K_rot, damp_ratio = 90.0, 30.0, 40.0, 0.6

    K = [K_xy, K_xy, K_z, K_rot, K_rot, K_rot]
    D = [k * damp_ratio for k in K]
    return K, D


def _ros_image_to_np(img_msg, scale: float = 1.0) -> np.ndarray:
    """Decode a sensor_msgs/Image into an HxWx3 uint8 array, optionally resized.

    The same INTER_AREA resize must be applied at eval time for train/eval
    preprocessing to match. RunACT.py uses this same operation. The active
    image_scale is stored as an HDF5 file attribute by _save_episode.
    """
    if img_msg is None or img_msg.height == 0 or img_msg.width == 0:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    arr = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(
        img_msg.height, img_msg.width, 3
    )
    if scale != 1.0:
        arr = cv2.resize(arr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return arr.copy()


class DataCollect(Policy):
    """CheatCode policy that also logs (observation, action) pairs to HDF5.

    Each call to insert_cable() runs one episode and writes a single
    .h5 file under AIC_DATA_DIR (default: ~/aic_data). Files are named
    episode_<unix-ms>_<uuid>.h5 to be safe under parallel runs.
    """

    def __init__(self, parent_node):
        self._tip_x_error_integrator = 0.0
        self._tip_y_error_integrator = 0.0
        self._max_integrator_windup = 0.05
        self._task = None

        # Per-episode buffers. Reset at the top of insert_cable().
        self._buf_left = []
        self._buf_center = []
        self._buf_right = []
        self._buf_wrench = []           # (6,)  fx,fy,fz,tx,ty,tz
        self._buf_joint_pos = []        # (J,)
        self._buf_joint_vel = []        # (J,)
        self._buf_joint_eff = []        # (J,)
        self._buf_act_translation = []   # (3,)
        self._buf_act_quaternion = []    # (4,) wxyz
        self._buf_act_stiffness_diag = []  # (6,)  [Kxx, Kyy, Kzz, Krx, Kry, Krz]
        self._buf_act_damping_diag = []    # (6,)  [Dxx, Dyy, Dzz, Drx, Dry, Drz]
        self._buf_z_offset = []          # ()  scheduling input, useful for debug
        self._buf_timestamp = []         # float seconds
        self._joint_names = None
        self._image_encoding = None

        # The current observation callback, set in insert_cable() so that the
        # recording wrapper can fetch the observation that goes with each action.
        self._get_observation: GetObservationCallback | None = None

        out_dir = os.environ.get("AIC_DATA_DIR", str(Path.home() / "ws_aic/data/episodes"))
        self._out_dir = Path(out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)

        super().__init__(parent_node)
        self.get_logger().info(f"DataCollect: writing HDF5 episodes to {self._out_dir}")

    # --------------------------------------------------------------- recording

    def _reset_buffers(self) -> None:
        self._buf_left.clear()
        self._buf_center.clear()
        self._buf_right.clear()
        self._buf_wrench.clear()
        self._buf_joint_pos.clear()
        self._buf_joint_vel.clear()
        self._buf_joint_eff.clear()
        self._buf_act_translation.clear()
        self._buf_act_quaternion.clear()
        self._buf_act_stiffness_diag.clear()
        self._buf_act_damping_diag.clear()
        self._buf_z_offset.clear()
        self._buf_timestamp.clear()
        self._joint_names = None
        self._image_encoding = None

    def _record_step(
        self,
        obs: Observation,
        pose: Pose,
        stiffness_diag: list[float],
        damping_diag: list[float],
        z_offset: float,
    ) -> None:
        """Append one (observation, action) pair to the in-memory buffers."""
        try:
            self._buf_left.append(_ros_image_to_np(obs.left_image, IMAGE_SCALE))
            self._buf_center.append(_ros_image_to_np(obs.center_image, IMAGE_SCALE))
            self._buf_right.append(_ros_image_to_np(obs.right_image, IMAGE_SCALE))
            if self._image_encoding is None:
                self._image_encoding = obs.left_image.encoding

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

            self._buf_act_translation.append(
                np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float32)
            )
            self._buf_act_quaternion.append(
                np.array(
                    [pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z],
                    dtype=np.float32,
                )
            )
            self._buf_act_stiffness_diag.append(np.asarray(stiffness_diag, dtype=np.float32))
            self._buf_act_damping_diag.append(np.asarray(damping_diag, dtype=np.float32))
            self._buf_z_offset.append(np.float32(z_offset))

            t = self.time_now()
            self._buf_timestamp.append(t.nanoseconds * 1e-9)
        except Exception as ex:
            # Never let logging fail the episode.
            self.get_logger().warn(f"DataCollect: failed to record step: {ex}")

    def _move_and_record(
        self,
        move_robot: MoveRobotCallback,
        pose: Pose,
        z_offset: float,
    ) -> None:
        """Compute phase-scheduled impedance, log (obs, action), then send the action."""
        K_diag, D_diag = _impedance_schedule(z_offset)
        if self._get_observation is not None:
            try:
                obs = self._get_observation()
                if obs is not None:
                    self._record_step(obs, pose, K_diag, D_diag, z_offset)
            except Exception as ex:
                self.get_logger().warn(f"DataCollect: get_observation failed: {ex}")
        self.set_pose_target(
            move_robot=move_robot,
            pose=pose,
            stiffness=K_diag,
            damping=D_diag,
        )

    def _save_episode(self, task: Task, success: bool) -> None:
        """Dump the per-step buffers for this episode into a single HDF5 file."""
        n = len(self._buf_timestamp)
        if n == 0:
            self.get_logger().warn("DataCollect: empty episode, nothing to save")
            return

        # Stack only if shapes are consistent across steps. Cameras and joint
        # states should be fixed-size; we trust that here.
        ts_ms = int(time.time() * 1000)
        fname = self._out_dir / f"episode_{ts_ms}_{uuid.uuid4().hex[:8]}.h5"

        try:
            ts = np.asarray(self._buf_timestamp, dtype=np.float64)
            duration = float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0
            control_hz = float(n / duration) if duration > 0 else 0.0

            with h5py.File(fname, "w") as f:
                f.attrs["success"] = bool(success)
                f.attrs["num_steps"] = n
                f.attrs["duration_s"] = duration
                f.attrs["control_hz"] = control_hz
                f.attrs["image_scale"] = IMAGE_SCALE
                f.attrs["cable_name"] = str(task.cable_name)
                f.attrs["plug_name"] = str(task.plug_name)
                f.attrs["target_module_name"] = str(task.target_module_name)
                f.attrs["port_name"] = str(task.port_name)
                f.attrs["image_encoding"] = self._image_encoding or ""
                f.attrs["action_layout"] = (
                    "translation[3] | quaternion_wxyz[4] | stiffness_diag[6] | damping_diag[6]"
                )
                if self._joint_names:
                    f.attrs["joint_names"] = np.array(self._joint_names, dtype="S")

                obs_g = f.create_group("observations")
                imgs_g = obs_g.create_group("images")
                # gzip is plenty for 8-bit images and keeps us off external codecs.
                imgs_g.create_dataset(
                    "left", data=np.stack(self._buf_left), compression="gzip", compression_opts=4
                )
                imgs_g.create_dataset(
                    "center", data=np.stack(self._buf_center), compression="gzip", compression_opts=4
                )
                imgs_g.create_dataset(
                    "right", data=np.stack(self._buf_right), compression="gzip", compression_opts=4
                )
                obs_g.create_dataset("wrench", data=np.stack(self._buf_wrench))
                obs_g.create_dataset("joint_position", data=np.stack(self._buf_joint_pos))
                obs_g.create_dataset("joint_velocity", data=np.stack(self._buf_joint_vel))
                obs_g.create_dataset("joint_effort", data=np.stack(self._buf_joint_eff))

                act_g = f.create_group("actions")
                translation = np.stack(self._buf_act_translation)
                quaternion = np.stack(self._buf_act_quaternion)
                stiffness_diag = np.stack(self._buf_act_stiffness_diag)
                damping_diag = np.stack(self._buf_act_damping_diag)
                act_g.create_dataset("translation", data=translation)
                act_g.create_dataset("quaternion_wxyz", data=quaternion)
                act_g.create_dataset("stiffness_diag", data=stiffness_diag)
                act_g.create_dataset("damping_diag", data=damping_diag)
                # Pre-concatenated 19-D action vector for direct ACT consumption.
                act_g.create_dataset(
                    "action_19d",
                    data=np.concatenate(
                        [translation, quaternion, stiffness_diag, damping_diag], axis=1
                    ).astype(np.float32),
                )
                act_g.create_dataset(
                    "z_offset", data=np.asarray(self._buf_z_offset, dtype=np.float32)
                )

                f.create_dataset("timestamps", data=ts)

            self.get_logger().info(
                f"DataCollect: wrote {n} steps -> {fname} (success={success})"
            )
        except Exception as ex:
            self.get_logger().error(f"DataCollect: failed to write {fname}: {ex}")

    # ------------------------------------------------------- original CheatCode

    def _wait_for_tf(
        self, target_frame: str, source_frame: str, timeout_sec: float = 10.0
    ) -> bool:
        """Wait for a TF frame to become available."""
        start = self.time_now()
        timeout = Duration(seconds=timeout_sec)
        attempt = 0
        while (self.time_now() - start) < timeout:
            try:
                self._parent_node._tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    Time(),
                )
                return True
            except TransformException:
                if attempt % 20 == 0:
                    self.get_logger().info(
                        f"Waiting for transform '{source_frame}' -> '{target_frame}'... -- are you running eval with `ground_truth:=true`?"
                    )
                attempt += 1
                self.sleep_for(0.1)
        self.get_logger().error(
            f"Transform '{source_frame}' not available after {timeout_sec}s"
        )
        return False

    def calc_gripper_pose(
        self,
        port_transform: Transform,
        slerp_fraction: float = 1.0,
        position_fraction: float = 1.0,
        z_offset: float = 0.1,
        reset_xy_integrator: bool = False,
    ) -> Pose:
        """Find the gripper pose that results in plug alignment."""
        q_port = (
            port_transform.rotation.w,
            port_transform.rotation.x,
            port_transform.rotation.y,
            port_transform.rotation.z,
        )
        plug_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
            "base_link",
            f"{self._task.cable_name}/{self._task.plug_name}_link",
            Time(),
        )
        q_plug = (
            plug_tf_stamped.transform.rotation.w,
            plug_tf_stamped.transform.rotation.x,
            plug_tf_stamped.transform.rotation.y,
            plug_tf_stamped.transform.rotation.z,
        )
        q_plug_inv = (
            -q_plug[0],
            q_plug[1],
            q_plug[2],
            q_plug[3],
        )
        q_diff = quaternion_multiply(q_port, q_plug_inv)
        gripper_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
            "base_link",
            "gripper/tcp",
            Time(),
        )
        q_gripper = (
            gripper_tf_stamped.transform.rotation.w,
            gripper_tf_stamped.transform.rotation.x,
            gripper_tf_stamped.transform.rotation.y,
            gripper_tf_stamped.transform.rotation.z,
        )
        q_gripper_target = quaternion_multiply(q_diff, q_gripper)
        q_gripper_slerp = quaternion_slerp(q_gripper, q_gripper_target, slerp_fraction)

        gripper_xyz = (
            gripper_tf_stamped.transform.translation.x,
            gripper_tf_stamped.transform.translation.y,
            gripper_tf_stamped.transform.translation.z,
        )
        port_xy = (
            port_transform.translation.x,
            port_transform.translation.y,
        )
        plug_xyz = (
            plug_tf_stamped.transform.translation.x,
            plug_tf_stamped.transform.translation.y,
            plug_tf_stamped.transform.translation.z,
        )
        plug_tip_gripper_offset = (
            gripper_xyz[0] - plug_xyz[0],
            gripper_xyz[1] - plug_xyz[1],
            gripper_xyz[2] - plug_xyz[2],
        )

        tip_x_error = port_xy[0] - plug_xyz[0]
        tip_y_error = port_xy[1] - plug_xyz[1]

        if reset_xy_integrator:
            self._tip_x_error_integrator = 0.0
            self._tip_y_error_integrator = 0.0
        else:
            self._tip_x_error_integrator = np.clip(
                self._tip_x_error_integrator + tip_x_error,
                -self._max_integrator_windup,
                self._max_integrator_windup,
            )
            self._tip_y_error_integrator = np.clip(
                self._tip_y_error_integrator + tip_y_error,
                -self._max_integrator_windup,
                self._max_integrator_windup,
            )

        self.get_logger().info(
            f"pfrac: {position_fraction:.3} xy_error: {tip_x_error:0.3} {tip_y_error:0.3}   integrators: {self._tip_x_error_integrator:.3} , {self._tip_y_error_integrator:.3}"
        )

        i_gain = 0.15

        target_x = port_xy[0] + i_gain * self._tip_x_error_integrator
        target_y = port_xy[1] + i_gain * self._tip_y_error_integrator
        target_z = port_transform.translation.z + z_offset - plug_tip_gripper_offset[2]

        blend_xyz = (
            position_fraction * target_x + (1.0 - position_fraction) * gripper_xyz[0],
            position_fraction * target_y + (1.0 - position_fraction) * gripper_xyz[1],
            position_fraction * target_z + (1.0 - position_fraction) * gripper_xyz[2],
        )

        return Pose(
            position=Point(
                x=blend_xyz[0],
                y=blend_xyz[1],
                z=blend_xyz[2],
            ),
            orientation=Quaternion(
                w=q_gripper_slerp[0],
                x=q_gripper_slerp[1],
                y=q_gripper_slerp[2],
                z=q_gripper_slerp[3],
            ),
        )

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        self.get_logger().info(f"DataCollect.insert_cable() task: {task}")
        self._task = task
        self._get_observation = get_observation
        self._reset_buffers()
        success = False

        try:
            port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
            cable_tip_frame = f"{task.cable_name}/{task.plug_name}_link"

            # Wait for both the port and cable tip TFs to become available.
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

            # Smoothly interpolate from the current position to a position above
            # the port. Each step is recorded as one (observation, action) pair.
            for t in range(0, 100):
                interp_fraction = t / 100.0
                try:
                    pose = self.calc_gripper_pose(
                        port_transform,
                        slerp_fraction=interp_fraction,
                        position_fraction=interp_fraction,
                        z_offset=z_offset,
                        reset_xy_integrator=True,
                    )
                    self._move_and_record(move_robot=move_robot, pose=pose, z_offset=z_offset)
                except TransformException as ex:
                    self.get_logger().warn(f"TF lookup failed during interpolation: {ex}")
                self.sleep_for(0.05)

            # Descend until the cable is inserted.
            while True:
                if z_offset < -0.015:
                    break
                z_offset -= 0.0005
                self.get_logger().info(f"z_offset: {z_offset:0.5}")
                try:
                    pose = self.calc_gripper_pose(port_transform, z_offset=z_offset)
                    self._move_and_record(move_robot=move_robot, pose=pose, z_offset=z_offset)
                except TransformException as ex:
                    self.get_logger().warn(f"TF lookup failed during insertion: {ex}")
                self.sleep_for(0.05)

            self.get_logger().info("Waiting for connector to stabilize...")
            self.sleep_for(5.0)

            self.get_logger().info("DataCollect.insert_cable() exiting...")
            success = True
            return True
        finally:
            # Always dump whatever we collected, even on early return / exception,
            # so partial trajectories are still usable for offline analysis.
            try:
                self._save_episode(task, success=success)
            except Exception as ex:
                self.get_logger().error(f"DataCollect: _save_episode failed: {ex}")
            self._get_observation = None
