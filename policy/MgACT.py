#
#  Copyright (C) 2026 Parth Maradia
#  Licensed under the Apache License, Version 2.0
#
"""MG-ACT v2 inference policy.

Loads a checkpoint trained by `mg_act/train_mg_act_v2.ipynb` and drives the AIC
robot through the standard `aic_model` lifecycle.

Heavy imports (torch, torchvision) are deferred to `MgACT.__init__` so module
discovery doesn't blow the AIC framework's 30-second budget.

Environment variables:
  AIC_MGACT_CHECKPOINT       Path to the .pt checkpoint.
  AIC_MGACT_DEVICE           'cuda' or 'cpu'. Default: cuda if available.
  AIC_MGACT_DTYPE            'bf16' / 'fp16' / 'fp32'. Default: bf16 cuda, fp32 cpu.
  AIC_MGACT_IMAGE_SCALE      Camera pre-resize scale before model resize. Default 0.25
                             for the original dataset; use 0.5 for DataCollectv2.
  AIC_MGACT_TE               '1' enable temporal ensembling (default), '0' chunk[0] only.
  AIC_MGACT_TE_M             TE exponential weight. Default 0.01 (per ACT paper).
  AIC_MGACT_CONTROL_HZ       Fixed control-loop rate in Hz. Default 16.7 (matches training).
  AIC_MGACT_REPLAN_EVERY     Re-run inference every N control steps. Default 3 — between
                             replans the loop advances through the TE-blended chunk, so
                             control commands always go out at CONTROL_HZ regardless of
                             how slow inference is (fixes the rate mismatch).
  AIC_MGACT_SUPERVISOR       '1' enable the phase supervisor (default), '0' raw model output.
  AIC_MGACT_Z_FLOOR          Workspace z floor in base_link. Default -0.015.
  AIC_MGACT_TASK_Z_FLOOR     '1' choose terminal z by task kind from demo priors. Default 1.
  AIC_MGACT_TASK_XY_PRIOR    '1' steer ALIGN/SETTLE xy to demo terminal priors. Default 1.
  AIC_MGACT_TASK_XY_ALPHA    Blend toward task xy prior in ALIGN/SETTLE. Default 0.25.
  AIC_MGACT_SFP_Z_FLOOR_0    SFP terminal z for nic_card_mount_0. Default 0.061.
  AIC_MGACT_SFP_Z_FLOOR_1    SFP terminal z for nic_card_mount_1. Default 0.064.
  AIC_MGACT_SC_Z_FLOOR       SC terminal z. Default -0.021.
  AIC_MGACT_Z_ALIGN          z below this -> enter ALIGN phase. Default 0.12.
  AIC_MGACT_Z_RATE_APPROACH  Supervisor-owned z descent rate (APPROACH/ALIGN). Default 0.0005 m/step.
  AIC_MGACT_Z_RATE_SETTLE    z descent per step in SETTLE. Default 0.00025 m/step.
  AIC_MGACT_XY_RATE_MAX      Cap on the model's xy velocity. Default 0.015 m/step.
  AIC_MGACT_XY_DAMP          xy low-pass coefficient in ALIGN. Default 0.85.
  AIC_MGACT_SETTLE_XY_DAMP   xy low-pass coefficient in SETTLE. Default 0.92.
  AIC_MGACT_SEARCH_RADIUS_MAX Expanding terminal xy search radius. Default 0.0 m.
  AIC_MGACT_SEARCH_START_STEPS Floor steps before search starts. Default 20.
  AIC_MGACT_SEARCH_PERIOD_STEPS Steps per search revolution. Default 180.
  AIC_MGACT_CONTACT_F        Baseline-subtracted |F_z| above this -> ALIGN->SETTLE. Default 8.0 N.
  AIC_MGACT_FORCE_MAX        Baseline-subtracted |F_z| above this -> retract z. Default 18.0 N.
  AIC_MGACT_FORCE_ABS_MAX    Raw |F_z| emergency retract threshold. Default 45.0 N.
  AIC_MGACT_STUCK_STEPS      No z progress for this many steps -> retract big & re-approach. Default 120.
  AIC_MGACT_Z_PROGRESS_EPS   Minimum z progress per step for stuck recovery. Default 0.0001.
  AIC_MGACT_SETTLE_KXY_MAX   Cap lateral stiffness in SETTLE. Default 90.0 N/m.
  AIC_MGACT_SETTLE_KZ_MAX    Cap vertical stiffness in SETTLE. Default 30.0 N/m.
  AIC_MGACT_SETTLE_D_MAX     Cap damping in SETTLE. Default 40.0.
  AIC_MGACT_END_ON_SETTLE_FLOOR End trial after reaching z floor. Default 1.
  AIC_MGACT_TERMINAL_SERVO   '1' enable 1x terminal crop residual servo. Default 0.
  AIC_MGACT_TERMINAL_CKPT    Path to terminal_servo_1x checkpoint.
  AIC_MGACT_TERMINAL_CROP    1x terminal crop size. Default 640.
  AIC_MGACT_TERMINAL_STEP_XY Per-step local xy correction cap. Default 0.001 m.
  AIC_MGACT_TERMINAL_STEP_Z  Per-step local z correction cap. Default 0.0005 m.
  AIC_MGACT_TERMINAL_MAX_TOTAL Max accumulated terminal correction. Default 0.025 m.
  AIC_MGACT_TERMINAL_SCAN    '1' run short high-Z confidence scan. Default 1.
"""
from __future__ import annotations

import math
import os
from collections import deque

import cv2
import numpy as np

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class _TemporalEnsembler:
    """ACT-style temporal ensembling in the network's 21-D action space.

    On each inference at step `t` we add the predicted (k, 21) chunk to a buffer.
    To get the action for step `t`, we average all stored chunks where row `i`
    corresponds to that step (i.e. the chunk was predicted `i` steps ago), with
    weight exp(-m * i). Older chunks fall out of the window after k steps.
    """

    def __init__(self, k: int, m: float = 0.01) -> None:
        self.k = k
        self.m = m
        self._buf: deque = deque(maxlen=k)  # (t_predicted, np.ndarray (k, 21))

    def reset(self) -> None:
        self._buf.clear()

    def add(self, t_step: int, chunk_21d: np.ndarray) -> None:
        # Drop chunks too old to contribute to the current or any future step
        while self._buf and (t_step - self._buf[0][0]) >= self.k:
            self._buf.popleft()
        self._buf.append((t_step, chunk_21d.astype(np.float32, copy=True)))

    def get(self, t_step: int) -> np.ndarray | None:
        if not self._buf:
            return None
        weights = []
        actions = []
        for t_at, chunk in self._buf:
            i = t_step - t_at
            if 0 <= i < self.k:
                weights.append(math.exp(-self.m * i))
                actions.append(chunk[i])
        if not actions:
            return None
        w = np.asarray(weights, dtype=np.float32)
        w /= w.sum()
        return np.einsum("i,ij->j", w, np.stack(actions))


class _PhaseSupervisor:
    """Thin state machine wrapped around the policy's raw pose output.

    The MG-ACT model learned CheatCode demos, which always had xy aligned before
    descending. At inference (vision-only) xy is slightly off, so when the arm
    descends it lands in states the model never saw and "bounces" back up — wasting
    the trial. This supervisor re-imposes structure: APPROACH -> ALIGN -> SETTLE.

    Design after the v1 post-mortem (it got stuck holding z at 17 N forever):
      - The supervisor OWNS the z descent. It descends at a fixed rate; it does
        NOT follow the model's bouncy z. The model only influences xy.
      - ONE force threshold `force_max`: above it, retract z (in any phase);
        below it, keep descending. No "hold on moderate force" dead zone.
      - Stuck recovery: if z hasn't progressed for `stuck_steps`, retract well
        above the jam and reset to APPROACH so the model re-aims xy.

    Phases:
      APPROACH : supervisor descends z at z_rate_approach; model owns xy but
                 rate-limited (can't drift wildly into walls).
                 -> ALIGN when z drops below z_align_enter.
      ALIGN    : xy low-passed toward the model (slow correction OK, no
                 oscillation); supervisor keeps descending z at z_rate_approach.
                 -> SETTLE when |F_z| > contact_force OR z near floor.
      SETTLE   : xy held at the anchor; z descends slowly at z_rate_settle;
                 impedance from the model.
    """

    APPROACH, ALIGN, SETTLE = "APPROACH", "ALIGN", "SETTLE"

    def __init__(
        self,
        z_floor: float = -0.015,
        z_align_enter: float = 0.12,
        z_rate_approach: float = 0.0005,  # supervisor-owned descent rate, m/step
        z_rate_settle: float = 0.00025,   # slower descent once in contact
        z_retract: float = 0.006,         # back-off per step when force is high
        z_retract_big: float = 0.06,      # big retract on stuck-recovery
        xy_rate_max: float = 0.015,       # cap on model's xy velocity, m/step
        xy_damp_alpha: float = 0.85,      # xy low-pass in ALIGN (0=free, 1=frozen)
        settle_xy_damp_alpha: float = 0.92,
        search_radius_max: float = 0.0,
        search_start_steps: int = 20,
        search_period_steps: int = 180,
        contact_force: float = 8.0,       # baseline-subtracted |F_z| -> SETTLE
        force_max: float = 18.0,          # baseline-subtracted |F_z| -> retract
        force_abs_max: float = 45.0,      # raw |F_z| emergency retract guard
        stuck_steps: int = 120,           # no z progress for this many steps -> recover
        z_progress_eps: float = 1e-4,     # min z drop per step to count as "progress"
    ) -> None:
        self.z_floor = z_floor
        self.z_align_enter = z_align_enter
        self.z_rate_approach = z_rate_approach
        self.z_rate_settle = z_rate_settle
        self.z_retract = z_retract
        self.z_retract_big = z_retract_big
        self.xy_rate_max = xy_rate_max
        self.xy_damp_alpha = xy_damp_alpha
        self.settle_xy_damp_alpha = settle_xy_damp_alpha
        self.search_radius_max = search_radius_max
        self.search_start_steps = search_start_steps
        self.search_period_steps = max(20, search_period_steps)
        self.contact_force = contact_force
        self.force_max = force_max
        self.force_abs_max = force_abs_max
        self.stuck_steps = stuck_steps
        self.z_progress_eps = z_progress_eps
        self.reset()

    def reset(self) -> None:
        self.phase = self.APPROACH
        self.last_z = None        # last commanded z
        self.last_xy = None       # last commanded (x, y) — for rate limiting
        self.z_floor_seen = float("inf")  # lowest z commanded so far
        self.xy_anchor = None     # (x, y) when ALIGN entered; low-pass updated after
        self.terminal_xy = None
        self.terminal_xy_alpha = 0.25
        self.steps_since_progress = 0
        self.z_at_progress_check = None
        self.settle_floor_steps = 0

    def seed_z(self, z_current: float | None, z_raw: float) -> None:
        """Initialize supervisor z from the measured TCP pose when available."""
        if self.last_z is not None:
            return
        if z_current is not None and math.isfinite(z_current) and 0.02 <= z_current <= 0.80:
            self.last_z = max(float(z_current), self.z_floor)
        else:
            self.last_z = max(float(z_raw), self.z_floor)
        self.z_at_progress_check = self.last_z

    def _rate_limited_xy(self, x_raw, y_raw):
        """Cap how far xy can move per step (prevents the model drifting into walls)."""
        if self.last_xy is None:
            return x_raw, y_raw
        lx, ly = self.last_xy
        dx, dy = x_raw - lx, y_raw - ly
        d = (dx * dx + dy * dy) ** 0.5
        if d > self.xy_rate_max and d > 1e-9:
            s = self.xy_rate_max / d
            return lx + dx * s, ly + dy * s
        return x_raw, y_raw

    def set_terminal_xy(self, xy: tuple[float, float] | None, alpha: float = 0.25) -> None:
        self.terminal_xy = xy
        self.terminal_xy_alpha = max(0.0, min(1.0, float(alpha)))

    def _prior_blend(self, x_raw: float, y_raw: float) -> tuple[float, float]:
        if self.terminal_xy is None or self.terminal_xy_alpha <= 0.0:
            return x_raw, y_raw
        px, py = self.terminal_xy
        a = self.terminal_xy_alpha
        return (1.0 - a) * x_raw + a * px, (1.0 - a) * y_raw + a * py

    def step(self, x_raw, y_raw, z_raw, fz_delta, fz_abs_raw, t_step):
        """Apply phase logic. Returns (x_cmd, y_cmd, z_cmd, phase, info_str)."""
        if self.last_z is None:
            self.last_z = z_raw
        if self.z_at_progress_check is None:
            self.z_at_progress_check = self.last_z

        high_force = fz_delta > self.force_max or fz_abs_raw > self.force_abs_max
        recovered = False

        # ---------- z command (supervisor-owned) ----------
        if high_force:
            # Too much axial resistance — back off to unload, regardless of phase.
            z_cmd = self.last_z + self.z_retract
            z_cmd = min(z_cmd, self.z_floor_seen + 5.0 * self.z_retract)  # bounded retract
        elif self.phase == self.SETTLE:
            z_cmd = max(self.last_z - self.z_rate_settle, self.z_floor)
        else:  # APPROACH or ALIGN
            z_cmd = max(self.last_z - self.z_rate_approach, self.z_floor)

        # ---------- xy command ----------
        if self.phase == self.APPROACH:
            x_cmd, y_cmd = self._rate_limited_xy(x_raw, y_raw)  # model owns xy, rate-limited
        elif self.phase == self.ALIGN:
            ax, ay = self.xy_anchor
            x_raw, y_raw = self._prior_blend(x_raw, y_raw)
            tx, ty = self._rate_limited_xy(x_raw, y_raw)
            a = self.xy_damp_alpha
            x_cmd = a * ax + (1.0 - a) * tx
            y_cmd = a * ay + (1.0 - a) * ty
            self.xy_anchor = (x_cmd, y_cmd)
        else:  # SETTLE
            # Keep correcting gently toward the raw model target. Freezing xy at
            # the ALIGN anchor was stable, but it also froze centimeter-scale
            # visual errors that are fatal for insertion.
            ax, ay = self.xy_anchor
            x_raw, y_raw = self._prior_blend(x_raw, y_raw)
            tx, ty = self._rate_limited_xy(x_raw, y_raw)
            a = self.settle_xy_damp_alpha
            cx = a * ax + (1.0 - a) * tx
            cy = a * ay + (1.0 - a) * ty
            self.xy_anchor = (cx, cy)

            at_floor = z_cmd <= self.z_floor + 1e-4
            self.settle_floor_steps = self.settle_floor_steps + 1 if at_floor else 0
            if at_floor and self.search_radius_max > 0.0 and self.settle_floor_steps >= self.search_start_steps:
                n = self.settle_floor_steps - self.search_start_steps
                grow = min(1.0, n / float(2 * self.search_period_steps))
                radius = self.search_radius_max * math.sqrt(grow)
                theta = 2.0 * math.pi * n / float(self.search_period_steps)
                x_cmd = cx + radius * math.cos(theta)
                y_cmd = cy + radius * math.sin(theta)
            else:
                x_cmd, y_cmd = cx, cy

        # ---------- phase transitions ----------
        if self.phase == self.APPROACH and z_cmd <= self.z_align_enter:
            self.phase = self.ALIGN
            self.xy_anchor = (x_cmd, y_cmd)
        elif self.phase == self.ALIGN and (fz_delta > self.contact_force or z_cmd <= self.z_floor + 0.005):
            self.phase = self.SETTLE

        # ---------- stuck recovery ----------
        # Every `stuck_steps` steps, check whether z has dropped. If not, the arm
        # is jammed (bad xy anchor, obstacle, etc.) — retract big and re-approach.
        if (t_step % self.stuck_steps == 0) and t_step > 0:
            at_settle_floor = self.phase == self.SETTLE and z_cmd <= self.z_floor + 1e-4
            if (not at_settle_floor) and (self.z_at_progress_check - z_cmd) < self.z_progress_eps * self.stuck_steps:
                # No meaningful progress over the window -> recover.
                z_cmd = min(self.last_z + self.z_retract_big, 0.30)  # back up high
                self.phase = self.APPROACH
                self.xy_anchor = None
                recovered = True
            self.z_at_progress_check = z_cmd

        self.z_floor_seen = min(self.z_floor_seen, z_cmd)
        self.last_z = z_cmd
        self.last_xy = (x_cmd, y_cmd)
        flags = (" BRAKE" if high_force else "") + (" RECOVER" if recovered else "")
        info = f"phase={self.phase}{flags} z_floor_seen={self.z_floor_seen:+.3f}"
        return x_cmd, y_cmd, z_cmd, self.phase, info


class MgACT(Policy):
    """MG-ACT v2 lifecycle policy.

    The class body and module imports stay light — torch / torchvision are
    pulled in inside `__init__` after the AIC framework has activated the node.
    """

    def __init__(self, parent_node):
        super().__init__(parent_node)

        # ------------------------------------------------------------- deferred heavy imports
        import torch  # noqa: WPS433
        from aic_example_policies.ros.mg_act_v2_model import (
            MGActV2,
            MGActV2Config,
            encode_task_dict,
            network_action_to_ros_payload,
        )

        self._torch = torch
        self._network_action_to_ros_payload = network_action_to_ros_payload
        self._encode_task_dict = encode_task_dict

        # ------------------------------------------------------------- device + dtype
        device_str = os.environ.get("AIC_MGACT_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        self._device = torch.device(device_str)
        dtype_str = os.environ.get("AIC_MGACT_DTYPE", "bf16" if self._device.type == "cuda" else "fp32")
        self._dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_str]
        self.get_logger().info(f"MgACT: device={self._device} dtype={self._dtype}")

        # ------------------------------------------------------------- checkpoint
        ckpt_path = os.environ.get(
            "AIC_MGACT_CHECKPOINT",
            # "/home/ubuntu/ws_aic/mg_act/mg_act_v2_last.pt",
            # "/home/ubuntu/ws_aic/mg_act/mg_act_v2_best.pt",
            "/home/ubuntu/ws_aic/mg_act/mg_act_v2_e11.pt",
        )
        self.get_logger().info(f"MgACT: loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self._device, weights_only=False)
        cfg_dict = ckpt["cfg"]
        # The cfg dict written by the notebook stores `cam_size` as a list after JSON
        # round-trip in some loaders; normalise to a tuple for the dataclass.
        if isinstance(cfg_dict.get("cam_size"), list):
            cfg_dict["cam_size"] = tuple(cfg_dict["cam_size"])
        # MGActV2Config gained task-conditioning fields in v3. Old checkpoints
        # don't have them — fall back to dataclass defaults for any missing keys.
        valid_keys = set(MGActV2Config.__dataclass_fields__.keys())
        cfg_dict_clean = {k: v for k, v in cfg_dict.items() if k in valid_keys}
        self._cfg = MGActV2Config(**cfg_dict_clean)
        self.get_logger().info(
            f"MgACT: cfg cam_size={self._cfg.cam_size} chunk={self._cfg.chunk_size} "
            f"haptic_T={self._cfg.haptic_window} proprio={self._cfg.proprio_dim} "
            f"task_cond={self._cfg.use_task_conditioning}"
        )

        # Task vocabulary — saved by the v3 training notebook alongside the
        # weights. Pre-v3 checkpoints don't have it; fall back to a permissive
        # vocab where every observed string maps to <UNK> (=0). If the model
        # was trained without task conditioning, this is harmless because
        # cfg.use_task_conditioning will be False.
        self._task_vocab = ckpt.get("task_vocab", {
            "module_class": {"<UNK>": 0},
            "port_class": {"<UNK>": 0},
            "plug_class": {"<UNK>": 0},
        })
        self.get_logger().info(
            f"MgACT: task vocab — module={list(self._task_vocab['module_class'].keys())} "
            f"port={list(self._task_vocab['port_class'].keys())} "
            f"plug={list(self._task_vocab['plug_class'].keys())}"
        )

        # ------------------------------------------------------------- model
        self._model = MGActV2(self._cfg).to(self._device)
        missing, unexpected = self._model.load_state_dict(ckpt["model_state"], strict=False)
        if missing:
            self.get_logger().warn(f"MgACT: missing keys when loading state_dict: {missing[:6]}{'...' if len(missing) > 6 else ''}")
        if unexpected:
            self.get_logger().warn(f"MgACT: unexpected keys when loading state_dict: {unexpected[:6]}{'...' if len(unexpected) > 6 else ''}")
        self._model.eval()
        # Keep weights in fp32; use torch.amp.autocast for low-precision activations.
        # This matches the training pattern in train_mg_act_v2.ipynb cell 25.
        self._use_autocast = self._device.type == "cuda" and self._dtype != torch.float32
        self._image_scale = float(os.environ.get("AIC_MGACT_IMAGE_SCALE", "0.25"))
        self._terminal_servo_requested = os.environ.get("AIC_MGACT_TERMINAL_SERVO", "0") != "0"
        self.get_logger().info(f"MgACT: image_scale={self._image_scale}")

        # ------------------------------------------------------------- preallocated input tensors
        H, W = self._cfg.cam_size  # 224, 224
        # Inputs stay fp32 — autocast handles the cast on entry to compute ops.
        self._img_buf = torch.empty(1, self._cfg.n_cameras, 3, H, W, device=self._device, dtype=torch.float32)
        self._mean = torch.tensor(_IMAGENET_MEAN, device=self._device, dtype=torch.float32).view(1, 3, 1, 1)
        self._std = torch.tensor(_IMAGENET_STD, device=self._device, dtype=torch.float32).view(1, 3, 1, 1)

        # ------------------------------------------------------------- temporal ensembling
        te_on = os.environ.get("AIC_MGACT_TE", "1") != "0"
        te_m = float(os.environ.get("AIC_MGACT_TE_M", "0.01"))
        self._te = _TemporalEnsembler(self._cfg.chunk_size, m=te_m) if te_on else None
        self.get_logger().info(f"MgACT: temporal_ensembling={'on (m=%.3f)' % te_m if te_on else 'off'}")

        # ------------------------------------------------------------- control rate + chunk playback
        # The control loop runs at a FIXED rate (matches training's 16.7 Hz) regardless
        # of inference speed. Inference happens every `replan_every` control steps; the
        # in-between steps just advance through the most recent action chunk (TE-blended).
        # This decouples control rate from inference rate -> no rate mismatch.
        control_hz = float(os.environ.get("AIC_MGACT_CONTROL_HZ",
                                          os.environ.get("AIC_MGACT_LOOP_HZ", "16.7")))
        self._control_dt = 1.0 / control_hz
        self._replan_every = max(1, int(os.environ.get("AIC_MGACT_REPLAN_EVERY", "3")))
        self.get_logger().info(
            f"MgACT: control_hz={control_hz:.1f} replan_every={self._replan_every} "
            f"(inference runs ~every {self._replan_every * self._control_dt * 1000:.0f}ms)"
        )

        # ------------------------------------------------------------- phase supervisor
        sup_on = os.environ.get("AIC_MGACT_SUPERVISOR", "1") != "0"
        if sup_on:
            self._supervisor = _PhaseSupervisor(
                z_floor=float(os.environ.get("AIC_MGACT_Z_FLOOR", "-0.015")),
                z_align_enter=float(os.environ.get("AIC_MGACT_Z_ALIGN", "0.12")),
                z_rate_approach=float(os.environ.get("AIC_MGACT_Z_RATE_APPROACH", "0.0005")),
                z_rate_settle=float(os.environ.get("AIC_MGACT_Z_RATE_SETTLE", "0.00025")),
                xy_rate_max=float(os.environ.get("AIC_MGACT_XY_RATE_MAX", "0.015")),
                xy_damp_alpha=float(os.environ.get("AIC_MGACT_XY_DAMP", "0.85")),
                settle_xy_damp_alpha=float(os.environ.get("AIC_MGACT_SETTLE_XY_DAMP", "0.92")),
                search_radius_max=float(os.environ.get("AIC_MGACT_SEARCH_RADIUS_MAX", "0.0")),
                search_start_steps=int(os.environ.get("AIC_MGACT_SEARCH_START_STEPS", "20")),
                search_period_steps=int(os.environ.get("AIC_MGACT_SEARCH_PERIOD_STEPS", "180")),
                contact_force=float(os.environ.get("AIC_MGACT_CONTACT_F", "8.0")),
                force_max=float(os.environ.get("AIC_MGACT_FORCE_MAX", "18.0")),
                force_abs_max=float(os.environ.get("AIC_MGACT_FORCE_ABS_MAX", "45.0")),
                stuck_steps=int(os.environ.get("AIC_MGACT_STUCK_STEPS", "120")),
                z_progress_eps=float(os.environ.get("AIC_MGACT_Z_PROGRESS_EPS", "0.0001")),
            )
            self._settle_kxy_max = float(os.environ.get("AIC_MGACT_SETTLE_KXY_MAX", "90.0"))
            self._settle_kz_max = float(os.environ.get("AIC_MGACT_SETTLE_KZ_MAX", "30.0"))
            self._settle_d_max = float(os.environ.get("AIC_MGACT_SETTLE_D_MAX", "40.0"))
            self._end_on_settle_floor = os.environ.get("AIC_MGACT_END_ON_SETTLE_FLOOR", "1") != "0"
            self._task_z_floor = os.environ.get("AIC_MGACT_TASK_Z_FLOOR", "1") != "0"
            # Terminal servo uses visual local corrections, so absolute xy priors
            # default off when the sidecar is enabled. Users can still override.
            default_task_xy_prior = "0" if self._terminal_servo_requested else "1"
            self._task_xy_prior = os.environ.get("AIC_MGACT_TASK_XY_PRIOR", default_task_xy_prior) != "0"
            self._task_xy_alpha = float(os.environ.get("AIC_MGACT_TASK_XY_ALPHA", "0.25"))
            self._sfp_z_floor_0 = float(os.environ.get("AIC_MGACT_SFP_Z_FLOOR_0", "0.061"))
            self._sfp_z_floor_1 = float(os.environ.get("AIC_MGACT_SFP_Z_FLOOR_1", "0.064"))
            self._sc_z_floor = float(os.environ.get("AIC_MGACT_SC_Z_FLOOR", "-0.021"))
            self.get_logger().info(
                f"MgACT: phase supervisor ON — z_align={self._supervisor.z_align_enter} "
                f"z_rate_approach={self._supervisor.z_rate_approach} z_rate_settle={self._supervisor.z_rate_settle} "
                f"xy_rate_max={self._supervisor.xy_rate_max} xy_damp={self._supervisor.xy_damp_alpha} "
                f"settle_xy_damp={self._supervisor.settle_xy_damp_alpha} "
                f"search_radius={self._supervisor.search_radius_max} "
                f"contact_F_delta={self._supervisor.contact_force} force_max_delta={self._supervisor.force_max} "
                f"force_abs_max={self._supervisor.force_abs_max} stuck_steps={self._supervisor.stuck_steps} "
                f"z_progress_eps={self._supervisor.z_progress_eps} "
                f"settle_Kxy_max={self._settle_kxy_max} settle_Kz_max={self._settle_kz_max} "
                f"settle_D_max={self._settle_d_max} end_on_floor={self._end_on_settle_floor} "
                f"task_z_floor={self._task_z_floor} task_xy_prior={self._task_xy_prior}"
            )
        else:
            self._supervisor = None
            self._settle_kxy_max = None
            self._settle_kz_max = None
            self._settle_d_max = None
            self._end_on_settle_floor = False
            self._task_z_floor = False
            self._task_xy_prior = False
            self.get_logger().info("MgACT: phase supervisor OFF (raw model output)")

        self._init_terminal_servo()
        self._warmup()
        self.get_logger().info("MgACT: ready")

    def _terminal_pose_prior_for_task(self, task_msg) -> tuple[tuple[float, float] | None, float | None]:
        """Return demo-derived terminal xy/z priors for the requested connector."""
        port_name = str(getattr(task_msg, "port_name", ""))
        module_name = str(getattr(task_msg, "target_module_name", ""))
        if port_name.startswith("sfp_port"):
            module_1 = module_name.endswith("_1")
            port_1 = port_name.endswith("_1")
            if module_1:
                xy = (-0.4142, 0.2446) if port_1 else (-0.3908, 0.2470)
                z_floor = self._sfp_z_floor_1
            else:
                xy = (-0.4126, 0.2079) if port_1 else (-0.3888, 0.2082)
                z_floor = self._sfp_z_floor_0
            return xy, z_floor
        if port_name.startswith("sc_port") or module_name.startswith("sc_port"):
            return (-0.4849, 0.2859), self._sc_z_floor
        return None, None

    def _init_terminal_servo(self) -> None:
        """Load the optional 1x terminal crop residual servo."""
        self._terminal_servo = None
        self._terminal_cfg = None
        self._terminal_vocab = None
        self._terminal_task_tensors_from_msg = None
        self._terminal_observation_crops = None
        self._terminal_crops_to_tensor = None
        self._terminal_local_to_base_delta = None
        self._terminal_last_pred = None
        self._terminal_last_info = "terminal=off"
        self._terminal_offset_base = np.zeros(3, dtype=np.float64)
        self._terminal_scan_offsets_local = []
        self._terminal_scan_best_conf = -1.0
        self._terminal_scan_best_offset = np.zeros(3, dtype=np.float64)
        self._terminal_scan_last_offset = np.zeros(3, dtype=np.float64)
        self._terminal_scan_idx = 0
        self._terminal_scan_done = False

        if not self._terminal_servo_requested:
            self.get_logger().info("MgACT: terminal servo OFF")
            return

        ckpt_path = os.environ.get(
            "AIC_MGACT_TERMINAL_CKPT",
            "/home/ubuntu/ws_aic/mg_act/terminal_servo_1x_best.pt",
        )
        if not os.path.exists(ckpt_path):
            self.get_logger().warn(
                f"MgACT: terminal servo requested but checkpoint not found: {ckpt_path}"
            )
            return

        try:
            from aic_example_policies.ros.terminal_servo_model import (
                load_terminal_servo_checkpoint,
                observation_crops,
                crops_to_tensor,
                task_tensors_from_msg,
                local_to_base_delta,
            )

            model, cfg, vocab = load_terminal_servo_checkpoint(ckpt_path, self._device, self._dtype)
            self._terminal_servo = model
            self._terminal_cfg = cfg
            self._terminal_vocab = vocab
            self._terminal_task_tensors_from_msg = task_tensors_from_msg
            self._terminal_observation_crops = observation_crops
            self._terminal_crops_to_tensor = crops_to_tensor
            self._terminal_local_to_base_delta = local_to_base_delta
            self._terminal_crop = int(os.environ.get("AIC_MGACT_TERMINAL_CROP", str(cfg.crop_size)))
            self._terminal_step_xy = float(os.environ.get("AIC_MGACT_TERMINAL_STEP_XY", "0.001"))
            self._terminal_step_z = float(os.environ.get("AIC_MGACT_TERMINAL_STEP_Z", "0.0005"))
            self._terminal_max_total = float(os.environ.get("AIC_MGACT_TERMINAL_MAX_TOTAL", "0.025"))
            self._terminal_scan = os.environ.get("AIC_MGACT_TERMINAL_SCAN", "1") != "0"
            self._terminal_scan_radius = float(os.environ.get("AIC_MGACT_TERMINAL_SCAN_RADIUS", "0.008"))
            self._terminal_replan_every = max(1, int(os.environ.get("AIC_MGACT_TERMINAL_REPLAN_EVERY", "2")))
            r = self._terminal_scan_radius
            self._terminal_scan_offsets_local = [
                np.array([0.0, 0.0, 0.0], dtype=np.float64),
                np.array([r, 0.0, 0.0], dtype=np.float64),
                np.array([-r, 0.0, 0.0], dtype=np.float64),
                np.array([0.0, r, 0.0], dtype=np.float64),
                np.array([0.0, -r, 0.0], dtype=np.float64),
            ]
            self.get_logger().info(
                f"MgACT: terminal servo ON ckpt={ckpt_path} crop={self._terminal_crop} "
                f"step_xy={self._terminal_step_xy:.4f} step_z={self._terminal_step_z:.4f} "
                f"max_total={self._terminal_max_total:.3f} scan={self._terminal_scan}"
            )
        except Exception as ex:
            self._terminal_servo = None
            self.get_logger().warn(f"MgACT: failed to load terminal servo: {ex}")

    def _reset_terminal_runtime(self) -> None:
        self._terminal_last_pred = None
        self._terminal_last_info = "terminal=idle"
        self._terminal_offset_base = np.zeros(3, dtype=np.float64)
        self._terminal_scan_best_conf = -1.0
        self._terminal_scan_best_offset = np.zeros(3, dtype=np.float64)
        self._terminal_scan_last_offset = np.zeros(3, dtype=np.float64)
        self._terminal_scan_idx = 0
        self._terminal_scan_done = not bool(getattr(self, "_terminal_scan", False))

    def _build_terminal_task_tensors(self, task_msg):
        if self._terminal_servo is None:
            return None
        return self._terminal_task_tensors_from_msg(task_msg, self._terminal_vocab, self._device)

    def _terminal_predict(self, obs: Observation, terminal_task) -> dict | None:
        if self._terminal_servo is None or terminal_task is None:
            return None
        torch = self._torch
        try:
            crops = self._terminal_observation_crops(
                obs,
                self._terminal_crop,
                self._terminal_cfg.camera_names,
            )
            crops_t = self._terminal_crops_to_tensor(
                crops,
                self._device,
                self._terminal_cfg.imagenet_mean,
                self._terminal_cfg.imagenet_std,
            )
            wrench_t = torch.from_numpy(self._wrench_row(obs)).to(self._device).float().unsqueeze(0)
            proprio_t = torch.from_numpy(self._proprio_from_obs(obs)).to(self._device).float().unsqueeze(0)
            with torch.inference_mode():
                if self._use_autocast:
                    with torch.amp.autocast(device_type="cuda", dtype=self._dtype):
                        out = self._terminal_servo(crops_t, wrench_t, proprio_t, terminal_task)
                else:
                    out = self._terminal_servo(crops_t, wrench_t, proprio_t, terminal_task)
            residual = out["residual_local"][0].float().cpu().numpy().astype(np.float64)
            conf = float(torch.sigmoid(out["confidence_logit"])[0].float().cpu())
            hold = float(torch.sigmoid(out["hold_logit"])[0].float().cpu())
            return {"residual_local": residual, "confidence": conf, "hold": hold}
        except Exception as ex:
            self.get_logger().warn(f"MgACT: terminal servo inference failed: {ex}")
            return None

    def _clip_terminal_step(self, residual_local: np.ndarray) -> np.ndarray:
        step = np.asarray(residual_local, dtype=np.float64).copy()
        xy_norm = float(np.linalg.norm(step[:2]))
        if xy_norm > self._terminal_step_xy and xy_norm > 1e-12:
            step[:2] *= self._terminal_step_xy / xy_norm
        step[2] = float(np.clip(step[2], -self._terminal_step_z, self._terminal_step_z))
        return step

    def _clamp_terminal_total(self, offset_base: np.ndarray) -> np.ndarray:
        out = np.asarray(offset_base, dtype=np.float64).copy()
        xy_norm = float(np.linalg.norm(out[:2]))
        if xy_norm > self._terminal_max_total and xy_norm > 1e-12:
            out[:2] *= self._terminal_max_total / xy_norm
        out[2] = float(np.clip(out[2], -self._terminal_max_total, self._terminal_max_total))
        return out

    def _apply_terminal_servo(
        self,
        obs: Observation,
        terminal_task,
        x_cmd: float,
        y_cmd: float,
        z_cmd: float,
        quat_wxyz: np.ndarray,
        phase: str,
        step: int,
    ) -> tuple[float, float, float]:
        if self._terminal_servo is None or terminal_task is None:
            return x_cmd, y_cmd, z_cmd
        if phase not in (_PhaseSupervisor.ALIGN, _PhaseSupervisor.SETTLE):
            return x_cmd, y_cmd, z_cmd

        if self._terminal_last_pred is None or (step % self._terminal_replan_every == 0):
            pred = self._terminal_predict(obs, terminal_task)
            if pred is not None:
                self._terminal_last_pred = pred
        pred = self._terminal_last_pred
        if pred is None:
            return x_cmd, y_cmd, z_cmd

        conf = float(pred["confidence"])
        hold = float(pred["hold"])

        if self._terminal_scan and not self._terminal_scan_done and phase == _PhaseSupervisor.ALIGN:
            if self._terminal_scan_idx > 0 and conf > self._terminal_scan_best_conf:
                self._terminal_scan_best_conf = conf
                self._terminal_scan_best_offset = self._terminal_scan_last_offset.copy()

            if self._terminal_scan_idx < len(self._terminal_scan_offsets_local):
                local_offset = self._terminal_scan_offsets_local[self._terminal_scan_idx]
                base_offset = self._terminal_local_to_base_delta(quat_wxyz, local_offset)
                self._terminal_scan_last_offset = base_offset.astype(np.float64)
                self._terminal_scan_idx += 1
                z_hold = z_cmd
                if self._supervisor is not None:
                    z_hold = max(z_hold, self._supervisor.z_floor + 0.035)
                self._terminal_last_info = (
                    f"term_scan={self._terminal_scan_idx}/{len(self._terminal_scan_offsets_local)} "
                    f"conf={conf:.2f}"
                )
                return x_cmd + float(base_offset[0]), y_cmd + float(base_offset[1]), z_hold

            self._terminal_scan_done = True
            self._terminal_offset_base = self._clamp_terminal_total(self._terminal_scan_best_offset)

        step_local = self._clip_terminal_step(pred["residual_local"])
        step_base = self._terminal_local_to_base_delta(quat_wxyz, step_local)
        self._terminal_offset_base = self._clamp_terminal_total(self._terminal_offset_base + step_base)
        x_cmd += float(self._terminal_offset_base[0])
        y_cmd += float(self._terminal_offset_base[1])
        z_cmd += float(self._terminal_offset_base[2])
        self._terminal_last_info = (
            f"term conf={conf:.2f} hold={hold:.2f} "
            f"res=({pred['residual_local'][0]*1000:+.1f},"
            f"{pred['residual_local'][1]*1000:+.1f},"
            f"{pred['residual_local'][2]*1000:+.1f})mm "
            f"off=({self._terminal_offset_base[0]*1000:+.1f},"
            f"{self._terminal_offset_base[1]*1000:+.1f},"
            f"{self._terminal_offset_base[2]*1000:+.1f})mm"
        )
        return x_cmd, y_cmd, z_cmd

    # ------------------------------------------------------------- helpers

    def _warmup(self) -> None:
        torch = self._torch
        H, W = self._cfg.cam_size
        imgs = torch.zeros(1, self._cfg.n_cameras, 3, H, W, device=self._device)
        wrench = torch.zeros(1, self._cfg.haptic_window, 6, device=self._device)
        proprio = torch.zeros(1, self._cfg.proprio_dim, device=self._device)
        # Build a dummy task (all zeros = <UNK> for every field) so the task
        # branch's autocast graph also gets compiled during warmup.
        if self._cfg.use_task_conditioning:
            dummy_task = {
                "module_class": torch.zeros(1, dtype=torch.long, device=self._device),
                "module_idx": torch.zeros(1, dtype=torch.float32, device=self._device),
                "port_class": torch.zeros(1, dtype=torch.long, device=self._device),
                "port_idx": torch.zeros(1, dtype=torch.float32, device=self._device),
                "plug_class": torch.zeros(1, dtype=torch.long, device=self._device),
            }
        else:
            dummy_task = None
        with torch.inference_mode():
            for _ in range(2):
                self._predict(imgs, wrench, proprio, task=dummy_task)
            if self._device.type == "cuda":
                torch.cuda.synchronize()

    def _predict(self, imgs, wrench, proprio, task=None):
        """One forward pass under autocast (matches train_mg_act_v2.ipynb)."""
        torch = self._torch
        if self._use_autocast:
            with torch.amp.autocast(device_type="cuda", dtype=self._dtype):
                return self._model.predict(imgs, wrench, proprio, task=task)
        return self._model.predict(imgs, wrench, proprio, task=task)

    def _build_task_tensors(self, task_msg):
        """Convert a Task ROS message into the dict-of-tensors the model expects.

        Returns None if task conditioning is disabled in cfg (older checkpoints).
        """
        if not self._cfg.use_task_conditioning:
            return None
        torch = self._torch
        task_strings = {
            "target_module_name": str(getattr(task_msg, "target_module_name", "")),
            "port_name": str(getattr(task_msg, "port_name", "")),
            "plug_name": str(getattr(task_msg, "plug_name", "")),
        }
        ids = self._encode_task_dict(task_strings, self._task_vocab)
        return {
            "module_class": torch.tensor([ids["module_class"]], dtype=torch.long, device=self._device),
            "module_idx": torch.tensor([ids["module_idx"]], dtype=torch.float32, device=self._device),
            "port_class": torch.tensor([ids["port_class"]], dtype=torch.long, device=self._device),
            "port_idx": torch.tensor([ids["port_idx"]], dtype=torch.float32, device=self._device),
            "plug_class": torch.tensor([ids["plug_class"]], dtype=torch.long, device=self._device),
        }

    def _img_msg_to_tensor(self, img_msg, cam_idx: int) -> None:
        """Decode a sensor_msgs/Image, pre-scale, resize to model input, write image tensor."""
        torch = self._torch
        if img_msg is None or img_msg.height == 0 or img_msg.width == 0:
            self._img_buf[0, cam_idx].zero_()
            return
        arr = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, 3)
        # Step 1: INTER_AREA downscale matching the collector's AIC_IMAGE_SCALE.
        # Original MgACT data used 0.25; DataCollectv2 uses 0.5.
        if self._image_scale > 0 and abs(self._image_scale - 1.0) > 1e-6:
            arr = cv2.resize(
                arr,
                None,
                fx=self._image_scale,
                fy=self._image_scale,
                interpolation=cv2.INTER_AREA,
            )
        # Step 2: resize to model input size — matches torchvision T.Resize(cam_size) bilinear.
        H, W = self._cfg.cam_size
        if arr.shape[0] != H or arr.shape[1] != W:
            arr = cv2.resize(arr, (W, H), interpolation=cv2.INTER_LINEAR)
        # HWC uint8 -> CHW float in [0,1] -> normalize, on device. Inputs stay fp32;
        # autocast will cast to bf16 internally where appropriate.
        t = torch.from_numpy(arr).to(self._device).permute(2, 0, 1).float().div(255.0)
        t = (t.unsqueeze(0) - self._mean) / self._std
        self._img_buf[0, cam_idx].copy_(t.squeeze(0))

    def _proprio_from_obs(self, obs: Observation) -> np.ndarray:
        """Concatenate joint position[7] | velocity[7] | effort[7] = 21-D, padding/truncating defensively."""
        js = obs.joint_states
        def _take7(arr) -> np.ndarray:
            a = np.asarray(arr, dtype=np.float32)
            if a.size >= 7:
                return a[:7]
            out = np.zeros(7, dtype=np.float32)
            out[: a.size] = a
            return out
        return np.concatenate([_take7(js.position), _take7(js.velocity), _take7(js.effort)])

    @staticmethod
    def _wrench_row(obs: Observation) -> np.ndarray:
        w = obs.wrist_wrench.wrench
        return np.array(
            [w.force.x, w.force.y, w.force.z, w.torque.x, w.torque.y, w.torque.z],
            dtype=np.float32,
        )

    @staticmethod
    def _tcp_z_from_obs(obs: Observation) -> float | None:
        try:
            return float(obs.controller_state.tcp_pose.position.z)
        except Exception:
            return None

    def _build_inputs(
        self,
        obs: Observation,
        wrench_buf: np.ndarray,
    ):
        """Return (images_tensor, wrench_tensor, proprio_tensor) ready for `model.predict`."""
        torch = self._torch
        # Images
        self._img_msg_to_tensor(obs.left_image, 0)
        self._img_msg_to_tensor(obs.center_image, 1)
        self._img_msg_to_tensor(obs.right_image, 2)
        # Wrench (fp32; autocast handles activation dtype)
        wrench_t = torch.from_numpy(wrench_buf).to(self._device).float().unsqueeze(0)
        # Proprio
        proprio = self._proprio_from_obs(obs)
        proprio_t = torch.from_numpy(proprio).to(self._device).float().unsqueeze(0)
        return self._img_buf, wrench_t, proprio_t

    def _net_action_to_components(self, action_21d: np.ndarray):
        """Convert one 21-D network action -> (trans[3], quat_wxyz[4], K[6], D[6]).

        Runs entirely on CPU (the conversion is tiny matrix math) — avoids a
        GPU round-trip + sync per control step. `network_action_to_ros_payload`
        is device-agnostic torch ops, so a CPU tensor works fine here.
        """
        torch = self._torch
        a = torch.from_numpy(action_21d.astype(np.float32, copy=False)).unsqueeze(0)  # CPU
        payload = self._network_action_to_ros_payload(a, self._cfg)
        trans = payload["translation"][0].numpy()
        quat = payload["quaternion_wxyz"][0].numpy()
        K = payload["stiffness_diag"][0].numpy()
        D = payload["damping_diag"][0].numpy()
        return trans, quat, K, D

    # ------------------------------------------------------------- main entry

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        torch = self._torch
        self.get_logger().info(f"MgACT.insert_cable() task: {task}")

        if self._te is not None:
            self._te.reset()
        if self._supervisor is not None:
            self._supervisor.reset()
        self._reset_terminal_runtime()

        # Wait for a valid first observation, then prime the wrench buffer with it.
        obs = None
        for _ in range(50):
            obs = get_observation()
            if obs is not None:
                break
            self.sleep_for(self._control_dt)
        if obs is None:
            self.get_logger().error("MgACT: no observation received after 50 polls; aborting trial")
            return False

        # The wrist F/T stream can start with a large static load. Estimate a
        # per-trial baseline before sending commands, then use force deltas for
        # contact/retract decisions while keeping raw force as an emergency guard.
        baseline_rows = [self._wrench_row(obs)]
        for _ in range(max(1, self._cfg.haptic_window - 1)):
            self.sleep_for(self._control_dt)
            maybe_obs = get_observation()
            if maybe_obs is not None:
                obs = maybe_obs
                baseline_rows.append(self._wrench_row(obs))
        wrench_baseline = np.mean(np.stack(baseline_rows), axis=0).astype(np.float32)
        wrench_buf = np.tile(wrench_baseline, (self._cfg.haptic_window, 1)).astype(np.float32)
        self.get_logger().info(
            f"MgACT: wrench baseline F=({wrench_baseline[0]:+.1f},"
            f"{wrench_baseline[1]:+.1f},{wrench_baseline[2]:+.1f})N"
        )

        # Build task tensors once per trial — task identity is fixed for the trial.
        task_tensors = self._build_task_tensors(task)
        terminal_task_tensors = self._build_terminal_task_tensors(task)
        if task_tensors is not None:
            self.get_logger().info(
                f"MgACT: task encoded — module_class_id={int(task_tensors['module_class'].item())} "
                f"module_idx={float(task_tensors['module_idx'].item()):.0f} "
                f"port_class_id={int(task_tensors['port_class'].item())} "
                f"port_idx={float(task_tensors['port_idx'].item()):.0f} "
                f"plug_class_id={int(task_tensors['plug_class'].item())}"
            )
        if self._supervisor is not None:
            task_xy, task_z_floor = self._terminal_pose_prior_for_task(task)
            if task_z_floor is not None and getattr(self, "_task_z_floor", False):
                self._supervisor.z_floor = float(task_z_floor)
                self.get_logger().info(f"MgACT: task terminal z_floor={self._supervisor.z_floor:+.3f}")
            if task_xy is not None and getattr(self, "_task_xy_prior", False):
                self._supervisor.set_terminal_xy(task_xy, self._task_xy_alpha)
                self.get_logger().info(
                    f"MgACT: task terminal xy prior=({task_xy[0]:+.4f},{task_xy[1]:+.4f}) "
                    f"alpha={self._task_xy_alpha:.2f}"
                )

        # Trial-time budget — sim-time aware. Stop a beat early so the engine
        # doesn't kill us mid-publish.
        from rclpy.duration import Duration
        time_limit_safe = Duration(seconds=max(1.0, float(task.time_limit) - 1.5))
        control_dt_dur = Duration(seconds=self._control_dt)
        t_start = self.time_now()

        # ----- Fixed-rate control loop with chunk playback -----
        # Control commands go out every `control_dt` (matches training's 16.7 Hz).
        # Heavy inference happens every `replan_every` steps; between replans we
        # advance through the TE-blended chunk. This decouples control rate from
        # inference rate -> no rate mismatch even if inference is slow.
        step = 0
        last_chunk0 = None
        last_published = False
        n_inferences = 0
        wall_t0 = self.time_now()  # for measured-Hz reporting
        settle_floor_steps = 0

        while True:
            loop_start = self.time_now()
            elapsed = loop_start - t_start
            if elapsed > time_limit_safe:
                self.get_logger().info(f"MgACT: trial time budget reached at step={step}")
                break

            # Always grab the latest observation — the supervisor needs F_z every step.
            obs = get_observation()
            if obs is None:
                self.sleep_for(self._control_dt)
                continue
            # Refresh wrench ring buffer (used both for inference and supervisor feedback).
            wrench_buf = np.roll(wrench_buf, -1, axis=0)
            wrench_buf[-1] = self._wrench_row(obs)
            fz_raw = float(wrench_buf[-1, 2])
            fz_abs_raw = abs(fz_raw)
            fz_delta = abs(fz_raw - float(wrench_baseline[2]))

            # ---- Inference (only on replan steps) ----
            do_replan = (step % self._replan_every == 0) or (last_chunk0 is None)
            if do_replan:
                try:
                    imgs_t, wrench_t, proprio_t = self._build_inputs(obs, wrench_buf)
                    with torch.inference_mode():
                        pred = self._predict(imgs_t, wrench_t, proprio_t, task=task_tensors)  # (1, k, 21)
                    chunk_21d = pred[0].float().cpu().numpy()
                    last_chunk0 = chunk_21d[0]
                    if self._te is not None:
                        self._te.add(step, chunk_21d)
                    n_inferences += 1
                except Exception as ex:
                    self.get_logger().warn(f"MgACT: inference exception: {ex}")
                    self.sleep_for(self._control_dt)
                    continue

            # ---- Pick the raw action for this control step ----
            if self._te is not None:
                action_21d = self._te.get(step)
                if action_21d is None:
                    action_21d = last_chunk0
            else:
                # No TE: if we just replanned use chunk[0]; otherwise advance the
                # last chunk by (step - replan_step). Simpler: just hold chunk[0].
                action_21d = last_chunk0

            # ---- Convert + supervise + publish ----
            try:
                trans, quat, K, D = self._net_action_to_components(action_21d)
                if self._supervisor is not None:
                    self._supervisor.seed_z(self._tcp_z_from_obs(obs), float(trans[2]))
                    x_cmd, y_cmd, z_cmd, phase, _ = self._supervisor.step(
                        float(trans[0]), float(trans[1]), float(trans[2]), fz_delta, fz_abs_raw, step
                    )
                    if phase == _PhaseSupervisor.SETTLE and self._settle_kz_max is not None:
                        if self._settle_kxy_max is not None:
                            K[0] = min(float(K[0]), self._settle_kxy_max)
                            K[1] = min(float(K[1]), self._settle_kxy_max)
                        K[2] = min(float(K[2]), self._settle_kz_max)
                        if self._settle_d_max is not None:
                            D[:] = np.minimum(D, self._settle_d_max)
                    if phase == _PhaseSupervisor.SETTLE and z_cmd <= self._supervisor.z_floor + 1e-4:
                        settle_floor_steps += 1
                    else:
                        settle_floor_steps = 0
                else:
                    x_cmd, y_cmd, z_cmd, phase = float(trans[0]), float(trans[1]), float(trans[2]), "RAW"
                if self._terminal_servo is not None:
                    x_cmd, y_cmd, z_cmd = self._apply_terminal_servo(
                        obs,
                        terminal_task_tensors,
                        x_cmd,
                        y_cmd,
                        z_cmd,
                        quat,
                        phase,
                        step,
                    )
                pose = Pose(
                    position=Point(x=x_cmd, y=y_cmd, z=z_cmd),
                    orientation=Quaternion(w=float(quat[0]), x=float(quat[1]), y=float(quat[2]), z=float(quat[3])),
                )
                self.set_pose_target(
                    move_robot=move_robot,
                    pose=pose,
                    stiffness=K.tolist(),
                    damping=D.tolist(),
                )
                last_published = True
                if step % 20 == 0:
                    secs = max(1e-6, (loop_start - wall_t0).nanoseconds * 1e-9)
                    hz = (step + 1) / secs
                    self.get_logger().info(
                        f"MgACT step={step} {phase} elapsed={elapsed.nanoseconds*1e-9:.1f}s "
                        f"hz={hz:.1f} infers={n_inferences}  "
                        f"trans=({x_cmd:+.3f},{y_cmd:+.3f},{z_cmd:+.3f})  z_raw={float(trans[2]):+.3f}  "
                        f"Fz={fz_raw:+.1f}N dFz={fz_delta:.1f}N  K_z={K[2]:.0f} D_z={D[2]:.0f} "
                        f"{self._terminal_last_info}"
                    )
                if self._end_on_settle_floor and settle_floor_steps >= max(10, int(3.0 / self._control_dt)):
                    self.get_logger().info(
                        f"MgACT: settle floor reached for {settle_floor_steps} steps; ending trial"
                    )
                    break
            except Exception as ex:
                self.get_logger().warn(f"MgACT: publish exception: {ex}")

            if step == 0:
                send_feedback("MgACT inference loop active")

            step += 1
            # Sleep to maintain the fixed control rate (sim-time aware).
            spent = self.time_now() - loop_start
            remaining = control_dt_dur - spent
            if remaining.nanoseconds > 0:
                self.get_clock().sleep_for(remaining)

        self.get_logger().info(
            f"MgACT.insert_cable() exiting after {step} steps, {n_inferences} inferences"
        )
        return last_published
