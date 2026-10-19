"""Isaac Sim stand-in for the YAM arm servers and cameras, reached over one portal.

yamlab's ``scripts/aspire_bridge_server.py`` runs a YAM task in Isaac Sim and serves
both arms, the cameras, and seeded scene resets from one RPC port. This module is the
client side: :class:`IsaacYamArm` duck-types the per-arm follower client and
:class:`IsaacCamera` the RealSense reader, so :func:`build_isaac_station` returns the
same :class:`~cap_harness.yam_real.env.RealYamEnv` the bench uses -- the level-1
control path, kinematics and recording are unchanged; only the bottom of the stack
is swapped, as with the MuJoCo station in :mod:`cap_harness.yam_real.sim`.

What differs from hardware, and is not hidden:

- The bridge ignores ``kp``/``kd``. Isaac Lab's joint actuators track position
  targets with their own gains, so commanded gains are not a property of this plant.
- The scene *can* be restored to a seeded state, unlike a bench. ``IsaacYamEnv``
  exposes that as ``reset_scene`` and ``task_info``; the adapter calls them.
- The calibration bundle is written from the simulator's true camera pose
  (:func:`write_isaac_station_profile`), so ``base_from_camera`` is exact, not solved.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from cap_harness.contracts import Observation
from cap_harness.yam_real.adapter import YamRealAdapter
from cap_harness.yam_real.cameras import CameraFrame, ThreadedCamera
from cap_harness.yam_real.config import load_yam_station_config
from cap_harness.yam_real.env import ARMS, RealYamEnv
from cap_harness.yam_real.registry import YAM_REAL_TASKS, YamRealTaskMetadata, YamRealTaskRegistry

DEFAULT_BRIDGE_HOST = "127.0.0.1"
DEFAULT_BRIDGE_PORT = 11400

#: Camera roles in the ASPIRE station profile, keyed by the bridge's camera name.
CAMERA_ROLES = {"top": "top", "left_wrist": "wrist_left", "right_wrist": "wrist_right"}

#: How long a reset may take: the bridge also lets objects settle before replying.
RESET_TIMEOUT_S = 120.0

#: Floor for time dilation: the controller clamps playback speed at 0.05.
MIN_REALTIME_FACTOR = 0.05

#: Polling rates for the threaded camera readers.
TOP_CAMERA_FPS = 15.0
AUX_CAMERA_FPS = 5.0


class IsaacBridgeClient:
    """One portal connection to the bridge, shared by both arms and every camera.

    portal clients are not documented as thread-safe, and the level-1 loop, the
    camera threads and the recorder all call in concurrently, so calls are serialized.
    """

    def __init__(self, host: str = DEFAULT_BRIDGE_HOST, port: int = DEFAULT_BRIDGE_PORT) -> None:
        import portal

        self.host = str(host)
        self.port = int(port)
        self._client = portal.Client(f"{self.host}:{self.port}")
        self._lock = threading.Lock()

    def call(self, method: str, *args: Any, timeout: float | None = 10.0) -> Any:
        with self._lock:
            future = getattr(self._client, method)(*args)
        return future.result(timeout=timeout) if timeout is not None else future.result()

    def send(self, method: str, *args: Any) -> None:
        """Fire and forget, as the hardware transport sends commands."""
        with self._lock:
            getattr(self._client, method)(*args)

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if close is not None:
            close()


class IsaacYamArm:
    """Per-arm follower facade over the bridge."""

    def __init__(self, bridge: IsaacBridgeClient, side: str) -> None:
        if side not in ARMS:
            raise ValueError(f"side must be 'left' or 'right'; got {side!r}")
        self.bridge = bridge
        self.side = side

    def get_observations(self) -> dict[str, np.ndarray]:
        return self.bridge.call("get_arm_observations", self.side)

    def get_joint_pos(self) -> np.ndarray:
        obs = self.get_observations()
        return np.concatenate([obs["joint_pos"], obs["gripper_pos"]])

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        self.bridge.send("command_arm", self.side, np.asarray(joint_pos, dtype=np.float64))

    def command_joint_state(self, joint_state: dict[str, Any]) -> None:
        self.command_joint_pos(joint_state["pos"])

    def close(self) -> None:
        """Client-side close; the bridge outlives any one arm facade."""


class IsaacCamera:
    """One bridge camera, read as RGB-D in the harness's units."""

    def __init__(self, bridge: IsaacBridgeClient, name: str) -> None:
        self.bridge = bridge
        self.name = str(name)

    def read(self) -> CameraFrame | None:
        frame = self.bridge.call("get_camera", self.name)
        return CameraFrame(
            rgb=np.asarray(frame["rgb"], dtype=np.uint8),
            depth_m=np.asarray(frame["depth_m"], dtype=np.float64),
            intrinsics=np.asarray(frame["intrinsics"], dtype=np.float64),
            timestamp_s=time.monotonic(),
        )

    def close(self) -> None:
        """The bridge owns the render product."""


class IsaacYamEnv(RealYamEnv):
    """RealYamEnv whose scene can be reset to a seed and report task success."""

    def __init__(self, *args: Any, bridge: IsaacBridgeClient, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.bridge = bridge

    def realtime_factor(self) -> float:
        """Sim seconds per wall second, as the bridge last measured it (at most 1)."""
        rtf = float(self.bridge.call("get_status").get("realtime_factor", 1.0))
        return min(1.0, max(MIN_REALTIME_FACTOR, rtf))

    def execute_action_batch(
        self,
        batch: Any,
        *,
        command_hz: float | None = None,
        start_interp_s: float = 0.0,
        settle_s: float = 0.2,
        playback_speed: float = 1.0,
    ) -> dict[str, Any]:
        """The chokepoint, time-dilated to the simulator's real-time factor.

        The level-1 loop streams by the wall clock, and the bench's arm keeps that
        clock. When the sim runs at a fraction ``r`` of real time, a trajectory played
        at full speed reaches the arm ``1/r`` times faster in sim time, and the
        settle hold ends long before the arm has caught up -- the adapter then
        reports a tracking failure for a motion the arm completes. Playing at ``r``
        and stretching the holds by ``1/r`` gives the sim the motion as planned.
        """
        rtf = self.realtime_factor()
        return super().execute_action_batch(
            batch,
            command_hz=command_hz,
            start_interp_s=start_interp_s / rtf,
            settle_s=settle_s / rtf,
            playback_speed=playback_speed * rtf,
        )

    def reset_scene(self, seed: int, *, settle_steps: int = 30) -> dict[str, Any]:
        result = self.bridge.call("reset", int(seed), int(settle_steps), timeout=RESET_TIMEOUT_S)
        if not result.get("ok"):
            raise RuntimeError(f"Isaac scene reset failed: {result.get('error')}")
        return result

    def task_info(self) -> dict[str, Any]:
        return self.bridge.call("get_task_info")

    def close(self) -> None:
        try:
            super().close()
        finally:
            self.bridge.close()


#: Tasks the bridge can serve, beyond the station tasks every YAM profile has.
#: ``task_name`` is ASPIRE's; the bridge must be launched with the matching yamlab task.
ISAAC_TASKS = (
    YamRealTaskMetadata(
        "gear_assembly",
        "Pick up the gear and insert it onto the middle of the three shafts on the gear base, "
        "seated all the way down.",
        family="yam_isaac",
        scene_reset="seeded",
    ),
)


class IsaacYamAdapter(YamRealAdapter):
    """The real-YAM adapter, plus what a simulator can do and a bench cannot.

    ``reset`` restores the scene from the seed before the inherited robot reset
    (homing), and ``check_success`` reports the task env's latched success instead
    of the hardware adapter's "cannot tell".
    """

    def __init__(self, env: IsaacYamEnv, **kwargs: Any) -> None:
        kwargs.setdefault("registry", YamRealTaskRegistry(YAM_REAL_TASKS + ISAAC_TASKS))
        super().__init__(env, **kwargs)
        self._scene_reset: dict[str, Any] | None = None

    def reset(self, task_ref: object = None, seed: int = 0) -> Observation:
        self._scene_reset = self.native_env.reset_scene(int(seed))
        return super().reset(task_ref, seed)

    def check_success(self) -> bool:
        info = self.native_env.task_info()
        return bool(info.get("intermediate_success", {}).get("task_success", False))


def build_isaac_station(
    station: str = "yam-isaac",
    *,
    config_root: str | None = None,
    host: str = DEFAULT_BRIDGE_HOST,
    port: int = DEFAULT_BRIDGE_PORT,
) -> IsaacYamEnv:
    """Build the YAM env against a running yamlab bridge."""
    config = load_yam_station_config(station, config_root=config_root)
    bridge = IsaacBridgeClient(host, port)
    names = {role: name for name, role in CAMERA_ROLES.items()}
    arms = {side: IsaacYamArm(bridge, side) for side in ARMS}
    # Each frame is a few MB over RPC; the bridge renders at the env rate anyway.
    camera = ThreadedCamera(IsaacCamera(bridge, names[config.camera.role]), max_fps=TOP_CAMERA_FPS)
    aux = {
        role: ThreadedCamera(IsaacCamera(bridge, names[role]), max_fps=AUX_CAMERA_FPS)
        for role in config.aux_cameras
        if role in names
    }
    return IsaacYamEnv(config, arms, camera=camera, aux_cameras=aux, bridge=bridge)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_isaac_station_profile(
    target: Path,
    *,
    template: Path,
    host: str = DEFAULT_BRIDGE_HOST,
    port: int = DEFAULT_BRIDGE_PORT,
    station: str = "yam-isaac",
) -> Path:
    """Write a station profile whose calibration is the simulator's ground truth.

    The arm, joint-limit, workspace, controller and model sections are copied from
    ``template`` (yam-example, whose station model the bridge's ``aspire`` arm layout
    matches). Cameras and the calibration bundle come from the live bridge.
    """
    import yaml

    bridge = IsaacBridgeClient(host, port)
    try:
        status = bridge.call("get_status")
        top = bridge.call("get_camera", "top")
    finally:
        bridge.close()

    data = yaml.safe_load(template.read_text(encoding="utf-8"))
    height, width = np.asarray(top["depth_m"]).shape
    bundle_id = f"{station}-sim-v1"
    bundle_dir = target / "calibration" / bundle_id
    bundle_dir.mkdir(parents=True, exist_ok=True)

    calibration = {
        "schema_version": 1,
        "complete": True,
        "cameras": {
            "top": {
                "camera_serial": "isaac-top",
                "parent_frame": "world",
                "transform_name": "T_base_from_camera",
                "T_base_from_camera": np.asarray(top["T_world_from_camera"]).tolist(),
            }
        },
    }
    calibration_path = bundle_dir / "calibration.json"
    calibration_path.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "bundle_id": bundle_id,
        "station": station,
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "calibration_sha256": _sha256(calibration_path),
        "cameras": {
            "top": {
                "serial": "isaac-top",
                "resolution": [int(width), int(height)],
                "intrinsics": np.asarray(top["intrinsics"]).tolist(),
            }
        },
        "provenance": {"method": "isaac-sim-ground-truth", "bridge_task": status.get("task")},
    }
    manifest_path = bundle_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    data["station"] = station
    data["calibration"] = {
        "bundle": f"calibration/{bundle_id}",
        "manifest_sha256": _sha256(manifest_path),
    }
    for side in ARMS:
        data["arms"][side]["host"] = host
        data["arms"][side]["port"] = int(port)
        data["arms"][side]["can_interface"] = "isaac-sim"
    data["camera"] = {
        "role": "top",
        "serial": "isaac-top",
        "resolution": [int(width), int(height)],
        "max_acquisition_elapsed_s": 0.5,
    }
    data["aux_cameras"] = [
        {
            "role": role,
            "serial": f"isaac-{role}",
            "resolution": [int(width), int(height)],
            "max_acquisition_elapsed_s": 0.5,
        }
        for name, role in CAMERA_ROLES.items()
        if role != "top" and name in status.get("cameras", ())
    ]
    target.mkdir(parents=True, exist_ok=True)
    station_path = target / "station.yaml"
    station_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return station_path


__all__ = [
    "CAMERA_ROLES",
    "ISAAC_TASKS",
    "IsaacBridgeClient",
    "IsaacYamAdapter",
    "IsaacCamera",
    "IsaacYamArm",
    "IsaacYamEnv",
    "build_isaac_station",
    "write_isaac_station_profile",
]
