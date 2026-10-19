"""
Serve a YAM task environment to ASPIRE's cap-harness over portal RPC.

ASPIRE (https://github.com/NVlabs/ASPIRE) drives a YAM station through per-arm clients
(joint targets + a normalized gripper) and an RGB-D camera with known intrinsics and pose.
This server runs the IsaacLab task env and exposes exactly that, so ASPIRE's
``build_isaac_station()`` can stand in for the real arm servers and cameras.

Differences from ``joylo/scripts/launch_follower.py``:
  - Defaults to GPU physics (the gear task's resting gear drifts on CPU; see GEAR_ASSEMBLY.md).
  - Cameras also render depth (``distance_to_image_plane``).
  - Terminations are disabled: the env never auto-resets under a running program. Success is
    reported through ``get_task_info`` and ASPIRE asks for resets with ``reset(seed)``.
  - No recording; nothing is written to disk.

Run:
  python scripts/aspire_bridge_server.py --task PutPotOnCooktop-v0 --headless
  python scripts/aspire_bridge_server.py --task PutPotOnCooktop-v0 --viz kit   # with GUI

RPC methods (all numpy, world frame, quaternions XYZW as Isaac Lab 3 returns them):
  get_arm_observations(side)       joint_pos(6), gripper_pos(1, 0=closed..1=open), joint_vel(7), joint_eff(7)
  command_arm(side, pos7)          6 joint targets + normalized gripper target
  get_camera(name)                 rgb (H,W,3) uint8, depth_m (H,W), intrinsics 3x3,
                                   T_world_from_camera 4x4 (OpenCV axes), step
  camera_names()                   names served by get_camera
  reset(seed, settle_steps)        reset the task with a seed, hold the arms, let objects settle
  debug_set_object_pose, debug_ontop_check, debug_grasp_state   test hooks for the task itself (not used by ASPIRE)
  get_task_info()                  object poses + stage success flags
  get_body_poses(side)             every arm body's world pose (for kinematics checks)
  get_status()                     step count, step_dt, device, task
"""

import argparse
import queue
import threading
import time

import numpy as np

from isaaclab.app import AppLauncher

from yamlab.configs import get_objects_randomization, get_task_kwargs, resolve_assets, parse_asset_args

parser = argparse.ArgumentParser(description="ASPIRE bridge: YAM task env over portal RPC.")
parser.add_argument("--task", type=str, default="PutPotOnCooktop-v0", help="Task environment to use")
parser.add_argument("--asset", action="append", default=[], metavar="NAME=PATH",
                    help="Object asset instance dir (repeatable). Overrides objects.<name>.asset in the task YAML.")
parser.add_argument("--port", type=int, default=11400, help="Port for the bridge RPC server")
parser.add_argument("--realtime", action=argparse.BooleanOptionalAction, default=True,
                    help="Pace env steps to step_dt (ASPIRE's YAM control loop assumes wall-clock time)")
parser.add_argument("--render_every", type=int, default=3,
                    help="Render and capture the cameras every N control steps. Rendering three RTX "
                         "cameras every step keeps the sim well below real time, and ASPIRE's YAM "
                         "control loop times motions by the wall clock")
parser.add_argument("--finger_effort_limit", type=float, default=20.0,
                    help="Cap each finger joint's actuator force (N); <= 0 keeps the YAML limit. The real "
                         "YAM gripper is torque-limited, while the sim's position-controlled fingers "
                         "squeeze toward a 'closed' target with up to 100 N and eject a small gear")
parser.add_argument("--grasp_pad_check", action=argparse.BooleanOptionalAction, default=False,
                    help="Keep grasp detection's contact-on-pad check. Off by default: on the GPU pipeline "
                         "the contact sensors' contact_pos_w is not where the contact is (points below the "
                         "table while the fingers hold the gear), so the check rejects every real grasp. "
                         "The force check stays, and pick also needs a sustained 5 cm lift")
parser.add_argument("--arm_layout", choices=("aspire", "yamlab"), default="aspire",
                    help="Arm base positions: 'aspire' matches ASPIRE's yam station model, which its "
                         "kinematics and cuRobo planner assume; 'yamlab' keeps configs/robot/yam.yaml")
parser.add_argument("--sam_overlay", action=argparse.BooleanOptionalAction, default=False,
                    help="Viewer aid: periodically segment the top camera with SAM3 and show the masks in a "
                         "live 'SAM3 · top camera' panel in the Isaac Sim window. A UI panel is never "
                         "rendered into camera sensors, so robot programs never see it (3D debug-draw "
                         "markers would be)")
parser.add_argument("--sam_url", type=str, default="http://127.0.0.1:8114", help="SAM3 service for --sam_overlay")
parser.add_argument("--sam_prompts", type=str, default="light blue gear;gear base plate",
                    help="';'-separated text prompts for --sam_overlay, one colour each")
parser.add_argument("--sam_period", type=float, default=1.5, help="Seconds between --sam_overlay updates")
parser.add_argument("--sam_top_k", type=int, default=3, help="Candidates drawn per prompt (best is brightest)")
AppLauncher.add_app_launcher_args(parser)
# Default to GPU physics rather than the teleop YAML's CPU (resting gear drift on CPU).
parser.set_defaults(device="cuda:0")
args = parser.parse_args()
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import portal
import torch
from isaaclab.managers import SceneEntityCfg

from yamlab.envs import yam_bimanual_scene
from yamlab.robot.yam import ROBOT
from yamlab.robot.yam.action import YamActionLayout
from yamlab.utils.lab3 import as_torch

ARMS = ("left", "right")
FINGER = YamActionLayout.ARM_GRIPPER_JOINT_INDEX
# Arm base positions in ASPIRE's station model (station_yam_example.xml / station.urdf).
ASPIRE_ARM_POSITIONS = {"left": [0.2525, 0.31, 0.75], "right": [0.2525, -0.31, 0.75]}
DEPTH_KEY = "distance_to_image_plane"

# Depth on every camera, and the arm layout. Task scene cfgs are instantiated when their
# modules are imported, so these must be set before task_creation (and the tasks) load.
yam_bimanual_scene.CAMERA_DATA_TYPES = ["rgb", DEPTH_KEY]
if args.arm_layout == "aspire":
    for _side, _position in ASPIRE_ARM_POSITIONS.items():
        ROBOT.raw["arms"][_side]["position"] = list(_position)
if not args.grasp_pad_check:
    ROBOT.raw["grasp"]["check_pad"] = False

from yamlab.utils.task_creation import create_task_environment  # noqa: E402


def _never(env, **kwargs):
    """Termination stand-in that never fires."""
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def _quat_xyzw_to_matrix(q):
    """Rotation matrix from an (x, y, z, w) quaternion."""
    x, y, z, w = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


SAM_COLORS = [(255, 0, 255), (255, 216, 0), (0, 255, 102), (255, 77, 0)]


class SamOverlay:
    """Viewer aid: a live Isaac Sim panel showing SAM3 masks on the top camera image.

    A worker thread posts the latest top frame to SAM3 and composites the masks and boxes into an
    RGBA image; the main loop pushes it into an omni.ui window. A UI panel is never rendered into
    camera sensors (3D debug-draw markers are, so they would corrupt what robot programs see).
    """

    def __init__(self, bridge, url, prompts, period, top_k):
        import base64
        import requests
        import struct
        import zlib
        import omni.ui as ui

        self._b64, self._struct, self._zlib = base64, struct, zlib
        self.bridge, self.url, self.prompts, self.period, self.top_k = bridge, url, prompts, period, top_k
        self.session = requests.Session()
        self._pending = None
        self._lock = threading.Lock()
        self.provider = ui.ByteImageProvider()
        self.window = ui.Window("SAM3 · top camera", width=680, height=600)
        with self.window.frame:
            with ui.VStack(spacing=4):
                ui.ImageWithProvider(self.provider, fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT)
                self.label = ui.Label("waiting for the first segmentation...", height=60, word_wrap=True)
        threading.Thread(target=self._loop, daemon=True).start()

    def _png_b64(self, rgb):
        chunk = lambda tag, data: (self._struct.pack(">I", len(data)) + tag + data
                                   + self._struct.pack(">I", self._zlib.crc32(tag + data) & 0xFFFFFFFF))
        h, w, _ = rgb.shape
        raw = b"".join(b"\x00" + row.tobytes() for row in rgb)
        png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", self._struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
               + chunk(b"IDAT", self._zlib.compress(raw, 1)) + chunk(b"IEND", b""))
        return self._b64.b64encode(png).decode("ascii")

    def _loop(self):
        while True:
            time.sleep(self.period)
            try:
                frame = self.bridge.get_camera("top")
            except KeyError:
                continue
            try:
                self._segment(frame)
            except Exception as exc:  # a viewer aid must never take the bridge down
                print(f"[SAM overlay] {type(exc).__name__}: {exc}")

    def _segment(self, frame):
        rgb = np.ascontiguousarray(frame["rgb"][..., :3]).astype(np.uint8)
        image = self._png_b64(rgb)
        canvas = rgb.astype(np.float32)
        lines = []
        for p, prompt in enumerate(self.prompts):
            r = self.session.post(f"{self.url}/segment", json={"image_base64": image, "text_prompt": prompt}, timeout=30)
            r.raise_for_status()
            results = sorted(r.json().get("results", []), key=lambda it: -float(it["score"]))[: self.top_k]
            color = np.array(SAM_COLORS[p % len(SAM_COLORS)], np.float32)
            scores = []
            for rank in reversed(range(len(results))):  # best candidate drawn last, on top
                it = results[rank]
                mask = np.frombuffer(self._b64.b64decode(it["mask_base64"]), np.uint8).reshape(it["shape"]).astype(bool)
                alpha = (0.55, 0.3, 0.18)[min(rank, 2)]
                canvas[mask] = (1 - alpha) * canvas[mask] + alpha * color
                x0, y0, x1, y1 = (int(round(v)) for v in it["box"])
                x0, x1 = max(0, x0), min(rgb.shape[1] - 1, x1)
                y0, y1 = max(0, y0), min(rgb.shape[0] - 1, y1)
                w = 2 if rank == 0 else 1
                canvas[y0:y0 + w, x0:x1 + 1] = canvas[max(y1 - w + 1, 0):y1 + 1, x0:x1 + 1] = color
                canvas[y0:y1 + 1, x0:x0 + w] = canvas[y0:y1 + 1, max(x1 - w + 1, 0):x1 + 1] = color
            scores = [f"#{k + 1} {float(it['score']):.2f}" for k, it in enumerate(results)]
            lines.append(f"[{('magenta', 'yellow', 'green', 'orange')[p % 4]}] \"{prompt}\": {', '.join(scores) or 'none'}")
        rgba = np.concatenate([np.clip(canvas, 0, 255).astype(np.uint8),
                               np.full(rgb.shape[:2] + (1,), 255, np.uint8)], axis=2)
        text = f"step {frame['step']}  ·  best candidate solid, next two faint\n" + "\n".join(lines)
        with self._lock:
            self._pending = (np.ascontiguousarray(rgba), text)
        print(f"[SAM overlay] {' | '.join(lines)}")

    def update_viewport(self):
        with self._lock:
            pending, self._pending = self._pending, None
        if pending is None:
            return
        rgba, text = pending
        self.provider.set_data_array(rgba, [rgba.shape[1], rgba.shape[0]])
        self.label.text = text


class AspireBridge:
    """Holds per-arm targets and the latest camera frames; the main loop owns the env."""

    def __init__(self, env):
        self.env = env
        self.scene = env.scene
        self.camera_names = [n for n in ROBOT.camera_names if f"{n}_camera" in self.scene.keys()]
        self._lock = threading.Lock()
        self._frames = {}
        self._targets = {}
        self.step_count = 0
        self.realtime_factor = 1.0  # sim seconds per wall second, measured by the main loop
        self._jobs = queue.Queue()
        self.hold_current()

    # -- gripper units: finger joint position <-> ASPIRE's normalized 0 (closed) .. 1 (open)

    @staticmethod
    def _finger_to_normalized(side, finger):
        closed, opened = ROBOT.finger_closed(side), ROBOT.finger_open(side)
        return float(np.clip((finger - closed) / (opened - closed), 0.0, 1.0))

    @staticmethod
    def _normalized_to_finger(side, value):
        closed, opened = ROBOT.finger_closed(side), ROBOT.finger_open(side)
        return closed + float(np.clip(value, 0.0, 1.0)) * (opened - closed)

    # -- arms

    def _arm(self, side):
        if side not in ARMS:
            raise ValueError(f"side must be 'left' or 'right'; got {side!r}")
        return self.scene[f"{side}_arm"]

    def hold_current(self):
        """Target the measured joints so the arms hold still (startup and after reset)."""
        with self._lock:
            for side in ARMS:
                q = as_torch(self._arm(side).data.joint_pos)[0].cpu().numpy()
                self._targets[side] = np.concatenate([q[:6], [q[FINGER]]]).astype(np.float64)

    def get_arm_observations(self, side):
        data = self._arm(side).data
        q = as_torch(data.joint_pos)[0].cpu().numpy().astype(np.float64)
        qd = as_torch(data.joint_vel)[0].cpu().numpy().astype(np.float64)
        tau = as_torch(data.applied_torque)[0].cpu().numpy().astype(np.float64)
        return {
            "joint_pos": q[:6].copy(),
            "gripper_pos": np.array([self._finger_to_normalized(side, q[FINGER])]),
            "joint_vel": np.concatenate([qd[:6], [qd[FINGER]]]),
            "joint_eff": np.concatenate([tau[:6], [tau[FINGER]]]),
        }

    def command_arm(self, side, pos7):
        pos7 = np.asarray(pos7, dtype=np.float64).reshape(7)
        if not np.all(np.isfinite(pos7)):
            raise ValueError("command contains non-finite values")
        target = np.concatenate([pos7[:6], [self._normalized_to_finger(side, pos7[6])]])
        self._arm(side)  # validates side
        with self._lock:
            self._targets[side] = target
        return True

    def action(self):
        with self._lock:
            left, right = self._targets["left"], self._targets["right"]
        action = torch.zeros(1, YamActionLayout.DIM, device=self.env.device)
        action[0, YamActionLayout.LEFT_ARM] = torch.as_tensor(left[:6], dtype=torch.float32)
        action[0, YamActionLayout.LEFT_GRIPPER] = float(left[6])
        action[0, YamActionLayout.RIGHT_ARM] = torch.as_tensor(right[:6], dtype=torch.float32)
        action[0, YamActionLayout.RIGHT_GRIPPER] = float(right[6])
        return action

    def get_body_poses(self, side):
        data = self._arm(side).data
        return {
            "names": list(self._arm(side).body_names),
            "pos": as_torch(data.body_pos_w)[0].cpu().numpy().astype(np.float64),
            "quat_xyzw": as_torch(data.body_quat_w)[0].cpu().numpy().astype(np.float64),
        }

    # -- cameras

    def capture_cameras(self):
        """Copy the latest render of every camera; called by the main loop after a step."""
        frames = {}
        for name in self.camera_names:
            cam = self.scene[f"{name}_camera"]
            rgb = as_torch(cam.data.output["rgb"])[0, ..., :3].cpu().numpy().astype(np.uint8)
            depth = as_torch(cam.data.output[DEPTH_KEY])[0].cpu().numpy().astype(np.float32)
            depth = depth.reshape(depth.shape[0], depth.shape[1])
            depth[~np.isfinite(depth)] = 0.0
            K = as_torch(cam.data.intrinsic_matrices)[0].cpu().numpy().astype(np.float64)
            T = np.eye(4)
            T[:3, :3] = _quat_xyzw_to_matrix(as_torch(cam.data.quat_w_ros)[0].cpu().numpy())
            T[:3, 3] = as_torch(cam.data.pos_w)[0].cpu().numpy()
            frames[name] = {
                "rgb": rgb,
                "depth_m": depth,
                "intrinsics": K,
                "T_world_from_camera": T,
                "step": self.step_count,
            }
        with self._lock:
            self._frames = frames

    def get_camera(self, name):
        with self._lock:
            frame = self._frames.get(name)
        if frame is None:
            raise KeyError(f"no frame for camera {name!r}; cameras: {self.camera_names}")
        return frame

    # -- task

    def get_task_info(self):
        info = self.env.get_current_task_info()
        info["step"] = self.step_count
        return info

    def get_status(self):
        return {
            "task": args.task,
            "device": str(self.env.device),
            "step": self.step_count,
            "step_dt": float(self.env.step_dt),
            "realtime_factor": float(self.realtime_factor),
            "cameras": list(self.camera_names),
        }

    # -- main-thread jobs: anything that writes sim state runs between env steps

    def _on_main(self, fn, *fn_args, timeout=120.0):
        """Run ``fn`` on the main loop's thread; block until done and return its result."""
        done, box = threading.Event(), {}
        self._jobs.put((fn, fn_args, done, box))
        if not done.wait(timeout=timeout):
            raise TimeoutError(f"{fn.__name__} did not complete within {timeout:.0f} s")
        if "error" in box:
            return {"ok": False, "error": box["error"]}
        return box["result"]

    def process_jobs(self):
        """Run pending jobs on the main thread. Returns True if any ran."""
        ran = False
        while True:
            try:
                fn, fn_args, done, box = self._jobs.get_nowait()
            except queue.Empty:
                return ran
            try:
                box["result"] = fn(*fn_args)
            except Exception as exc:  # reported to the caller, not raised in the sim loop
                import traceback
                traceback.print_exc()
                box["error"] = f"{type(exc).__name__}: {exc}"
            done.set()
            ran = True

    def _step(self, n):
        for _ in range(int(n)):
            self.env.step(self.action())
            self.step_count += 1
        self.capture_cameras()

    def reset(self, seed, settle_steps=30):
        """Reset the task with a seed, hold the arms, and let the objects settle."""
        def job():
            self.env.reset(seed=int(seed))
            self.hold_current()
            self._step(settle_steps)
            return {"ok": True, "seed": int(seed), **self.get_task_info()}
        return self._on_main(job)

    # -- debug hooks (test the task itself; ASPIRE programs never reach these)

    def debug_set_object_pose(self, name, pos, quat_xyzw, settle_steps=60):
        """Teleport a rigid object (zero velocity), then step so it settles."""
        def job():
            obj = self.scene[name]
            pose = np.concatenate([np.asarray(pos, dtype=np.float32).reshape(3),
                                   np.asarray(quat_xyzw, dtype=np.float32).reshape(4)])
            pose = torch.as_tensor(pose[None], device=self.env.device)
            obj.write_root_pose_to_sim(pose)
            obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=self.env.device))
            self._step(settle_steps)
            return {"ok": True, **self.get_task_info()}
        return self._on_main(job)

    def debug_grasp_state(self):
        """Grasp detection and the picked object's lift, as the pick check sees them."""
        def job():
            left, right = self.env.robot.is_grasping()
            lift = float(self.scene["pot"].data.root_pos_w.torch[0, 2]) - float(self.env.pot_init_z)
            sensors = {}
            for key, sensor in self.scene.sensors.items():
                if "finger" not in key and "contact" not in key:
                    continue
                net = as_torch(sensor.data.net_forces_w)
                matrix = as_torch(sensor.data.force_matrix_w)
                sensors[key] = {
                    "net_N": float(torch.norm(net[0].sum(dim=0))) if net is not None else None,
                    "filtered_N": [float(v) for v in torch.norm(matrix[0].sum(dim=0), dim=-1)] if matrix is not None else None,
                    "filters": list(sensor.cfg.filter_prim_paths_expr),
                }
            pads = {}
            env_ids = torch.arange(1, device=self.env.device)
            for side in ARMS:
                for finger in getattr(self.env.robot, f"{side}_arm").end_effector.fingers:
                    fraction, valid = finger.contact_pad_fraction("pot", env_ids)
                    arm = self.scene[f"{side}_arm"]
                    contact = as_torch(finger._sensor.data.contact_pos_w)
                    pads[f"{side}/{finger.link}"] = {
                        "fraction": float(fraction[0]), "valid": bool(valid[0]),
                        "contact_w": contact[0, 0, finger._filter_index("pot")].tolist() if contact is not None else None,
                        "finger_w": arm.data.body_link_pose_w.torch[0, arm.data.body_names.index(finger.link), :3].tolist(),
                        "keypoints": finger._keypoints.tolist(),
                    }
            return {"left_grasping": bool(left[0]), "right_grasping": bool(right[0]), "lift_m": lift,
                    "pick": bool(self.env.stage1_success[0]), "sensors": sensors, "pads": pads}
        return self._on_main(job)

    def debug_ontop_check(self):
        """Evaluate the placement check now, with the configured XY points and with origins."""
        from yamlab.envs.tasks import put_pot_on_cooktop_manager as m
        from yamlab.utils.task_logic import check_ontop_success

        def job():
            env = self.env
            common = dict(
                pot_height=env.pot_height, cooktop_height=env.cooktop_height,
                height_tolerance=m.ONTOP_HEIGHT_TOLERANCE_M, xy_tolerance=env.ontop_xy_threshold,
                expected_z_offset=m.ONTOP_Z_OFFSET_M,
                orientation_tolerance=m.ONTOP_ORIENTATION_TOLERANCE_RAD,
            )
            pot, cooktop = SceneEntityCfg("pot"), SceneEntityCfg("cooktop")
            configured = check_ontop_success(env, pot, cooktop, pot_xy_point=m.ONTOP_POT_XY_POINT_M,
                                             cooktop_xy_point=m.ONTOP_COOKTOP_XY_POINT_M, **common)
            origins = check_ontop_success(env, pot, cooktop, **common)
            return {"configured": bool(configured[0]), "origins": bool(origins[0]), **self.get_task_info()}
        return self._on_main(job)


def main():
    task_kwargs = get_task_kwargs(args.task, "evaluation")
    assets_instance_paths = resolve_assets(args.task, cli_assets=parse_asset_args(args.asset))
    if not assets_instance_paths:
        raise SystemExit("No object assets specified (--asset NAME=PATH or objects.<name>.asset in the task YAML).")

    # Evaluation mode, one env: teleop mode swaps the camera sensors for bare prims (nothing
    # renders), and evaluation keeps them. Its 30-step placement latch also applies.
    env = create_task_environment(
        task_name=args.task,
        assets_instance_paths=assets_instance_paths,
        objects_randomization=get_objects_randomization(args.task),
        init_joint_pos_randomization=0.0,
        mode="evaluation",
        observation_modalities=[],
        num_envs=1,
        device=args.device,
        **task_kwargs,
    )

    # Never auto-reset under a running program.
    tm = env.termination_manager
    for name in list(tm.active_terms):
        cfg = tm.get_term_cfg(name)
        cfg.func = _never
        cfg.params = {}
        tm.set_term_cfg(name, cfg)
    print(f"[INFO] Disabled terminations: {list(tm.active_terms)}")

    # Isaac Lab reads render_interval (in physics substeps) every step.
    env.cfg.sim.render_interval = env.cfg.decimation * max(1, args.render_every)

    if args.finger_effort_limit > 0:
        for side in ARMS:
            arm = env.scene[f"{side}_arm"]
            finger_ids, finger_names = arm.find_joints(["left_finger", "right_finger"])
            arm.write_joint_effort_limit_to_sim_index(limits=float(args.finger_effort_limit), joint_ids=finger_ids)
        print(f"[INFO] Finger effort limit: {args.finger_effort_limit} N ({finger_names})")
    print(f"[INFO] Grasp detection pad check: {ROBOT.raw['grasp']['check_pad']}")

    env.reset()
    bridge = AspireBridge(env)

    server = portal.Server(args.port)
    server.bind("get_arm_observations", bridge.get_arm_observations)
    server.bind("command_arm", bridge.command_arm)
    server.bind("get_camera", bridge.get_camera)
    server.bind("camera_names", lambda: list(bridge.camera_names))
    server.bind("reset", bridge.reset)
    server.bind("get_task_info", bridge.get_task_info)
    server.bind("get_body_poses", bridge.get_body_poses)
    server.bind("get_status", bridge.get_status)
    server.bind("debug_set_object_pose", bridge.debug_set_object_pose)
    server.bind("debug_ontop_check", bridge.debug_ontop_check)
    server.bind("debug_grasp_state", bridge.debug_grasp_state)
    threading.Thread(target=server.start, daemon=True).start()

    overlay = None
    if args.sam_overlay:
        prompts = [p.strip() for p in args.sam_prompts.split(";") if p.strip()]
        overlay = SamOverlay(bridge, args.sam_url, prompts, args.sam_period, args.sam_top_k)
        print(f"[INFO] SAM overlay: {prompts} every {args.sam_period}s from {args.sam_url}")

    control_dt = float(env.step_dt)
    print(f"[INFO] ASPIRE bridge: task={args.task} device={env.device} step_dt={control_dt:.4f}s "
          f"cameras={bridge.camera_names}")
    print(f"[INFO] ASPIRE bridge ready on port {args.port}")

    last_time = time.time()
    timing = {"step": 0.0, "capture": 0.0, "n": 0, "since": time.time()}
    while simulation_app.is_running():
        if bridge.process_jobs():
            last_time = time.time()
            timing = {"step": 0.0, "capture": 0.0, "n": 0, "since": time.time()}  # jobs step on their own
            continue
        t0 = time.perf_counter()
        env.step(bridge.action())
        t1 = time.perf_counter()
        bridge.step_count += 1
        if env._sim_step_counter % env.cfg.sim.render_interval == 0:  # rendered this step
            bridge.capture_cameras()
        if overlay is not None:
            overlay.update_viewport()
        timing["step"] += t1 - t0
        timing["capture"] += time.perf_counter() - t1
        timing["n"] += 1
        if timing["n"] == 150:
            n = timing.pop("n")
            wall = time.time() - timing.pop("since", time.time())
            if wall > 0:
                bridge.realtime_factor = min(1.0, n * control_dt / wall)
            print(f"[TIMING] per step: env.step {1000 * timing['step'] / n:.1f} ms, "
                  f"capture {1000 * timing['capture'] / n:.1f} ms (budget {1000 * control_dt:.1f} ms), "
                  f"real-time factor {bridge.realtime_factor:.2f}")
            timing = {"step": 0.0, "capture": 0.0, "n": 0, "since": time.time()}
        if args.realtime:
            sleep_time = control_dt - (time.time() - last_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
        last_time = time.time()

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
