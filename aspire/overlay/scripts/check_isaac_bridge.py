"""Check a yamlab Isaac Sim bridge against ASPIRE's YAM station model.

Run with the bridge up (yamlab: scripts/aspire_bridge_server.py):

    .venv-yam-real/bin/python scripts/check_isaac_bridge.py [--host 127.0.0.1 --port 11400] [--move]

1. connect     status, both arms' observations, every camera's frame shape
2. kinematics  ASPIRE's MuJoCo station model vs Isaac's world link poses, same joints
3. camera      the picked object's true position projected through the station
               calibration: pixel in bounds, rendered depth vs geometric depth
4. motion      (--move) a bounded joint-1 move on the right arm through the bridge, tracked

Exits non-zero when a check fails.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from cap_harness.yam_real.config import load_yam_station_config
from cap_harness.yam_real.isaac import IsaacBridgeClient

ARMS = ("left", "right")
KINEMATICS_TOL_M = 0.005
DEPTH_TOL_M = 0.03
MOTION_DELTA_RAD = 0.2
MOTION_TOL_RAD = 0.03


def check(name: str, ok: bool, detail: str) -> bool:
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    return ok


def kinematics(bridge: IsaacBridgeClient, model_xml: str) -> bool:
    import mujoco

    model = mujoco.MjModel.from_xml_path(model_xml)
    data = mujoco.MjData(model)
    joints = {side: bridge.call("get_arm_observations", side)["joint_pos"] for side in ARMS}
    for side in ARMS:
        for index, value in enumerate(joints[side], start=1):
            data.qpos[model.jnt_qposadr[model.joint(f"{side}_joint{index}").id]] = value
    mujoco.mj_kinematics(model, data)

    ok = True
    for side in ARMS:
        isaac = bridge.call("get_body_poses", side)
        errors = {}
        for name, pos in zip(isaac["names"], isaac["pos"]):
            if _has_body(model, f"{side}_{name}"):
                errors[name] = float(np.linalg.norm(np.asarray(pos) - data.body(f"{side}_{name}").xpos))
        # The arm chain is what planning and IK use. The wrist-camera mount bodies also
        # match by name but are modelled ~1 cm apart; ASPIRE uses wrist images only.
        arm = {n: e for n, e in errors.items() if n == "arm" or n.startswith("link_")}
        other = {n: e for n, e in errors.items() if n not in arm}
        worst = max(arm.items(), key=lambda item: item[1], default=("none", float("inf")))
        ok &= check(
            f"kinematics {side}",
            len(arm) == 7 and worst[1] < KINEMATICS_TOL_M,
            f"{len(arm)} arm links, worst {worst[1] * 1000:.1f} mm ({worst[0]}); "
            f"not checked: " + ", ".join(f"{n} {e * 1000:.1f} mm" for n, e in other.items()),
        )
    return ok


def _has_body(model, name: str) -> bool:
    try:
        model.body(name)
        return True
    except KeyError:
        return False


def camera(bridge: IsaacBridgeClient, station) -> bool:
    frame = bridge.call("get_camera", station.camera.role)
    info = bridge.call("get_task_info")
    point = np.asarray(info["pot_position"], dtype=np.float64)
    T = np.asarray(station.camera.base_from_camera, dtype=np.float64)
    K = np.asarray(station.camera.intrinsics, dtype=np.float64)
    p_cam = np.linalg.inv(T) @ np.append(point, 1.0)
    u, v, w = K @ p_cam[:3]
    u, v = u / w, v / w
    depth = np.asarray(frame["depth_m"])
    h, wd = depth.shape
    inside = 0 <= u < wd and 0 <= v < h
    ok = check("camera projection", inside and p_cam[2] > 0,
               f"picked object at world {np.round(point, 3)} -> pixel ({u:.0f}, {v:.0f}) in {wd}x{h}")
    if inside:
        r = 3
        patch = depth[max(0, int(v) - r): int(v) + r + 1, max(0, int(u) - r): int(u) + r + 1]
        rendered = float(np.median(patch[patch > 0])) if np.any(patch > 0) else float("nan")
        # The object's origin is inside it; the rendered surface is nearer the camera.
        ok &= check("camera depth", abs(rendered - p_cam[2]) < DEPTH_TOL_M,
                    f"rendered {rendered:.3f} m vs origin {p_cam[2]:.3f} m (tol {DEPTH_TOL_M} m)")
    return ok


def motion(bridge: IsaacBridgeClient) -> bool:
    start = bridge.call("get_arm_observations", "right")
    target = np.concatenate([start["joint_pos"], start["gripper_pos"]])
    target[0] += MOTION_DELTA_RAD
    steps = 60
    for i in range(1, steps + 1):  # ~2 s ramp, well under the station's velocity limit
        cmd = np.concatenate([start["joint_pos"], start["gripper_pos"]])
        cmd[0] += MOTION_DELTA_RAD * i / steps
        bridge.send("command_arm", "right", cmd)
        time.sleep(1 / 30)
    time.sleep(1.0)
    reached = bridge.call("get_arm_observations", "right")["joint_pos"]
    err = float(abs(reached[0] - target[0]))
    ok = check("motion track", err < MOTION_TOL_RAD,
               f"joint1 {start['joint_pos'][0]:.3f} -> target {target[0]:.3f}, reached {reached[0]:.3f}")
    back = np.concatenate([start["joint_pos"], start["gripper_pos"]])
    for i in range(steps, -1, -1):
        cmd = back.copy()
        cmd[0] += MOTION_DELTA_RAD * i / steps
        bridge.send("command_arm", "right", cmd)
        time.sleep(1 / 30)
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11400)
    parser.add_argument("--station", default="yam-isaac")
    parser.add_argument("--move", action="store_true", help="also command a small arm motion")
    args = parser.parse_args()

    station = load_yam_station_config(args.station)
    bridge = IsaacBridgeClient(args.host, args.port)
    ok = True
    try:
        status = bridge.call("get_status")
        ok &= check("connect", True, f"{status}")
        for side in ARMS:
            obs = bridge.call("get_arm_observations", side)
            ok &= check(f"arm {side}", obs["joint_pos"].shape == (6,),
                        f"joints {np.round(obs['joint_pos'], 3)}, gripper {obs['gripper_pos'][0]:.2f}")
        for name in status["cameras"]:
            frame = bridge.call("get_camera", name)
            depth = np.asarray(frame["depth_m"])
            ok &= check(f"camera {name}", frame["rgb"].ndim == 3 and np.any(depth > 0),
                        f"rgb {frame['rgb'].shape}, depth {depth[depth > 0].min():.3f}-{depth.max():.3f} m")
        ok &= kinematics(bridge, str(station.model_xml))
        ok &= camera(bridge, station)
        if args.move:
            ok &= motion(bridge)
    finally:
        bridge.close()
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
