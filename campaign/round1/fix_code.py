# v05 — ablation of v04: jaw fully open (1.0) before the grasp descent instead of pre-narrowed
# to 0.75. One variable changed; measures whether the pre-close matters.
# v04 — grasp retention: shallower top-down hub grasp (10 mm below the top face). At 16 mm the
# jaw closed on the gear teeth (width 0.65 vs 0.47-0.51 on the hub) and the gear jammed tilted.
# A width that closes wider than the first grasp by >25% also triggers the single retry.
# v03 — grasp retention: deeper top-down hub grasp (16 mm), jaw pre-narrowed to 0.75 before the
# descent, and the relay re-grasp is verified against the first grasp's jaw width (re-grasp once
# if it closed much narrower, which is what a gear tipped by the closing jaw looks like).
# v02 — execution time: v01's strategy with far fewer motion segments (each move_to_pose costs
# ~3.7 s of sim time regardless of length; v01 needed 121 motions / 559 s per seed).
# v01: park both arms out of the top view before perceiving, search grasp orientations
# (top-down first, then tilted forward), relay through a table point the inserting arm
# reaches top-down, low hover heights the inserting arm can reach.
#
# Strategy: localize the gear and the gear base with the top RGB-D camera, find the three raised
# shafts on the base from its point cloud and pick the middle one (the shaft NOT in the
# farthest-apart pair). Insert top-down in small vertical steps, release, retreat.
import math

CAM = "top"
MS = MotionStrategy(ik_solver="mink", trajectory_planner="interpolation")
state0 = get_robot_state()
BASE = state0.base_frame
ARMS = ("left", "right")
OUTWARD = {"left": 1.0, "right": -1.0}   # joint-0 sign that swings each arm away from the centre
report = {"task": get_task_context().language, "log": []}

GRASP_DEPTH = 0.010      # TCP below the gear's top face when grasping the hub top-down
TILT_DEPTH = 0.010       # same, for tilted grasps
PRE = 0.08               # pre-grasp stand-off along the approach axis
STEP = 0.008             # Cartesian step for straight-line segments
FINE = 0.004             # vertical step for the final insertion


def log(msg):
    report["log"].append(msg)


def quant(vals, f):
    s = sorted(vals)
    return s[int((len(s) - 1) * f)]


def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


def tool_quat(arm, pitch_deg, yaw_deg):
    # home orientation points the tool forward (+x); pitch it down about world y, then yaw about z
    p = math.radians(pitch_deg) / 2.0
    y = math.radians(yaw_deg) / 2.0
    qp = (math.cos(p), 0.0, math.sin(p), 0.0)
    qy = (math.cos(y), 0.0, 0.0, math.sin(y))
    return qmul(qy, qmul(qp, tuple(float(v) for v in state0.end_effector_poses[arm].quaternion_wxyz)))


def approach_dir(qt):
    w, x, y, z = qt
    return (2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y))


def reachable(arm, p, qt):
    ik = solve_ik(Pose(p, qt, BASE), arm=arm, backend="mink")
    return bool(ik.ok)


def move(arm, p, qt, tol=0.006, what="move"):
    r = move_to_pose(Pose(p, qt, BASE), tolerance=tol, max_steps=250, arm=arm, strategy=MS)
    if not r.ok:
        log(what + " failed: " + str(r.status) + " " + str(r.error)[:160])
    return bool(r.ok)


def line(arm, p0, p1, qt, step=STEP, tol=0.005, what="line"):
    d = math.sqrt(sum((p1[i] - p0[i]) ** 2 for i in range(3)))
    n = max(1, int(math.ceil(d / step)))
    ok = True
    for k in range(1, n + 1):
        f = k / n
        p = tuple(p0[i] + (p1[i] - p0[i]) * f for i in range(3))
        ok = move(arm, p, qt, tol=tol, what=what) and ok
    return ok


def ee(arm):
    return tuple(float(v) for v in get_robot_state().end_effector_poses[arm].position)


def park(arm):
    j = [float(v) for v in state0.joint_positions[arm]]
    j[0] = j[0] + OUTWARD[arm] * 0.7
    r = move_to_joints(j, tolerance=0.02, max_steps=200, arm=arm)
    if not r.ok:
        log("park " + arm + " failed: " + str(r.status))


def xyz_lists(points):
    return ([float(p[0]) for p in points], [float(p[1]) for p in points], [float(p[2]) for p in points])


def candidates(prompts):
    out = []
    for prompt in prompts:
        seg = segment_text(CAM, prompt)
        if not seg.ok:
            continue
        for cand in seg.segmentations[:8]:
            try:
                pc = mask_to_point_cloud(cand, CAM, target_frame=BASE)
            except Exception:
                continue
            if len(pc.points) < 30:
                continue
            xs, ys, zs = xyz_lists(pc.points)
            out.append({"score": float(cand.score), "xs": xs, "ys": ys, "zs": zs, "prompt": prompt,
                        "ext_x": quant(xs, 0.95) - quant(xs, 0.05),
                        "ext_y": quant(ys, 0.95) - quant(ys, 0.05)})
    return out


def find_gear():
    best = None
    for c in candidates(("light blue gear", "small gear")):
        if c["ext_x"] < 0.015 or c["ext_x"] > 0.07 or c["ext_y"] < 0.015 or c["ext_y"] > 0.07:
            continue
        ztop = quant(c["zs"], 0.98)
        zlow = quant(c["zs"], 0.02)
        if ztop - zlow < 0.015 or ztop - zlow > 0.08:
            continue
        top = [i for i in range(len(c["zs"])) if c["zs"][i] > ztop - 0.004]
        gx = sum(c["xs"][i] for i in top) / len(top)
        gy = sum(c["ys"][i] for i in top) / len(top)
        if best is None or c["score"] > best["score"]:
            best = {"score": c["score"], "x": gx, "y": gy, "ztop": ztop, "zlow": zlow, "ntop": len(top)}
    return best


def cluster_raised(c, surface):
    idx = [i for i in range(len(c["zs"])) if c["zs"][i] > surface + 0.012]
    idx = sorted(idx, key=lambda i: -c["zs"][i])
    clusters = []
    for i in idx:
        placed = None
        for cl in clusters:
            if math.hypot(c["xs"][i] - cl["sx"], c["ys"][i] - cl["sy"]) < 0.015:
                placed = cl
                break
        if placed is None:
            clusters.append({"sx": c["xs"][i], "sy": c["ys"][i], "members": [i]})
        else:
            placed["members"].append(i)
    shafts = []
    for cl in clusters:
        if len(cl["members"]) < 5:
            continue
        zmax = max(c["zs"][i] for i in cl["members"])
        top = [i for i in cl["members"] if c["zs"][i] > zmax - 0.004]
        shafts.append({"x": sum(c["xs"][i] for i in top) / len(top),
                       "y": sum(c["ys"][i] for i in top) / len(top),
                       "ztop": zmax, "n": len(cl["members"])})
    return shafts


def find_base():
    best = None
    for c in candidates(("gear base plate", "blue plate")):
        if c["ext_x"] < 0.04 or c["ext_y"] < 0.04 or c["ext_x"] > 0.35 or c["ext_y"] > 0.35:
            continue
        surface = quant(c["zs"], 0.5)
        if max(c["zs"]) - surface > 0.08:
            continue   # robot links / tall clutter
        shafts = cluster_raised(c, surface)
        if len(shafts) < 3:
            continue
        shafts = sorted(shafts, key=lambda s: -s["n"])[:3]
        tops = [s["ztop"] for s in shafts]
        if max(tops) - min(tops) > 0.01 or min(tops) - surface < 0.015:
            continue   # three shafts of a common height stand on the plate
        if best is None or c["score"] > best["score"]:
            best = {"score": c["score"], "surface": surface, "shafts": shafts,
                    "zlow": quant(c["zs"], 0.02), "xs": c["xs"], "ys": c["ys"]}
    return best


def middle_shaft(shafts):
    far = None
    for i in range(3):
        for j in range(i + 1, 3):
            d = math.hypot(shafts[i]["x"] - shafts[j]["x"], shafts[i]["y"] - shafts[j]["y"])
            if far is None or d > far[0]:
                far = (d, i, j)
    k = [m for m in range(3) if m != far[1] and m != far[2]][0]
    return shafts[k]


ORIENTS = [(90, 0)] + [(p, y) for p in (75, 60, 45, 30) for y in (0, 30, -30)]


def grasp_plan(arm, g, top_down_only=False):
    """First orientation whose grasp and pre-grasp poses both solve; None if none."""
    for (pitch, yaw) in ORIENTS:
        if top_down_only and pitch != 90:
            continue
        qt = tool_quat(arm, pitch, yaw)
        depth = GRASP_DEPTH if pitch == 90 else TILT_DEPTH
        gp = (g["x"], g["y"], g["ztop"] - depth)
        a = approach_dir(qt)
        pre = (gp[0] - PRE * a[0], gp[1] - PRE * a[1], gp[2] - PRE * a[2])
        if reachable(arm, gp, qt) and reachable(arm, pre, qt):
            return {"arm": arm, "pitch": pitch, "yaw": yaw, "q": qt, "grasp": gp, "pre": pre}
    return None


def do_grasp(plan):
    arm = plan["arm"]
    open_gripper(arm=arm)
    move(arm, plan["pre"], plan["q"], what="pre-grasp")
    line(arm, plan["pre"], plan["grasp"], plan["q"], step=0.05, what="grasp approach")
    close_gripper(arm=arm)
    width = float(get_robot_state().gripper_positions[arm])
    log(arm + " closed at width " + str(round(width, 3)))
    gp = plan["grasp"]
    up = (gp[0], gp[1], gp[2] + 0.05)
    line(arm, gp, up, plan["q"], step=0.05, what="lift")
    return width


# ---------------- perception (arms parked out of view) ----------------
for arm in ARMS:
    park(arm)
gear = find_gear()
if gear is None:
    raise ValueError("gear not found")
gb = find_base()
if gb is None:
    raise ValueError("gear base with three shafts not found")
target = middle_shaft(gb["shafts"])
table_z = min(gear["zlow"], gb["zlow"])
report["gear"] = {k: round(gear[k], 4) for k in ("x", "y", "ztop", "zlow", "score")}
report["shafts"] = [{k: round(s[k], 4) for k in ("x", "y", "ztop")} for s in gb["shafts"]]
report["target"] = {k: round(target[k], 4) for k in ("x", "y", "ztop")}
report["surface"] = round(gb["surface"], 4)
report["table_z"] = round(table_z, 4)

gear_h = gear["ztop"] - table_z
tcp_above_bottom = gear_h - GRASP_DEPTH      # top-down TCP height above the gear's bottom face
seat_z = gb["surface"] + tcp_above_bottom - 0.002
hover_z = target["ztop"] + 0.02 + tcp_above_bottom
tx, ty = target["x"], target["y"]

# ---------------- arm assignment ----------------
insert_arms = [a for a in ARMS if reachable(a, (tx, ty, seat_z), tool_quat(a, 90, 0))
               and reachable(a, (tx, ty, hover_z), tool_quat(a, 90, 0))]
report["insert_arms"] = insert_arms
if not insert_arms:
    raise ValueError("no arm reaches the middle shaft top-down")
# prefer the inserting arm nearer the shaft (smaller |dy| to its home position)
insert_arms = sorted(insert_arms, key=lambda a: abs(float(state0.end_effector_poses[a].position[1]) - ty))
insert_arm = insert_arms[0]
DOWN = tool_quat(insert_arm, 90, 0)

direct = grasp_plan(insert_arm, gear, top_down_only=True)
if direct is not None:
    report["mode"] = "direct:" + insert_arm
    do_grasp(direct)
else:
    pick = grasp_plan(insert_arm, gear)
    if pick is None:
        other = [a for a in ARMS if a != insert_arm][0]
        pick = grasp_plan(other, gear)
    if pick is None:
        raise ValueError("no arm can grasp the gear")
    pick_arm = pick["arm"]
    report["mode"] = "relay:" + pick_arm + "(" + str(pick["pitch"]) + "," + str(pick["yaw"]) + ")->" + insert_arm
    hl = state0.end_effector_poses["left"].position
    hr = state0.end_effector_poses["right"].position
    mx = (float(hl[0]) + float(hr[0])) / 2.0
    my = (float(hl[1]) + float(hr[1])) / 2.0
    grid = []
    for dx in (0.0, 0.06, -0.06):
        for dy in (0.0, 0.08, -0.08, 0.16, -0.16):
            grid.append((mx + dx, my + dy))
    grid = sorted(grid, key=lambda p: math.hypot(p[0] - mx, p[1] - my))
    relay = None
    place_z = pick["grasp"][2] + 0.003
    for (rx, ry) in grid:
        clear = True
        for i in range(0, len(gb["xs"]), 5):
            if math.hypot(gb["xs"][i] - rx, gb["ys"][i] - ry) < 0.08:
                clear = False
                break
        if not clear:
            continue
        if not reachable(pick_arm, (rx, ry, place_z), pick["q"]):
            continue
        if not reachable(pick_arm, (rx, ry, place_z + 0.06), pick["q"]):
            continue
        gzi = gear["ztop"] - GRASP_DEPTH
        if reachable(insert_arm, (rx, ry, gzi), DOWN) and reachable(insert_arm, (rx, ry, gzi + PRE), DOWN):
            relay = (rx, ry)
            break
    if relay is None:
        raise ValueError("no relay point reachable by both grasps")
    report["relay"] = [round(relay[0], 4), round(relay[1], 4)]
    first_width = do_grasp(pick)
    cur = ee(pick_arm)
    lifted = (cur[0], cur[1], place_z + 0.06)
    line(pick_arm, cur, lifted, pick["q"], step=0.1, what="relay lift")
    line(pick_arm, lifted, (relay[0], relay[1], place_z + 0.06), pick["q"], step=0.2, what="relay transit")
    line(pick_arm, (relay[0], relay[1], place_z + 0.06), (relay[0], relay[1], place_z), pick["q"], step=0.1, what="relay place")
    open_gripper(arm=pick_arm)
    a = approach_dir(pick["q"])
    line(pick_arm, (relay[0], relay[1], place_z),
         (relay[0] - 0.06 * a[0], relay[1] - 0.06 * a[1], place_z - 0.06 * a[2] + 0.02), pick["q"],
         step=0.1, what="relay retreat")
    park(pick_arm)
    gear2 = find_gear()
    if gear2 is None:
        raise ValueError("gear lost after relay")
    report["gear_relay"] = {k: round(gear2[k], 4) for k in ("x", "y", "ztop")}
    plan2 = grasp_plan(insert_arm, gear2, top_down_only=True)
    if plan2 is None:
        raise ValueError("insert arm cannot grasp the relayed gear top-down")
    gear_h = gear2["ztop"] - table_z
    tcp_above_bottom = gear_h - GRASP_DEPTH
    seat_z = gb["surface"] + tcp_above_bottom - 0.002
    hover_z = target["ztop"] + 0.02 + tcp_above_bottom
    width2 = do_grasp(plan2)
    report["regrasp_widths"] = [round(first_width, 3), round(width2, 3)]
    if width2 < 0.7 * first_width or width2 > 1.25 * first_width:
        # jaw closed far narrower than on the first grasp: the gear tipped; put it back and retry
        cur = ee(insert_arm)
        line(insert_arm, cur, (cur[0], cur[1], cur[2] - 0.045), DOWN, step=0.05, what="regrasp lower")
        open_gripper(arm=insert_arm)
        cur = ee(insert_arm)
        line(insert_arm, cur, (cur[0], cur[1], cur[2] + 0.06), DOWN, step=0.1, what="regrasp clear")
        park(insert_arm)
        gear2 = find_gear()
        if gear2 is None:
            raise ValueError("gear lost after failed regrasp")
        plan2 = grasp_plan(insert_arm, gear2, top_down_only=True)
        if plan2 is None:
            raise ValueError("insert arm cannot reach the gear for the retry")
        gear_h = gear2["ztop"] - table_z
        tcp_above_bottom = gear_h - GRASP_DEPTH
        seat_z = gb["surface"] + tcp_above_bottom - 0.002
        hover_z = target["ztop"] + 0.02 + tcp_above_bottom
        width2 = do_grasp(plan2)
        report["regrasp_widths"].append(round(width2, 3))

# ---------------- insertion ----------------
cur = ee(insert_arm)
lift_z = max(cur[2], hover_z)
if lift_z - cur[2] > 0.005:
    line(insert_arm, cur, (cur[0], cur[1], lift_z), DOWN, step=0.1, what="transit up")
line(insert_arm, (cur[0], cur[1], lift_z), (tx, ty, hover_z), DOWN, step=0.15, what="transit")
move(insert_arm, (tx, ty, hover_z), DOWN, tol=0.003, what="hover shaft")
engage_z = target["ztop"] + 0.004 + tcp_above_bottom    # gear bottom just above the shaft tip
move(insert_arm, (tx, ty, engage_z), DOWN, tol=0.003, what="engage")
line(insert_arm, (tx, ty, engage_z), (tx, ty, seat_z), DOWN, step=0.02, tol=0.003, what="insert")
report["seat_ee"] = [round(v, 4) for v in ee(insert_arm)]
report["seat_target_z"] = round(seat_z, 4)
open_gripper(arm=insert_arm)
line(insert_arm, (tx, ty, seat_z), (tx, ty, seat_z + 0.05), DOWN, step=0.1, what="retreat")
try:
    report["steps_used"] = get_task_context().metadata["episode_steps_used"]
except Exception:
    pass
result = report
