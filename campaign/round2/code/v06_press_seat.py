# v06 - not-fully-seated + consolidation: v05 with roll 0 (as v00/v01); full held descent, one lift-and-retry,
# release, then press the hub down with a part-open gripper (fixes the 3.5 mm shallow stall of v01 seed 56).
# v05 - relay reach: the relay pick keeps roll 0 (the table re-flattens the gear); the inserting arm may
# regrasp at the relay with any yaw at the insertion pitch/roll (a yaw about world z keeps the gear level);
# wider relay grid; relay failures are counted in the report.
# v04 - insertion jam (tilted grasps): roll the tool 90 deg about its approach axis so the jaw closing axis
# (tool y) stays horizontal at any pitch/yaw; no in-hand vision correction (top-view x reading biased);
# full held descent, one lift-and-retry on a stall, then release + press.
# (v03 - insertion jam: (a) insertion orientation must solve with IK error <= 2 mm, with a +-2 cm margin
# for corrections; (b) held-gear hub centre measured in a z band [EE-hold, EE+depth] with the top camera,
# EE corrected until the residual is < 1.5 mm; (c) full held descent (v00 style), one re-align retry on
# a stall, then release and press the hub down at the corrected xy with a part-open gripper.
# (v01: insertion orientation searched over pitch/yaw; gear grasped in that same orientation.)
# Park arms -> perceive gear / plate / shafts -> choose inserting arm (top-down IK at middle shaft)
# -> pick (top-down if reachable, else tilted pick + table relay + top-down regrasp) -> insert -> release.
import math

state0 = get_robot_state()
BASE = state0.base_frame
home_joints = {a: [float(v) for v in state0.joint_positions[a]] for a in ("left", "right")}
home_pos = {a: tuple(float(v) for v in state0.end_effector_poses[a].position) for a in ("left", "right")}
home_q = {a: tuple(float(v) for v in state0.end_effector_poses[a].quaternion_wxyz) for a in ("left", "right")}
STRAT = MotionStrategy(ik_solver="mink", trajectory_planner="interpolation")
report = {"task": get_task_context().language, "log": []}

def log(*items):
    report["log"].append(" ".join(str(i) for i in items))

def qtile(vals, f):
    s = sorted(vals)
    return s[int((len(s) - 1) * f)]

# ---------------- motion helpers ----------------
OUTWARD = {"left": 1.0, "right": -1.0}
PARK_SWING_RAD = 0.7

def park(arm):
    target = list(home_joints[arm])
    target[0] = target[0] + OUTWARD[arm] * PARK_SWING_RAD
    r = move_to_joints(target, tolerance=0.02, max_steps=200, arm=arm)
    if not r.ok:
        r = move_to_joints(target, tolerance=0.02, max_steps=200, arm=arm)
    return bool(r.ok)

def ee(arm):
    return tuple(float(v) for v in get_robot_state().end_effector_poses[arm].position)

def mv(arm, pos, q, label, tol=0.01, steps=250):
    r = move_to_pose(Pose(tuple(pos), q, BASE), tolerance=tol, max_steps=steps, arm=arm, strategy=STRAT)
    got = ee(arm)
    err = ""
    if not r.ok:
        try:
            err = str(r.error.message)[:120]
        except Exception:
            err = str(r.status)
    log(label, arm, "ok" if r.ok else "FAIL", [round(v, 4) for v in pos], "got", [round(v, 4) for v in got], err)
    return bool(r.ok)

def ik_ok(arm, pos, q):
    return bool(solve_ik(Pose(tuple(pos), q, BASE), arm=arm, backend="mink").ok)

def ik_tight(arm, pos, q, max_err=0.002):
    r = solve_ik(Pose(tuple(pos), q, BASE), arm=arm, backend="mink")
    if not r.ok:
        return False
    try:
        return float(r.diagnostics["position_error_m"]) <= max_err
    except Exception:
        return True

def qmul(a, b):
    return (a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3], a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
            a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1], a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0])

TOOL_ROLL_DEG = 0.0    # roll 90 (level jaw axis) was tried in v04/v05: jaws closed on the teeth (width 0.58-0.77), no gain

def tool_quat(arm, pitch_deg, yaw_deg, roll_deg=None):   # home tool points +x; pitch 90 = straight down
    if roll_deg is None:
        roll_deg = TOOL_ROLL_DEG
    p = math.radians(pitch_deg) / 2.0
    y = math.radians(yaw_deg) / 2.0
    r = math.radians(roll_deg) / 2.0
    base = qmul((math.cos(y), 0.0, 0.0, math.sin(y)), qmul((math.cos(p), 0.0, math.sin(p), 0.0), home_q[arm]))
    return qmul(base, (math.cos(r), 0.0, 0.0, math.sin(r)))

def approach_dir(q):   # tool z axis in world
    w, x, y, z = q
    return (2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y))

def grasp_plan(arm, gx, gy, gz, stand_off=0.08):
    for pitch, yaw in [(90, 0)] + [(p, y) for p in (75, 60, 45, 30) for y in (0, 30, -30)]:
        q = tool_quat(arm, pitch, yaw, 0.0)
        a = approach_dir(q)
        pre = (gx - stand_off * a[0], gy - stand_off * a[1], gz - stand_off * a[2])
        if ik_ok(arm, (gx, gy, gz), q) and ik_ok(arm, pre, q):
            return {"q": q, "pre": pre, "grasp": (gx, gy, gz), "pitch": pitch, "yaw": yaw, "a": a}
    return None

# ---------------- perception helpers ----------------
def find_gear():
    seg = segment_text("top", "light blue gear")
    best = None
    if seg.ok:
        for cand in seg.segmentations[:8]:
            cloud = mask_to_point_cloud(cand, "top", target_frame=BASE)
            if len(cloud.points) < 30:
                continue
            xs = [float(p[0]) for p in cloud.points]
            ys = [float(p[1]) for p in cloud.points]
            ext_x = qtile(xs, 0.95) - qtile(xs, 0.05)
            ext_y = qtile(ys, 0.95) - qtile(ys, 0.05)
            if 0.015 < ext_x < 0.07 and 0.015 < ext_y < 0.07 and (best is None or float(cand.score) > best[0]):
                best = (float(cand.score), cloud)
    if best is None:
        return None
    pts = best[1].points
    zs = [float(p[2]) for p in pts]
    hub_top = qtile(zs, 0.98)
    top = [p for p in pts if float(p[2]) > hub_top - 0.004]
    gx = sum(float(p[0]) for p in top) / len(top)
    gy = sum(float(p[1]) for p in top) / len(top)
    return {"x": gx, "y": gy, "hub_top": hub_top, "n": len(pts), "score": best[0]}

def raised_clusters(xs, ys, zs, surface_z, min_rise=0.012, join_r=0.015, min_pts=8):
    order = sorted([i for i in range(len(zs)) if zs[i] > surface_z + min_rise], key=lambda i: -zs[i])
    groups = []
    for i in order:
        home = None
        for g in groups:
            if math.hypot(xs[i] - g["sx"], ys[i] - g["sy"]) < join_r:
                home = g
                break
        if home is None:
            groups.append({"sx": xs[i], "sy": ys[i], "m": [i]})
        else:
            home["m"].append(i)
    out = []
    for g in groups:
        if len(g["m"]) >= min_pts:
            zmax = max(zs[i] for i in g["m"])
            topi = [i for i in g["m"] if zs[i] > zmax - 0.004]
            out.append({"x": sum(xs[i] for i in topi) / len(topi), "y": sum(ys[i] for i in topi) / len(topi), "ztop": zmax, "n": len(g["m"])})
    return out

def find_plate(gear):
    for prompt in ("gear base plate", "blue plate"):
        s = segment_text("top", prompt)
        if not s.ok:
            continue
        for cand in s.segmentations[:6]:
            cloud = mask_to_point_cloud(cand, "top", target_frame=BASE)
            if len(cloud.points) < 100:
                continue
            xs = [float(p[0]) for p in cloud.points]
            ys = [float(p[1]) for p in cloud.points]
            zs = [float(p[2]) for p in cloud.points]
            span = max(qtile(xs, 0.95) - qtile(xs, 0.05), qtile(ys, 0.95) - qtile(ys, 0.05))
            if span < 0.06 or span > 0.30:
                continue
            surface_z = qtile(zs, 0.5)
            cl = raised_clusters(xs, ys, zs, surface_z)
            if gear is not None:
                cl = [c for c in cl if math.hypot(c["x"] - gear["x"], c["y"] - gear["y"]) > 0.05]
            if len(cl) != 3:
                continue
            tops = [c["ztop"] for c in cl]
            if max(tops) - min(tops) > 0.01 or min(tops) < surface_z + 0.015:
                continue
            return {"prompt": prompt, "surface_z": surface_z, "shafts": cl,
                    "xmin": qtile(xs, 0.02), "xmax": qtile(xs, 0.98), "ymin": qtile(ys, 0.02), "ymax": qtile(ys, 0.98)}
    return None

def middle_of_three(shafts):
    pairs = [(math.hypot(shafts[i]["x"] - shafts[j]["x"], shafts[i]["y"] - shafts[j]["y"]), i, j) for i in range(3) for j in range(i + 1, 3)]
    far = max(pairs)
    return shafts[[k for k in range(3) if k != far[1] and k != far[2]][0]]

def table_z():
    s = segment_text("top", "wooden table")
    if s.ok:
        for cand in s.segmentations[:3]:
            cloud = mask_to_point_cloud(cand, "top", target_frame=BASE)
            if len(cloud.points) > 1000:
                return qtile([float(p[2]) for p in cloud.points], 0.5)
    return None

# ---------------- program ----------------
HUB_GRASP_DEPTH = 0.010
ORIENTS = [(90, 0)] + [(p, y) for p in (80, 75, 70, 65, 60, 55, 50, 45) for y in (0, -20, 20, -40, 40)]

def grasp_with(arm, q, gx, gy, gz, label):
    a = approach_dir(q)
    pre = (gx - 0.08 * a[0], gy - 0.08 * a[1], gz - 0.08 * a[2])
    open_gripper(arm=arm)
    if not mv(arm, pre, q, label + "-pre"):
        pre = (gx - 0.05 * a[0], gy - 0.05 * a[1], gz - 0.05 * a[2])
        mv(arm, pre, q, label + "-pre-short")
    if not mv(arm, (gx, gy, gz), q, label + "-grasp", tol=0.008):
        raise ValueError(label + " grasp approach failed")
    close_gripper(arm=arm)
    width = float(get_robot_state().gripper_positions[arm])
    log("grip width", arm, round(width, 3))
    mv(arm, pre, q, label + "-retreat")
    return width

def reach_ok(arm, q, gx, gy, gz, stand_off=0.08):
    a = approach_dir(q)
    return ik_ok(arm, (gx, gy, gz), q) and ik_ok(arm, (gx - stand_off * a[0], gy - stand_off * a[1], gz - stand_off * a[2]), q)


PRESS_WIDTH = 0.35   # part-open jaw: fingertips straddle the bore and land on the hub's top face

def find_held_gear(near, min_z, max_z=None):
    # gear points with min_z < z < max_z within 4.5 cm (xy) of `near`; returns the hub-top centre
    seg = segment_text("top", "light blue gear")
    best = None
    if seg.ok:
        for cand in seg.segmentations[:8]:
            cloud = mask_to_point_cloud(cand, "top", target_frame=BASE)
            pts = [p for p in cloud.points if float(p[2]) > min_z and (max_z is None or float(p[2]) < max_z) and
                   math.hypot(float(p[0]) - near[0], float(p[1]) - near[1]) < 0.045]
            if len(pts) < 30:
                continue
            if best is None or float(cand.score) > best[0]:
                best = (float(cand.score), pts)
    if best is None:
        return None
    best = best[1]
    zs = [float(p[2]) for p in best]
    top_z = qtile(zs, 0.98)
    top = [p for p in best if float(p[2]) > top_z - 0.004]
    xs = sorted(float(p[0]) for p in best)
    ys = sorted(float(p[1]) for p in best)
    return {"x": sum(float(p[0]) for p in top) / len(top), "y": sum(float(p[1]) for p in top) / len(top),
            "hub_top": top_z, "n": len(best), "ntop": len(top),
            "bx": 0.5 * (qtile(xs, 0.03) + qtile(xs, 0.97)), "by": 0.5 * (qtile(ys, 0.03) + qtile(ys, 0.97))}

def held_band(arm, hold):
    e = ee(arm)
    return find_held_gear(e, e[2] - hold - 0.004, e[2] + HUB_GRASP_DEPTH + 0.004), e

ALIGN_MARGIN = 0.02   # room left around the shaft for in-hand offset corrections

def margin_ok(arm, q, x, y, zs, tol):
    for ddx, ddy in ((ALIGN_MARGIN, 0.0), (-ALIGN_MARGIN, 0.0), (0.0, ALIGN_MARGIN), (0.0, -ALIGN_MARGIN)):
        for z in zs:
            if not ik_tight(arm, (x + ddx, y + ddy, z), q, max(tol, 0.003)):
                return False
    return True

def main():
    park("left")
    park("right")
    gear = find_gear()
    if gear is None:
        raise ValueError("gear not found")
    plate = find_plate(gear)
    if plate is None:
        raise ValueError("gear base with three shafts not found")
    shaft = middle_of_three(plate["shafts"])
    tz = table_z()
    if tz is None:
        raise ValueError("table not found")
    report["gear"] = {k: round(float(v), 4) for k, v in gear.items()}
    report["shafts"] = [[round(c["x"], 4), round(c["y"], 4), round(c["ztop"], 4)] for c in plate["shafts"]]
    report["middle"] = [round(shaft["x"], 4), round(shaft["y"], 4), round(shaft["ztop"], 4)]
    report["plate_surface_z"] = round(plate["surface_z"], 4)
    report["table_z"] = round(tz, 4)

    grasp_z = gear["hub_top"] - HUB_GRASP_DEPTH
    hold = grasp_z - tz                      # EE height above the gear's bottom face
    seat_z = plate["surface_z"] + hold       # EE z with the gear bottom on the plate top
    entry_z = shaft["ztop"] + hold + 0.004
    above_z = shaft["ztop"] + hold + 0.02
    engage_z = shaft["ztop"] + hold - 0.008  # gear bottom 8 mm onto the shaft
    seated_hub_top = plate["surface_z"] + (gear["hub_top"] - tz)
    press_hi = seated_hub_top + 0.03
    report["hold"] = round(hold, 4)

    # insertion orientation: highest pitch whose seat / entry / above poses all solve
    insert_arm = None
    q_ins = None
    for ik_tol in (0.002, 0.010):
        best_pitch = -1
        for arm in ("right", "left"):
            for pitch, yaw in ORIENTS:
                if pitch <= best_pitch:
                    break
                q = tool_quat(arm, pitch, yaw)
                if ik_tight(arm, (shaft["x"], shaft["y"], seat_z - 0.003), q, ik_tol) and ik_tight(arm, (shaft["x"], shaft["y"], engage_z), q, ik_tol) \
                   and ik_tight(arm, (shaft["x"], shaft["y"], above_z), q, ik_tol) and ik_tight(arm, (shaft["x"], shaft["y"], press_hi), q, ik_tol):
                    insert_arm = arm
                    q_ins = q
                    best_pitch = pitch
                    report["insert"] = {"arm": arm, "pitch": pitch, "yaw": yaw, "ik_tol": ik_tol}
                    break
        if insert_arm is not None:
            break
    if insert_arm is None:
        raise ValueError("no arm reaches the middle shaft")
    other = "left" if insert_arm == "right" else "right"

    # --- pick in the insertion orientation, directly or via a relay ---
    if reach_ok(insert_arm, q_ins, gear["x"], gear["y"], grasp_z):
        report["direct_pick"] = True
        width = grasp_with(insert_arm, q_ins, gear["x"], gear["y"], grasp_z, "pick")
    else:
        report["direct_pick"] = False
        plan = None
        pick_arm = None
        for arm in (other, insert_arm):
            p = grasp_plan(arm, gear["x"], gear["y"], grasp_z)
            if p is not None and (plan is None or p["pitch"] > plan["pitch"]):
                plan = p
                pick_arm = arm
        if plan is None:
            raise ValueError("gear unreachable by either arm")
        report["pick"] = {"arm": pick_arm, "pitch": plan["pitch"], "yaw": plan["yaw"]}
        mid_x = 0.5 * (home_pos["left"][0] + home_pos["right"][0])
        mid_y = 0.5 * (home_pos["left"][1] + home_pos["right"][1])
        cands = []
        for ix in range(-4, 5):
            for iy in range(-6, 7):
                cands.append((mid_x + 0.04 * ix, mid_y + 0.04 * iy))
        cands = sorted(cands, key=lambda c: math.hypot(c[0] - mid_x, c[1] - mid_y))
        relay = None
        q_regrasp = None
        a = plan["a"]
        fails = {"near_gear": 0, "near_plate": 0, "pick_arm": 0, "insert_arm": 0}
        ins = report["insert"]
        for cx, cy in cands:
            if math.hypot(cx - gear["x"], cy - gear["y"]) < 0.08:
                fails["near_gear"] += 1
                continue
            if plate["xmin"] - 0.08 < cx < plate["xmax"] + 0.08 and plate["ymin"] - 0.08 < cy < plate["ymax"] + 0.08:
                fails["near_plate"] += 1
                continue
            if not (reach_ok(pick_arm, plan["q"], cx, cy, grasp_z + 0.003) and ik_ok(pick_arm, (cx, cy, grasp_z + 0.05), plan["q"])):
                fails["pick_arm"] += 1
                continue
            for yaw in [ins["yaw"]] + [v for v in (0, -30, 30, -60, 60, -90, 90) if v != ins["yaw"]]:
                qg = tool_quat(insert_arm, ins["pitch"], yaw)
                if reach_ok(insert_arm, qg, cx, cy, grasp_z):
                    q_regrasp = qg
                    break
            if q_regrasp is None:
                fails["insert_arm"] += 1
                continue
            relay = (cx, cy)
            break
        report["relay_fails"] = fails
        if relay is None:
            raise ValueError("no relay point")
        report["relay"] = [round(relay[0], 3), round(relay[1], 3)]
        grasp_with(pick_arm, plan["q"], gear["x"], gear["y"], grasp_z, "pick")
        mv(pick_arm, (relay[0], relay[1], grasp_z + 0.05), plan["q"], "relay-above")
        mv(pick_arm, (relay[0], relay[1], grasp_z + 0.003), plan["q"], "relay-down")
        open_gripper(arm=pick_arm)
        mv(pick_arm, (relay[0] - 0.08 * a[0], relay[1] - 0.08 * a[1], grasp_z + 0.003 - 0.08 * a[2]), plan["q"], "relay-retreat")
        park(pick_arm)
        gear2 = find_gear()
        if gear2 is None:
            raise ValueError("gear lost after relay")
        report["gear_relay"] = {k: round(float(v), 4) for k, v in gear2.items()}
        g2z = gear2["hub_top"] - HUB_GRASP_DEPTH
        width = grasp_with(insert_arm, q_regrasp, gear2["x"], gear2["y"], g2z, "regrasp")
    report["grip_width"] = round(width, 3)

    # --- insert ---
    a = approach_dir(q_ins)

    def align(tx, ty, z):
        mv(insert_arm, (tx, ty, z), q_ins, "above-shaft")
        return tx, ty

    report["align"] = []
    tx, ty = align(shaft["x"], shaft["y"], above_z)
    seated = False
    for trial in range(2):
        mv(insert_arm, (tx, ty, entry_z), q_ins, "shaft-entry", tol=0.006)
        mv(insert_arm, (tx, ty, seat_z - 0.003), q_ins, "seat", tol=0.005)
        e = ee(insert_arm)
        report["ee_at_seat_%d" % trial] = [round(v, 4) for v in e]
        if e[2] < seat_z + 0.006:
            seated = True
            break
        if trial == 0:
            log("stalled", round(e[2] - seat_z, 4), "above seat: lift and retry")
            tx, ty = align(tx, ty, above_z)
    report["seated_by_ee"] = seated
    open_gripper(arm=insert_arm)
    mv(insert_arm, (tx - 0.03 * a[0], ty - 0.03 * a[1], seat_z - 0.03 * a[2]), q_ins, "release-retreat")
    set_gripper(PRESS_WIDTH, arm=insert_arm)
    mv(insert_arm, (tx, ty, press_hi), q_ins, "press-above")
    mv(insert_arm, (tx, ty, seated_hub_top - 0.006), q_ins, "press", tol=0.004)
    report["ee_at_press"] = [round(v, 4) for v in ee(insert_arm)]
    mv(insert_arm, (tx, ty, press_hi), q_ins, "press-retreat")
    park(insert_arm)
    gear3 = find_held_gear((shaft["x"], shaft["y"], 0.0), plate["surface_z"] + 0.02)
    if gear3 is not None:
        report["gear_final"] = {k: round(float(v), 4) for k, v in gear3.items()}
        report["seated_hub_top_expected"] = round(seated_hub_top, 4)

try:
    main()
except Exception as exc:
    report["error"] = str(exc)
    log("ERROR", str(exc))
result = report
