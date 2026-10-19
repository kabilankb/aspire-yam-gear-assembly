# Pick the medium gear and seat it on the medium shaft of the gear base (yam-isaac station).
#
# Perception: SAM3 + top-camera depth. "Light blue" matches both the gear and the base with
# close scores whose order flips between frames, so each object is chosen from SAM3's
# candidates by its expected size rather than by rank. The gear axis is the centroid of the
# gear's top face; the medium shaft is the middle of the base's three shafts, located from
# their top caps (the camera sees each shaft's near side, so whole-shaft centroids are biased).
import math

ARM = "right"
DOWN = [0.0, 0.707107, -0.707107, 0.0]  # grasp site approach axis pointing -z (wxyz)
STRATEGY = MotionStrategy(ik_solver="mink", trajectory_planner="interpolation")
SHAFT_HEIGHT = 0.0375                      # factory base: shafts rise this far above its bottom
CAP_BAND = 0.004                           # points this close to an object's top count as its cap
GRASP_DEPTH = 0.02  # grasp site this far below the gear's top: deep enough that the pads hold it rigidly
HOVER_Z = 0.845  # grasp-site height for approach/lift: a vertical YAM grip reaches ~0.85 m here
report = {}


def find_object(query, min_size, max_size):
    """Point cloud of the best-scoring SAM3 candidate whose largest XY extent fits the size."""
    found = segment_text("top", query)
    if not found.ok:
        return None, str(found.error)
    seen = []
    for candidate in found.segmentations:
        if float(candidate.score) < 0.1:
            continue
        cloud = mask_to_point_cloud(candidate, "top", target_frame="world")
        size = max(float(v) for v in estimate_geometry(cloud).extents[:2])
        seen.append((round(float(candidate.score), 3), round(size, 3)))
        if min_size <= size <= max_size:
            return cloud.points, seen
    return None, seen


def top_cap_xy(points, band):
    top = max(float(p[2]) for p in points)
    cap = [p for p in points if float(p[2]) > top - band]
    return (sum(float(p[0]) for p in cap) / len(cap), sum(float(p[1]) for p in cap) / len(cap), top, len(cap))


def medium_shaft(points):
    """Middle of the three shaft caps along the base's long axis."""
    n = len(points)
    cx = sum(float(p[0]) for p in points) / n
    cy = sum(float(p[1]) for p in points) / n
    sxx = sum((float(p[0]) - cx) ** 2 for p in points) / n
    syy = sum((float(p[1]) - cy) ** 2 for p in points) / n
    sxy = sum((float(p[0]) - cx) * (float(p[1]) - cy) for p in points) / n
    ang = 0.5 * math.atan2(2 * sxy, sxx - syy)
    ux, uy = math.cos(ang), math.sin(ang)
    bottom = min(float(p[2]) for p in points)
    caps = sorted(
        ((float(p[0]) - cx) * ux + (float(p[1]) - cy) * uy, p)
        for p in points
        if float(p[2]) > bottom + SHAFT_HEIGHT - CAP_BAND
    )
    clusters = []
    for t, p in caps:
        if clusters and t - clusters[-1][-1][0] < 0.012:
            clusters[-1].append((t, p))
        else:
            clusters.append([(t, p)])
    if len(clusters) != 3:
        return None, [len(c) for c in clusters]
    mid = clusters[1]
    x = sum(float(p[0]) for t, p in mid) / len(mid)
    y = sum(float(p[1]) for t, p in mid) / len(mid)
    gaps = [round(sum(t for t, p in clusters[i + 1]) / len(clusters[i + 1]) - sum(t for t, p in clusters[i]) / len(clusters[i]), 4) for i in range(2)]
    return (x, y, bottom + SHAFT_HEIGHT), gaps


def yawed(q, yaw):
    """World-frame rotation about +z applied to quaternion q (wxyz)."""
    c, s = math.cos(yaw / 2), math.sin(yaw / 2)
    w, x, y, z = q
    return [c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w]


def reachable_grip(points):
    """First top-down grip yaw for which IK reaches every point (the gear is round)."""
    for deg in (0, 30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180):
        q = yawed(DOWN, math.radians(deg))
        if all(solve_ik(Pose(position=p, quaternion_wxyz=q, frame="world"), arm=ARM, backend="mink").ok
               for p in points):
            return q, deg
    return None, None


GRIP = DOWN


def move(name, x, y, z, steps=300, tol=0.005):
    result = move_to_pose(Pose(position=(x, y, z), quaternion_wxyz=GRIP, frame="world"),
                          tolerance=tol, max_steps=steps, arm=ARM, strategy=STRATEGY)
    report[name] = result.ok
    if not result.ok:
        print("FAILED", name, result.error)
    return result.ok


def site():
    p = get_robot_state().end_effector_poses[ARM].position
    return float(p[0]), float(p[1]), float(p[2])


gear, gear_seen = find_object("round light blue part", 0.04, 0.09)
base, base_seen = find_object("metal base plate", 0.18, 0.30)
print("gear candidates (score, size)", gear_seen, "base candidates", base_seen)
ok = gear is not None and base is not None
if ok:
    gx, gy, gtop, gcap = top_cap_xy(gear, CAP_BAND)
    shaft, gaps = medium_shaft(base)
    print("gear axis", round(gx, 4), round(gy, 4), "top", round(gtop, 4), "cap points", gcap)
    print("medium shaft", shaft, "cap gaps", gaps)
    ok = shaft is not None
if ok:
    GRIP, grip_yaw = reachable_grip([(gx, gy, HOVER_Z), (gx, gy, gtop - GRASP_DEPTH),
                                     (shaft[0], shaft[1], HOVER_Z), (shaft[0], shaft[1], shaft[2] + 0.02)])
    print("grip yaw", grip_yaw)
    ok = GRIP is not None
    report["reachable"] = ok

if ok:
    open_gripper(arm=ARM)
    grasp_z = gtop - GRASP_DEPTH
    ok = move("pregrasp", gx, gy, HOVER_Z) and move("grasp", gx, gy, grasp_z, tol=0.004)
if ok:
    sx, sy, sz = site()
    close_gripper(arm=ARM)
    width = float(get_robot_state().gripper_positions[ARM])
    print("site at grasp", round(sx, 4), round(sy, 4), round(sz, 4), "gripper after close", round(width, 3))
    ok = width > 0.05  # fully closed means the fingers missed the gear
    report["holding"] = ok
if ok:
    ok = move("lift", sx, sy, HOVER_Z)
if ok:
    # Keep the gear-in-hand offset from the grasp; the gear bottom starts at gtop - 0.0375.
    off_x, off_y = sx - gx, sy - gy
    gear_bottom_below_site = sz - (gtop - 0.0375)
    shaft_x, shaft_y, shaft_top = shaft
    ok = move("over shaft", shaft_x + off_x, shaft_y + off_y, shaft_top + 0.03 + gear_bottom_below_site)
    # Drop, don't push: lower until the gear's bore is just above the shaft tip and let go.
    # A light gear held in a force-limited grip pivots in the fingers when pushed against the
    # shaft edge; released aligned, it falls and slides down the shaft on its own.
    ok = ok and move("align", shaft_x + off_x, shaft_y + off_y, shaft_top + 0.01 + gear_bottom_below_site,
                     tol=0.004)
if ok:
    open_gripper(arm=ARM)
    sx, sy, sz = site()
    move("retreat", sx, sy, HOVER_Z)

print(report)
result = report
