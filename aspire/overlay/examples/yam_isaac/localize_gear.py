# Localize the gear and the gear base from the top camera (SAM3 + depth).
report = {}
for query in ("small metal gear", "gear base with shafts"):
    found = localize_object(query, camera_name="top")
    if found.ok:
        report[query] = {
            "center": [round(float(v), 4) for v in found.geometry.pose.position],
            "extents": [round(float(v), 4) for v in found.geometry.extents],
            "points": found.geometry.point_count,
            "score": round(float(found.segmentation.score), 3),
        }
    else:
        report[query] = {"error": str(found.error)}
print(report)
result = report
