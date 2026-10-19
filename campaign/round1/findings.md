# Findings: yam_real/gear_assembly/task_0

(Written by the coordinator from the worker's hand-back; the worker's Write of this file was refused.)

## Root causes observed
- reach (v00: 51, 52, 53, 55): 10 cm hover above the shaft outside the right arm's top-down envelope; gear sometimes beyond both arms' top-down reach (53: 0.155 m short; 54: 3 cm short).
- perception (v00: 54): right arm at home occludes one end of the base in `top`; only 2/3 shafts found.
- execution time (v01: 52 killed): 121 motions x ~3.7 s = 559 s/seed, over the 10-min command limit.
- grasp retention (v02: 51; v03: 52): re-grasp tipped the gear; 16 mm-deep grasp closed on the teeth; tilted gear jammed ~1.2 cm onto the shaft.

## What fixed them
- v01: park arms first, search grasp orientations, relay via table point, hover 2 cm above shaft tips.
- v02: ~25 motion segments, 182 s/seed.
- v04: 10 mm hub grasp + jaw-width check vs first grasp, one retry -> 5/5.
- v05 (selected): ablation of v04 with jaw fully open before descent; also 5/5.

## Development vs unseen
Dev 5/5, unseen 3/5. Residual regimes: shaft beyond top-down envelope (66, no fallback); insertion jam with no recovery (70).

## Generalizable patterns
See skill_report.json proposed_new[0..3]: park YAM arms out of top view; find raised shafts & pick middle; reach beyond top-down envelope via pitch/yaw search + relay re-grasp; hubbed-gear grasp at 10 mm depth.

## Blocked modes
none

## Non-privileged API audit
PASS (worker). Coordinator spot-check of fix_code.py: no world coordinates, seed branches, success/reward reads or bridge access.
