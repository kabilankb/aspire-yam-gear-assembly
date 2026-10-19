# Blocked: insertion jam when the middle shaft needs a tilted tool (pitch 60-80)

## Root Cause
When the middle shaft is beyond top-down mink reach of the right arm (shaft x >= ~0.76 m on yam-isaac; dev 59 60 61 62,
unseen 71), the only solving insertion orientations are pitched 60-80 deg. A gear held that way does not go down the
shaft: the descent stalls 11-14 mm above the seat (EE drifts +3..+5 mm in x), and the gear either jams tilted or pivots
out of the two-point grip and drops beside the plate (wrist video, v01 seed 60). Measured contributing factors:
- The jaw closing axis is tool y, which at yaw 0 / roll 0 is world x. `tool_quat` pitches about world y, so at pitch 75 the
  closing axis is 15 deg off horizontal. The gear hangs from a tilted contact line, so it sits tilted in the hand.
- The gear also slides/pivots in the grasp during transport: a set-down calibration (probe_calib.py, seed 60, pitch 75)
  showed the in-hand y offset growing from +1.4 to +13.6 mm over the trajectory, with a true set-down offset of +11.8 mm in y.

## Details
- IK selection: mink reports `position_error_m` even when `ok`; v01 accepted poses up to 9 mm off. v02+ required <= 2 mm,
  which removed the IK error but did not fix the jam.
- The top-camera reading of a HELD gear (hub-top band, z within [EE-hold, EE+depth]) is biased -18..-25 mm in x
  (position dependent) while y is within ~2 mm, judged against set-down ground truth. Correcting with it (v03) put the gear 2 cm off
  the shaft. Do not servo on it without a calibration.

## What Was Tried
- v01: orientation search over pitch/yaw, grasp in the same orientation: 61 passed once (pitch 80); 59 60 62 jammed.
- v02: in-hand visual alignment (measurement invalid: robot points in the mask) + engage 8 mm, release, press: 0/4 tilted;
  the early release also cost two top-down seeds.
- v03: z-banded measurement + servo alignment: alignment moved the gear off the shaft (bias above).
- v04/v05: roll the tool 90 deg so the closing axis stays horizontal: relay pick needed roll 0 (v04 had no relay point);
  with roll 90 the regrasp closed wide (0.58-0.77, apparently on the teeth) and the seat still stalled 12.5 mm high, twice.
- v06: roll 0 + lift-and-retry + release/press: still 0/4 tilted.
Untested ideas: deeper hub grasp for tilted inserts (12-14 mm), a press that is checked by re-perception, a regrasp at a
relay point nearer the plate so the insertion pitch stays above 85 deg.
