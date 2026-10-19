# Curriculum — gear-isaac campaign

## Campaign setup and deviations from the standard protocol

- Benchmark: `yam_real` against a yamlab Isaac Sim 6 bridge (`scripts/aspire_bridge_server.py`,
  port 11400), station profile `yam-isaac`, task `gear_assembly:0` (yamlab `PutPotOnCooktop-v0`
  with the factory medium gear and gear base). One task.
- Reduced partition, chosen by the user (option B): development 51–55, unseen-check 66–70,
  held-out 1–10. Not comparable to the paper's 15/15/50 numbers.
- One simulator (single bridge, single GPU): all work is serial. Deviation from Single-GPU
  Operating Mode: the worker runs its own episodes (always `--workers 1`, one cap-harness process
  at a time) instead of handing every command to the coordinator; with one task and one active
  agent this keeps the same one-episode-at-a-time guarantee. The verifier, if any, runs only after
  the worker returns and never alongside Stage 2.
- Bridge settings during this campaign (bridge-only, documented in the bridge's --help): arm
  bases at ASPIRE's station layout; cameras render RGB-D every 3rd control step; finger force
  capped at 20 N (the sim's position-controlled gripper otherwise crushes and ejects the gear);
  grasp detection's pad check off (contact positions are wrong on the GPU pipeline; force check
  and the sustained 5 cm lift remain); placement success compares the gear axis with the medium
  shaft (yamlab task fix). Motions are time-dilated to the measured real-time factor (~0.25–0.3).
- Seeding audited directly on the bridge (2026-10-19): distinct layouts per seed, exact repeats.
  `audit_seeding.py` is LIBERO-specific and was not used.
- Coordinator-side knowledge not given to the worker: a hand-written program
  (`examples/yam_isaac/gear_assembly.py`) and development runs exist; the worker is forbidden to
  read them and gets only the public API docs, the task language and the skill library.

## Round 1 — 2026-10-19T09:05Z

Library now covers: the shipped library (localize, grasp, transport, manipulation, task
exploration), all of it learned on LIBERO-Pro / Robosuite single-arm Panda scenes.
Thin or missing: anything YAM-specific (bimanual reach, vertical top-down grips near the reach
limit, a force-limited gripper on a light part) and peg-in-hole / shaft insertion.
Remaining tasks: 1

Picked: yam_real/gear_assembly/task_0
Why: the only task. Expected to draw on localize.md (SAM3 prompts, disambiguating two light-blue
     parts) and grasp.md / transport.md (top-down grasp, carrying); expected to teach back an
     insertion/placement pattern and YAM reach constraints.
Deferred: none

## Round 1 results — 2026-10-19
- Stage 1 (worker, v05): dev 5/5, unseen 3/5 (66 reach, 70 insertion jam). Audit PASS (coordinator spot-checked fix_code.py).
- Stage 2 held-out 1–10: **3/10** (4, 6, 10 pass).
  - 5 failures (3, 5, 7, 8, 9) raise "no arm reaches the middle shaft top-down". Coordinator truth check: the
    middle shaft is ≥0.486 m from the right arm base on every such seed (66 too); all dev seeds were ≤0.48 m,
    so the development partition never exercised far bases. Likely beyond a vertical-grip insertion envelope;
    needs a non-top-down insertion grasp (gear axis kept vertical) or the task spawn range narrowed.
  - 2 failures (1, 2): insertion stalled ~15 mm above the seat with the TCP 4–6 mm off the shaft axis; seed 4
    stalled identically but passed when the gear dropped on release. Perception was accurate (~3 mm).
- Verifier dispatched after Stage 2 (prompts/verify_gear_assembly_0.txt; addenda_isaac.md appended).
- Verifier: **verified**, 5/5 dev on attempt 1 from the library + proposals alone. All 5 proposals promoted
  (localize.md: park arms; raised shafts / middle of three. grasp.md: tilted search + relay; hubbed-gear
  hub grasp; Key Conventions yam_real jaw-depth note), each with provenance and the verifier's gaps.
  Note: the staged proposed_skills.md said "3/50 held-out" (dispatcher default); the actual figure is 3/10.
- Open for a next round: insertion beyond the top-down envelope (half the base spawn region), and an
  insertion-jam recovery (stall ~15 mm above the seat).

## Round 2 — 2026-10-19 (rerun_02: insertion failures)
Picked: yam_real/gear_assembly/task_0, rerun into task_0/rerun_02 against the updated library.
Why: user asked for a round on the insertion failures (far bases beyond top-down insertion reach;
     insertion jam). Library changed (5 promotions), so it is also the phase-3 rerun candidate.
Coordinator feasibility check (truth poses, scratchpad only): ~45% of seeds 56-100 put the middle
     shaft >= 0.486 m from the right arm base; right-arm mink IK reaches every probed far seat
     (59, 60, 62, 64, 71, 85) with the tool tilted 10-30 deg from vertical. So the task is solvable.
Seed bands (round-1 bands retired; saved in campaign_round1.json):
     dev 56-63 (contiguous, unselected; 59-62 happen to be far bases), unseen-check 71-75.
Held-out: 1-10 rerun for comparability — DISCLOSED REUSE: round 1's held-out failure analysis
     motivated this round (and the coordinator's reach note in grasp.md), so 1-10 is no longer
     clean. Fresh held-out 11-20 is run as well and is the clean number.
The worker is not told about held-out seeds or failure positions; it sees only the library.
- Round 2 Stage 1 (worker, v06 selected): dev 56-63 4/8, unseen 71-75 3/5, audit PASS. Every seed needing a
  tilted insertion fails (dev 59-62, unseen 71): BLOCKED "insertion-jam-tilted" (jaw closing axis tilts with
  pitch; gear pivots in the two-point grip). Top-down seeds pass except unseen 75 (3 mm shallow stall).
  Worker stopped v03-v05 sweeps early after both targeted seeds failed (disclosed in its CHANGELOG).
  Proposed 1 new + 2 edits (mink IK residual check; tilted-insertion limit; held-object top-view bias).
- Bridge restarted twice for viewing (GUI relaunch; then --sam_overlay). A first overlay version drew 3D
  debug-draw markers, which DO render into the robot cameras (5356 magenta px in top); it ran only while
  no episode was running and was replaced by a Kit UI panel, verified 0 overlay pixels in all 3 cameras.
- Stage 2 for rerun_02: seeds 1-20 (1-10 disclosed-reuse, 11-20 clean), output in rerun_02/validation.
- campaign.json heldout_count 10 -> 20 (round 2) so fresh held-out 11-20 can be scored for both rounds.

## Round 2 held-out results (2026-10-19)
- rerun_02/fix_code.py (v06_press_seat), run 97539f1d4326408d: **11/20**. Seeds 1–10: 7/10 (reused seeds; round 1 scored 3/10). Seeds 11–20 (clean): 4/10.
- Same-seed comparison: round-1 fix_code.py on 11–20, run 9d93a251d3300d13 (validation_r1_seeds11_20): **6/10**. Its 4 failures are all far seeds (12, 14, 16, 18), where the program raises an error (program_ok=False).
- Split by ground-truth distance from shaft to right base (>= 0.485 m is far):
  - Round 2 far 3/9 (passes 3, 8, 12); near 8/11.
  - Round 2 **regressed on near seeds 11, 17, 19**, which round 1 passes.
- Keyframes for seeds 11 and 19: after release and press-seat the gear ends up on the table, off the shafts. Likely cause: the new press/retract sequence pulls the gear off when it isn't fully seated. Not confirmed.
- Promotion of the round-2 proposals is on hold until this regression is resolved.
