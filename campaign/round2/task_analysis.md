# Initial Task Analysis (seed 56, probes probe_scene.py / probe_geometry.py)

- Actual task language: "Pick up the gear and insert it onto the middle of the three shafts on the gear base, seated all the way down."
- Cameras: `top` (metric RGB-D, frame world), `wrist_left`, `wrist_right` (image only). Base frame `world`.
  Home EE: left (0.4975, +0.31, 0.914), right (0.4975, -0.31, 0.914), q = (0.5,-0.5,0.5,-0.5) both.
- Manipulated object and likely prompts: small light-blue hubbed gear lying flat. Prompt "light blue gear"
  (score 0.36; the plate scores higher, 0.41, so bound the projected extent 1.5-7 cm). Radial z-profile
  from the hub centre: hub top flat out to r~20 mm (z 0.788), teeth top 22.5 mm lower (0.766), table 0.7507.
  Gear height ~37.5 mm.
- Goal/handle and likely prompts: "gear base plate" (fallback "blue plate"), plate top (median z) 0.758,
  three shafts (tops 0.788, ~30 mm above plate top) found by raised-cluster search; on seed 56 shafts lie
  in a line along y (-0.395, -0.350, -0.274), middle one at y=-0.350 (positional middle, uneven spacing).
  The robot mounting plates and the gear also answer the plate prompts -> require exactly 3 clusters.
- Visible obstacles and plausible approach: both arms partly cover the near table; park them (joint-0
  swing) before perceiving. Gear at x=0.734 is outside top-down mink IK reach for BOTH arms (0.036 m / 0.091 m
  short). Middle shaft (x 0.69) is reachable top-down by the right arm only, and only up to ~8 cm above the
  shaft top (+0.10 fails by 1.3 cm) -> tight vertical envelope for the insertion approach.
  Plan: tilted pick (pitch search) -> relay set-down near table centre -> top-down re-grasp by the inserting
  arm -> insert on middle shaft.
- Interaction type: multi-stage (pick, relay/regrasp, peg-in-hole insertion).
- Uncertainties to verify from later seeds: which arm reaches the gear / shaft on each seed; bore clearance
  (library reports insertion stalls ~15 mm above the seat with TCP 4-6 mm off axis); mink IK tolerance is
  1 cm, so commanded vs achieved EE xy may differ by several mm; seat height = plate top or a shaft shoulder.
