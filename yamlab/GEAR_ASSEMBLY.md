# Gear assembly in PutPotOnCooktop

`PutPotOnCooktop-v0` now uses the Isaac Lab factory gear-assembly assets in place of the pot
and cooktop. The task is "pick the gear and seat it on its shaft". The assets are the ones
Isaac Lab's `Isaac-Deploy-GearAssembly-UR10e` task spawns (see the Isaac Lab tutorial
`policy_deployment/02_gear_assembly/gear_assembly_policy.rst`).

| Scene key | Asset | Size (x × y × z) | Mass |
|---|---|---|---|
| `pot` (picked) | `factory_gear_medium` | 6.3 × 6.3 × 3.75 cm | 0.026 kg (USD value) |
| `cooktop` (base) | `factory_gear_base` | 22.8 × 10.8 × 3.75 cm | 5.0 kg (heavy so insertion doesn't shove it) |

The scene keys are still `pot`/`cooktop`. This keeps `--asset` flags, recorded datasets and the
mimic env working without a rename.

> **Status (2026-10-19): runs in teleop (GUI) and headless.** The gear spawns and settles on the
> table, and the success check detects a seated gear. **On CPU the resting gear can drift
> 15–38 mm** across resets; on GPU it doesn't. See [Known issues](#known-issues).

| File | Role |
|---|---|
| `scripts/download_gear_assets.py` | Downloads the gear and base, writes `asset_size.json` |
| `yamlab/configs/tasks/put_pot_on_cooktop.yaml` | Gear assets, masses, spawn ranges, success thresholds |
| `yamlab/utils/assets.py` | `resting_origin_z()`: rests objects whose origin is off-center |
| `yamlab/utils/task_logic.py` | `check_ontop_success(expected_z_offset=..., pot_xy_point=..., cooktop_xy_point=...)` for seated placement |
| `yamlab/envs/tasks/put_pot_on_cooktop_manager.py` | Reads the new optional success keys |
| `yamlab/envs/mimic/put_pot_on_cooktop_mimic_env.py` | Respects `pick_by_two_hands` from the YAML |

---

## 1. Setup

```bash
conda activate env_yamlab6
cd ~/yamlab6
python scripts/download_gear_assets.py            # medium gear + base (default)
python scripts/download_gear_assets.py --gear large   # other sizes: small | medium | large
```

The script fetches the USDs and their `configuration/` layers from the Isaac Sim 6.0 asset
server into:

```
yamlab_datasets/tasks_data/PutPotOnCooktop/objects/
├── Gear/factory_gear_medium/      factory_gear_medium.usd, configuration/, asset_size.json
└── GearBase/factory_gear_base/    factory_gear_base.usd,   configuration/, asset_size.json
```

It also makes two changes to the downloaded assets:

- **Instancing off.** The factory USDs mark their visuals and collisions instanceable, so
  nothing can be bound below them. The script writes `instanceable = false` overrides into the
  top-level USD so the task can bind its higher-friction physics material.
- **Extents recorded.** `asset_size.json` stores `min`/`max` as well as `size`. The task needs
  them because the factory origins aren't at the bbox center (next section).

To use the large gear, point the YAML `objects.pot.asset` at `Gear/factory_gear_large`, or pass
`--asset pot=<path>`.

## 2. Launch

Teleop with the GUI (terminal 1). No `--asset` flags are needed; the YAML supplies them.

```bash
python joylo/scripts/launch_follower.py --task PutPotOnCooktop-v0 --port 11333 \
  --demos_per_asset 25 --data_output_dir ./recorded_data --data_output_filename gear_demos \
  --enable_gripper_clamp --enable_cameras --viz kit
```

It's ready when `ss -ltn | grep 11333` shows a line. Then start the leaders in a real terminal
window (terminal 2), as in [TELEOP_QUICKSTART.md](TELEOP_QUICKSTART.md):

```bash
python joylo/scripts/launch_so101.py --left_port $L --right_port $R --enable_recording
```

Demos are saved to `recorded_data/gear_demos`, separate from the pot demos. Stop the follower
with `pkill -9 -f launch_follower.py`.

## 3. How the task works

**Shared frame.** The factory assets share one coordinate frame: a gear fully seated on its
shaft has the same origin as the base. Isaac Lab's medium-shaft offset (x = 0.030375 m in the
base frame) matches the gear's bbox center, which confirms this. Two things follow:

- **Spawn height.** The gear's local bbox starts at z = 0.0075, not at −height/2 like the pot.
  `resting_origin_z()` uses `min.z` when `asset_size.json` has it, so the gear rests on the
  table instead of floating. Assets without `min` behave as before.
- **Success.** The gear's origin is on its rim, 0.030375 m from its axis. Comparing origins
  would therefore also require the gear's yaw to match the base's (within about 9° at 5 mm).
  The check compares the gear's axis with the medium shaft instead, which are both at local
  (0.030375, 0) in the factory frame, so a seated gear passes at any yaw.

**Stages:**

1. **Pick:** either arm lifts the gear 5 cm above its resting height for 8 steps
   (`pick_by_two_hands: false`, since the gear is too small for two grippers).
2. **Place (seated):** with both grippers released, for 8 steps in teleop or 30 in evaluation:

| Check | Value | YAML key (`success:`) |
|---|---|---|
| Gear − base origin height | 0 ± 5 mm | `ontop_z_offset_m`, `ontop_height_tolerance_m` |
| Gear axis − shaft XY distance | < 5 mm | `ontop_xy_tolerance_m`, `ontop_pot_xy_point_m`, `ontop_cooktop_xy_point_m` |
| Gear tilt from upright | < 0.1 rad | `ontop_orientation_tolerance_rad` |

A gear resting on the shaft tip sits about 3 cm high, so it doesn't count as seated. Without
`ontop_z_offset_m` / `ontop_xy_tolerance_m`, the check falls back to the pot behavior (stack
on top, XY tolerance from asset sizes). Without the `*_xy_point_m` keys, it compares origins.

To use the small or large gear, set both XY points to that gear's shaft offset
(Isaac Lab `gear_offsets`: small 0.076125, large −0.045375).

**Spawn randomization.** Gear yaw is ±180° (it's round); the base keeps ±5°. The XY regions
are unchanged from the pot task.

## 4. Verification (2026-10-19)

Headless runs in teleoperation mode:

| Test | CPU | GPU |
|---|---|---|
| Gear and base settle on the table | Yes (base z 0.750, gear 7.5 mm lower) | Yes |
| Gear dropped 2 cm above the shaft slides down; reported seated (origin check, before the axis-point keys) | Yes (dz 0.0, dxy 1.0 mm) | — |
| Resting drift over 2 s, 5–8 resets | 0–38 mm, most resets drift | 0 mm on all 5 |
| Physics material binds to gear and base | Yes | Yes |

The GUI follower launches and opens port 11333. A full teleop demo hasn't been recorded yet.

## Known issues

- **Gear drift on CPU.** A resting gear can creep a few mm/s while slowly yawing. Teleop, replay
  and MimicGen default to CPU. A heavier gear (0.1 and 0.3 kg) and tighter contact offsets
  (0.005 m) didn't help, so the likely cause is PhysX's CPU handling of the gear's SDF
  collision mesh (Isaac Lab runs its gear task on GPU). Options:
  - Run this task on GPU: set `modes.<mode>.sim.device: cuda` in the task YAML. `defaults.yaml`
    pins CPU for "grasp raycast correctness", which hasn't been checked on GPU.
  - Keep CPU and switch the gear's collision to a convex decomposition. More stable, but less
    accurate for the shaft fit.
- **MimicGen isn't ready.** The mimic env's subtask signals
  (`left_grasping → both_grasping → pot_ready → above_obj1`) assume a two-hand pot grasp. The
  gear needs its own subtask breakdown and new annotated demos.
- **Pot data doesn't transfer.** Existing pot demos, annotations and ACT checkpoints don't
  apply to the gear task.

## Going back to the pot

Pass the pot assets explicitly, and restore the pot settings in the task YAML:

```bash
D=yamlab_datasets/tasks_data/PutPotOnCooktop/objects
... --asset pot=$D/Pot/pot_000 --asset cooktop=$D/Cooktop/cooktop_000
```

In `put_pot_on_cooktop.yaml`: remove `ontop_z_offset_m`, `ontop_xy_tolerance_m`,
`ontop_pot_xy_point_m` and `ontop_cooktop_xy_point_m`, set
`ontop_height_tolerance_m: 0.02`, `ontop_orientation_tolerance_rad: 0.2`,
`pick_by_two_hands: true`, pot `mass: 1.3` and pot `orientation_range_deg: 37.5`.
Or restore the file from git: `git checkout 488be69 -- yamlab/configs/tasks/put_pot_on_cooktop.yaml`.
