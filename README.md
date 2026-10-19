# aspire-yam-gear-assembly

Run [NVIDIA ASPIRE](https://github.com/NVlabs/ASPIRE) on YAM bimanual arms in Isaac Sim for a gear-insertion task.

ASPIRE (agentic code-as-policies) has an AI agent write the robot's control program as Python. The agent runs it, reads the failures, fixes the code, and turns lessons that hold up into verified, reusable skills. This package connects ASPIRE's YAM station API to a simulated YAM in Isaac Lab, so the whole loop runs in simulation.

**Task:** pick up the medium factory gear and seat it on the middle shaft of the gear base.

```
ASPIRE cap-harness ──portal RPC :11400──▶ aspire_bridge_server.py ──▶ Isaac Lab (yamlab)
  agent program                           joints, gripper, RGB-D        YAM bimanual + gear task
  SAM3 · mink IK · interpolation          reset(seed), task info
```

## Contents

| Path | What it is |
|---|---|
| `yamlab/scripts/aspire_bridge_server.py` | Serves the Isaac Lab task to ASPIRE: arm commands, RGB-D cameras, seeded resets, task success. `--sam_overlay` adds a live SAM3 panel to the GUI. |
| `yamlab/scripts/download_gear_assets.py` | Downloads the Isaac Lab factory gear and gear base USDs |
| `yamlab/patches/0001-gear-assembly-task.patch` | Turns yamlab's `PutPotOnCooktop-v0` into the gear task (see `yamlab/GEAR_ASSEMBLY.md`) |
| `aspire/overlay/` | New ASPIRE files: the `yam-isaac` station profile and calibration, `yam_real/isaac.py` (bridge client and adapter), `scripts/check_isaac_bridge.py`, example programs |
| `aspire/patches/0001-cap-harness-isaac-bridge.patch` | Adds `--isaac-bridge` / `CAP_HARNESS_ISAAC_BRIDGE` to `cap-harness run` and `run-batch` |
| `aspire/patches/0002-learned-skills-round1.patch` | Skills learned and verified in round 1, added to `grasp.md` and `localize.md` |
| `campaign/` | Programs, analyses and results from the two ASPIRE rounds |

Tested against ASPIRE `fa00f0e` and yamlab `488be69`.

## Install

```bash
git clone https://github.com/kabilankb/aspire-yam-gear-assembly.git
cd aspire-yam-gear-assembly
./install.sh --aspire ~/ASPIRE --yamlab ~/yamlab6     # add --no-skills to leave ASPIRE's skill library untouched
```

The script copies the new files and applies the patches with `git apply`. Running it again is safe.

## Run

```bash
# 1. Gear assets (yamlab env)
conda activate env_yamlab6 && cd ~/yamlab6
python scripts/download_gear_assets.py

# 2. Bridge: headless, or with the GUI and a live SAM3 panel
python scripts/aspire_bridge_server.py --task PutPotOnCooktop-v0 --headless
python scripts/aspire_bridge_server.py --task PutPotOnCooktop-v0 --viz kit --sam_overlay

# 3. ASPIRE model services (SAM3, mink, ...), in another shell
cd ~/ASPIRE && scripts/supervise_services.sh --profile rtx5090

# 4. Check the bridge against ASPIRE's station model
.venv-yam-real/bin/python scripts/check_isaac_bridge.py

# 5. Run a program on one seed
export CAP_HARNESS_ISAAC_BRIDGE=127.0.0.1:11400 CAP_HARNESS_YAM_STATION=yam-isaac
.venv-yam-real/bin/cap-harness run --benchmark yam_real --suite gear_assembly --task-id 0 \
    --seed 13 --program examples/yam_isaac/gear_assembly.py
```

The simulation never needs `--allow-motion`. The `yam-isaac` station only talks to the bridge, never to hardware.

## Results

All results are from simulation, with held-out seeds that the agent never saw during development.

| Program | Seeds 1–10 | Seeds 11–20 (clean) |
|---|---|---|
| Round 1 (`campaign/round1/fix_code.py`) | 3/10 | **6/10** |
| Round 2 (`campaign/round2/fix_code.py`, v06) | 7/10 † | 4/10 |

† Seeds 1–10 were also used for round 1's held-out run, so treat seeds 11–20 as the fair comparison.

- **Round 1** added five skills, each confirmed by a separate verifier agent: parking the idle arm, finding the middle of three raised shafts, a tilted search with an arm-to-arm relay, the reach limit for far bases (≥ ~0.485 m from the right arm), and a grasp for gears with a hub. Its failures are far shafts (the program refuses) and insertion jams.
- **Round 2** added a tilted reach for far shafts, which passes 3/9 far seeds where round 1 passes none. But it regressed on near seeds (11, 17, 19): after release, the press-and-retract step pulls the gear off a shaft it hasn't fully seated on. Its skills have not been promoted.
- **Open problem:** tilted insertion jams 11–14 mm above the seat, because the gear pivots in the two-finger grip. See `campaign/round2/insertion-jam-tilted.md`.

Full notes are in `campaign/curriculum.md`.

## Motion stack

SAM3 text-prompted segmentation and top-camera depth give object point clouds. mink differential IK (a DAQP QP) turns grasp and insert poses into joint targets. Joint-space interpolation produces the trajectory. Isaac Lab joint-position PD actuators execute it. `move_to_pose` results are checked against `diagnostics["position_error_m"]`, because the IK reports success up to about 10 mm off.

## License

Apache-2.0, the same as ASPIRE. The patches modify ASPIRE source © NVIDIA Corporation.
