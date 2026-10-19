"""Download the Isaac Lab factory gear-assembly assets as YAMLab object instances.

Fetches a factory gear and the gear base from the Isaac Sim 6.0 Nucleus asset
server (the same USDs Isaac Lab's ``Isaac-Deploy-GearAssembly`` task spawns),
including their ``configuration/`` payload layers, and writes an
``asset_size.json`` next to each so the task can size its spawn region. Layout::

    <out>/Gear/factory_gear_<size>/factory_gear_<size>.usd
    <out>/GearBase/factory_gear_base/factory_gear_base.usd

The factory assets share one frame: a gear fully seated on its shaft has the
same origin as the base. Their origins are therefore NOT at the bbox center, so
asset_size.json also records the local ``min``/``max`` extents, which the task
uses to rest each object on the table.

Examples:
    python scripts/download_gear_assets.py
    python scripts/download_gear_assets.py --gear large --out ./my_objects
"""

import argparse
import json
import os
import urllib.request

from pxr import Sdf, Usd, UsdGeom

ASSET_ROOT = ("https://omniverse-content-production.s3-us-west-2.amazonaws.com"
              "/Assets/Isaac/6.0/Isaac/Props/Factory/gear_assets")


def fetch_usd_tree(base_url: str, local_dir: str, top: str):
    """Download ``top`` and every relative USD layer it composes, recursively.

    Args:
        base_url (str): URL of the asset folder on the server.
        local_dir (str): Local folder mirroring ``base_url``.
        top (str): Top-level USD file name inside the folder.
    """
    todo, seen = [top], set()
    while todo:
        rel = os.path.normpath(todo.pop())
        if rel in seen:
            continue
        seen.add(rel)
        dst = os.path.join(local_dir, rel)
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            urllib.request.urlretrieve(f"{base_url}/{rel}", dst)
            print(f"[INFO] Downloaded {rel}")
        layer = Sdf.Layer.FindOrOpen(dst)
        for dep in layer.GetCompositionAssetDependencies():
            # Absolute/URL refs resolve on their own; MDLs (OmniPBR) ship with Isaac Sim.
            if "://" not in dep and not os.path.isabs(dep) and dep.lower().endswith((".usd", ".usda", ".usdc")):
                todo.append(os.path.join(os.path.dirname(rel), dep))


def disable_instancing(usd_path: str):
    """Make the asset's instanceable prims editable in the top-level layer.

    The factory USDs mark their body prim instanceable, so nothing can be bound
    below it; the task binds its friction material onto the collision prims.

    Args:
        usd_path (str): Path to the asset's top-level USD.
    """
    stage = Usd.Stage.Open(usd_path)
    stage.SetEditTarget(stage.GetRootLayer())
    changed = []
    # De-instancing a prim can expose nested instances, so repeat until none are left.
    while True:
        instances = [p for p in Usd.PrimRange(stage.GetPseudoRoot()) if p.IsInstance()]
        if not instances:
            break
        for prim in instances:
            prim.SetInstanceable(False)
            changed.append(str(prim.GetPath()))
    if changed:
        stage.GetRootLayer().Save()
        print(f"[INFO] Disabled instancing on {changed}")


def write_asset_size(usd_path: str) -> dict:
    """Write asset_size.json (size + local min/max extents) next to ``usd_path``.

    Args:
        usd_path (str): Path to the asset's top-level USD.

    Returns:
        dict: The written size info.
    """
    stage = Usd.Stage.Open(usd_path)
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(
        stage.GetDefaultPrim()).ComputeAlignedRange()
    bmin, bmax = bbox.GetMin(), bbox.GetMax()
    info = {
        "size": {axis: float(bmax[i] - bmin[i]) for i, axis in enumerate("xyz")},
        "min": {axis: float(bmin[i]) for i, axis in enumerate("xyz")},
        "max": {axis: float(bmax[i]) for i, axis in enumerate("xyz")},
    }
    with open(os.path.join(os.path.dirname(usd_path), "asset_size.json"), "w") as f:
        json.dump(info, f, indent=2)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gear", choices=["small", "medium", "large"], default="medium",
                        help="Gear size to download (default: medium).")
    parser.add_argument("--out", default="yamlab_datasets/tasks_data/PutPotOnCooktop/objects",
                        help="Objects root to place the Gear/ and GearBase/ categories in.")
    args = parser.parse_args()

    for category, name in (("Gear", f"factory_gear_{args.gear}"), ("GearBase", "factory_gear_base")):
        local_dir = os.path.join(args.out, category, name)
        fetch_usd_tree(f"{ASSET_ROOT}/{name}", local_dir, f"{name}.usd")
        disable_instancing(os.path.join(local_dir, f"{name}.usd"))
        info = write_asset_size(os.path.join(local_dir, f"{name}.usd"))
        print(f"[INFO] {local_dir}: size {info['size']}")


if __name__ == "__main__":
    main()
