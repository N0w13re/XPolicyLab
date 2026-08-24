"""Minimal Isaac Sim render smoke test, isolating GPU/render failures from the policy stack.

Renders a few camera frames headless. This exercises the same texture-upload,
block-compression and raytracing-pipeline path that the RoboDojo eval client hits, but
reaches it in ~20s instead of ~2min, which makes it usable for bisecting driver and Kit
settings. Prints RENDER_OK on success.

Usage:
    source scripts/robodojo_sim_env.sh /path/to/RoboDojo-eval
    python scripts/robodojo_render_smoke.py --frames 3 --kit-args="$ROBODOJO_KIT_ARGS"
"""

from __future__ import annotations

import argparse
import os
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--kit-args", type=str, default="")
    parser.add_argument(
        "--material",
        type=str,
        default="",
        help="Optional .mdl file to bind, forcing a textured material upload.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # SimulationApp forwards leftover sys.argv straight to the Kit kernel, so this is the
    # only way to set carb settings that are read before the app finishes booting.
    kit_args = [a for a in args.kit_args.split(" ") if a]
    sys.argv = [sys.argv[0]] + kit_args
    print(f"[repro] kit args: {kit_args}", flush=True)

    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "enable_cameras": True,
            "width": args.width,
            "height": args.height,
        }
    )

    import omni.kit.commands
    import omni.replicator.core as rep
    import omni.usd
    from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

    stage = omni.usd.get_context().get_stage()

    UsdGeom.Xform.Define(stage, "/World")
    plane = UsdGeom.Cube.Define(stage, "/World/Cube")
    plane.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))

    light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    light.CreateIntensityAttr(1000.0)

    if args.material:
        mtl_path = Sdf.Path("/World/Looks/ReproMaterial")
        omni.kit.commands.execute(
            "CreateMdlMaterialPrim",
            mtl_url=args.material,
            mtl_name=os.path.splitext(os.path.basename(args.material))[0],
            mtl_path=str(mtl_path),
        )
        material = UsdShade.Material.Get(stage, mtl_path)
        if material:
            UsdShade.MaterialBindingAPI(plane.GetPrim()).Bind(material)
            print(f"[repro] bound material {args.material}", flush=True)

    camera = rep.create.camera(position=(4.0, 4.0, 4.0), look_at=(0.0, 0.0, 0.0))
    render_product = rep.create.render_product(camera, (args.width, args.height))
    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    annotator.attach([render_product])

    for i in range(args.frames):
        rep.orchestrator.step(rt_subframes=1)
        data = annotator.get_data()
        shape = getattr(data, "shape", None)
        print(f"[repro] frame {i} rgb={shape}", flush=True)

    print("[repro] RENDER_OK", flush=True)
    app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
