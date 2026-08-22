from __future__ import annotations

import argparse
from pathlib import Path

from astrovox.viewer import KinematicVolumeViewer, load_fits_cube


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive 3D viewer for volumetric FITS cubes.")
    parser.add_argument("cube_path", type=Path, help="Path to a FITS cube")
    parser.add_argument("--vel-scale", type=float, default=None, help="Scale factor for the spectral axis")
    parser.add_argument("--opacity", default="sigmoid", help="PyVista opacity transfer function")
    parser.add_argument("--cmap", default="plasma", help="Colormap name")
    parser.add_argument(
        "--moment0",
        action="store_true",
        help="Overlay a moment-0 (integrated intensity) plane above the volume",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cube = load_fits_cube(args.cube_path)
    viewer = KinematicVolumeViewer(
        cube,
        vel_scale=args.vel_scale,
        opacity=args.opacity,
        cmap=args.cmap,
        show_moment0=args.moment0,
    )
    viewer.show()
    return 0
