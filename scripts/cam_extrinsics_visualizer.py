#!/usr/bin/env python3
"""Visualize camera extrinsics from config/config.yaml in 3D."""

import argparse
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R

from common import CONFIG_PATH, load_yaml, sorted_cam_names

# ── colour palette (one per camera, up to 8) ──────────────────────────────
CAM_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45",
]

AXIS_COLORS = {
    "x": "#d62728",
    "y": "#2ca02c",
    "z": "#1f77b4",
}


def draw_camera(
    ax,
    pos,
    rot: R,
    label: str,
    color: str,
    scale: float = 0.08,
    axis_scale_ratio: float = 0.65,
):
    """Draw a camera as a coloured frustum + RGB local axes."""
    o = np.asarray(pos, dtype=float)

    # ── local-axis arrows (RGB = XYZ) ─────────────────────────────────────
    for axis_vec, c in [([1, 0, 0], AXIS_COLORS["x"]),
                        ([0, 1, 0], AXIS_COLORS["y"]),
                        ([0, 0, 1], AXIS_COLORS["z"])]:
        v = rot.apply(axis_vec) * scale * axis_scale_ratio
        ax.quiver(o[0], o[1], o[2], v[0], v[1], v[2],
                  color=c, arrow_length_ratio=0.15, linewidth=1.2)

    # ── frustum pyramid ───────────────────────────────────────────────────
    w, h, d = scale * 0.45, scale * 0.34, scale * 0.9   # 4:3-ish
    corners_local = np.array([[-w, -h, d],
                               [ w, -h, d],
                               [ w,  h, d],
                               [-w,  h, d]])
    corners_world = o + rot.apply(corners_local)

    # side faces (triangles from origin to each edge of the base)
    for i in range(4):
        j = (i + 1) % 4
        tri = Poly3DCollection([[o, corners_world[i], corners_world[j]]],
                               facecolors=color, edgecolors=color,
                               alpha=0.22, linewidths=0.6)
        ax.add_collection3d(tri)

    # base quad
    base = Poly3DCollection([corners_world],
                            facecolors=color, edgecolors=color,
                            alpha=0.30, linewidths=0.8)
    ax.add_collection3d(base)

    # ── label ─────────────────────────────────────────────────────────────
    ax.text(o[0], o[1], o[2] - scale * 0.35, label,
            fontsize=9, fontweight="bold", color=color,
            ha="center", va="top",
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": color, "alpha": 0.75})


def set_equal_aspect(ax, positions: np.ndarray, margin: float = 0.88):
    """Force equal aspect ratio on all three axes."""
    center = positions.mean(axis=0)
    span = max(np.ptp(positions, axis=0).max(), 0.05) / 2.0 * margin
    for setter, c in [(ax.set_xlim, center[0]),
                      (ax.set_ylim, center[1]),
                      (ax.set_zlim, center[2])]:
        setter(c - span, c + span)
    ax.set_box_aspect((1, 1, 1))


def add_legend(ax):
    legend_elements = [
        Line2D([0], [0], color=AXIS_COLORS["x"], lw=2, label="Cam X"),
        Line2D([0], [0], color=AXIS_COLORS["y"], lw=2, label="Cam Y"),
        Line2D([0], [0], color=AXIS_COLORS["z"], lw=2, label="Cam Z (Forward)"),
        Line2D([0], [0], marker="o", color="k", lw=0, markersize=6, label="World Origin"),
        Line2D([0], [0], color="#666666", lw=1.4, ls="--", label="Camera Center Line"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8, frameon=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize camera extrinsics in 3D")
    parser.add_argument("--cam-scale", type=float, default=0.08,
                        help="Camera frustum scale (default: 0.08)")
    parser.add_argument("--axis-scale-ratio", type=float, default=0.65,
                        help="Axis length relative to frustum scale (default: 0.65)")
    parser.add_argument("--margin", type=float, default=0.88,
                        help="Axis range margin; smaller means visually bigger (default: 0.88)")
    parser.add_argument("--zoom", type=float, default=7.8,
                        help="Initial camera distance in Matplotlib 3D (default: 7.8)")
    return parser.parse_args()


def main():
    args = parse_args()

    if not CONFIG_PATH.is_file():
        print(f"Config not found: {CONFIG_PATH}")
        return

    cfg = load_yaml(CONFIG_PATH)
    extrinsics = cfg.get("camera", {}).get("extrinsics")
    if not extrinsics:
        print("No camera extrinsics in config.")
        return

    # ── set up figure ─────────────────────────────────────────────────────
    plt.style.use("default")
    fig = plt.figure(figsize=(11, 9), dpi=120)
    fig.patch.set_facecolor("#fafafa")
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    ax.set_facecolor("#fafafa")

    sorted_names = sorted_cam_names(extrinsics)
    positions = []

    for idx, name in enumerate(sorted_names):
        data = extrinsics[name]
        pos = np.array(data["translation"])
        rot = R.from_quat(data["rotation"])       # [x, y, z, w]
        color = CAM_COLORS[idx % len(CAM_COLORS)]

        draw_camera(
            ax,
            pos,
            rot,
            name,
            color,
            scale=args.cam_scale,
            axis_scale_ratio=args.axis_scale_ratio,
        )
        positions.append(pos)

    positions = np.array(positions)

    if len(positions) >= 2:
        ax.plot(
            positions[:, 0], positions[:, 1], positions[:, 2],
            linestyle="--", linewidth=1.4, color="#666666", alpha=0.8, zorder=3
        )

    # ── world origin marker ───────────────────────────────────────────────
    ax.scatter(0, 0, 0, c="k", s=40, marker="o", zorder=5)
    ax.text(0, 0, 0, "  world", color="k", fontsize=8)

    # ── styling ───────────────────────────────────────────────────────────
    set_equal_aspect(ax, positions, margin=args.margin)

    ax.set_xlabel("X  (m)", labelpad=8)
    ax.set_ylabel("Y  (m)", labelpad=8)
    ax.set_zlabel("Z  (m)", labelpad=8)
    ax.set_title("Camera Extrinsics", fontsize=14, pad=12)

    # transparent panes + light grid
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1, 1, 1, 0))
        axis._axinfo["grid"]["color"] = (0.85, 0.85, 0.85, 0.6)

    add_legend(ax)

    ax.view_init(elev=-75, azim=-90)
    if hasattr(ax, "dist"):
        ax.dist = args.zoom
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
