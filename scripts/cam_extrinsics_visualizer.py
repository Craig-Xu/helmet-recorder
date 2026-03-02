#!/usr/bin/env python3
"""Visualize camera extrinsics from config/config.yaml in 3D."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R

from common import CONFIG_PATH, load_yaml, sorted_cam_names

CAM_COLORS = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231',
    '#911eb4', '#42d4f4', '#f032e6', '#bfef45',
]

AXIS_COLORS = {
    'x': '#d62728',
    'y': '#2ca02c',
    'z': '#1f77b4',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Visualize camera extrinsics in 3D')
    parser.add_argument('--scale', type=float, default=0.08, help='Camera frustum scale')
    parser.add_argument('--elev', type=float, default=-80.0, help='View elevation angle')
    parser.add_argument('--azim', type=float, default=-90.0, help='View azimuth angle')
    return parser.parse_args()


def draw_camera(
    ax,
    pos: np.ndarray,
    rot: R,
    label: str,
    color: str,
    scale: float,
    axis_scale_ratio: float = 0.65,
) -> None:
    o = np.asarray(pos, dtype=float)

    for axis_vec, axis_color in [([1, 0, 0], AXIS_COLORS['x']), ([0, 1, 0], AXIS_COLORS['y']), ([0, 0, 1], AXIS_COLORS['z'])]:
        v = rot.apply(axis_vec) * scale * axis_scale_ratio
        ax.quiver(o[0], o[1], o[2], v[0], v[1], v[2], color=axis_color, arrow_length_ratio=0.15, linewidth=1.2)

    w, h, d = scale * 0.45, scale * 0.34, scale * 0.9
    corners_local = np.array([
        [-w, -h, d],
        [w, -h, d],
        [w, h, d],
        [-w, h, d],
    ])
    corners_world = o + rot.apply(corners_local)

    for i in range(4):
        j = (i + 1) % 4
        tri = Poly3DCollection(
            [[o, corners_world[i], corners_world[j]]],
            facecolors=color,
            edgecolors=color,
            alpha=0.22,
            linewidths=0.6,
        )
        ax.add_collection3d(tri)

    base = Poly3DCollection([corners_world], facecolors=color, edgecolors=color, alpha=0.30, linewidths=0.8)
    ax.add_collection3d(base)

    ax.text(
        o[0],
        o[1],
        o[2] - scale * 0.35,
        label,
        fontsize=9,
        fontweight='bold',
        color=color,
        ha='center',
        va='top',
        bbox={'boxstyle': 'round,pad=0.2', 'fc': 'white', 'ec': color, 'alpha': 0.75},
    )


def set_equal_aspect(ax, positions: np.ndarray, margin: float = 0.88) -> None:
    center = positions.mean(axis=0)
    span = max(np.ptp(positions, axis=0).max(), 0.05) / 2.0 * margin

    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)
    ax.set_box_aspect((1, 1, 1))


def add_legend(ax) -> None:
    legend_elements = [
        Line2D([0], [0], color=AXIS_COLORS['x'], lw=2, label='Cam X'),
        Line2D([0], [0], color=AXIS_COLORS['y'], lw=2, label='Cam Y'),
        Line2D([0], [0], color=AXIS_COLORS['z'], lw=2, label='Cam Z (Forward)'),
        Line2D([0], [0], marker='o', color='k', lw=0, markersize=6, label='World Origin'),
        Line2D([0], [0], color='#666666', lw=1.4, ls='--', label='Camera Center Line'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)


def main() -> None:
    args = parse_args()

    if not CONFIG_PATH.is_file():
        print(f'Config not found: {CONFIG_PATH}')
        return

    cfg = load_yaml(CONFIG_PATH)
    extrinsics = cfg.get('camera', {}).get('extrinsics')
    if not extrinsics:
        print('No camera extrinsics in config.')
        return

    sorted_names = sorted_cam_names(extrinsics)
    fig = plt.figure(figsize=(11, 9), dpi=120)
    fig.patch.set_facecolor('#fafafa')
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    ax.set_facecolor('#fafafa')

    positions = []
    for i, cam_name in enumerate(sorted_names):
        cam_data = extrinsics[cam_name]
        trans = np.array(cam_data['translation'], dtype=float)
        quat = np.array(cam_data['rotation'], dtype=float)
        rot = R.from_quat(quat)

        positions.append(trans)
        draw_camera(ax, trans, rot, cam_name, CAM_COLORS[i % len(CAM_COLORS)], args.scale)

    positions_np = np.array(positions)

    ax.scatter([0], [0], [0], color='k', s=45, marker='o')
    ax.text(0, 0, -args.scale * 0.35, 'Origin', color='k', fontsize=8, ha='center')

    if len(positions_np) > 0:
        ax.plot(positions_np[:, 0], positions_np[:, 1], positions_np[:, 2], '--', color='#666666', linewidth=1.4, alpha=0.9)
        set_equal_aspect(ax, positions_np)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Camera Extrinsics Visualization', fontsize=14)

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    add_legend(ax)
    ax.view_init(elev=args.elev, azim=args.azim)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
