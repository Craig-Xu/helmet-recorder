#!/usr/bin/env python3
import yaml
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R
import os

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def plot_camera_pose(ax, position, rotation, label, color='b', scale=0.1):
    # Camera coordinate system: Z forward, X right, Y down (standard OpenCV)
    o = position
    r = rotation
    
    # Axis vectors in global frame (for quiver)
    # Scale down axes for cleaner look
    axis_scale = scale * 0.8
    x_axis = r.apply([1, 0, 0]) * axis_scale
    y_axis = r.apply([0, 1, 0]) * axis_scale
    z_axis = r.apply([0, 0, 1]) * axis_scale # Forward
    
    # Plot axes (RGB = XYZ)
    ax.quiver(o[0], o[1], o[2], x_axis[0], x_axis[1], x_axis[2], color='r', alpha=0.8, lw=1)
    ax.quiver(o[0], o[1], o[2], y_axis[0], y_axis[1], y_axis[2], color='g', alpha=0.8, lw=1)
    ax.quiver(o[0], o[1], o[2], z_axis[0], z_axis[1], z_axis[2], color='b', alpha=0.8, lw=1)
    
    # Camera Frustum (Pyramid)
    w = scale * 0.5  # half width
    h = scale * 0.4  # half height (4:3 aspect ratio ish)
    d = scale * 0.8  # depth
    
    # Corners in local frame
    tl = np.array([-w, -h, d])
    tr = np.array([ w, -h, d])
    br = np.array([ w,  h, d])
    bl = np.array([-w,  h, d])
    
    # Transform to global
    points = [tl, tr, br, bl]
    verts_global = [o + r.apply(p) for p in points]
    
    # Define faces
    # 4 triangular faces connecting origin to base + 1 base face
    faces = [
        [o, verts_global[0], verts_global[1]], # Top
        [o, verts_global[1], verts_global[2]], # Right
        [o, verts_global[2], verts_global[3]], # Bottom
        [o, verts_global[3], verts_global[0]], # Left
        [verts_global[0], verts_global[1], verts_global[2], verts_global[3]] # Base
    ]
    
    # Create Poly3DCollection
    pc = Poly3DCollection(faces, facecolors=color, linewidths=1, edgecolors=color, alpha=0.25)
    ax.add_collection3d(pc)

    # Add label with offset
    ax.text(o[0], o[1], o[2] - scale*0.2, label, color='k', fontsize=9, 
            horizontalalignment='center', verticalalignment='top')



def main():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(script_dir, '..', 'config.yaml')
    
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return

    config = load_config(config_path)
    
    if 'camera' not in config or 'extrinsics' not in config['camera']:
        print("No camera extrinsics found in config")
        return

    extrinsics = config['camera']['extrinsics']
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    positions = []
    
    sorted_cams = sorted(extrinsics.keys(), key=lambda x: int(x.replace('cam', '')))
    
    # Use a colormap
    colors = plt.cm.tab10(np.linspace(0, 1, len(sorted_cams)))
    
    for i, cam_name in enumerate(sorted_cams):
        cam_start = time.time()
        print(f"Processing {cam_name}")
        cam_data = extrinsics[cam_name]
        
        trans = cam_data['translation'] # [x, y, z]
        rot_quat = cam_data['rotation'] # [x, y, z, w]
        
        pos = np.array(trans)
        positions.append(pos)
        
        rot = R.from_quat(rot_quat)
        
        plot_camera_pose(ax, pos, rot, cam_name, color=colors[i], scale=0.08)
    
    # Plot World Origin
    ax.scatter([0], [0], [0], color='k', s=50, marker='o', label='Origin')
    ax.text(0, 0, -0.02, '(0,0,0)', color='k', fontsize=8)
    
    positions = np.array(positions)
    
    # Determine plot limits
    if len(positions) > 0:
        center = np.mean(positions, axis=0)
        max_range = np.max(np.ptp(positions, axis=0)) / 2.0
        # If max_ran (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Camera Extrinsics Visualization', fontsize=14)
    
    # Make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    
    # Hide grid lines if desired (optional)
    # ax.grid(False)
    
    # Invert Y and Z to match typical 3D viewer conventions if needed
    # Usually Z is up in matplotlib, but in CV, Y is down.
    # Let's keep it standard 3D plot for now.
    
    # To make it look nice:
    ax.view_init(elev=-70, azim=-90) # Top-ish view
    # Let's visualize in a way that makes sense.
    # If we want to see the "top down" view of the helmet rig:
    # Usually X is left-right, Z is forward-back in head coords? NO.
    # Let's just plot and let user interact.
    
    # To make it look nice:
    ax.view_init(elev=-80, azim=-90) # Top-ish view? Experiment.
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    import time
    main()
