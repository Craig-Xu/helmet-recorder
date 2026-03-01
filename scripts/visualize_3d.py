'''
  ******************************************************************************
  * @file    visualize_3d.py
  * @brief   3D Coordinate System Visualization for Yesense IMU data.
  ******************************************************************************    
'''
# -*- encoding:utf-8 -*-
#!/usr/bin/env python3

import sys
import os
import argparse
import time
import threading
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation
from port_manager import *
from yis_std_dec import *

# Yesense output dictionary structure
yis_out = {'tid':1, 'roll':0.0, 'pitch':0.0, 'yaw':0.0, \
			'q0':1.0, 'q1':0.0, 'q2':0.0, 'q3':0.0, \
			'sensor_temp':25.0, 'acc_x':0.0, 'acc_y':0.0, 'acc_z':1.0, \
			'gyro_x':0.0, 'gyro_y':0.0, 'gyro_z':0.0,	\
			'norm_mag_x':0.0, 'norm_mag_y':0.0, 'norm_mag_z':0.0,	\
			'raw_mag_x':0.0, 'raw_mag_y':0.0, 'raw_mag_z':0.0,	\
			'lat':0.0, 'longt':0.0, 'alt':0.0,	\
			'vel_e':0.0, 'vel_n':0.0, 'vel_u':0.0, \
			'ms':0, 'year': 2022, 'month':8, 'day': 31, \
			'hour':12, 'minute':0, 'second':0,	\
			'smp_timestamp':0, 'ready_timestamp':0 ,'status':0
			}

dec_buf = bytearray()
data_lock = threading.Lock()
running = True

def quaternion_to_rotation_matrix(q):
    """
    Convert a quaternion into a rotation matrix.
    q = [w, x, y, z]
    """
    w, x, y, z = q
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [2*x*y + 2*w*z,     1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y,     2*y*z + 2*w*x,     1 - 2*x**2 - 2*y**2]
    ])

def serial_reader(port, baudrate):
    global dec_buf, running
    decoder = std_decoder()
    ser = open_port(port, baudrate)
    
    while running:
        try:
            data = rd_data(ser)
            if data:
                dec_buf.extend(bytearray(data))
            
            if len(dec_buf) > 0:
                while len(dec_buf) > 0:
                    ret = decoder.proc_data(dec_buf, len(dec_buf), yis_out, False)
                    if not ret:
                        break
            time.sleep(0.001)
        except Exception as e:
            print(f"Serial reader error: {e}")
            break
            
    close_port(ser)

def update_plot(frame, ax):
    with data_lock:
        # Get current quaternion
        q = [yis_out['q0'], yis_out['q1'], yis_out['q2'], yis_out['q3']]
        
        # Calculate rotation matrix
        R = quaternion_to_rotation_matrix(q)
        
        # The columns of R are the local X, Y, Z axes in global frame
        x_axis = R[:, 0]
        y_axis = R[:, 1]
        z_axis = R[:, 2]
        
        # Clear and redraw axes
        ax.cla()
        
        # Set static limits
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_zlim([-1.5, 1.5])
        
        # Labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"Yesense 3D Orientation (TID: {yis_out['tid']})\n"
                     f"Pitch: {yis_out['pitch']:.2f}, Roll: {yis_out['roll']:.2f}, Yaw: {yis_out['yaw']:.2f}")

        # Draw origin
        ax.scatter([0], [0], [0], color='black', s=50)
        
        # 1. Draw Reference Gravity (Downward in Global Frame) - Purple
        # This shows where "Down" is in the world
        ax.quiver(0, 0, 0, 0, 0, -1, color='purple', length=1.2, linestyle='--', label='Gravity (Ref)')

        # 2. Draw Sensor Body (Rectangular Board in Local XY Plane)
        # Based on user request: X positive (0.3), Y negative (0.2)
        v_local = np.array([
            [0, 0, 0],
            [0.4, 0, 0],
            [0.4, -0.3, 0],
            [0, -0.3, 0]
        ])
        v_global = (R @ v_local.T).T
        body = Poly3DCollection([v_global], alpha=0.4, facecolor='cyan', edgecolor='darkblue', label='Sensor Body')
        ax.add_collection3d(body)

        # 3. Draw Measured Acceleration Vector - Yellow
        # This is the real-time gravity + motion vector felt by the sensor
        acc_vec = np.array([yis_out['acc_x'], yis_out['acc_y'], yis_out['acc_z']])
        acc_norm = np.linalg.norm(acc_vec)
        if acc_norm > 0.1: # Only draw if there's a significant signal
            # Transform local acceleration to global frame for display
            acc_global = R @ acc_vec
            # Normalize for visualization if it's too large, or just scale it
            acc_disp = acc_global / max(acc_norm, 9.8) * 1.0 
            ax.quiver(0, 0, 0, acc_disp[0], acc_disp[1], acc_disp[2], color='orange', length=1.0, label='Measured Acc')

        # 3. Draw Local axes arrows
        # X Axis - Red
        ax.quiver(0, 0, 0, x_axis[0], x_axis[1], x_axis[2], color='r', length=1.0, normalize=True, label='Local X')
        # Y Axis - Green
        ax.quiver(0, 0, 0, y_axis[0], y_axis[1], y_axis[2], color='g', length=1.0, normalize=True, label='Local Y')
        # Z Axis - Blue
        ax.quiver(0, 0, 0, z_axis[0], z_axis[1], z_axis[2], color='b', length=1.0, normalize=True, label='Local Z')
        
        ax.legend(loc='lower left', fontsize='small')

def main():
    parser = argparse.ArgumentParser(description='YESENSE IMU 3D Visualization')
    parser.add_argument('--port', type=str, required=True, help='Serial port name')
    parser.add_argument('--bps', type=int, default=460800, help='Baud rate (default: 460800)')
    args = parser.parse_args()

    # Setup 3D plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    fig.canvas.manager.set_window_title('Yesense 3D Coordinate System')
    
    # Set initial camera view (rotated 180 degrees around Z from previous 45)
    ax.view_init(elev=20, azim=225)

    # Start serial thread
    thread = threading.Thread(target=serial_reader, args=(args.port, args.bps), daemon=True)
    thread.start()

    # Start animation
    ani = FuncAnimation(fig, update_plot, fargs=(ax,), interval=30, cache_frame_data=False)
    
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        global running
        running = False
        thread.join(timeout=1.0)

if __name__ == '__main__':
    main()
