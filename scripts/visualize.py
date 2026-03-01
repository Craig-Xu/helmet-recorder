'''
  ******************************************************************************
  * @file    visualize.py
  * @brief   Real-time visualization for Yesense IMU data.
  ******************************************************************************    
'''
# -*- encoding:utf-8 -*-
#!/usr/bin/env python3

import sys
import os
import argparse
import time
import threading
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from port_manager import *
from yis_std_dec import *

# Max number of points to display in the plot
MAX_POINTS = 100

# Global storage for plot data
data_history = {
    'tid': deque(maxlen=MAX_POINTS),
    'pitch': deque(maxlen=MAX_POINTS),
    'roll': deque(maxlen=MAX_POINTS),
    'yaw': deque(maxlen=MAX_POINTS),
    'acc_x': deque(maxlen=MAX_POINTS),
    'acc_y': deque(maxlen=MAX_POINTS),
    'acc_z': deque(maxlen=MAX_POINTS),
    'gyro_x': deque(maxlen=MAX_POINTS),
    'gyro_y': deque(maxlen=MAX_POINTS),
    'gyro_z': deque(maxlen=MAX_POINTS),
}

# Yesense output dictionary structure
yis_out = {'tid':1, 'roll':0.0, 'pitch':0.0, 'yaw':0.0, \
			'q0':1.0, 'q1':1.0, 'q2':0.0, 'q3':0.0, \
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
                # Update data in a loop to catch up with buffer
                while len(dec_buf) > 0:
                    ret = decoder.proc_data(dec_buf, len(dec_buf), yis_out, False)
                    if ret:
                        with data_lock:
                            data_history['tid'].append(yis_out['tid'])
                            data_history['pitch'].append(yis_out['pitch'])
                            data_history['roll'].append(yis_out['roll'])
                            data_history['yaw'].append(yis_out['yaw'])
                            data_history['acc_x'].append(yis_out['acc_x'])
                            data_history['acc_y'].append(yis_out['acc_y'])
                            data_history['acc_z'].append(yis_out['acc_z'])
                            data_history['gyro_x'].append(yis_out['gyro_x'])
                            data_history['gyro_y'].append(yis_out['gyro_y'])
                            data_history['gyro_z'].append(yis_out['gyro_z'])
                    else:
                        break # No more complete packets
            time.sleep(0.001)
        except Exception as e:
            print(f"Serial reader error: {e}")
            break
            
    close_port(ser)

def update_plot(frame, lines, axes):
    with data_lock:
        if not data_history['tid']:
            return lines
        
        x = list(range(len(data_history['tid'])))
        
        # Euler Angles
        lines[0].set_data(x, list(data_history['pitch']))
        lines[1].set_data(x, list(data_history['roll']))
        lines[2].set_data(x, list(data_history['yaw']))
        
        # Accelerometer
        lines[3].set_data(x, list(data_history['acc_x']))
        lines[4].set_data(x, list(data_history['acc_y']))
        lines[5].set_data(x, list(data_history['acc_z']))
        
        # Gyroscope
        lines[6].set_data(x, list(data_history['gyro_x']))
        lines[7].set_data(x, list(data_history['gyro_y']))
        lines[8].set_data(x, list(data_history['gyro_z']))
        
        # Adjust axes
        for ax in axes:
            ax.relim()
            ax.autoscale_view()
            
    return lines

def main():
    parser = argparse.ArgumentParser(description='YESENSE IMU Data Visualization')
    parser.add_argument('--port', type=str, required=True, help='Serial port name')
    parser.add_argument('--bps', type=int, default=460800, help='Baud rate (default: 460800)')
    args = parser.parse_args()

    # Setup plots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
    fig.canvas.manager.set_window_title('Yesense IMU Real-time Data')
    
    # Ax1: Euler Angles
    line_pitch, = ax1.plot([], [], label='Pitch')
    line_roll, = ax1.plot([], [], label='Roll')
    line_yaw, = ax1.plot([], [], label='Yaw')
    ax1.set_ylabel('Degrees')
    ax1.legend(loc='upper right')
    ax1.grid(True)
    ax1.set_title('Euler Angles')

    # Ax2: Accelerometer
    line_accx, = ax2.plot([], [], label='Acc X')
    line_accy, = ax2.plot([], [], label='Acc Y')
    line_accz, = ax2.plot([], [], label='Acc Z')
    ax2.set_ylabel('m/s²')
    ax2.legend(loc='upper right')
    ax2.grid(True)
    ax2.set_title('Accelerometer')

    # Ax3: Gyroscope
    line_gyrox, = ax3.plot([], [], label='Gyro X')
    line_gyroy, = ax3.plot([], [], label='Gyro Y')
    line_gyroz, = ax3.plot([], [], label='Gyro Z')
    ax3.set_ylabel('deg/s')
    ax3.legend(loc='upper right')
    ax3.grid(True)
    ax3.set_title('Gyroscope')

    lines = [line_pitch, line_roll, line_yaw, line_accx, line_accy, line_accz, line_gyrox, line_gyroy, line_gyroz]
    axes = [ax1, ax2, ax3]

    plt.tight_layout()

    # Start serial thread
    thread = threading.Thread(target=serial_reader, args=(args.port, args.bps), daemon=True)
    thread.start()

    # Start animation
    ani = FuncAnimation(fig, update_plot, fargs=(lines, axes), interval=50, blit=False)
    
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
