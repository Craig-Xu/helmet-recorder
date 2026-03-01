"""
IMU 模块管理器
提供 IMU 传感器数据采集和处理功能
"""

import sys
import os
import threading
import time
from typing import Optional, Dict, Callable
from port_manager import open_port, close_port, rd_data
from yis_std_dec import std_decoder


class IMUManager:
    """
    IMU 管理器，负责 IMU 传感器的数据采集和解析
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化 IMU 管理器
        
        Args:
            config: IMU 配置字典
        """
        self.config = config or {}
        
        # 从配置中获取参数
        self.port = self.config.get('port', 'COM6')
        self.baudrate = self.config.get('baudrate', 460800)
        self.debug = self.config.get('debug', False)
        
        # 串口和解析器
        self.serial = None
        self.decoder = std_decoder()
        self.dec_buf = bytearray()
        
        # 运行状态
        self.is_running = False
        self.read_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # 数据存储
        self.yis_out = {
            'tid': 1, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'q0': 1.0, 'q1': 1.0, 'q2': 0.0, 'q3': 0.0,
            'sensor_temp': 25.0, 'acc_x': 0.0, 'acc_y': 0.0, 'acc_z': 1.0,
            'gyro_x': 0.0, 'gyro_y': 0.0, 'gyro_z': 0.0,
            'norm_mag_x': 0.0, 'norm_mag_y': 0.0, 'norm_mag_z': 0.0,
            'raw_mag_x': 0.0, 'raw_mag_y': 0.0, 'raw_mag_z': 0.0,
            'lat': 0.0, 'longt': 0.0, 'alt': 0.0,
            'vel_e': 0.0, 'vel_n': 0.0, 'vel_u': 0.0,
            'ms': 0, 'year': 2022, 'month': 8, 'day': 31,
            'hour': 12, 'minute': 0, 'second': 0,
            'smp_timestamp': 0, 'ready_timestamp': 0, 'status': 0
        }
        
        # 回调函数
        self.data_callback: Optional[Callable] = None
        
    def set_data_callback(self, callback: Callable[[Dict], None]):
        """
        设置数据回调函数，每次解析到新数据时调用
        
        Args:
            callback: 回调函数，接收 yis_out 字典作为参数
        """
        self.data_callback = callback
    
    def connect(self) -> bool:
        """
        连接 IMU 设备
        
        Returns:
            是否成功连接
        """
        try:
            print(f"🔌 正在连接 IMU 设备...")
            print(f"   端口: {self.port}")
            print(f"   波特率: {self.baudrate}")
            
            self.serial = open_port(self.port, self.baudrate)
            
            if self.serial:
                print("✅ IMU 设备连接成功")
                return True
            else:
                print("❌ IMU 设备连接失败")
                return False
                
        except Exception as e:
            print(f"❌ 连接 IMU 设备时出错: {e}")
            return False
    
    def start(self) -> bool:
        """
        开始采集 IMU 数据
        
        Returns:
            是否成功启动
        """
        with self._lock:
            if self.is_running:
                print("⚠️ IMU 数据采集已在运行中")
                return False
            
            if not self.serial:
                if not self.connect():
                    return False
            
            print("▶️ 启动 IMU 数据采集...")
            self.is_running = True
            self.dec_buf = bytearray()
            
            # 在单独的线程中运行数据读取
            self.read_thread = threading.Thread(
                target=self._read_loop,
                daemon=True
            )
            self.read_thread.start()
            
            print("✅ IMU 数据采集已启动")
            return True
    
    def _read_loop(self):
        """数据读取和解析循环"""
        try:
            while self.is_running:
                # 读取数据
                data = rd_data(self.serial)
                num = len(data)
                
                if num > 0:
                    # 更新解析缓冲区
                    self.dec_buf.extend(bytearray(data))
                    
                    if self.debug:
                        print(f'读取长度 {num}, 总长度 {len(self.dec_buf)}')
                        self.decoder.hex_show(self.dec_buf, len(self.dec_buf))
                
                # 解析数据
                if len(self.dec_buf) > 0:
                    ret = self.decoder.proc_data(
                        self.dec_buf, 
                        len(self.dec_buf), 
                        self.yis_out, 
                        self.debug
                    )
                    
                    if ret:
                        # 解析成功，打印数据
                        self._print_data()
                        
                        # 调用回调函数
                        if self.data_callback:
                            try:
                                self.data_callback(self.yis_out.copy())
                            except Exception as e:
                                print(f"⚠️ 数据回调函数出错: {e}")
                
                time.sleep(0.001)
                
        except Exception as e:
            print(f"❌ IMU 数据读取循环出错: {e}")
            self.is_running = False
    
    def _print_data(self):
        """打印 IMU 数据"""
        print(
            f"[IMU] tid:{self.yis_out['tid']:5d} | "
            f"姿态(°) pitch:{self.yis_out['pitch']:7.2f} roll:{self.yis_out['roll']:7.2f} yaw:{self.yis_out['yaw']:7.2f} | "
            f"加速度(g) x:{self.yis_out['acc_x']:7.3f} y:{self.yis_out['acc_y']:7.3f} z:{self.yis_out['acc_z']:7.3f} | "
            f"角速度(°/s) x:{self.yis_out['gyro_x']:7.2f} y:{self.yis_out['gyro_y']:7.2f} z:{self.yis_out['gyro_z']:7.2f}"
        )
    
    def get_data(self) -> Dict:
        """
        获取最新的 IMU 数据
        
        Returns:
            IMU 数据字典的副本
        """
        return self.yis_out.copy()
    
    def stop(self):
        """停止 IMU 数据采集"""
        with self._lock:
            if not self.is_running:
                return
            
            print("⬛ 正在停止 IMU 数据采集...")
            self.is_running = False
            
            # 等待线程结束
            if self.read_thread and self.read_thread.is_alive():
                self.read_thread.join(timeout=2.0)
            
            # 关闭串口
            if self.serial:
                try:
                    close_port(self.serial)
                    self.serial = None
                except Exception as e:
                    print(f"⚠️ 关闭串口时出错: {e}")
            
            print("✅ IMU 数据采集已停止")
    
    def is_active(self) -> bool:
        """检查 IMU 是否正在运行"""
        return self.is_running
    
    def get_status(self) -> Dict:
        """
        获取 IMU 状态信息
        
        Returns:
            状态字典
        """
        return {
            'is_running': self.is_running,
            'port': self.port,
            'baudrate': self.baudrate,
            'connected': self.serial is not None,
        }


def main():
    """独立运行 IMU 模块的示例"""
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import IMU_CONFIG
    
    manager = IMUManager(IMU_CONFIG)
    
    # 启动数据采集
    if manager.start():
        print("IMU 数据采集已启动，按 Ctrl+C 退出...")
        try:
            # 保持主线程运行
            while manager.is_active():
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n收到中断信号...")
        finally:
            manager.stop()
    else:
        print("无法启动 IMU 数据采集")
    
    print("程序结束")


if __name__ == "__main__":
    main()
