#!/usr/bin/env python3
"""
测量相机话题的实际 FPS
使用正确的 QoS 策略 (BEST_EFFORT) 以匹配发布者
"""

import time
from collections import defaultdict, deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image


class FPSMeasure(Node):
    def __init__(self, topics):
        super().__init__('fps_measure')
        
        # 使用与发布者匹配的 QoS
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        
        # 为每个话题创建订阅
        self.subscribers = []
        self.stats = {}
        
        for topic in topics:
            sub = self.create_subscription(
                Image,
                topic,
                self.make_callback(topic),
                qos_profile
            )
            self.subscribers.append(sub)
            self.stats[topic] = {
                'timestamps': deque(maxlen=100),
                'frame_count': 0,
                'start_time': None,
            }
        
        # 定时打印统计信息
        self.timer = self.create_timer(2.0, self.print_stats)
        
        self.get_logger().info(f'开始测量 {len(topics)} 个话题的 FPS...')
        for topic in topics:
            self.get_logger().info(f'  - {topic}')
    
    def make_callback(self, topic):
        def callback(msg):
            now = time.time()
            stats = self.stats[topic]
            
            if stats['start_time'] is None:
                stats['start_time'] = now
            
            stats['timestamps'].append(now)
            stats['frame_count'] += 1
        
        return callback
    
    def print_stats(self):
        print('\n' + '='*70)
        print(f'FPS 统计 (最近 100 帧)')
        print('='*70)
        
        for topic, stats in self.stats.items():
            timestamps = list(stats['timestamps'])
            frame_count = stats['frame_count']
            
            if len(timestamps) < 2:
                print(f'{topic:30s}: 等待数据...')
                continue
            
            # 计算最近 100 帧的平均 FPS
            time_span = timestamps[-1] - timestamps[0]
            recent_fps = (len(timestamps) - 1) / time_span if time_span > 0 else 0
            
            # 计算总体平均 FPS
            if stats['start_time']:
                total_time = time.time() - stats['start_time']
                avg_fps = frame_count / total_time if total_time > 0 else 0
            else:
                avg_fps = 0
            
            # 计算延迟（相邻帧的时间间隔）
            intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
            if intervals:
                avg_interval = sum(intervals) / len(intervals)
                min_interval = min(intervals)
                max_interval = max(intervals)
            else:
                avg_interval = min_interval = max_interval = 0
            
            print(f'{topic:30s}: {recent_fps:6.2f} fps (瞬时) | '
                  f'{avg_fps:6.2f} fps (平均) | '
                  f'总帧数: {frame_count:5d}')
            print(f'{"":30s}  帧间隔: {avg_interval*1000:5.1f}ms (avg) | '
                  f'{min_interval*1000:5.1f}ms (min) | '
                  f'{max_interval*1000:5.1f}ms (max)')


def main(args=None):
    rclpy.init(args=args)
    
    # 可以从命令行参数指定话题，或使用默认话题
    import sys
    if len(sys.argv) > 1:
        topics = sys.argv[1:]
    else:
        # 默认测量所有相机话题
        topics = [f'/helmet/cam{i}/image_raw' for i in range(8)]
    
    node = FPSMeasure(topics)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\n')
        node.get_logger().info('停止测量')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
