#!/usr/bin/env python3
"""通过拔插对比方式查找 IMU 串口。"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

try:
    from serial.tools import list_ports
except Exception as exc:  # pragma: no cover
    print(f"导入 pyserial 失败: {exc}")
    print("请先安装依赖: uv sync")
    sys.exit(1)


@dataclass(frozen=True)
class PortInfo:
    device: str
    description: str
    hwid: str
    vid: int | None
    pid: int | None
    serial_number: str | None
    manufacturer: str | None
    product: str | None
    interface: str | None


def snapshot_ports() -> dict[str, PortInfo]:
    result: dict[str, PortInfo] = {}
    for p in list_ports.comports():
        result[p.device] = PortInfo(
            device=p.device,
            description=getattr(p, 'description', ''),
            hwid=getattr(p, 'hwid', ''),
            vid=getattr(p, 'vid', None),
            pid=getattr(p, 'pid', None),
            serial_number=getattr(p, 'serial_number', None),
            manufacturer=getattr(p, 'manufacturer', None),
            product=getattr(p, 'product', None),
            interface=getattr(p, 'interface', None),
        )
    return result


def format_port(port: PortInfo) -> str:
    usb_id = f"VID:PID={port.vid:04X}:{port.pid:04X}" if port.vid is not None and port.pid is not None else "VID:PID=未知"
    serial_num = port.serial_number or '未知'
    product = port.product or port.description or '未知设备'
    return f"{port.device:<18} {usb_id:<18} SN={serial_num:<16} {product}"


def print_ports(title: str, ports: dict[str, PortInfo]) -> None:
    print(f"\n{title}")
    print('-' * 88)
    if not ports:
        print('(无串口设备)')
        return
    for key in sorted(ports.keys()):
        print(format_port(ports[key]))


def diff_ports(before: dict[str, PortInfo], after: dict[str, PortInfo]) -> tuple[list[PortInfo], list[PortInfo]]:
    removed = [before[k] for k in sorted(before.keys() - after.keys())]
    added = [after[k] for k in sorted(after.keys() - before.keys())]
    return removed, added


def wait_enter(prompt: str) -> None:
    print(prompt)
    input('完成后按回车继续...')


def main() -> int:
    parser = argparse.ArgumentParser(description='通过拔插检测 IMU 串口')
    parser.add_argument('--pause', type=float, default=0.8, help='每次回车后等待系统刷新设备列表的秒数 (默认: 0.8)')
    args = parser.parse_args()

    print('=== IMU 串口拔插检测工具 ===')
    print('说明: 建议只操作目标 IMU，避免同时插拔其他 USB 设备。')

    initial = snapshot_ports()
    print_ports('当前串口列表 (初始):', initial)

    wait_enter('\n步骤1: 请先拔掉 IMU。')
    time.sleep(max(args.pause, 0.0))
    after_unplug = snapshot_ports()
    print_ports('拔掉后串口列表:', after_unplug)

    removed, _ = diff_ports(initial, after_unplug)
    if removed:
        print('\n拔掉时消失的端口:')
        for p in removed:
            print(f'  - {format_port(p)}')
    else:
        print('\n未检测到端口消失（可能设备原本未连接，或系统未及时刷新）。')

    wait_enter('\n步骤2: 请重新插上 IMU。')
    time.sleep(max(args.pause, 0.0))
    after_replug = snapshot_ports()
    print_ports('重新插上后串口列表:', after_replug)

    _, added = diff_ports(after_unplug, after_replug)
    if added:
        print('\n重新插上后新增端口（最可能是 IMU 端口）:')
        for p in added:
            print(f'  ✅ {p.device}')
            print(f'     {format_port(p)}')
    else:
        print('\n未检测到新增端口。')
        print('可尝试:')
        print('  1) 提高等待时间: --pause 2.0')
        print('  2) 检查权限: 当前用户是否在 dialout / tty / uucp 组')
        print('  3) 查看内核日志: dmesg | tail -n 50')
        return 1

    print('\n完成。把上面输出的设备路径（如 /dev/ttyUSB0）填到 config/config.yaml 的 imu.port 即可。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
