from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def resolve_config_path(config_path: str | None = None) -> Path:
    candidates = []
    if config_path:
        candidates.append(Path(config_path).expanduser())

    # 通过 rospkg 定位包真实路径（.resolve() 跟随软链接），
    # 进而找到项目根目录（包目录的上一级）下的 config/config.yaml
    try:
        import rospkg
        pkg_path = Path(rospkg.RosPack().get_path('helmet_recorder_ros')).resolve()
        candidates.append(pkg_path.parent / 'config' / 'config.yaml')  # 项目根/config/
        candidates.append(pkg_path / 'config' / 'config.yaml')          # 包内 config/（备用）
    except Exception:
        pass

    # 回退：相对 cwd（独立脚本模式）
    candidates.append(Path.cwd() / 'config' / 'config.yaml')
    candidates.append(Path.cwd() / 'config.yaml')

    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    with path.open('w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)


def resolve_cam_to_video(camera_ids: list[int], raw_index_map: dict | None) -> dict[int, int]:
    ids = [int(x) for x in camera_ids]
    id_set = set(ids)
    raw = {int(k): int(v) for k, v in (raw_index_map or {}).items()}
    if not raw:
        return {cid: cid for cid in ids}

    keys = set(raw.keys())
    vals = set(raw.values())

    if vals.issubset(id_set):
        return {cam: vid for cam, vid in raw.items() if vid in id_set}
    if keys.issubset(id_set):
        return {cam: vid for vid, cam in raw.items() if vid in id_set}
    return {cam: vid for cam, vid in raw.items()}
