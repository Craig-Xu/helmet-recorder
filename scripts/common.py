#!/usr/bin/env python3
"""Shared helpers for project scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / 'config' / 'config.yaml'


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    with path.open('w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)


def sorted_cam_names(extrinsics: dict[str, Any]) -> list[str]:
    def cam_index(name: str) -> int:
        try:
            return int(name.replace('cam', ''))
        except Exception:
            return 10**9

    return sorted(extrinsics.keys(), key=cam_index)


def resolve_cam_to_video(camera_ids: list[int], raw_index_map: dict | None) -> dict[int, int]:
    """Normalize index_map to cam_id -> /dev/video_id."""
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
