from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

DEFAULT_GRID_LAYOUT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "grid_layouts.json"


def _normalize_visible_columns(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        column_name = item.strip()
        if not column_name or column_name in normalized:
            continue
        normalized.append(column_name)
    return normalized


def _normalize_layouts(raw: object) -> dict[str, dict[str, list[str]]]:
    if not isinstance(raw, dict):
        return {}

    normalized: dict[str, dict[str, list[str]]] = {}
    for raw_surface_id, raw_payload in raw.items():
        if not isinstance(raw_surface_id, str):
            continue
        surface_id = raw_surface_id.strip()
        if not surface_id or not isinstance(raw_payload, dict):
            continue
        raw_visible_columns = raw_payload.get("visible_columns")
        if not isinstance(raw_visible_columns, list):
            continue
        visible_columns = _normalize_visible_columns(raw_visible_columns)
        normalized[surface_id] = {"visible_columns": visible_columns}
    return normalized


def load_grid_layouts(path: Path | str | None = None) -> dict[str, dict[str, list[str]]]:
    config_path = Path(path) if path else DEFAULT_GRID_LAYOUT_CONFIG_PATH
    if not config_path.exists():
        return {}

    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return _normalize_layouts(raw)


def save_grid_layouts(
    layouts: dict[str, dict[str, list[str]]],
    path: Path | str | None = None,
) -> Path:
    config_path = Path(path) if path else DEFAULT_GRID_LAYOUT_CONFIG_PATH
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _normalize_layouts(layouts)
    config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return config_path


def load_grid_layout(surface_id: str, path: Path | str | None = None) -> Optional[list[str]]:
    surface_key = str(surface_id).strip()
    if not surface_key:
        return None
    payload = load_grid_layouts(path).get(surface_key)
    if not isinstance(payload, dict):
        return None
    visible_columns = payload.get("visible_columns")
    if not isinstance(visible_columns, list):
        return None
    return list(visible_columns)


def save_grid_layout(
    surface_id: str,
    visible_columns: list[str],
    path: Path | str | None = None,
) -> Path:
    surface_key = str(surface_id).strip()
    if not surface_key:
        raise ValueError("surface_id must be non-empty")
    layouts = load_grid_layouts(path)
    layouts[surface_key] = {"visible_columns": _normalize_visible_columns(visible_columns)}
    return save_grid_layouts(layouts, path)


def clear_grid_layout(surface_id: str, path: Path | str | None = None) -> Optional[Path]:
    surface_key = str(surface_id).strip()
    if not surface_key:
        return None
    layouts = load_grid_layouts(path)
    if surface_key not in layouts:
        return None
    layouts.pop(surface_key, None)
    return save_grid_layouts(layouts, path)
