"""Load YAML or JSON experiment configs without requiring PyYAML."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".json"}:
        return json.loads(text)
    try:
        import yaml

        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError(f"config must be a mapping: {path}")
        return data
    except ImportError:
        return _minimal_yaml(text)


def _minimal_yaml(text: str) -> dict[str, Any]:
    """Flat and one-level nested YAML (key: value / key: \\n  sub: value)."""
    root: dict[str, Any] = {}
    current: dict[str, Any] | None = None
    current_key = None
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        if ":" not in raw:
            continue
        key, val = raw.split(":", 1)
        key, val = key.strip(), val.strip()
        parsed = _parse_scalar(val)
        if indent == 0:
            if val == "":
                current = {}
                root[key] = current
                current_key = key
            else:
                root[key] = parsed
                current = None
        else:
            if current is None:
                current = {}
                root[current_key or key] = current
            current[key] = parsed
    return root


def _parse_scalar(val: str) -> Any:
    if val == "":
        return ""
    if val.lower() in {"true", "yes"}:
        return True
    if val.lower() in {"false", "no"}:
        return False
    if val.lower() in {"null", "none"}:
        return None
    try:
        if "." in val:
            return float(val)
        return int(val)
    except ValueError:
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            return val[1:-1]
        return val
