from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    p = ensure_parent(path)
    torch.save(payload, p)
