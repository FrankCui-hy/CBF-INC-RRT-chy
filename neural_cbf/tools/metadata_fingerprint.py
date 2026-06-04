import hashlib
import json
from typing import Any

import torch


def _stable_jsonable(value: Any) -> Any:
	if isinstance(value, torch.Tensor):
		return _stable_jsonable(value.detach().cpu().tolist())
	if isinstance(value, dict):
		return {str(k): _stable_jsonable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
	if isinstance(value, (list, tuple)):
		return [_stable_jsonable(v) for v in value]
	if isinstance(value, float):
		return float(value)
	if isinstance(value, int):
		return int(value)
	if isinstance(value, bool) or value is None or isinstance(value, str):
		return value
	return str(value)


def compute_metadata_fingerprint(metadata: dict) -> str:
	"""Stable sha256 fingerprint for cache/model geometry metadata."""
	payload = json.dumps(_stable_jsonable(metadata), sort_keys=True, separators=(",", ":"))
	return hashlib.sha256(payload.encode("utf-8")).hexdigest()
