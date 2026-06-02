from __future__ import annotations

from typing import Dict, Iterable, List, Union

import torch
import torch.nn as nn


PANDA_LINK_ID_TO_NAME = {
    -1: "panda_link0",
    0: "panda_link1",
    1: "panda_link2",
    2: "panda_link3",
    3: "panda_link4",
    4: "panda_link5",
    5: "panda_link6",
    6: "panda_link7",
    7: "panda_link8",
    8: "panda_hand",
    9: "panda_leftfinger",
    10: "panda_rightfinger",
    11: "panda_grasptarget",
}
PANDA_LINK_NAME_TO_ID = {v: k for k, v in PANDA_LINK_ID_TO_NAME.items()}


def _as_link_ids(link_ids_or_names: Iterable[Union[int, str]]) -> List[int]:
    out = []
    for item in link_ids_or_names:
        if isinstance(item, str):
            if item not in PANDA_LINK_NAME_TO_ID:
                raise KeyError(f"Unknown Panda link name: {item}")
            out.append(int(PANDA_LINK_NAME_TO_ID[item]))
        else:
            out.append(int(item))
    return out


class TorchPandaFK(nn.Module):
    """Differentiable FK for the repository's Panda URDF link frames.

    The implementation follows utils/robot/franka_panda/panda.urdf and the
    PyBullet joint/link indexing used by environment.FrankaPanda.
    """

    def __init__(self, finger_open: float = 0.04) -> None:
        super().__init__()
        self.finger_open = float(finger_open)

    @staticmethod
    def _eye(batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(batch, 1, 1)

    @staticmethod
    def _transform_xyz_rpy(
        batch: int,
        xyz: tuple[float, float, float],
        rpy: tuple[float, float, float],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        roll, pitch, yaw = [torch.tensor(v, device=device, dtype=dtype) for v in rpy]
        cr, sr = torch.cos(roll), torch.sin(roll)
        cp, sp = torch.cos(pitch), torch.sin(pitch)
        cy, sy = torch.cos(yaw), torch.sin(yaw)

        Rx = torch.stack(
            [
                torch.stack([torch.ones_like(cr), torch.zeros_like(cr), torch.zeros_like(cr)]),
                torch.stack([torch.zeros_like(cr), cr, -sr]),
                torch.stack([torch.zeros_like(cr), sr, cr]),
            ]
        )
        Ry = torch.stack(
            [
                torch.stack([cp, torch.zeros_like(cp), sp]),
                torch.stack([torch.zeros_like(cp), torch.ones_like(cp), torch.zeros_like(cp)]),
                torch.stack([-sp, torch.zeros_like(cp), cp]),
            ]
        )
        Rz = torch.stack(
            [
                torch.stack([cy, -sy, torch.zeros_like(cy)]),
                torch.stack([sy, cy, torch.zeros_like(cy)]),
                torch.stack([torch.zeros_like(cy), torch.zeros_like(cy), torch.ones_like(cy)]),
            ]
        )
        R = Rz @ Ry @ Rx
        T = TorchPandaFK._eye(batch, device, dtype)
        T[:, :3, :3] = R
        T[:, :3, 3] = torch.tensor(xyz, device=device, dtype=dtype)
        return T

    @staticmethod
    def _rot_z(theta: torch.Tensor) -> torch.Tensor:
        B = int(theta.shape[0])
        T = TorchPandaFK._eye(B, theta.device, theta.dtype)
        c = torch.cos(theta)
        s = torch.sin(theta)
        T[:, 0, 0] = c
        T[:, 0, 1] = -s
        T[:, 1, 0] = s
        T[:, 1, 1] = c
        return T

    @staticmethod
    def _trans_batch(
        x: Union[float, torch.Tensor],
        y: Union[float, torch.Tensor],
        z: Union[float, torch.Tensor],
        batch: int,
        device,
        dtype,
    ) -> torch.Tensor:
        T = TorchPandaFK._eye(batch, device, dtype)
        vals = []
        for v in (x, y, z):
            if isinstance(v, torch.Tensor):
                vals.append(v.to(device=device, dtype=dtype).reshape(batch))
            else:
                vals.append(torch.full((batch,), float(v), device=device, dtype=dtype))
        T[:, 0, 3] = vals[0]
        T[:, 1, 3] = vals[1]
        T[:, 2, 3] = vals[2]
        return T

    def all_link_transforms(self, q: torch.Tensor) -> Dict[int, torch.Tensor]:
        if q.ndim != 2 or q.shape[1] != 7:
            raise ValueError(f"Expected q shape [B, 7], got {tuple(q.shape)}")
        B = int(q.shape[0])
        device = q.device
        dtype = q.dtype
        links: Dict[int, torch.Tensor] = {}
        links[-1] = self._eye(B, device, dtype)

        T = links[-1]
        T = T @ self._transform_xyz_rpy(B, (0.0, 0.0, 0.333), (0.0, 0.0, 0.0), device, dtype) @ self._rot_z(q[:, 0])
        links[0] = T
        T = T @ self._transform_xyz_rpy(B, (0.0, 0.0, 0.0), (-1.57079632679, 0.0, 0.0), device, dtype) @ self._rot_z(q[:, 1])
        links[1] = T
        T = T @ self._transform_xyz_rpy(B, (0.0, -0.316, 0.0), (1.57079632679, 0.0, 0.0), device, dtype) @ self._rot_z(q[:, 2])
        links[2] = T
        T = T @ self._transform_xyz_rpy(B, (0.0825, 0.0, 0.0), (1.57079632679, 0.0, 0.0), device, dtype) @ self._rot_z(q[:, 3])
        links[3] = T
        T = T @ self._transform_xyz_rpy(B, (-0.0825, 0.384, 0.0), (-1.57079632679, 0.0, 0.0), device, dtype) @ self._rot_z(q[:, 4])
        links[4] = T
        T = T @ self._transform_xyz_rpy(B, (0.0, 0.0, 0.0), (1.57079632679, 0.0, 0.0), device, dtype) @ self._rot_z(q[:, 5])
        links[5] = T
        T = T @ self._transform_xyz_rpy(B, (0.088, 0.0, 0.0), (1.57079632679, 0.0, 0.0), device, dtype) @ self._rot_z(q[:, 6])
        links[6] = T

        links[7] = links[6] @ self._transform_xyz_rpy(B, (0.0, 0.0, 0.107), (0.0, 0.0, 0.0), device, dtype)
        links[8] = links[7] @ self._transform_xyz_rpy(B, (0.0, 0.0, 0.0), (0.0, 0.0, -0.785398163397), device, dtype)
        finger = torch.full((B,), self.finger_open, device=device, dtype=dtype)
        finger_origin = self._transform_xyz_rpy(B, (0.0, 0.0, 0.0584), (0.0, 0.0, 0.0), device, dtype)
        links[9] = links[8] @ finger_origin @ self._trans_batch(0.0, finger, 0.0, B, device, dtype)
        links[10] = links[8] @ finger_origin @ self._trans_batch(0.0, -finger, 0.0, B, device, dtype)
        links[11] = links[8] @ self._transform_xyz_rpy(B, (0.0, 0.0, 0.105), (0.0, 0.0, 0.0), device, dtype)
        return links

    def forward(self, q: torch.Tensor, link_ids_or_names: Iterable[Union[int, str]]) -> torch.Tensor:
        link_ids = _as_link_ids(link_ids_or_names)
        links = self.all_link_transforms(q)
        missing = [idx for idx in link_ids if idx not in links]
        if missing:
            raise KeyError(f"Unsupported Panda link ids: {missing}")
        return torch.stack([links[idx] for idx in link_ids], dim=1)
