from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils import configclass


class PiperSymmetricGripperAction(ActionTerm):
    """Map a single scalar gripper command to PiPER's two mirrored finger joints."""

    cfg: "PiperSymmetricGripperActionCfg"
    _asset: Articulation

    def __init__(self, cfg: "PiperSymmetricGripperActionCfg", env) -> None:
        super().__init__(cfg, env)

        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.joint_names, preserve_order=True)
        if len(self._joint_ids) != 2:
            raise ValueError(
                f"PiperSymmetricGripperAction expects exactly 2 mirrored joints, got {self._joint_names}"
            )

        self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, 2, device=self.device)

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        joint7_pos = torch.clamp(actions[:, 0], min=self.cfg.lower_limit, max=self.cfg.upper_limit)
        self._processed_actions[:, 0] = joint7_pos
        self._processed_actions[:, 1] = -joint7_pos

    def apply_actions(self):
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0


@configclass
class PiperSymmetricGripperActionCfg(ActionTermCfg):
    """Configuration for PiPER's mirrored finger action term."""

    class_type: type[ActionTerm] = PiperSymmetricGripperAction
    joint_names: list[str] = MISSING
    lower_limit: float = 0.0
    upper_limit: float = 0.05
