import numpy as np
import os
import torch

from gym import spaces
from enum import Enum
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.tasks.quadruped_amp_base import QuadrupedAMPBase

from isaacgym.torch_utils import *
from isaacgymenvs.utils.torch_jit_utils import *

from typing import Tuple, Dict


class QuadrupedGetup(QuadrupedAMPBase):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        super().__init__(cfg=cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)
        self.target_height = self.cfg["env"]["targetHeight"]
        self.target_height_diff_eps = self.cfg["env"]["targetHeightDiffEps"]

    def compute_reward(self):
        self.rew_buf[:] = compute_getup_reward(
            # tensors
            self.root_states,
            self.target_height
        )

    def compute_reset(self):
        self.reset_buf, self._terminate_buf = compute_getup_reset(
            self.progress_buf,
            self._terminate_buf,
            self.root_states, 
            self.contact_forces,
            self.max_episode_length, 
            self.target_height,
            self.target_height_diff_eps,
            self._termination_height,
            self._enable_early_termination
        )

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_getup_reward(root_states, target_height):
    # type: (Tensor, Tensor) -> Tensor
    body_height = root_states[:, 2]
    reward = torch.exp(-(body_height - target_height)**2)
    return reward

@torch.jit.script
def compute_getup_reset(
    # tensors
    reset_buf, 
    progress_buf, 
    root_states,
    max_episode_length, 
    target_height,
    target_height_eps,
    termination_height, # unused
    enable_early_termination
):
    # type: (Tensor, Tensor, Tensor, int, float, float, float, bool) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)
    if (enable_early_termination):
        # terminated = terminated | (torch.norm(contact_forces[:, base_index, :], dim=1) > 1.)
        # terminated = terminated | torch.any(torch.norm(contact_forces[:, knee_indices, :], dim=2) > 1., dim=1)
        body_height = root_states[:, 2]
        terminated = terminated | torch.norm((body_height - target_height), dim=-1) < target_height_eps

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)
    return reset, terminated