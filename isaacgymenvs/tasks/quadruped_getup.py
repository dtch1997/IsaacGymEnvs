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

def random_uniform(lower, upper, device):
    return (upper - lower) * torch.rand_like(upper, device=device)

def random_uniform_quaternion(n: int, device) -> torch.Tensor:
    """
    Reference: Top answer to https://stackoverflow.com/questions/31600717/how-to-generate-a-random-quaternion-quickly
    """
    two_pi = np.pi * 2
    u = torch.zeros(n).uniform_(0., 1)
    v = torch.zeros(n).uniform_(0., 1)
    w = torch.zeros(n).uniform_(0., 1)

    qx = torch.sqrt(1-u) * torch.sin(two_pi * v)
    qy = torch.sqrt(1-u) * torch.cos(two_pi * v)
    qz = torch.sqrt(u) * torch.sin(two_pi * w)
    qw = torch.sqrt(u) * torch.cos(two_pi * w)
    q = torch.stack([qx, qy, qz, qw], dim=-1)
    return q.to(device)

class QuadrupedGetup(QuadrupedAMPBase):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        super().__init__(cfg=cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)
        self.target_height = self.cfg["env"]["targetHeight"]
        self.target_height_diff_eps = self.cfg["env"]["targetHeightDiffEps"]

    def _reset_actors(self, env_ids):
        # TODO: Reset to a fallen state by dropping the robot from random height, orientation, and dof pos. 
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)


        self.dof_pos[env_ids] = random_uniform(self.dof_limits_lower, self.dof_limits_upper, device=self.device)
        self.dof_vel[env_ids] = torch.zeros_like(self.default_dof_vel).uniform_(-0.2, 0.2) # m/s
        root_h = torch.zeros_like(self.initial_root_states[:,2]).uniform_(0.2, 0.8) # m
        root_orn = random_uniform_quaternion(self.initial_root_states.shape[0], device=self.device)
        root_lin_vel = torch.zeros_like(self.initial_root_states[:,7:10]).uniform_(-0.1, 0.1)
        root_ang_vel = torch.zeros_like(self.initial_root_states[:,10:13]).uniform_(-0.1, 0.1)

        self.initial_root_states[:,2] = root_h
        self.initial_root_states[:,3:7] = root_orn 
        self.initial_root_states[:,7:10] = root_lin_vel
        self.initial_root_states[:,10:13] = root_ang_vel

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.initial_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self._terminate_buf[env_ids] = 0

    def compute_reward(self):
        self.rew_buf[:] = compute_getup_reward(
            # tensors
            self.root_states,
            self.target_height
        )

    def compute_reset(self):
        
        self.reset_buf, self._terminate_buf = compute_getup_reset(
            self.reset_buf,
            self.progress_buf,
            self.root_states, 
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
    # type: (Tensor, float) -> Tensor
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