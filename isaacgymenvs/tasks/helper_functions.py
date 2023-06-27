import abc 
import torch

from enum import Enum
from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.tasks.quadruped_tasks import TargetVelocity


class MetricEvaluation:
    # TODO: Refactor this into a class
    def compute_reward_target_location(root_states, target_pos):
        """
        args:
            root_states - robot root states in world frame
            target_location - desired location in world frame
        """
        root_pos = root_states[: ,:3]
        return torch.exp(exp_neg_sq(torch.norm(root_pos - target_pos)))

    def compute_observation_target_location(root_states, target_pos):
        root_pos = root_states[: ,:3]
        root_rot = root_states[:, 3:7]
        heading_rot = calc_heading_quat_inv(root_rot)
        goal_pos = target_pos - root_pos
        goal_pos_local = my_quat_rotate(goal_pos, heading_rot)
        return goal_pos_local

