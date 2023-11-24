import isaacgym

from omegaconf import DictConfig, OmegaConf
import isaacgymenvs
from isaacgymenvs.learning.quadruped_amp_players import QuadrupedAMPPlayerContinuous
from rl_games.common import env_configurations, vecenv
from typing import Dict
import os
import hydra
import torch
from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver
from isaacgymenvs.learning import (
    amp_continuous,
    amp_players,
    amp_models,
    amp_network_builder,
    quadruped_amp_players,
)
from rl_games.algos_torch import model_builder
from DDPG import DDPG
from buffer import ReplayBuffer


class HL_Player:
    def __init__(self, path):
        self.path = path

    def _get_config(self):
        """Load train and taskconfig files for low-level player"""

        "Train config "
        cfg_train_file = os.path.join(self.path, "train/QuadrupedAMPPPO.yaml")
        cfg_train = OmegaConf.load(cfg_train_file)

        "Task config"
        cfg_task_file = os.path.join(self.path, "task/QuadrupedAMP.yaml")
        cfg_task = OmegaConf.load(cfg_task_file)

        cfg_all_file = os.path.join(self.path, "config.yaml")
        cfg_all = OmegaConf.load(cfg_all_file)

        self.cfg = OmegaConf.create(self.merge_configs(cfg_all, cfg_task, cfg_train))

    def merge_configs(self, config1, config2, config3):
        # Merge dictionaries based on sections
        merged_cfg = {
            **config1,
            "task": {**config2},
            "train": {**config3},
        }
        return merged_cfg

    def omegaconf_to_dict(self, d: DictConfig) -> Dict:
        """Converts an omegaconf DictConfig to a python Dict, respecting variable interpolation."""
        ret = {}
        for k, v in d.items():
            if isinstance(v, DictConfig):
                ret[k] = self.omegaconf_to_dict(v)
            else:
                ret[k] = v
        return ret

    def _create_env(self):
        envs = isaacgymenvs.make(
            self.cfg.seed,
            self.cfg.task_name,
            self.cfg.task.env.numEnvs,
            self.cfg.sim_device,
            self.cfg.rl_device,
            self.cfg.graphics_device_id,
            self.cfg.headless,
            self.cfg.multi_gpu,
            self.cfg.capture_video,
            self.cfg.force_render,
            self.cfg,
        )
        self.envs = envs

    def _build_runner(self, algo_observer):
        from rl_games.torch_runner import Runner

        runner = Runner(algo_observer)
        runner.algo_factory.register_builder(
            "amp_continuous", lambda **kwargs: amp_continuous.AMPAgent(**kwargs)
        )
        runner.player_factory.register_builder(
            "amp_continuous", lambda **kwargs: amp_players.AMPPlayerContinuous(**kwargs)
        )
        model_builder.register_model(
            "continuous_amp",
            lambda network, **kwargs: amp_models.ModelAMPContinuous(network),
        )
        model_builder.register_network(
            "amp", lambda **kwargs: amp_network_builder.AMPBuilder()
        )
        # runner.algo name is used to initialize both agent and player
        # therefore we need to register an agent for 'quadruped_amp' in order to use custom player
        runner.algo_factory.register_builder(
            "quadruped_amp", lambda **kwargs: amp_continuous.AMPAgent(**kwargs)
        )
        runner.player_factory.register_builder(
            "quadruped_amp",
            lambda **kwargs: quadruped_amp_players.QuadrupedAMPPlayerContinuous(
                **kwargs
            ),
        )
        return runner

    def _get_runner(self):
        self.runner = self._build_runner(RLGPUAlgoObserver())
        self.runner.load(self.omegaconf_to_dict(self.cfg.train))
        # self.runner.reset()

    def _get_players(self) -> None:
        self.players = [self.runner.create_player() for i in range(3)]
        [player.restore(self._load_checkpoint[6]) for player in self.players]

        RuntimeError("Fix for many players")

    def _load_checkpoint(self):
        """load the checkpoints"""
        checkpoints = [f'checkpoint_vel{i+1}.pth' for i in range(7)]
        return checkpoints

    def _step_simulation(self):
        is_deterministic = True
        max_terations = 1000

        self.obs_dict = self.envs.reset()

        for i in range(max_terations):
            self.obs_dict = self.envs.reset_done()
            action = self.player.get_action(self.obs_dict[0],is_deterministic)
            self.obs_dict, rew_buf, reset_buf, extras = self.envs.step(action)



    def train(self):
        max_timesteps = 1000000
        num_of_low_level_policies = 3
        H = 20  # time horizon to achieve subgoal
        lr = 0.001  # learning rate
        n_training_iter = 50  # number of training iterations
        batch_size = 1000  # batch size
        gamma = 0.99  # discount factor

        # Create a high level DDPG policy
        hlp = DDPG(self.envs.state_dim, num_of_low_level_policies, lr, H, self.players, gamma)

        # Create replay buffer
        buffer = ReplayBuffer()

        obs = self.envs.reset()
        for i in range(max_timesteps):
            # select action
            policy_to_use_dist = hlp.select_action(obs)

            # From policy_to_use distribution, select highest probability action
            policy_to_use = policy_to_use_dist.argmax()

            # select action from low level policy without accumulating gradients
            with torch.no_grad():
                action = self.players[policy_to_use].select_action(obs).detach()

            # take action in env
            next_obs, reward, done, _ = self.envs.step(action)

            # add to buffer
            buffer.add((obs, action, reward, next_obs, done))

            # update obs
            obs = next_obs

            # if episode ends
            if done:
                obs = self.envs.reset()

            # if buffer is full then update high level policy
            if buffer.size() >= batch_size:
                hlp.update(buffer, n_training_iter, batch_size)


path = os.path.join(os.getcwd(), "cfg/")
HL = HL_Player(path)
HL._get_config()
HL._get_runner()
# HL._get_player()
# HL._create_env()
# HL._step_simulation()


# HL.train()
