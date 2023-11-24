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
    quadruped_amp_players
)
from rl_games.algos_torch import model_builder
from isaacgymenvs.learning.DDPG import DDPG
from isaacgymenvs.learning.replay_buffer import ReplayBuffer
import numpy as np
import wandb
from torch.utils.tensorboard import SummaryWriter



class HL_Player:

    def __init__(self,path_cfg, path_checkpoint):
        self.path_cfg = path_cfg
        self.path_checkpoint = path_checkpoint
        self.save_checkpoint = f'{self.path_checkpoint}/trained_checkpoints'

    def _get_config(self):
        '''Load train and taskconfig files for low-level player'''

        'Train config '
        cfg_train_file = os.path.join(self.path_cfg,'train/QuadrupedAMPPPO.yaml')
        cfg_train = OmegaConf.load(cfg_train_file)

        'Task config'
        cfg_task_file = os.path.join(self.path_cfg, 'task/QuadrupedAMP.yaml')
        cfg_task = OmegaConf.load(cfg_task_file)

        cfg_all_file= os.path.join(self.path_cfg, 'config.yaml')
        cfg_all = OmegaConf.load(cfg_all_file)

        self.cfg = OmegaConf.create(self.merge_configs(cfg_all,cfg_task,cfg_train))


    def merge_configs(self,config1, config2, config3):
        # Merge dictionaries based on sections
        merged_cfg = {
            **config1,
            'task': {**config2},
            'train': {**config3},
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
            self.cfg)
        self.envs = envs


    def _build_runner(self,algo_observer):
        from rl_games.torch_runner import Runner
        runner = Runner(algo_observer)
        runner.algo_factory.register_builder('amp_continuous', lambda **kwargs: amp_continuous.AMPAgent(**kwargs))
        runner.player_factory.register_builder('amp_continuous',
                                               lambda **kwargs: amp_players.AMPPlayerContinuous(**kwargs))
        model_builder.register_model('continuous_amp', lambda network, **kwargs: amp_models.ModelAMPContinuous(network))
        model_builder.register_network('amp', lambda **kwargs: amp_network_builder.AMPBuilder())
        # runner.algo name is used to initialize both agent and player
        # therefore we need to register an agent for 'quadruped_amp' in order to use custom player
        runner.algo_factory.register_builder('quadruped_amp', lambda **kwargs: amp_continuous.AMPAgent(**kwargs))
        runner.player_factory.register_builder('quadruped_amp',
                                               lambda **kwargs: quadruped_amp_players.QuadrupedAMPPlayerContinuous(
                                                   **kwargs))
        return runner


    def _get_runner(self):
        self.runner = self._build_runner(RLGPUAlgoObserver())
        self.runner.load(self.omegaconf_to_dict(self.cfg.train))
        # self.runner.reset()


    def _get_player(self):
        ' get sigle player to check and stpe the simulation'

        self.player = self.runner.create_player()
        self.player.restore(self._load_checkpoint()[2])

    def _get_players(self):
        self.players = []
        self.num_agents = self.cfg.task.env.numEnvs
        ' initialise all players evaluated with corresponding  checkpoints'
        players = [self.runner.create_player() for i in range(8)]
        [players[i].restore(self._load_checkpoint()[i]) for i in range(len(players))]

        self.players = players


    def _load_checkpoint(self):
       ''' load the checkpoints '''
       filenames = [f'{self.path_checkpoint}/checkpoint_vel{vel+1}.pth' for vel in range(8)]
       return filenames

    def _step_simulation(self):
        is_deterministic = True
        max_terations = 1000
        velocity = -1

        player = self.players[velocity]

        self.obs_dict = self.envs.reset()

        for i in range(max_terations):
            self.obs_dict = self.envs.reset_done()
            action = player.get_action(self.obs_dict[0],is_deterministic)
            self.obs_dict, rew_buf, reset_buf, extras = self.envs.step(action)


    def train(self):

        log_dir = self.save_checkpoint
        writer = SummaryWriter(log_dir=log_dir)

        max_timesteps = 10000
        num_of_low_level_policies = 7
        H = 20  # time horizon to achieve subgoal
        lr = 0.001  # learning rate
        n_training_iter = 50  # number of training iterations
        batch_size = 32768 # batch size
        gamma = 0.99  # discount factor
        buffer_size = 100000
        device ='cuda:0'



        buffer = ReplayBuffer(buffer_size,device)

        obs = self.envs.reset()

        last_action = torch.zeros(self.num_agents,num_of_low_level_policies).cuda(0)


        obs_hl = torch.cat([obs['obs'][:,[7,-2]], last_action],1)

        state_dim = obs_hl.shape[1]

        # Create a high level DDPG policy
        hlp = DDPG(state_dim,num_of_low_level_policies, lr, H, self.players, gamma)

        # Create replay buffer

        self.input_dict = {
            'obs' : None,
            'action' : None,
            'reward': None,
            'next_obs': None,
            'done': None,
            'hl_obs': None,
            'hl_next_obs': None,
            'hl_action': None,
            'hl_last_action':None

        }

        wandb.init(project="HRL-DDPG", name="hrl")

        for iteration in range(max_timesteps):


            # select action
            policy_to_use_dist = hlp.select_action(obs_hl)
            # From policy_to_use distribution, select highest probability action
            policy_to_use_dist = policy_to_use_dist.reshape(self.num_agents,num_of_low_level_policies)
            policy_to_use = policy_to_use_dist.argmax(axis=1)
            policy_index = np.array(policy_to_use)

            NN_indices = []
            actions = []
            for i in range(num_of_low_level_policies):
                target_number = i
                NN_index = np.where(policy_index==target_number)[0]
                if len(NN_index) !=0:
                    NN_index_tensor = torch.tensor(NN_index).cuda(0)
                    NN_obs = obs['obs'][NN_index_tensor,:]
                    NN_obs_dict = {'obs': NN_obs}
                    NN_action = self.players[i].get_action(NN_obs_dict)
                    if NN_action.ndim == 1:
                        NN_action = torch.unsqueeze(NN_action, dim=0).cuda(0)
                    actions.append(NN_action)

            action = torch.cat(actions,dim=0)
            next_obs, reward, done, _ = self.envs.step(action)


            #Low level inputs
            self.input_dict['obs'] = obs['obs']
            self.input_dict['action'] = action
            self.input_dict['reward'] = torch.unsqueeze(reward, dim=-1)
            self.input_dict['next_obs'] = next_obs['obs']
            self.input_dict['done'] = torch.unsqueeze(done,dim=-1)

            # #High Level Inputs
            current_action = torch.tensor(policy_to_use_dist).cuda(0)
            next_obs_hl = torch.cat([next_obs['obs'][:,[7,-2]], current_action], -1)


            self.input_dict['hl_obs'] = obs_hl
            self.input_dict['hl_next_obs'] =next_obs_hl
            self.input_dict['hl_action'] = current_action
            self.input_dict['hl_last_action'] = last_action

            # add to buffer
            buffer.store(self.input_dict)

            # update obs and last action
            obs = next_obs
            obs_hl = next_obs_hl
            last_action = current_action

            #reset done environemnts
            done_ids = done.nonzero(as_tuple=False).squeeze(-1)
            if len(done_ids) > 0:
                self.envs.reset_idx(done_ids)

            # if buffer is full then update high level policy
            if buffer._total_count >= batch_size:
                hlp.update(buffer, n_training_iter, batch_size)


            print(f'Iteration {iteration}/{max_timesteps}')

            if iteration%50==0:
                hlp.save(self.save_checkpoint, f'hl_{iteration}')
                wandb.log({"reward": torch.mean(reward).item()})
                writer.add_scalar("reward", torch.mean(reward), iteration)
                print('Checkpoint Saved!')






path_cfg = os.path.join(os.getcwd(), 'cfg/')
path_checkpoint = os.path.join(os.getcwd(), 'learning/saved_checkpoints')
HL = HL_Player(path_cfg,path_checkpoint)
HL._get_config()
HL._get_runner()
HL._get_players()
HL._create_env()
#HL._step_simulation()
HL.train()
#



