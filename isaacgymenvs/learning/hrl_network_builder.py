from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import layers
from rl_games.algos_torch import network_builder

import torch
import torch.nn as nn
import numpy as np

from isaacgymenvs.learning.npmp.npmp.model import Actor

class HRLBuilder(network_builder.A2CBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    class Network(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            env_actions_num = kwargs.pop('actions_num')
            self.input_shape = kwargs['input_shape']
            # TODO: remove hardcoded constant
            kwargs['actions_num'] = 32 # latent dim
            super().__init__(params, **kwargs)

            if self.is_continuous:
                if (not self.space_config['learn_sigma']):
                    sigma_init = self.init_factory.create(**self.space_config['sigma_init'])
                    self.llc_sigma = nn.Parameter(torch.zeros(env_actions_num, requires_grad=False, dtype=torch.float32), requires_grad=False)
                    sigma_init(self.llc_sigma)

            self._build_llc()
            self._load_llc_from_checkpoint()
            return

        def load(self, params):
            super().load(params)

            # TODO: load params from config
            self.state_dim = 45
            self.action_dim = 12 # 12 dof pos
            self.latent_dim = 32
            self.num_future_states = 2
            self.hidden_dim = 32
            self.batch_size = 8
            return

        def _build_llc(self):
            self.llc = Actor(self.input_shape, self.action_dim, self.latent_dim, self.hidden_dim)
            
            mlp_init = self.init_factory.create(**self._disc_initializer)
            for m in self.llc.modules():
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias) 
            return

        def _load_llc_from_checkpoint(self):
            # TODO: remove hardcoded path
            checkpoint_path = '/home/daniel/Documents/github/IsaacGymEnvs/isaacgymenvs/data/checkpoints/llc/actor.pth'
            self.llc.load_state_dict(torch.load(checkpoint_path))
            for p in self.llc.parameters():
                p.requires_grad=False

        def forward(self, obs_dict):
            latent, _, value, states = super().forward(obs_dict)
            obs = obs_dict['obs']
            mu = self.actor(obs, latent)
            sigma = self.llc_sigma
            return mu, sigma, value, states

    def build(self, name, **kwargs):
        net = HRLBuilder.Network(self.params, **kwargs)
        return net