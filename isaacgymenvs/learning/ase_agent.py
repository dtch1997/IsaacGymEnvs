# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from isaacgymenvs.learning import amp_continuous

import torch
from isaacgym.torch_utils import *
from rl_games.algos_torch import torch_ext
from rl_games.common import a2c_common

class ASEAgent(amp_continuous.AMPAgent):
    def __init__(self, base_name, params):
        super().__init__(base_name, params)
        return

    def init_tensors(self):
        super().init_tensors()
        
        batch_shape = self.experience_buffer.obs_base_shape
        self.experience_buffer.tensor_dict['ase_latents'] = torch.zeros(batch_shape + (self._latent_dim,),
                                                                dtype=torch.float32, device=self.ppo_device)
        
        self._ase_latents = torch.zeros((batch_shape[-1], self._latent_dim), dtype=torch.float32,
                                         device=self.ppo_device)
        
        self.tensor_list += ['ase_latents']

        self._latent_reset_steps = torch.zeros(batch_shape[-1], dtype=torch.int32, device=self.ppo_device)
        num_envs = self.vec_env.env.num_envs
        env_ids = to_torch(np.arange(num_envs), dtype=torch.long, device=self.ppo_device)
        self._reset_latent_step_count(env_ids)

        return

    def _reset_latent_step_count(self, env_ids):
        self._latent_reset_steps[env_ids] = torch.randint_like(self._latent_reset_steps[env_ids], low=self._latent_steps_min, 
                                                         high=self._latent_steps_max)
        return

    def _load_config_params(self, config):
        super()._load_config_params(config)
        
        self._latent_dim = config['latent_dim']
        self._latent_steps_min = config.get('latent_steps_min', np.inf)
        self._latent_steps_max = config.get('latent_steps_max', np.inf)
        self._latent_dim = config['latent_dim']
        self._amp_diversity_bonus = config['amp_diversity_bonus']
        self._amp_diversity_tar = config['amp_diversity_tar']
        
        self._enc_coef = config['enc_coef']
        self._enc_weight_decay = config['enc_weight_decay']
        self._enc_reward_scale = config['enc_reward_scale']
        self._enc_grad_penalty = config['enc_grad_penalty']

        self._enc_reward_w = config['enc_reward_w']

        return

    def _build_net_config(self):
        config = super()._build_net_config()
        config['ase_latent_shape'] = (self._latent_dim,)
        return config

    def _reset_latents(self, env_ids):
        n = len(env_ids)
        z = self._sample_latents(n)
        self._ase_latents[env_ids] = z

        if (self.vec_env.env.viewer):
            self._change_char_color(env_ids)

        return

    def _sample_latents(self, n):
        z = self.model.a2c_network.sample_latents(n)
        return z

    def _update_latents(self):
        new_latent_envs = self._latent_reset_steps <= self.vec_env.env.progress_buf

        need_update = torch.any(new_latent_envs)
        if (need_update):
            new_latent_env_ids = new_latent_envs.nonzero(as_tuple=False).flatten()
            self._reset_latents(new_latent_env_ids)
            self._latent_reset_steps[new_latent_env_ids] += torch.randint_like(self._latent_reset_steps[new_latent_env_ids],
                                                                               low=self._latent_steps_min, 
                                                                               high=self._latent_steps_max)
            if (self.vec_env.env.viewer):
                self._change_char_color(new_latent_env_ids)

        return