import torch
import torch.nn as nn

import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
        
def sample_ase_latent(num_envs, latent_dim, device='cpu'):
    # Sample a standard Gaussian
    z = torch.normal(
        mean = torch.zeros(size=(num_envs, latent_dim), device=device),
        std = torch.ones(size=(num_envs, latent_dim), device=device)
    )
    # Project to unit hypersphere
    z_normalized = z / torch.sum(z, dim=-1, keepdim=True)
    # The resulting distribution is uniform over the unit hypersphere
    return z_normalized

def calc_enc_error(z_pred, z_true): 
    # We assume that z_true is distributed uniformly on the unit hypersphere
    # The posterior distribution is modeled as a Von Mises-Fisher distribution
    err = z_pred * z_true
    err = - torch.sum(err, dim=-1, keepdim=True)
    return err

def calc_enc_loss(z_pred, z_true):
    enc_err = calc_enc_error(z_pred, z_true)
    enc_loss = torch.mean(enc_err)
    return enc_loss

def calc_enc_rewards(enc, s_curr, s_next, z_true):
    with torch.no_grad():
        z_pred = enc(s_curr, s_next)
        err = calc_enc_error(z_pred, z_true)
        enc_r = torch.clamp_min(-err, 0.0)
        enc_r *= 5 # TODO: Refactor enc_reward_scale into argparse arg
    return enc_r

class Encoder(nn.Module):
    def __init__(self, envs, hidden_dim, latent_dim):
        super().__init__()
        self.enc_mlp = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod() * 2, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, latent_dim), std=1.0),
        )

    def cat_s_curr_next(self, s_curr, s_next):
        return torch.cat([s_curr, s_next], dim=-1)

    def get_enc_pred(self, s_curr, s_next):
        combined_s = self.cat_s_curr_next(s_curr, s_next)
        return self.enc_mlp(combined_s)

def combine_rewards(task_rewards, url_rewards, task_reward_w, url_reward_w):
    return task_rewards * task_reward_w + url_rewards * url_reward_w