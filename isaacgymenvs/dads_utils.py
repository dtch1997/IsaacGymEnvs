import torch
import torch.nn as nn

import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
        
def sample_latent(num_envs, latent_dim, device='cpu'):
    # Sample a standard Gaussian
    z = torch.normal(
        mean = torch.zeros(size=(num_envs, latent_dim), device=device),
        std = torch.ones(size=(num_envs, latent_dim), device=device)
    )
    # Project to unit hypersphere
    z_normalized = z / torch.linalg.norm(z, dim=-1, keepdim=True)
    # The resulting distribution is uniform over the unit hypersphere
    return z_normalized

def build_enc_obs(prev_enc_state, curr_enc_state):
    return torch.cat([prev_enc_state, curr_enc_state], dim=-1)

def calc_enc_error(z_pred, z_true): 
    # We assume that z_true is distributed uniformly on the unit hypersphere
    # The posterior distribution is modeled as a Von Mises-Fisher distribution
    err = z_pred * z_true
    err = - torch.sum(err, dim=-1)
    return err

def calc_enc_loss(z_pred, z_true):
    enc_err = calc_enc_error(z_pred, z_true)
    enc_loss = torch.mean(enc_err)
    return enc_loss

def calc_enc_rewards(enc, enc_obs, z_true, alpha=5):
    with torch.no_grad():
        z_pred = enc.get_enc_pred(enc_obs)
        err = calc_enc_error(z_pred, z_true)
        enc_r = torch.clamp_min(-err, 0.0)
        enc_r *= alpha
    return enc_r

class Encoder(nn.Module):
    def __init__(self, enc_obs_dim, hidden_dim, latent_dim):
        super().__init__()
        self.enc_mlp = nn.Sequential(
            layer_init(nn.Linear(enc_obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, latent_dim), std=1.0),
        )

    def get_enc_pred(self, enc_obs):
        return self.enc_mlp(enc_obs)
