import torch 
import torch.nn as nn 

import numpy as np

class Encoder(nn.Module):
    """ q(z_t | z_{t-1}, x_t) """
    
    def forward(self, ref_states: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        return self.net(torch.hstack([ref_states.flatten(), latent]))

class Actor(nn.Module):
    """ pi(a_t | s_t, z_t) """
    def __init__(self, state_dim: int, action_dim: int, latent_dim: int, hidden_dim: int = 256):
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
        # Net returns outputs in [-1, 1]

    def forward(self, state: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        return self.net(torch.hstack([state, latent]))

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state).cpu().data.numpy().flatten()