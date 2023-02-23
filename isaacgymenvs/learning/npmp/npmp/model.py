import torch 
import torch.nn as nn 

import numpy as np

class Encoder(nn.Module):
    """ q(z_t | s_t, x_t), following approach in CoMiC """
    def __init__(self, state_dim: int, latent_dim: int, 
                num_future_states: int = 1,
                hidden_dim: int = 256):

        super(Encoder, self).__init__()
        self.state_dim = state_dim 
        self.latent_dim = latent_dim 
        self.num_future_states = num_future_states
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(state_dim * (num_future_states + 1), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, ref_states: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        ref_states: [batch, num_future_states, state_dim]
        state: [batch, state_dim]
        """
        return self.net(torch.cat([ref_states, state], dim=-1))

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
        return self.net(torch.cat([state, latent], dim=-1))

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state).cpu().data.numpy().flatten()