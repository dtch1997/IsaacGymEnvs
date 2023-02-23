import torch.optim as optim
import pytorch_lightning as pl

from npmp.model import Encoder, Actor 
from npmp.dataset import Dataset

class BehaviourCloning(pl.LightningModule):
    
    def __init__(self, encoder, actor):
        super(BehaviourCloning, self).__init__()
        self.encoder = encoder 
        self.actor = actor

    def training_step(self, batch, batch_idx):
        x = batch['future_states']
        s = batch['states']
        a = batch['actions']
        z = batch['prev_latent']

        z = self.encoder()
        breakpoint()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

if __name__ == "__main__":

    state_dim = 4 + 3 + 3 + 12 + 12 # root_rot, root_lin_vel, root_ang_vel, dof_pos, dof_vel
    action_dim = 12 # 12 dof pos
    latent_dim = 32
    num_future_states = 2
    hidden_dim = 32

    encoder = Encoder(state_dim, latent_dim, num_future_states, hidden_dim = hidden_dim)
    actor = Actor(state_dim, action_dim, latent_dim, hidden_dim= hidden_dim)
    bc_module = BehaviourCloning(encoder, actor)
