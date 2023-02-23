""" Export Pytorch Lightning checkpoint to vanilla PyTorch checkpoint """
import torch
import pathlib
from npmp.model import Encoder, Actor
from npmp.train import BehaviourCloning

if __name__ == "__main__":

    state_dim = 4 + 3 + 3 + 12 + 12 # root_rot, root_lin_vel, root_ang_vel, dof_pos, dof_vel
    action_dim = 12 # 12 dof pos
    latent_dim = 32
    num_future_states = 2
    hidden_dim = 32
    batch_size = 8

    encoder = Encoder(state_dim, latent_dim, num_future_states, hidden_dim = hidden_dim)
    actor = Actor(state_dim, action_dim, latent_dim, hidden_dim= hidden_dim)
    
    root_dir = pathlib.Path(__file__).absolute().parent.parent
    checkpoint_path = root_dir / 'QuadrupedASE' / '7uoolrxw' / 'checkpoints' / 'epoch=49-step=71700.ckpt' 
    model = BehaviourCloning(encoder, actor)
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])

    torch.save(model.encoder.state_dict(), 'encoder.pth')
    torch.save(model.actor.state_dict(), 'actor.pth')