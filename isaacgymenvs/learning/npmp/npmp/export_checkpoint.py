""" Export Pytorch Lightning checkpoint to vanilla PyTorch checkpoint """
import torch
import pathlib
import argparse
from npmp.model import Encoder, Actor
from npmp.train import BehaviourCloning

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-filepath", type=str)
    parser.add_argument("-o", "--output-filepath", type=str, default=".")
    args = parser.parse_args()

    state_dim = 45
    action_dim = 12 # 12 dof pos
    latent_dim = 32
    num_future_states = 2
    hidden_dim = 32
    batch_size = 8

    encoder = Encoder(state_dim, latent_dim, num_future_states, hidden_dim = hidden_dim)
    actor = Actor(state_dim, action_dim, latent_dim, hidden_dim= hidden_dim)
    
    root_dir = pathlib.Path(__file__).absolute().parent.parent
    model = BehaviourCloning(encoder, actor)
    
    checkpoint = torch.load(args.input_filepath)
    model.load_state_dict(checkpoint['state_dict'])

    output_dir = pathlib.Path(args.output_filepath)
    output_dir.mkdir(exist_ok=True, parents=True)
    torch.save(model.encoder.state_dict(), output_dir / 'encoder.pth')
    torch.save(model.actor.state_dict(), output_dir / 'actor.pth')