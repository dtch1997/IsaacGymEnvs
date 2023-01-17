import torch
from diffuse_rl import net

def test_temporal_unet():
    # Arbitrary values
    batch_size=16
    horizon=128
    transition_dim = 1024

    unet = net.TemporalUnet(
        horizon=horizon, 
        transition_dim=transition_dim,
        cond_dim=1000 # Unused atm
    )

    x = torch.zeros((batch_size, horizon, transition_dim)) 
    t = torch.zeros((batch_size))
    out = unet(x, None, t)

    assert out.shape == (batch_size, horizon, transition_dim)
    