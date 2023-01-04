import torch

        
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