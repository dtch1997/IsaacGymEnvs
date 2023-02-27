import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from torch.utils.data import DataLoader, random_split
from npmp.model import Encoder, Actor 
from npmp.dataset import NPMPDataset

class BCDataModule(pl.LightningDataModule):
    def __init__(self, 
        filepath: str, 
        num_future_states: int = 1, 
        batch_size: int = 32,
        train_frac: float = 0.7,
        val_frac: float = 0.2,
    ):
        super(BCDataModule, self).__init__()
        self.dataset = NPMPDataset(filepath, num_future_states = num_future_states)
        self.batch_size = batch_size
        self.train_frac = train_frac
        self.val_frac = val_frac

    def setup(self, stage: str):
        num_train_episodes = int(self.train_frac * len(self.dataset))
        num_val_episodes = int(self.val_frac * len(self.dataset))
        num_test_episodes = len(self.dataset) - num_train_episodes - num_val_episodes

        self.dataset_train, self.dataset_val, self.dataset_test = random_split(
            self.dataset, lengths=(num_train_episodes, num_val_episodes, num_test_episodes))

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train, 
            batch_size = self.batch_size, 
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size = self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size = self.batch_size)

class BehaviourCloning(pl.LightningModule):
    
    def __init__(self, encoder, actor):
        super(BehaviourCloning, self).__init__()
        self.encoder = encoder 
        self.actor = actor

    def _calc_batch_dict(self, batch, batch_idx):
        s = batch['state'] # [b, T, s_dim]
        a_true = batch['action'] # [b, T, a_dim]
        x = batch['future_state'] # [b, T, k, s_dim]
        x = x.flatten(start_dim = -2, end_dim=-1)

        z = self.encoder(x, s)
        a_pred = self.actor(s, z)

        behaviour_cloning_loss = nn.functional.mse_loss(a_pred, a_true, reduction="mean")
        # Regularize z by making it more similar to unit Gaussian
        regularization_loss = self.kl_loss(z)
        # TODO: Make regularization loss coefficient configurable
        loss = behaviour_cloning_loss + 1e-2 * regularization_loss
        
        return {
            "loss": loss, 
            "bc_loss": behaviour_cloning_loss, 
            "reg_loss": regularization_loss,
            "state": s,
            "action": a_true, 
            "action_pred": a_pred,
            "latent": z
        }

    def _calc_batch_stats(self, batch_dict):
        """ Calculate various statistics of interest """
        for key in ["state", "action", "action_pred", "latent"]:
            value = batch_dict.pop(key)
            mean = torch.mean(value, dim=-1)
            std = torch.std(value, dim=-1)
            min = torch.min(value, dim=-1)
            max = torch.max(value, dim=-1)
            batch_dict[f"{key}_mean"] = mean.mean()
            batch_dict[f"{key}_std"] = std.mean()
            batch_dict[f"{key}_min"] = min[0].mean()
            batch_dict[f"{key}_max"] = max[0].mean()
        return batch_dict

    def _log_batch_stats(self, batch_stats, prefix="info"):
        for k, v in batch_stats.items():
            self.log(f"{prefix}/{k}", v)

    def training_step(self, batch, batch_idx):
        batch_dict = self._calc_batch_dict(batch, batch_idx)
        batch_stats = self._calc_batch_stats(batch_dict)
        self._log_batch_stats(batch_stats, "train")
        return batch_dict["loss"]

    def validation_step(self, batch, batch_idx):
        batch_dict = self._calc_batch_dict(batch, batch_idx)
        batch_stats = self._calc_batch_stats(batch_dict)
        self._log_batch_stats(batch_stats, "val")
        return batch_dict["loss"]

    def test_step(self, batch, batch_idx):
        batch_dict = self._calc_batch_dict(batch, batch_idx)
        batch_stats = self._calc_batch_stats(batch_dict)
        self._log_batch_stats(batch_stats, "test")
        return batch_dict["loss"]

    def on_train_end(self):
        torch.save(self.encoder.state_dict(), 'encoder.pth')
        torch.save(self.actor.state_dict(), 'actor.pth')

    def kl_loss(self, latent, variance=None):
        """
        Assume the HLC action distribution is a Gaussian N(latent, Sigma)
        Calculate KL divergence with a unit prior

        Reference: https://stats.stackexchange.com/questions/318184/kl-loss-with-a-unit-gaussian
        """
        if variance is None:
            variance = torch.ones_like(latent)
        loss = torch.log(variance) + variance + latent**2 - 1
        return loss.mean()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

if __name__ == "__main__":

    state_dim = 45
    action_dim = 12 # 12 dof pos
    latent_dim = 32
    num_future_states = 2
    hidden_dim = 32
    batch_size = 8

    # Define the architecture
    encoder = Encoder(state_dim, latent_dim, num_future_states, hidden_dim = hidden_dim)
    actor = Actor(state_dim, action_dim, latent_dim, hidden_dim= hidden_dim)
    bc_module = BehaviourCloning(encoder, actor)

    # Define a dataset
    import pathlib
    root_dir = pathlib.Path(__file__).absolute().parent.parent
    dataset_path = root_dir / 'data' / 'dataset_small.h5'
    datamodule = BCDataModule(dataset_path, num_future_states, batch_size=batch_size)

    from pytorch_lightning.loggers import WandbLogger
    wandb_logger = WandbLogger(
        project='QuadrupedASE',
        group='CoMiC',
        name='comic_bc',
    )
    trainer = pl.Trainer(max_epochs=1, logger=wandb_logger)
    trainer.fit(model=bc_module, datamodule = datamodule)
    trainer.test(model=bc_module, datamodule = datamodule)
    # Save checkpoint to WandB 
    import wandb 
    wandb.save('encoder.pth')
    wandb.save('actor.pth')
    # Delete old checkpoint
    import pathlib 
    pathlib.Path('encoder.pth').unlink()
    pathlib.Path('actor.pth').unlink()
