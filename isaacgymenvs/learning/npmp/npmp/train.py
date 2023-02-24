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

    def training_step(self, batch, batch_idx):
        s = batch['state'] # [b, T, s_dim]
        a_true = batch['action'] # [b, T, a_dim]
        x = batch['future_state'] # [b, T, k, s_dim]
        x = x.flatten(start_dim = -2, end_dim=-1)

        z = self.encoder(x, s)
        a_pred = self.actor(s, z)

        loss = nn.functional.mse_loss(a_pred, a_true)
        self.log("info/train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        s = batch['state'] # [b, T, s_dim]
        a_true = batch['action'] # [b, T, a_dim]
        x = batch['future_state'] # [b, T, k, s_dim]
        x = x.flatten(start_dim = -2, end_dim=-1)

        z = self.encoder(x, s)
        a_pred = self.actor(s, z)

        loss = nn.functional.mse_loss(a_pred, a_true)
        self.log("info/val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        # this is the validation loop
        s = batch['state'] # [b, T, s_dim]
        a_true = batch['action'] # [b, T, a_dim]
        x = batch['future_state'] # [b, T, k, s_dim]
        x = x.flatten(start_dim = -2, end_dim=-1)

        z = self.encoder(x, s)
        a_pred = self.actor(s, z)

        loss = nn.functional.mse_loss(a_pred, a_true)
        self.log("info/test_loss", loss)
        return loss

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
    trainer = pl.Trainer(max_epochs=20, logger=wandb_logger)
    trainer.fit(model=bc_module, datamodule = datamodule)
    trainer.test(model=bc_module, datamodule = datamodule)
