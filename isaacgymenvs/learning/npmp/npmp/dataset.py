import h5py
import torch
import numpy as np
from tensordict import MemmapTensor, TensorDict
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

class NPMPDataset(Dataset):
    """ 
    Dataset of noisy expert trajectories
    """

    def __init__(self, filepath: str, device = None, num_future_states: int = 1):
        file = h5py.File(filepath)
        if device is None: 
            device = torch.device('cpu')

        self.num_future_states = num_future_states
        self.attrs = {k: v for k, v in file.attrs.items()}
        
        self.file = file
        self.dataset_size = file["root_states"].shape[0]
        self.root_states = file["root_states"]
        self.dof_states = file["dof_states"]
        self.actions = file["actions"]

    def _num_samples_per_trajectory(self):
        return self.file.attrs["max_episode_length"] - self.num_future_states - 1

    def _num_trajectories(self):
        return self.file["root_states"].shape[0]

    def __len__(self):
        return self._num_trajectories()

    def __getitem__(self, idx):
        trajectory_idx = idx 
        root_states = self.root_states[trajectory_idx, :, 3:]
        dof_states = self.dof_states[trajectory_idx, :, :]
        actions = self.actions[trajectory_idx, :]
        
        states = np.concatenate([root_states, dof_states], axis=-1)
        s = states[:,:self._num_samples_per_trajectory()] # [b, T, s]
        a = actions[:,:self._num_samples_per_trajectory()] # [b, T, a]
        x = np.expand_dims(s, 2) # [b, T, 1, s]
        x = np.tile(x, (1, 1, self.num_future_states, 1)) # [b, T, k, s]
        
        # breakpoint()
        for i in range(self.num_future_states):
            x[:, :, i] = states[:, i + 1 : i + self._num_samples_per_trajectory() + 1]
        
        return TensorDict({'state': s, 'action': a, 'future_state': x}, [])

if __name__ == "__main__":
    # Do some debugging
    import pathlib
    root_dir = pathlib.Path(__file__).absolute().parent.parent
    dataset_path = root_dir / 'data' / 'dataset_small.h5'
    print(dataset_path)
    dataset = NPMPDataset(dataset_path, num_future_states = 4)

    print(len(dataset))
    sample = dataset[:2]
    print(list(sample.keys()))
    for k, v in sample.items():
        print(f"{k}: {v.shape}")

    # Simple sanity check
    s1 = sample['state'][0,1]
    s1_x = sample['future_state'][0][0, 0]
    assert torch.all(s1 == s1_x)