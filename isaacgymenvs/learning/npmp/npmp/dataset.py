import h5py
import torch
from tensordict import MemmapTensor, TensorDict
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

class NPMPDataset(Dataset):
    """ 
    Dataset of noisy expert trajectories
    """

    def __init__(self, filepath: str, device = None):
        file = h5py.File(filepath)
        if device is None: 
            device = torch.device('cpu')

        self.attrs = {k: v for k, v in file.attrs.items()}
        
        dataset_size = file['root_states'].shape[0]

        self.tensors = TensorDict(
            {
                "root_states": MemmapTensor(
                    file["root_states"].shape, 
                    dtype=torch.float32
                ),
                "dof_states": MemmapTensor(
                    file["dof_states"].shape, 
                    dtype=torch.float32
                ),
                "actions": MemmapTensor(
                    file["actions"].shape, 
                    dtype=torch.float32
                ),
            },
            batch_size = dataset_size,
            device = device
        )


    def __len__(self):
        return self.tensors['root_states'].shape[0]

    def __getitem__(self, idx):
        return self.tensors[idx]

if __name__ == "__main__":
    # Do some debugging
    import pathlib
    root_dir = pathlib.Path(__file__).absolute().parent.parent
    dataset_path = root_dir / 'data' / 'dataset_small.h5'
    print(dataset_path)
    dataset = NPMPDataset(dataset_path)

    print(len(dataset))
    sample = dataset[torch.randint(low=0, high=len(dataset), size=(1,))]
    print(list(sample.keys()))
    for k, v in sample.items():
        print(f"{k}: {v.shape}")