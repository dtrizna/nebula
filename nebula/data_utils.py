import numpy as np
from scipy.sparse import csr_matrix
from typing import Union

import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader


class CSRTensorDataset(Dataset):
    def __init__(self, csr_data, labels=None):
        if labels is not None:
            assert csr_data.shape[0] == len(labels)
        self.csr_data = csr_data
        self.labels = labels

    def get_tensor(self, row):
        assert isinstance(row, (np.ndarray, torch.Tensor)),\
            "Expected row to be a numpy array or torch tensor, but got {}".format(type(row))
        if isinstance(row, np.ndarray):
            data = torch.from_numpy(row).float()
        elif isinstance(row, torch.Tensor):
            data = row.clone().detach().float()
        return data

    def __len__(self):
        return self.csr_data.shape[0]

    def __getitem__(self, index):
        # Convert the sparse row to a dense numpy array
        row = self.csr_data[index].toarray().squeeze()
        x_data = self.get_tensor(row)
        if self.labels is not None:
            y_data = self.get_tensor(self.labels[index])
            return x_data, y_data
        else:
            return x_data,


def create_dataloader(
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor] = None,
        batch_size: int = 1024,
        shuffle: bool = False,
        workers: int = 4
) -> DataLoader:
    # Convert numpy arrays to torch tensors
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).long()
    if y is not None and isinstance(y, np.ndarray):
        y = torch.from_numpy(y).float()
    
    if y is not None:
        # y = y.reshape(-1, 1) if len(y.shape) == 1 else y
        assert X.shape[0] == y.shape[0], "X and y must have the same number of rows"

    # Handle csr_matrix case
    if isinstance(X, csr_matrix):
        dataset = CSRTensorDataset(X, y)
    # Handle torch.Tensor case
    elif isinstance(X, torch.Tensor):
        dataset = TensorDataset(X, y) if y is not None else TensorDataset(X)
    else:
        raise ValueError("Unsupported type for X. Supported types are numpy arrays, torch tensors, and scipy CSR matrices.")
    
    persistent_workers = True if workers > 0 else False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        # NOTE: these are important for lightning to iterate quickly
        # see: https://lightning.ai/docs/pytorch/stable/advanced/speed.html
        persistent_workers=persistent_workers,
        pin_memory=True
    )
