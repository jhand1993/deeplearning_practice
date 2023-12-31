from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST, KMNIST, QMNIST, EMNIST
from torch.utils.data import DataLoader
from torch.nn.functional import one_hot
from torch import tensor
from typing import Callable

from pathlib import Path


def load_MNISTlike(
        target_set: str = 'MNIST',
        batch_size_train: int = 64,
        batch_size_validation: int = 128,
        transform: Callable = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        target_transform: Callable = lambda x: one_hot(tensor(x), 10)
) -> tuple:
    """ Wrapper to load MNIST-compatible data sets from torchvision.

    Args:
        target_set (str): MNST-like target data set.  Defaults to 'MNIST'
        batch_size_train (int): Training batch size.  Defaults to 64.
        batch_size_validation (int): Validation batch size.  Defaults to
            128.
        transform (Callable): Transforms on input feature data.  Defaults to
            transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
            ]).
        target_transform (Callable): Transform for output label data.
            Defaults to lambda x: one_hot(tensor(x), 10).

    Returns:
        tuple: tuple of DataLoader instances -- one for training set,
            another for testing
    """
    # Select appropriate DataSet object.
    if target_set == 'MNIST':
        target_obj = MNIST

    elif target_set == 'FashionMNIST':
        target_obj = FashionMNIST

    elif target_set == 'QMNIST':
        target_obj = QMNIST

    elif target_set == 'KMNIST':
        target_obj = KMNIST

    elif target_set == 'EMNIST':
        target_obj = EMNIST

    else:
        raise ValueError(f'Invalid value {target_set} for target_set.')

    # Load the training and validation DataLoaders.
    train_loader = DataLoader(
        target_obj(
            str(Path(f'datasets/{target_set}')), train=True, download=True,
            transform=transform, target_transform=target_transform,
        ), batch_size=batch_size_train, shuffle=True
    )

    valid_loader = DataLoader(
        target_obj(
            str(Path(f'datasets/{target_set}')), train=False, download=True,
            transform=transform, target_transform=target_transform,
        ), batch_size=batch_size_validation, shuffle=True
    )

    return train_loader, valid_loader
