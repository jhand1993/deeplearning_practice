from torch.utils.data import Dataset, DataLoader
import torch


def random_image_set(
    shape: tuple[int], seed: int = 0
) -> torch.Tensor:
    """ Generates a white noise data set of the requested shape.

    Args:
        shape (tuple[int]): Shape of the data set
        seed (int, optional): Random seed. Defaults to 0.

    Returns:
        torch.Tensor: White noise data set.
    """
    torch.manual_seed(seed)
    return torch.randn(shape)


class TestDataSet(Dataset):
    """ Test datat set consisting of white noise.
    """
    def __init__(
            self,
            set_size: int,
            feature_shape: tuple[int],
            label_shape: tuple[int],
            seed: int = 0
    ) -> None:
        """
        Args:
            set_size (int): Number of data set elements.
            feature_shape (tuple[int]): Shape of data set features.
            label_shape (tuple[int]): Shape of data set labels.
            seed (int, optional): Random seed. Defaults to 0.
        """
        super().__init__()
        self.set_size = set_size
        self.seed = seed

        # All data is Gaussian white noise from the seed provided.
        self.feature_tensor = random_image_set(
            (set_size, *feature_shape), seed=seed
        )
        self.label_tensor = random_image_set(
            (set_size, *label_shape), seed=seed
        )

    def __len__(self) -> int:
        """ Returns length of data set, which is equal to self.set_size.

        Returns:
            int: self.set_size
        """
        return self.set_size

    def __getitem__(self, idx: int) -> tuple:
        """ Returns feature, label pair for requested index.

        Args:
            idx (int): Index to get.

        Returns:
            tuple: feature, label tuple pair for index idx.
        """
        return (self.feature_tensor[idx], self.label_tensor[idx])


def load_TestDataSets(
    test_size: int,
    validation_size: int,
    feature_shape: tuple[int],
    label_shape: tuple[int],
    batch_size_test: int,
    batch_size_validation: int,
    seed: int = 0
) -> tuple:
    """ Loads a test and validation DataLoader from TestDataSet.
        All DataLoaders shuffle input DataSets.

    Args:
        test_size (int): Size of test DataSet.
        validation_size (int): Size of validation DataSet.
        feature_shape (tuple[int]): Shape of data set features.
        label_shape (tuple[int]): Shape of data set labels.
        batch_size_test (int): Test DataLoader batch size.
        batch_size_validation (int): Validation DataLoader batch
            size.
        seed (int, optional): Random seed. Defaults to 0.

    Returns:
        tuple: Test and validation DataLoaders.
    """
    test_ds = TestDataSet(
        test_size, feature_shape, label_shape, seed=seed
    )
    validation_ds = TestDataSet(
        validation_size, feature_shape, label_shape, seed=seed
    )

    test_dl = DataLoader(
        test_ds, batch_size=batch_size_test,
        shuffle=True
    )
    validation_dl = DataLoader(
        validation_ds, batch_size=batch_size_validation,
        shuffle=True
    )

    return test_dl, validation_dl
