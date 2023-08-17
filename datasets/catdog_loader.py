from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch import tensor
from skimage import io
from pathlib import Path


class CatDogDataSet(Dataset):
    """ PyTorch DataSet class for loading cat and dog Kaggle training set.
        If there are 1000 total images indexed 0, 1, 2, ..., 999, then the
        first 500are in the cat subpath, and the remainder in the dog
        subpath.
    """
    def __init__(
            self, img_dir, transform=None, target_transform=None
    ) -> None:
        """
        Args:
            img_dir (str): Data directoy.
            transform (callable, optional): Data transformation function.
                Defaults to None.
            target_transform (callable, optional): Target Transformation
                function. Defaults to None.
        """
        super().__init__()
        if isinstance(img_dir, Path):
            self.img_path = img_dir
        else:
            self.img_path = Path(img_dir)

        self.cat_path = self.img_path / 'Cat'
        self.dog_path = self.img_path / 'Dog'

        paths = list(self.cat_path.glob('*.jpg')) +\
            list(self.dog_path.glob('*.jpg'))

        self.idx_to_name = dict(
            zip(
                range(len(paths)), paths,
            )
        )
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        """ Returns length of the data set as determined by the file counts in
            in self.catpath and self.dogpath
        """
        return len(list(self.cat_path.glob('*.jpg'))) +\
            len(list(self.dog_path.glob('*.jpg')))

    def __getitem__(self, idx: int) -> tuple:
        """ Return elements 'idx' of the data set

        Args:
            idx (List[int]): Indices to return
        """
        idx_path = self.idx_to_name[idx]

        # This might seem odd to use, but torchvision.io.read_image() is
        # not working for this set of jpg images. As such, we load it
        # using scikit-image and then transform the numpy array into a
        # PyTorch Tensor.
        image = io.imread(str(idx_path))
        image = ToTensor()(image)
        label = self._get_label(idx_path)

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def _get_label(self, idx_path) -> float:
        """ Gets target label from path information.  Specifically, return 1
            if the path's directory is self.cat_path, else return 0.

        Args:
            idx_path (Path): Path to check.
        """
        if idx_path.parts[-2] == 'Cat':
            return tensor(1.)

        else:
            return tensor(0.)
