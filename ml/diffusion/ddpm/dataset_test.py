import unittest

import torch
from torch.utils.data.dataset import Dataset

from ml.diffusion.ddpm.dataset import IMG_SIZE, create_datasets


class TestDataset(unittest.TestCase):
    """This will run slowly if the torchvision datasets have not been downloaded
    locally yet.
    """

    def test_create_datasets(self):
        train_dataset, test_dataset = create_datasets()
        assert isinstance(train_dataset, Dataset)
        assert isinstance(test_dataset, Dataset)

        # Do a few checks on an example data point
        x, y = train_dataset[0]

        # Check image shape
        assert x.shape == (3, IMG_SIZE, IMG_SIZE)

        # Check label is integer
        assert isinstance(y, int)

        # Check image values are in [-1, 1]
        assert torch.all(x <= 1.0)
        assert torch.all(x >= -1.0)
