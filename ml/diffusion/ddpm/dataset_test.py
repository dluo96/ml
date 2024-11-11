import unittest

from torch.utils.data.dataset import ConcatDataset, Dataset

from ml.diffusion.ddpm.dataset import create_datasets


class TestDataset(unittest.TestCase):
    def test_create_datasets(self):
        train_dataset, test_dataset = create_datasets()
        assert isinstance(train_dataset, Dataset)
        assert isinstance(test_dataset, Dataset)
