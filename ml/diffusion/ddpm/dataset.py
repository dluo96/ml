from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torchvision as tv
from PIL import Image
from torch.utils.data import Dataset

from ml.tensor import Tensor

IMG_SIZE = 64


def create_datasets() -> (
    tuple[Dataset[tuple[Tensor, int]], Dataset[tuple[Tensor, int]]]
):
    """Loads a torchvision dataset, performs transformations on it, and creates
    training and test sets.

    Returns:
        A concatenation of two `Dataset`s, specifically the train and test sets.
        Each `Dataset` contains elements that consist of:
        - Image,
        - Label.
    """
    # Define transformations
    data_transform = tv.transforms.Compose(
        [
            tv.transforms.Resize((IMG_SIZE, IMG_SIZE)),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),  # Scale data to be in [0, 1]
            tv.transforms.Lambda(lambda t: (t * 2) - 1),  # Scale data to be in [-1, 1]
        ]
    )

    # Load datasets, applying transformations
    data_dir = Path(__file__).parent
    train_dataset = tv.datasets.OxfordIIITPet(
        root=data_dir, download=True, transform=data_transform
    )
    test_dataset = tv.datasets.OxfordIIITPet(
        root=data_dir, download=True, transform=data_transform, split="test"
    )

    return train_dataset, test_dataset


def show_tensor_image(image: Image) -> None:
    """Visualisation function that is useful for debugging."""
    reverse_transforms = tv.transforms.Compose(
        [
            tv.transforms.Lambda(lambda t: (t + 1) / 2),
            tv.transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # (C, H, W)->(H, W, C)
            tv.transforms.Lambda(lambda t: t * 255.0),
            tv.transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
            tv.transforms.ToPILImage(),
        ]
    )

    # Pick the first image in the batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]

    plt.imshow(reverse_transforms(image))
    plt.show()
