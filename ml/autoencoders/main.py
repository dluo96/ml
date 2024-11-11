import argparse
import logging
import pathlib

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader

from ml.autoencoders.autoencoder import Autoencoder
from ml.autoencoders.conv_autoencoder import ConvAutoencoder
from ml.autoencoders.conv_vae import ConvVAE
from ml.autoencoders.vae import VAE, loss_fn_vae
from ml.tensor import Tensor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def visualize_single_mnist_image(image_tensor: Tensor):
    """Visualise a single MNIST image.

    Args:
        image_tensor (torch.Tensor): Single MNIST image of shape (1, 28, 28) or (28, 28).
    """
    image = image_tensor.squeeze()  # Remove the channel dimension if present
    plt.imshow(image, cmap="gray")
    plt.axis("off")  # Hide axis for a cleaner look
    plt.show()


if __name__ == "__main__":
    # fmt: off
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train an autoencoder")

    # System and I/O
    parser.add_argument("--device", type=str, default="cpu", help="Device to use: cpu|gpu)")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of dataloader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # Model
    parser.add_argument("--type", type=str, default="standard", help="Model to use: autoencoder|vae|conv-autoencoder|conv-vae")
    parser.add_argument("--d-x", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--d-hidden", type=int, default=64, help="Second embedding dimension")
    parser.add_argument("--d-z", type=int, default=16, help="Number of consecutive transformer blocks")

    # Optimization
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--num-epochs", type=int, default=10_000, help="Number of complete passes through the dataset")
    # fmt: on

    args = parser.parse_args()
    for arg_name, arg_value in vars(args).items():
        logging.info(f"{arg_name}: {arg_value}")

    # Set seed for reproducibility
    torch.manual_seed(args.seed)

    # Create MNIST dataset and dataloader
    data_dir = pathlib.Path(__file__).parent
    mnist_dataset = torchvision.datasets.MNIST(
        data_dir, download=True, transform=torchvision.transforms.ToTensor()
    )
    dataloader = torch.utils.data.DataLoader(
        mnist_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    # Extract channel, height, and width
    C, H, W = mnist_dataset[0][0].shape

    # Model and optimizer
    d_x = C * H * W  # Needed for autoencoder and VAE that use fully connected layers
    if args.type == "autoencoder":
        model = Autoencoder(d_x=d_x, d_hidden=args.d_hidden, d_z=args.d_z)
    elif args.type == "vae":
        model = VAE(d_x=d_x, d_hidden=args.d_hidden, d_z=args.d_z)
    elif args.type == "conv-autoencoder":
        model = ConvAutoencoder()
    elif args.type == "conv-vae":
        model = ConvVAE()
    else:
        raise ValueError(f"Invalid model type: {args.type}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.99),
        eps=1e-8,
    )

    # Training
    for epoch in range(args.num_epochs):
        epoch_loss = 0
        for batch in dataloader:
            # Autoencoder doesn't need labels as it is unsupervised
            X, _ = batch

            if args.type in ["standard", "variational"]:
                # For the autoencoder and VAE that use fully connected layers for
                # their encoder and decoder, we need to flatten each image into a
                # vector: (B, C, H, W) -> (B, C*H*W)
                X = X.view(X.size(0), d_x)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            if args.type in ["autoencoder", "conv-autoencoder"]:
                output = model(X)
                loss = F.mse_loss(output, X, reduction="sum")  # Reconstruction error
            elif args.type in ["vae", "conv-vae"]:
                output, z, mu, log_var = model(X)
                loss = loss_fn_vae(output, X, mu, log_var)
            else:
                raise ValueError(f"Invalid model type: {args.type}")

            # Backward pass and parameter updates
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        logging.info(f"Epoch {epoch} | Loss: {epoch_loss / len(dataloader)}")
