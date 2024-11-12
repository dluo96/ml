"""Module for training. This implements Algorithm 1 in the DDPM paper."""
import argparse
import logging

import torch
from torch.utils.data import DataLoader

from ml.diffusion.ddpm.dataset import create_datasets
from ml.diffusion.ddpm.diffuser import DDPM
from ml.diffusion.ddpm.u_net import Unet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


if __name__ == "__main__":
    # fmt: off
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train DDPM")

    # System and I/O
    parser.add_argument("--device", type=str, default="cpu", help="Device to use: cpu|gpu)")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of dataloader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # Model
    parser.add_argument("--T", type=int, default=100, help="Number of time steps")

    # Optimization
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--num-epochs", type=int, default=10_000, help="Number of complete passes through the dataset")
    # fmt: on

    args = parser.parse_args()
    for arg_name, arg_value in vars(args).items():
        logging.info(f"{arg_name}: {arg_value}")

    # Set seed for reproducibility
    torch.manual_seed(args.seed)

    # Determine device
    if args.device == "gpu" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError(f"Device {args.device} is not recognized!")
    logging.info(f"Using device: {device}")

    """Implement Algorithm 1 in the DDPM paper."""
    # Data
    train_dataset, test_dataset = create_datasets()
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Diffuser and denoising model
    diffuser = DDPM(args.T, device)
    denoising_model = Unet()

    # Move model to device
    denoising_model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        params=denoising_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.99),
        eps=1e-8,
    )

    for epoch in range(args.num_epochs):
        for step, batch in enumerate(train_dataloader):
            # Set the gradients to zero before doing the backpropagation step
            # This is necessary because, by default, PyTorch accumulates the
            # gradients on subsequent backward passes i.e. subsequent calls
            # of loss.backward()
            optimizer.zero_grad()

            # Algorithm 1 line 3: sample `t` from a discrete uniform distribution
            t = torch.randint(
                low=0, high=args.T, size=(args.batch_size,), device=device
            ).long()

            # Extract the image (index 0 is the image and index 1 the label)
            x_0 = batch[0]

            # From the original image x_0, sample a noised image at timestep t
            x_noisy, noise = diffuser.noising_step(x_0, t)

            # Denoising process
            #   1. Predict the noise (forward pass through denoising model)
            #   2. Compute the L1 loss between the actual noise and the predicted noise
            #   3. Backward pass (through the denoising model)
            #   4. Update parameters of the denoising model
            pred_noise = denoising_model(x_noisy, t)
            loss = torch.nn.functional.l1_loss(noise, pred_noise)
            loss.backward()
            optimizer.step()

            if epoch % 5 == 0 and step == 0:
                logging.info(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
                # sample_plot_image()
