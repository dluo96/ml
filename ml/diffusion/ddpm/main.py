import argparse
import logging

import torch
from torch.utils.data import DataLoader

from ml.diffusion.ddpm.dataset import create_datasets
from ml.diffusion.ddpm.diffuser import DiffuserDDPM
from ml.diffusion.ddpm.trainer import Trainer
from ml.diffusion.ddpm.u_net import Unet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


if __name__ == "__main__":
    # fmt: off
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train DDPM diffusion model")

    # System and I/O
    parser.add_argument("--device", type=str, default="cpu", help="Device to use: cpu|gpu)")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of dataloader workers")
    parser.add_argument("--shuffle", type=bool, default=True, help="Enable shuffling in dataloader")
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
    train_dataset, val_dataset = create_datasets()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
    )

    # Diffuser and denoising model
    diffuser = DiffuserDDPM(args.T, device)
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

    # Consolidate everything in the trainer
    trainer = Trainer(
        num_epochs=args.num_epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        T=args.T,
        diffuser=diffuser,
        denoising_model=denoising_model,
        optimizer=optimizer,
        device=device,
    )

    # Launch training
    trainer.train()
