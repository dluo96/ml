import argparse
import logging

import torch

from ml.diffusion.ddpm.dataset import show_tensor_image
from ml.diffusion.ddpm.diffuser import DiffuserDDPM
from ml.diffusion.ddpm.u_net import Unet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


if __name__ == "__main__":
    # fmt: off
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Inference (DDPM diffusion model)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use: cpu|gpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--T", type=int, default=100, help="Number of time steps")
    # fmt: on

    args = parser.parse_args()
    for arg_name, arg_value in vars(args).items():
        logging.info(f"{arg_name}: {arg_value}")

    """Inference (Algorithm 2 in the paper)"""
    T = 100
    diffuser = DiffuserDDPM(T=args.T)

    # Define the neural network which predicts noise
    denoising_model = Unet()

    # Algorithm 2 line 1: sample pure noise at t=T from N(0, I)
    B, C, H, W = 1, 3, 128, 128
    x_t = torch.randn(size=(B, C, H, W), device=args.device)

    # Algorithm 2 for-loop
    for timestep in range(T, 0):
        t = torch.Tensor([timestep]).type(torch.int64)

        # Algorithm 2 line 4
        x_t = diffuser.denoising_step(x_t, t, denoising_model)

    # Plot the generated/denoised image
    show_tensor_image(x_t)
