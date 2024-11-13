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
    parser.add_argument("--path-checkpoint", type=str, help="Path to checkpoint file (.pt)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use: cpu|gpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--T", type=int, default=100, help="Number of time steps")
    # fmt: on

    args = parser.parse_args()
    for arg_name, arg_value in vars(args).items():
        logging.info(f"{arg_name}: {arg_value}")

    # Determine device
    if args.device == "gpu" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError(f"Device {args.device} is not recognized!")
    logging.info(f"Using device: {device}")

    """Inference (Algorithm 2 in the paper)"""
    # Diffuser and denoising model
    diffuser = DiffuserDDPM(T=args.T, device=device)
    denoising_model = Unet()

    # Load checkpoint (for inference, we only need the model parameters)
    checkpoint = torch.load(args.path_checkpoint)
    denoising_model.load_state_dict(checkpoint["model_state_dict"])

    # Batch size and dimensions of image we want to generate
    B = 1  # We only want 1 image
    C, H, W = 3, 128, 128

    # Algorithm 2 line 1: sample pure noise at t=T from N(0, I)
    x_t = torch.randn(size=(B, C, H, W), device=device)

    # Algorithm 2 for-loop
    for timestep in range(args.T - 1, 1, -1):
        t = torch.tensor([timestep])
        x_t = diffuser.denoising_step(x_t, t, denoising_model)  # Algorithm 2 line 4

        # Plot the generated/denoised image
        show_tensor_image(x_t)
