"""Shared device selection for MARINN training and evaluation."""
import os
import torch


def get_device(requested=None):
    requested = requested or os.environ.get("MARINN_DEVICE", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Use auto, cpu, cuda, or cuda:<index> for MARINN_DEVICE.")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Install CUDA-enabled PyTorch or use --device cpu / MARINN_DEVICE=cpu.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {device.index} does not exist.")
    return device


def select_device(requested):
    device = get_device(requested)
    os.environ["MARINN_DEVICE"] = str(device)
    print(f"MARINN device: {device}", flush=True)
    return device
