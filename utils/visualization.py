from PIL import Image
import numpy as np
import torch


def as_image_array(tensor: torch.Tensor) -> torch.Tensor:
    """Converts a torch.Tensor to a numpy.ndarray.

    Args:
        tensor (torch.Tensor): input tensor of shape (H, W, C).

    Returns:
        numpy.ndarray: output image array.
    """
    return (tensor.cpu().clip(0, 1) * 255).numpy().astype(np.uint8)


def as_image(tensor: torch.Tensor) -> Image:
    """Converts a torch.Tensor to a PIL.Image.

    Args:
        tensor (torch.Tensor): input tensor of shape (H, W, C).

    Returns:
        Image: output image.
    """
    return Image.fromarray(as_image_array(tensor))


def save_image(tensor: torch.Tensor, path: str) -> None:
    """Saves a torch.Tensor as an image.

    Args:
        tensor (torch.Tensor): input tensor of shape (H, W, C).
        path (str): path to save the image.
    """
    as_image(tensor).save(path)
