import torch.utils.data as data
import cv2
import numpy as np
import torch
from utils import ray


class SingleImageDataset(data.Dataset):
    """Load a single image as a NeRF dataset.

    Args:
        image (str): path to image file.
    """

    def __init__(self, image):
        super().__init__()
        self.image = image
        self.image = cv2.imread(image)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB) / 255.0

        # Create N x 2 array of pixel coordinates
        self.points = np.indices(self.image.shape[:2]).transpose(1, 2, 0)
        self.points = self.points.reshape(-1, 2).astype(np.float32)
        # Normalize pixel coordinates
        self.points[:, 0] /= self.image.shape[0]
        self.points[:, 1] /= self.image.shape[1]
        # Convert to tensor
        self.points = torch.tensor(self.points, dtype=torch.float32)
        # Convert image to tensor
        self.image = torch.tensor(self.image, dtype=torch.float32)
        # Flatten image
        self.image = self.image.reshape(-1, 3)

    def __len__(self):
        return self.image.shape[0]

    def __getitem__(self, index):
        return self.points[index], self.image[index]


def getSingleImageDataLoader(image, batch_size=1024):
    """Get a data loader for a single image.

    Args:
        image (str): path to image file.
        batch_size (int, optional): batch size. Defaults to 1024.

    Returns:
        torch.utils.data.DataLoader: data loader.
    """
    dataset = SingleImageDataset(image)
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


class SingleImageTestingDataset(data.Dataset):
    """Generate a N by N grid of points for testing."""

    def __init__(self, resolution_x, resolution_y):
        self.points = np.indices((resolution_x, resolution_y)).transpose(1, 2, 0)
        self.points = self.points.reshape(-1, 2).astype(np.float32)
        # Normalize pixel coordinates
        self.points[:, 0] /= resolution_x
        self.points[:, 1] /= resolution_y
        # Convert to tensor
        self.points = torch.tensor(self.points, dtype=torch.float32)

    def __len__(self):
        return self.points.shape[0]

    def __getitem__(self, index):
        return self.points[index]


def getSingleImageTestingDataLoader(resolution_x, resolution_y, batch_size=1024):
    """Get a data loader for a single image.

    Args:
        image (str): path to image file.
        batch_size (int, optional): batch size. Defaults to 1024.

    Returns:
        torch.utils.data.DataLoader: data loader.
    """
    dataset = SingleImageTestingDataset(resolution_x, resolution_y)
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


class RaysDataset(data.Dataset):
    """Dataset for rays.

    Args:
        nerf_data_path (str): path to nerf data.
    """

    def __init__(self, images: np.ndarray, extrinsics: np.ndarray, focal: float):
        """Initialize a dataset for rays.

        Args:
            images (np.ndarray): dataset of images of shape (N, H, W, 3).
            extrinsics (np.ndarray): dataset of extrinsics of shape (N, 4, 4).
            focal (float): focal length.
        """
        super().__init__()
        self.images = torch.tensor(images, dtype=torch.float32)
        self.extrinsics = torch.tensor(extrinsics, dtype=torch.float32)
        self.focal = focal

        self.image_height = images.shape[1]
        self.image_width = images.shape[2]
        self.image_num = images.shape[0]

        self.pixel_rgb = self.images.reshape(-1, 3)
        self.ray_directions, self.ray_origins = ray.get_rays(
            self.image_num,
            self.image_height,
            self.image_width,
            self.focal,
            self.extrinsics,
        )

    def __len__(self):
        return self.pixel_rgb.shape[0]

    def __getitem__(self, index):
        return (
            self.pixel_rgb[index],
            self.ray_directions[index],
            self.ray_origins[index],
        )


class RaysTestingDataset(data.Dataset):
    """Testing dataset for rays.

    Args:
        nerf_data_path (str): path to nerf data.
    """

    def __init__(self, height: int, width: int, extrinsics: np.ndarray, focal: float):
        """Initialize a dataset for rays.

        Args:
            extrinsics (np.ndarray): dataset of extrinsics of shape (N, 4, 4).
            focal (float): focal length.
        """
        super().__init__()
        self.extrinsics = torch.tensor(extrinsics, dtype=torch.float32)
        self.focal = focal
        self.image_height = height
        self.image_width = width

        self.image_num = extrinsics.shape[0]

        self.ray_directions, self.ray_origins = ray.get_rays(
            self.image_num,
            self.image_height,
            self.image_width,
            self.focal,
            self.extrinsics,
        )

    def __len__(self):
        return self.ray_directions.shape[0]

    def __getitem__(self, index):
        return (
            self.ray_directions[index],
            self.ray_origins[index],
        )


def getRaysDataLoaders(rays_data_path, batch_size=1024):
    """Get data loaders for rays.

    Args:
        rays_data_path (str): path to rays data.
        batch_size (int, optional): batch size. Defaults to 1024.

    Returns:
        torch.utils.data.DataLoader: train data loader.
        torch.utils.data.DataLoader: validation data loader.
    """

    rays_data = np.load(rays_data_path)

    train_images = rays_data["images_train"] / 255.0
    train_extrinsics = rays_data["c2ws_train"]

    val_images = rays_data["images_val"] / 255.0
    val_extrinsics = rays_data["c2ws_val"]

    focal = rays_data["focal"]

    train_dataset = RaysDataset(train_images, train_extrinsics, focal)
    val_dataset = RaysDataset(val_images, val_extrinsics, focal)
    return (
        data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
    )

def getRaysTestingDataLoaders(rays_data_path, batch_size=1024):
    """Get a data loader for testing NeRF 

    Args:
        image (str): path to image file.
        batch_size (int, optional): batch size. Defaults to 1024.

    Returns:
        torch.utils.data.DataLoader: data loader.
    """
    rays_data = np.load(rays_data_path)

    test_extrinsics = rays_data["c2ws_test"]

    focal = rays_data["focal"]

    test_dataset = RaysTestingDataset(200, 200, test_extrinsics, focal)
    return data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)