import torch
import numpy as np
import torch, torch.nn as nn
import pdb


def apply_transform(
    transform_matrices: torch.Tensor, coordinates: torch.Tensor
) -> torch.Tensor:
    """Applies transform matrices to a set of coordinates.

    Args:
        transform_matrix (torch.Tensor): transform matrix of shape (B, 4, 4).
        coordinates (torch.Tensor): coordinates of shape (B, 4).

    Returns:
        torch.Tensor: transformed and normalized coordinates of shape (B, 4).
    """

    # Apply transform matrix
    coordinates = torch.matmul(transform_matrices, coordinates.unsqueeze(-1)).squeeze(
        -1
    )  # (B, 4, 4) * (B, 4, 1) -> (B, 4, 1) -> (B, 4)

    # Normalize coordinates
    coordinates = coordinates / coordinates[:, -1].unsqueeze(
        -1
    )  # (B, 4) / (B, 1) -> (B, 4)

    return coordinates  # (B, 4)


def pixel_to_camera(
    intrinsics: torch.Tensor,
    pixel_coordinates: torch.Tensor,
    depth: torch.Tensor = None,
) -> torch.Tensor:
    """Converts pixel coordinates to camera coordinates.

    Args:
        intrinsics (torch.Tensor): intrinsics matrix of shape (B, 3, 3).
        pixel_coordinates (torch.Tensor): pixel coordinates of shape (B, 2).
        depth (torch.Tensor, optional): depth. Defaults to None.

    Returns:
        torch.Tensor: camera coordinates of shape (B, 3).
    """

    bs = pixel_coordinates.shape[0]
    # Prepare pixel coordinates
    pixel_coordinates = torch.cat(
        [pixel_coordinates, torch.ones(bs, 1)], dim=-1
    )  # (B, 3)

    # Transform pixel coordinates to camera coordinates
    camera_coordinates = apply_transform(
        intrinsics.inverse(), pixel_coordinates
    )  # (B, 3)

    # Scale camera coordinates by depth
    if depth is not None:
        camera_coordinates = camera_coordinates * depth.unsqueeze(-1)  # (B, 3)

    return camera_coordinates  # (B, 3)


def pixel_to_rays(
    intrinsics: torch.Tensor,
    camera_to_world: torch.Tensor,
    pixel_coordinates: torch.Tensor,
) -> torch.Tensor:
    """Converts pixel coordinates to ray directions and oringins.

    Args:
        intrinsics (torch.Tensor): intrinsics matrix of shape (B, 3, 3).
        camera_to_world (torch.Tensor): camera to world matrix of shape (B, 4, 4).
        pixel_coordinates (torch.Tensor): pixel coordinates of shape (B, 2).

    Returns:
        torch.Tensor: ray directions of shape (B, 3).
        torch.Tensor: ray origins of shape (B, 3).
    """
    batch_size = pixel_coordinates.shape[0]
    # Convert pixel coordinates to camera coordinates
    camera_coordinates = pixel_to_camera(intrinsics, pixel_coordinates)  # (B, 3)

    # Prepare camera coordinates
    camera_coordinates = torch.cat(
        [camera_coordinates, torch.ones(batch_size, 1)], dim=-1
    )  # (B, 4)

    # Convert camera coordinates to world coordinates
    world_coordinates = apply_transform(camera_to_world, camera_coordinates)  # (B, 4)

    # Get ray origins
    ray_origins = camera_to_world[:, :3, 3]  # (B, 3)

    # Get ray directions
    ray_directions = world_coordinates[:, :3] - ray_origins  # (B, 3)
    ray_directions = ray_directions / torch.norm(
        ray_directions, dim=-1, keepdim=True
    )  # (B, 3)

    return ray_directions, ray_origins


def sample_to_weights(sigmas: torch.Tensor, step_size: float) -> torch.Tensor:
    """Converts sigmas to weights.

    Args:
        sigmas (torch.Tensor): density of shape (N, n_sample).
        step_size (torch.Tensor): step size between sample of shape (n_sample).

    Returns:
        torch.Tensor: weights of shape (N, n_sample).
    """
    t = torch.exp(torch.cumsum(-sigmas * step_size, dim=-1))  # (N, n_sample)
    t = torch.cat([torch.ones_like(t[:, :1]), t[:, :-1]], dim=-1)  # (N, n_sample)

    # Get weights
    weights = 1 - torch.exp(-sigmas * step_size)  # (N, n_sample)
    weights = weights * t  # (N, n_sample)
    return weights  # (N, n_sample)


def volumn_render(
    sigmas: torch.Tensor,
    rgbs: torch.Tensor,
    step_size: torch.Tensor,
    white_background=False,
) -> torch.Tensor:
    """Volumn rendering.

    Args:
        sigmas (torch.Tensor): density of shape (N, n_sample).
        rgbs (torch.Tensor): rgb of shape (N, n_sample, 3).
        step_size (torch.Tensor): step size between sample of shape (N, n_sample).
        white_background (bool, optional): whether to use white background. Defaults to False.

    Returns:
        torch.Tensor: rendered rgb of shape (N, 3).
    """
    if white_background:
        sigmas = torch.cat([sigmas, 9999 * torch.ones_like(sigmas[:, :1])], dim=-1)
        rgbs = torch.cat([rgbs, torch.ones_like(rgbs[:, :1])], dim=-2)
        step_size = torch.cat([step_size, torch.ones_like(step_size[:, :1])], dim=-1)

    weights = sample_to_weights(sigmas, step_size)  # (N, n_sample)

    # Get rgb
    result_rgb = torch.sum(weights.unsqueeze(-1) * rgbs, dim=-2)  # (N, 3)
    return result_rgb  # (N, 3)


def sample_t_vals(
    near=2.0, far=6.0, batch_size=2048, n_sample=64, perturb=True, device="cuda"
):
    """Sample t values.

    Args:
        near (float, optional): near plane. Defaults to 2.0.
        far (float, optional): far plane. Defaults to 6.0.
        batch_size (int, optional): batch size. Defaults to 2048.
        n_sample (int, optional): number of samples. Defaults to 64.
        perturb (bool, optional): whether to perturb the samples. Defaults to True.
        device (str, optional): device. Defaults to "cuda".

    Returns:
        torch.Tensor: t values of shape (batch_size, n_sample)
    """
    t_vals = torch.linspace(near, far, steps=n_sample)  # (n_sample)
    t_vals = t_vals.expand(batch_size, n_sample)  # (batch_size, n_sample)
    if perturb:
        width = (far - near) / n_sample
        t_vals = t_vals + torch.rand(batch_size, n_sample) * width
    t_vals = t_vals.to(device)
    return t_vals  # (batch_size, n_sample)


def sample_on_rays(
    t_vals: torch.Tensor,
    ray_direction: torch.Tensor,
    ray_origins: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample points on rays.

    Args:
        t_vals (torch.Tensor): t values of shape (N, n_sample).
        ray_direction (torch.Tensor): ray direction of shape (N, 3).
        ray_origins (torch.Tensor): ray origin of shape (N, 3).

    Returns:
        torch.Tensor: sample points of shape (N, n_sample, 3).
        torch.Tensor: step size of shape (N, n_sample).
    """

    # Get step size between t_vals
    step_size = torch.cat(
        [
            t_vals[:, 1:] - t_vals[:, :-1],
            torch.zeros_like(t_vals[:, :1]),
        ],
        dim=-1,
    )  # (N, n_sample)

    # Get sample points
    sample_points = ray_origins.unsqueeze(1) + ray_direction.unsqueeze(
        1
    ) * t_vals.unsqueeze(
        -1
    )  # (N, n_sample, 3)

    return sample_points, step_size


def get_rays(
    image_num: int, height: int, width: int, focal: float, extrinsics: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get rays from mesh grid.

    Args:
        image_num (int): number of images.
        height (int): image height.
        width (int): image width.
        focal (float): focal length.
        extrinsics (np.ndarray): extrinsics of shape (image_num, 4, 4).

    Returns:
        torch.Tensor: ray directions of shape (image_num * height * width, 3).
        torch.Tensor: ray origins of shape (image_num * height * width, 3).
    """

    x, y = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="xy",
    )  # (H, W)
    pixel_coordinates = torch.stack([x, y], dim=-1).reshape(-1, 2)  # (H * W, 2)

    intrinsics = np.array(
        [
            [focal, 0, height / 2],
            [0, focal, width / 2],
            [0, 0, 1],
        ]
    )
    intrinsics = torch.tensor(intrinsics, dtype=torch.float32)

    ray_directions = []
    ray_origins = []

    # Convert pixel coordinates to rays
    for image_index in range(image_num):
        current_extrinsics = extrinsics[image_index]
        current_extrinsics = current_extrinsics.repeat(height * width, 1, 1)
        current_ray_directions, current_ray_origins = pixel_to_rays(
            intrinsics, current_extrinsics, pixel_coordinates
        )
        ray_directions.append(current_ray_directions)
        ray_origins.append(current_ray_origins)

    ray_directions = torch.cat(ray_directions, dim=0)
    ray_origins = torch.cat(ray_origins, dim=0)

    return ray_directions, ray_origins


def sample_pdf(bins: torch.Tensor, weights: torch.Tensor, n_samples: int, det=False):
    """Sample points from pdf. Reference: https://github.com/bmild/nerf

    Args:
        bins (torch.Tensor): bins of shape (N, n_c - 1). n_c is sample number of coarse model
        weights (torch.Tensor): weights of shape (N, n_c - 2).
        n_samples (int): number of output samples, n_f in the paper.
        det (bool, optional): whether to use deterministic sampling. Defaults to False.

    Returns:
        torch.Tensor: samples of shape (N, n_f).
    """

    weights = weights + 1e-5  # (N, n_c - 2)
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)  # (N, n_c - 2)
    cdf = torch.cumsum(pdf, dim=-1)  # (N, n_c - 2)
    cdf = torch.cat(
        [torch.zeros_like(cdf[..., :1]), cdf], dim=-1
    )  # (N, n_c - 1), cdf[0] = 0

    if det:
        u = torch.linspace(0, 1.0, n_samples)  # (n_f)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])  # (N, n_f)
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples])  # (N, n_f)

    u = u.to(bins.device)
    u = u.contiguous()

    idxs = torch.searchsorted(cdf, u, right=True)  # (Batch, Nf)
    below = torch.max(torch.zeros_like(idxs), idxs - 1)
    above = torch.min(torch.ones_like(idxs) * (bins.shape[1] - 1), idxs)
    inds_g = torch.stack([below, above], dim=-1)  # (Batch, Nf, 2)

    matched_shape = list(inds_g.shape[:-1]) + [bins.shape[1]]  # (Batch, Nf, Nc-1)

    cdf_g = torch.gather(cdf[..., None, :].expand(matched_shape), dim=-1, index=inds_g)
    bins_g = torch.gather(
        bins[..., None, :].expand(matched_shape), dim=-1, index=inds_g
    )

    denom = cdf_g[..., 1] - cdf_g[..., 0]  # (Batch, Nf)
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)

    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
    return samples
