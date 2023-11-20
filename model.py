from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from data_loader import getSingleImageTestingDataLoader, getRaysTestingDataLoaders
import torch, torch.nn as nn
import torch.nn.functional as F
import lightning as L
from losses import PSNRLoss
import pdb
import copy
from utils import ray, visualization
import os


class PositionalEncoding(nn.Module):
    """Positional encoding module.

    Args:
        level (int, optional): level of positional encoding. Defaults to 10.
    """

    def __init__(self, level=10):
        super(PositionalEncoding, self).__init__()

        self.div_term = nn.Parameter(
            2 ** torch.arange(0, level).float() * torch.pi, requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Positional encoding x

        Args:
            x (torch.Tensor): input tensor of shape (N, 1).

        Returns:
            torch.Tensor: output tensor of shape (N, level * 2 + 1).
        """
        return torch.cat(
            [x, torch.sin(x * self.div_term), torch.cos(x * self.div_term)], dim=-1
        )


class SingleImageModule(L.LightningModule):
    """Module for training a NeRF on a single image.

    Args:
        lr (float, optional): learning rate. Defaults to 1e-2.
    """

    def __init__(self, lr=1e-2, pe_level=10, hidden_dim=256):
        super().__init__()
        self.lr = lr
        self.pe = PositionalEncoding(level=pe_level)

        self.model = nn.Sequential(
            nn.Linear(2 * (pe_level * 2 + 1), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid(),
        )

        self.PSNR_loss = PSNRLoss()
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): input tensor of shape (N, 1).

        Returns:
            torch.Tensor: output tensor of shape (N, 3).
        """
        batch_size, _ = x.shape
        x = x.reshape(-1, 1)
        x = self.pe(x).reshape(batch_size, -1)
        return self.model(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Training step"""
        x, y = batch
        y_hat = self.forward(x)
        PSNR_loss = self.PSNR_loss(y_hat, y)
        self.log("PSNR", PSNR_loss)
        return PSNR_loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def on_train_epoch_start(self) -> None:
        """Generate a low resolution image for visualization."""
        test_data_loader = getSingleImageTestingDataLoader(256, 256, 65536)
        test_result = []
        with torch.no_grad():
            for x in test_data_loader:
                x = x.to(self.device)
                test_result.append(self.forward(x))
        test_result = torch.cat(test_result, dim=0)
        result_image = test_result.reshape(256, 256, 3).permute(2, 0, 1)

        # Log image
        self.logger.experiment.add_image(
            "test_result", result_image, self.current_epoch
        )


class InputProcess(nn.Module):
    """Input process module for NeRF.

    Args:
        pe_level (int, optional): level of positional encoding. Defaults to 8.
    """

    def __init__(self, pe_level=8):
        super().__init__()
        self.pe = PositionalEncoding(level=pe_level)
        self.input_dim = 3 * (pe_level * 2 + 1)
        self.layer1 = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(256 + self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): input tensor of shape (N, 3).

        Returns:
            torch.Tensor: output tensor of shape (N, 256).
        """

        x = self.pe(x.view(-1, 1)).view(-1, self.input_dim)
        out1 = self.layer1(x)
        out2 = self.layer2(torch.cat([x, out1], dim=-1))
        return out2


class DensityLayer(nn.Module):
    """Density layer for NeRF."""

    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): input tensor of shape (N, 256).

        Returns:
            torch.Tensor: output tensor of shape (N, 1).
        """
        return self.layer(x)


class RGBLayer(nn.Module):
    """RGB layer for NeRF."""

    def __init__(self, pe_level=8):
        super().__init__()
        self.input_dim = 3 * (pe_level * 2 + 1)
        self.pe = PositionalEncoding(level=pe_level)
        self.layer = nn.Sequential(
            nn.Linear(256 + self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, ray_direction: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): input tensor of shape (N, 256).
            ray_direction (torch.Tensor): ray direction of shape (N, 3).

        Returns:
            torch.Tensor: output tensor of shape (N, 3).
        """
        pe_ray_direction = self.pe(ray_direction.view(-1, 1)).view(-1, self.input_dim)
        return self.layer(torch.cat([x, pe_ray_direction], dim=-1))


class NeRFModule(L.LightningModule):
    """NeRF module."""

    def __init__(self, lr=1e-2, n_sample=64, near=2.0, far=6.0, height=200, width=200):
        super().__init__()
        self.lr = lr
        self.n_sample = n_sample
        self.near = near
        self.far = far
        self.height = height
        self.width = width

        self.input_process = InputProcess()
        self.density_layer = DensityLayer()
        self.rgb_layer = RGBLayer()

        self.PSNR_loss = PSNRLoss()

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor, ray_direction: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): input tensor of shape (N, 3).
            ray_direction (torch.Tensor): ray direction of shape (N, 3).

        Returns:
            torch.Tensor: output tensor of shape (N, 3).
        """
        x = self.input_process(x)
        density = self.density_layer(x)
        rgb = self.rgb_layer(x, ray_direction)
        return density, rgb

    def render_rays(
        self, ray_direction: torch.Tensor, ray_origins: torch.Tensor, perturb=True
    ) -> torch.Tensor:
        """Render rays.

        Args:
            ray_direction (torch.Tensor): ray direction of shape (N, 3).
            ray_origins (torch.Tensor): ray origins of shape (N, 3).
        Returns:
            torch.Tensor: output RGB tensor of shape (N, 3).
        """
        sample_points = ray.sample_on_rays(
            ray_direction,
            ray_origins,
            near=self.near,
            far=self.far,
            n_sample=self.n_sample,
            perturb=perturb,
        )
        density, rgb = self.forward(
            sample_points,
            ray_direction.unsqueeze(1).repeat(1, self.n_sample, 1).reshape(-1, 3),
        )
        density = density.reshape(-1, self.n_sample)
        rgb = rgb.reshape(-1, self.n_sample, 3)
        output_rgb = ray.volumn_render(
            density, rgb, (self.far - self.near) / self.n_sample
        )
        return output_rgb

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Training step"""
        pixel_rgb, ray_direction, ray_origins = batch
        output_rgb = self.render_rays(ray_direction, ray_origins, perturb=True)
        PSNR_loss = self.PSNR_loss(output_rgb, pixel_rgb)
        self.log("PSNR", PSNR_loss)
        return PSNR_loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> STEP_OUTPUT:
        """Validation step"""
        pixel_rgb, ray_direction, ray_origins = batch
        output_rgb = self.render_rays(ray_direction, ray_origins, perturb=False)
        PSNR_loss = self.PSNR_loss(output_rgb, pixel_rgb)
        self.log("val_PSNR", PSNR_loss)
        return {"val_PSNR": PSNR_loss}

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def on_train_epoch_start(self) -> None:
        """Generate spherical image for visualization."""
        if self.current_epoch == 0:
            return

        test_data_loader = getRaysTestingDataLoaders(
            "data/lego_200x200.npz", batch_size=4096
        )
        test_result = []
        with torch.no_grad():
            for ray_direction, ray_origins in test_data_loader:
                ray_direction = ray_direction.to(self.device)
                ray_origins = ray_origins.to(self.device)
                test_result.append(
                    self.render_rays(ray_direction, ray_origins, perturb=False)
                )
        test_result = torch.cat(test_result, dim=0)
        result_images = test_result.reshape(-1, self.height, self.width, 3)

        save_path = (
            f"result/version_{self.logger.version}/epoch_{self.current_epoch:06}"
        )
        os.makedirs(save_path, exist_ok=True)
        for index, result_image in enumerate(result_images):
            visualization.save_image(
                result_image, os.path.join(save_path, f"{index:06}.png")
            )
        # Log image
        self.logger.experiment.add_image(
            "test_result", result_images[0].permute(2, 0, 1), self.current_epoch
        )


class CoarseToFineNeRFModule(L.LightningModule):
    """Coarse to fine NeRF module."""

    def __init__(
        self, lr=1e-2, n_c=32, n_f=64, near=2.0, far=6.0, height=200, width=200
    ):
        super().__init__()
        self.lr = lr
        self.n_c = n_c
        self.n_f = n_f
        self.near = near
        self.far = far
        self.height = height
        self.width = width

        self.coarse_model = NeRFModule.load_from_checkpoint(rf"lightning_logs\version_36\checkpoints\epoch=84-step=166090.ckpt")
        self.fine_model = NeRFModule(n_sample=n_f, near=near, far=far)
        self.PSNR_loss = PSNRLoss()

    def render_rays(
        self, ray_direction: torch.Tensor, ray_origins: torch.Tensor, perturb=True
    ) -> torch.Tensor:
        """Render rays using coarse-to-find strategy.
        Reference: https://github.com/kwea123/nerf_pl/

        Args:
            ray_direction (torch.Tensor): ray direction of shape (N, 3).
            ray_origins (torch.Tensor): ray origins of shape (N, 3).

        Returns:
            torch.Tensor: output RGB of shape (N, 3).
        """
        batch_size, _ = ray_direction.shape
        t_vals = ray.sample_t_vals(
            near=self.near,
            far=self.far,
            batch_size=batch_size,
            n_sample=self.n_c,
            perturb=perturb,
        )  # (n_sample)
        coarse_sample_points, coarse_step_size = ray.sample_on_rays(
            t_vals, ray_direction, ray_origins
        )
        coarse_density, coarse_rgb = self.coarse_model(
            coarse_sample_points,
            ray_direction.unsqueeze(1).repeat(1, self.n_c, 1).reshape(-1, 3),
        )
        coarse_density = coarse_density.reshape(-1, self.n_c)  # (N, n_c)
        coarse_rgb = coarse_rgb.reshape(-1, self.n_c, 3)  # (N, n_c, 3)
        coarse_weights = ray.sample_to_weights(
            coarse_density, coarse_step_size
        )  # (N, n_c)
        t_mid = 0.5 * (t_vals[:, 1:] + t_vals[:, :-1])  # (N, n_c - 1)
        t_fine = ray.sample_pdf(t_mid, coarse_weights[:, 1:-1], self.n_f).detach()  # (N, n_f)

        t_fine = torch.sort(torch.cat([t_vals, t_fine], dim=-1), dim=-1)[
            0
        ]  # (N, n_c + n_f)
        fine_sample_points, fine_step_size = ray.sample_on_rays(
            t_fine, ray_direction, ray_origins
        )

        fine_density, fine_rgb = self.fine_model(
            fine_sample_points,
            ray_direction.unsqueeze(1).repeat(1, self.n_c + self.n_f, 1).reshape(-1, 3),
        )

        fine_density = fine_density.reshape(-1, self.n_c + self.n_f)
        fine_rgb = fine_rgb.reshape(-1, self.n_c + self.n_f, 3)

        fine_output = ray.volumn_render(fine_density, fine_rgb, fine_step_size)

        coarse_output = ray.volumn_render(coarse_density, coarse_rgb, coarse_step_size)

        result = {
            "coarse_output": coarse_output,
            "fine_output": fine_output,
        }

        return result

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Training step"""
        pixel_rgb, ray_direction, ray_origins = batch
        result = self.render_rays(ray_direction, ray_origins, perturb=True)

        coarse_PSNR_loss = self.PSNR_loss(result["coarse_output"], pixel_rgb)
        fine_PSNR_loss = self.PSNR_loss(result["fine_output"], pixel_rgb)
        self.log("coarse_PSNR", coarse_PSNR_loss)
        self.log("fine_PSNR", fine_PSNR_loss)

        return coarse_PSNR_loss + fine_PSNR_loss
        # return coarse_PSNR_loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> STEP_OUTPUT:
        """Validation step"""
        pixel_rgb, ray_direction, ray_origins = batch
        result = self.render_rays(ray_direction, ray_origins, perturb=False)

        coarse_PSNR_loss = self.PSNR_loss(result["coarse_output"], pixel_rgb)
        fine_PSNR_loss = self.PSNR_loss(result["fine_output"], pixel_rgb)
        self.log("val_coarse_PSNR", coarse_PSNR_loss)
        self.log("val_fine_PSNR", fine_PSNR_loss)

        return {"val_coarse_PSNR": coarse_PSNR_loss, "val_fine_PSNR": fine_PSNR_loss}
        # return {"val_coarse_PSNR": coarse_PSNR_loss}

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def on_train_epoch_start(self) -> None:
        """Generate spherical image for visualization."""
        if self.current_epoch == 0:
            return

        test_data_loader = getRaysTestingDataLoaders(
            "data/lego_200x200.npz", batch_size=4096
        )
        fine_test_result = []
        coarse_test_result = []
        with torch.no_grad():
            for ray_direction, ray_origins in test_data_loader:
                ray_direction = ray_direction.to(self.device)
                ray_origins = ray_origins.to(self.device)
                result = self.render_rays(ray_direction, ray_origins, perturb=False)
                fine_test_result.append(result["fine_output"])
                coarse_test_result.append(result["coarse_output"])
        fine_test_result = torch.cat(fine_test_result, dim=0)
        coarse_test_result = torch.cat(coarse_test_result, dim=0)

        fine_result_images = fine_test_result.reshape(-1, self.height, self.width, 3)
        coarse_result_images = coarse_test_result.reshape(
            -1, self.height, self.width, 3
        )

        save_path = (
            f"result/version_{self.logger.version}/epoch_{self.current_epoch:06}"
        )
        os.makedirs(save_path, exist_ok=True)
        for index, result_image in enumerate(fine_result_images):
            visualization.save_image(
                result_image, os.path.join(save_path, f"fine_{index:06}.png")
            )
        for index, result_image in enumerate(coarse_result_images):
            visualization.save_image(
                result_image, os.path.join(save_path, f"coarse_{index:06}.png")
            )
        # Log image
        self.logger.experiment.add_image(
            "test_result", fine_result_images[0].permute(2, 0, 1), self.current_epoch
        )
