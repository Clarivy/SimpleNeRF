import lightning as L
import torch
import numpy as np
from data_loader import getRaysDataLoaders
from model import NeRFModule

torch.set_float32_matmul_precision("medium")

train_data_loader, val_data_loader = getRaysDataLoaders(
    "data/lego_200x200.npz", batch_size=4096
)


trainer = L.Trainer()

nerf_module = NeRFModule(lr=5e-4)

trainer.fit(nerf_module, train_data_loader, val_data_loader)