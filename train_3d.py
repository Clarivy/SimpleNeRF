import lightning as L
import torch
import numpy as np
from data_loader import getRaysDataLoaders
from model import CoarseToFineNeRFModule

train_data_loader, val_data_loader = getRaysDataLoaders(
    "data/lego_200x200.npz", batch_size=1024
)


trainer = L.Trainer()

nerf_module = CoarseToFineNeRFModule(lr=5e-4)

trainer.fit(nerf_module, train_data_loader, val_data_loader)
