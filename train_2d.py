import lightning as L
from data_loader import getSingleImageDataLoader
import torch
from model import SingleImageModule

train_data_loader = getSingleImageDataLoader("data/fox.jpg", batch_size=16384)
torch.set_float32_matmul_precision("medium")


trainer = L.Trainer(max_epochs=300)

single_image_module = SingleImageModule(
    lr=1e-2,
    pe_level=10,
    hidden_dim=410,
)

trainer.fit(single_image_module, train_data_loader)
