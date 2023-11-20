import lightning as L
from data_loader import getSingleImageDataLoader
from model import SingleImageModule

train_data_loader = getSingleImageDataLoader("data/Berkeley_glade_afternoon.jpg", batch_size=16384)

trainer = L.Trainer(max_epochs=300, default_root_dir="2d_lightning_logs")

single_image_module = SingleImageModule(
    lr=1e-2,
    pe_level=10,
    hidden_dim=410,
)

trainer.fit(single_image_module, train_data_loader)
