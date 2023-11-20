import viser
import time
import numpy as np
import torch
from data_loader import getRaysDataLoaders
import utils.ray as ray
import cv2
from matplotlib import pyplot as plt

def main():
    plt.imshow(np.load("data/lego_200x200.npz")["images_train"][0][:100])
    plt.show()
    train_data_loader, val_data_loader = getRaysDataLoaders(
        "data/lego_200x200.npz", batch_size=100
    )
    pixels, rays_d, rays_o = next(iter(train_data_loader)) # Should expect (B, 3)
    points = ray.sample_on_rays(rays_d, rays_o, near=2.0, far=6.0, n_sample=64)
    H, W = 200, 200

    server = viser.ViserServer(share=True)
    for i, (o, d) in enumerate(zip(rays_o, rays_d)):
        o = o.numpy()
        d = d.numpy()
        server.add_spline_catmull_rom(
            f"/rays/{i}", positions=np.stack((o, o + d * 6.0)),
        )
    points = points.numpy()
    server.add_point_cloud(
        f"/samples",
        colors=np.zeros_like(points).reshape(-1, 3),
        points=points.reshape(-1, 3),
        point_size=0.02,
    )
    time.sleep(1000)

if __name__ == "__main__":
    main()