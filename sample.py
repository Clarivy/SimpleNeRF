import lightning as L
import torch
import numpy as np
from data_loader import getRaysTestingDataLoaders
from model import CoarseToFineNeRFModule
import os
from utils import visualization
from tqdm import tqdm
import pdb


nerf_module = CoarseToFineNeRFModule.load_from_checkpoint(
    rf"lightning_logs\version_61\checkpoints\epoch=43-step=171908.ckpt"
)
nerf_module = nerf_module.cuda()
nerf_module.eval()

test_data_loader = getRaysTestingDataLoaders("data/lego_200x200.npz", batch_size=4000)
fine_test_result = []
coarse_test_result = []
depth_test_result = []
with torch.no_grad():
    for ray_direction, ray_origins in tqdm(test_data_loader):
        ray_direction = ray_direction.cuda()
        ray_origins = ray_origins.cuda()
        result = nerf_module.render_rays(
            ray_direction, ray_origins, perturb=False, white_background=True
        )
        output_depth = nerf_module.render_depth_map(
            ray_direction, ray_origins, perturb=False
        )
        fine_test_result.append(result["fine_output"])
        coarse_test_result.append(result["coarse_output"])
        depth_test_result.append(output_depth)
fine_test_result = torch.cat(fine_test_result, dim=0)
coarse_test_result = torch.cat(coarse_test_result, dim=0)
depth_test_result = torch.cat(depth_test_result, dim=0)

fine_result_images = fine_test_result.reshape(-1, 200, 200, 3)
coarse_result_images = coarse_test_result.reshape(-1, 200, 200, 3)
depth_result_images = depth_test_result.reshape(-1, 200, 200) / depth_test_result.max()

save_path = f"result/version_61/test"
os.makedirs(save_path, exist_ok=True)
for index, result_image in enumerate(fine_result_images):
    visualization.save_image(
        result_image, os.path.join(save_path, f"fine_{index:06}.png")
    )
for index, result_image in enumerate(coarse_result_images):
    visualization.save_image(
        result_image, os.path.join(save_path, f"coarse_{index:06}.png")
    )
for index, result_image in enumerate(depth_result_images):
    visualization.save_image(result_image, os.path.join(save_path, f"{index:06}.png"))
