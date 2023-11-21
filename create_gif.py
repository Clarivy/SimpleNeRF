import os
import imageio
from glob import glob


def create_gif(directory, output_filename, duration):
    # Get list of image files in the directory, sorted alphabetically
    image_files = sorted(glob(directory))

    # Load images
    images = []
    for filename in image_files:
        images.append(imageio.v2.imread(filename))

    # Write images to a gif
    imageio.mimsave(output_filename, images, duration=duration)


directory_path = "result/version_61/depth/*.png"
gif_filename = "images/coarse_to_fine/depth_000044.gif"
frame_duration = 0.03
create_gif(directory_path, gif_filename, frame_duration)
