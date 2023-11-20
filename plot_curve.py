import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import os


def plot_tensorboard_data(filenames):
    plt.figure(figsize=(10, 6))

    for filename in filenames:
        # Read the CSV file
        data = pd.read_csv(filename)
        data_label = os.path.basename(filename).split(".")[0]
        # Plot the data
        plt.plot(data["Step"], data["Value"], label=data_label)

    # Labeling the axes and title
    plt.xlabel("Step")
    plt.ylabel("Value (dB)")
    plt.title("PSNR over Steps")

    # Adding a legend
    plt.legend()

    # Show the plot
    plt.show()

# best result
data_path = [
    rf"result\2d_images\best\lr=1e-3,pe=10.csv",
]
plot_tensorboard_data(data_path)

# Learning rate
data_path = [
    rf"result\2d_images\lr\lr=1e-1.csv",
    rf"result\2d_images\lr\lr=5e-2.csv",
    rf"result\2d_images\lr\lr=1e-2.csv",
    rf"result\2d_images\lr\lr=1e-3.csv",
    rf"result\2d_images\lr\lr=1e-4.csv",
    rf"result\2d_images\lr\lr=1e-5.csv",
]
plot_tensorboard_data(data_path)

# Positional encoding
data_path = [
    rf"result\2d_images\pe\PE=6.csv",
    rf"result\2d_images\pe\PE=8.csv",
    rf"result\2d_images\pe\PE=10.csv",
    rf"result\2d_images\pe\PE=12.csv",
    rf"result\2d_images\pe\PE=14.csv",
]
plot_tensorboard_data(data_path)
