import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def generate_thin_horizontal_swiss_roll(n_samples, noise=0.2, scale=3, length_factor=2, thickness_factor=0.5):
    dataset = []
    for i in range(n_samples):
        t = 1.5 * np.pi * (1 + 2 * random.random())
        x = length_factor * scale * random.random()
        y = thickness_factor * t * np.cos(t)
        z = thickness_factor * t * np.sin(t)

        x += noise * np.random.randn()
        y += noise * np.random.randn()

        dataset.append([x, y, z])

    return np.array(dataset, dtype='float32')

def swiss_role(data_pts):
  n_samples = data_pts
  thin_horizontal_swiss_roll_data = generate_thin_horizontal_swiss_roll(n_samples)
  return thin_horizontal_swiss_roll_data


def plot_swiss_roll(data, filename='swiss_roll_plot.png'):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    x, y, z = data.T
    scatter = ax.scatter(x, y, z, c=z, cmap='viridis')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Swiss Roll')
    
    plt.colorbar(scatter)
    
    # Ensure the save directory exists
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the figure
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    # Close the figure to free up memory
    plt.close(fig)

# Example usage:
data = swiss_role(8000)
plot_swiss_roll(data, 'swissrole8k.png')