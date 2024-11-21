"""Visualizes the defined pattern."""

from pathlib import Path

import numpy as np
import nibabel
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from medclassx.mask_trafo import mask_vector

outfolder = Path("examples/exp_lateNC/out")

####################################LOAD DATA##########################################

print("Loading Data.")
pattern = np.load(outfolder / "pattern.npy")

#####################################Visualize#########################################

colors = [
    (0.0, (0.2, 0.2, 0.6)),  # Muted blue
    (0.33, (0.2, 0.6, 0.2)),  # Muted green
    (0.66, (0.9, 0.9, 0.3)),  # Muted yellow
    (1.0, (0.8, 0.2, 0.2))    # Muted red
]

# Create a colormap from the colors
custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)
custom_cmap.set_bad(color='black')
cmap = custom_cmap

def rotate_180(array):
     return np.flipud(np.fliplr(array))

a, b, c = pattern.shape
pattern = pattern.reshape(-1)

mask = nibabel.load(outfolder / "mask_resampled.nii").get_fdata()
mask = mask.reshape(-1)
pattern, unmask = mask_vector(pattern, mask)

global_min = np.nanmin(pattern)  # Ignore NaN values
global_max = np.nanmax(pattern)  # Ignore NaN values

# shift pattern to [0, max - min]
#pattern = pattern - global_min

pattern = unmask(pattern, padding=np.nan)
pattern= pattern.reshape(a, b, c)

n_rows, n_cols = 5, 6
N = n_rows * n_cols
H = pattern.shape[2]
H = pattern.shape[2]
cuts = np.linspace(10, H-10, N)

images = []
for cut in cuts:
    img = rotate_180(pattern[:, :, int(cut)]).transpose()
    images.append(img)

rows = [np.concatenate(images[i * n_cols:(i + 1) * n_cols], axis=1) for i in range(n_rows)]
final_image = np.concatenate(rows, axis=0)

# Plot the large concatenated image
fig, ax = plt.subplots(figsize=(25, 30))  # Adjust the figure size as needed
im = ax.imshow(final_image, cmap=cmap)#, vmin=-0.1, vmax=global_max - global_min)

ax.set_xticks([])  # Remove x-axis ticks
ax.set_yticks([])  # Remove y-axis ticks
ax.axis('off')  

# Adjust the position to make space for the colorbar
pos = ax.get_position()
# This ensures the colorbar is under the image and the same width
cbar_ax = fig.add_axes([pos.x0 - 0.125, pos.y0 - 0.079, 1.29 * pos.width, 0.02])

# Add horizontal colorbar below the image
cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')

# Set font size of the colorbar
cbar.ax.tick_params(labelsize=30)

fig.suptitle(f"Pattern", fontsize=50)

# Remove unnecessary space around the image and colorbar
plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0.11)

plt.savefig(outfolder / f"pattern_5x6.png", bbox_inches="tight", pad_inches=0)
plt.close()