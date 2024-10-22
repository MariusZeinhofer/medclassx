"""Visualizes the computed principle components."""

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import nibabel
from matplotlib import pyplot as plt
from scipy import ndimage
from matplotlib.colors import LinearSegmentedColormap
from medclassx.mask_trafo import mask_vector

outfolder = Path("examples/exp_lateNC/out")

####################################LOAD DATA##########################################

print("Loading Data.")
pcs = []
for i in range(0, 19):
    with open(outfolder / f"pc_{i+1}.json", "r") as file:
        pc = json.load(file)
        pcs.append(pc)

###################################SCREE PLOT##########################################

print("Assembling scree plot.")
singular_values = [pc["sv"] for pc in pcs]
vafs = [pc["vaf"] for pc in pcs]
x_axis = range(1, len(singular_values) + 1)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x_axis, singular_values, marker="o", label="singular values")

# Annotate each point with its value accounted for
for i, _ in enumerate(singular_values):
    # print vaf only for every second singular value
    if i % 2 == 0:
        plt.annotate(
            f"{vafs[i] * 100:.2f}%",
            (x_axis[i], singular_values[i]),
            textcoords="offset points",
            xytext=(20, 5),
            ha="center",
            bbox=dict(
                boxstyle="round,pad=0.3",
                edgecolor="black",
                facecolor="white",
            ),
        )

# Adding labels and title
plt.xticks(ticks=range(1, 21, 2))
plt.xlabel("Principle Component Number")
plt.ylabel("Singular Values")
plt.title("Scree Plot -- Singular Value Decay")
plt.legend()
plt.savefig(outfolder / "screeplot.png")
plt.clf()


##########################################################

# Define the custom colormap (black -> red -> orange -> yellow -> white)
colors = [
    (0.0, (0, 0, 0)),    # 0% black
    (0.4, (0.5, 0, 0)),  # 20% dark red
    (0.5, (0.8, 0, 0)),    # 40% red
    (0.6, (0.8, 0.5, 0)),  # 60% orange
    (0.7, (1, 1, 0)),    # 80% yellow
    (0.8, (1, 1, 0.75)),  # 80% light yellow
    (1.0, (1, 1, 1)),    # 100% white
]

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

for i in range(0, 18):
    path = outfolder / f"zscore_pc_{i+1}.nii"
    nifti_img = nibabel.load(path)
    data = nifti_img.get_fdata()

    a, b, c = data.shape
    data = -data.reshape(-1)

    mask = nibabel.load(outfolder / "mask_resampled.nii").get_fdata()
    mask = mask.reshape(-1)
    data, unmask = mask_vector(data, mask)

    global_min = np.nanmin(data)  # Ignore NaN values
    global_max = np.nanmax(data)  # Ignore NaN values
    
    # shift data to [0, max - min]
    #data = data - global_min

    data = unmask(data, padding=np.nan)
    data= data.reshape(a, b, c)
    

    n_rows, n_cols = 5, 6
    N = n_rows * n_cols
    H = data.shape[2]
    H = data.shape[2]
    cuts = np.linspace(10, H-10, N)

    images = []
    for cut in cuts:
        img = rotate_180(data[:, :, int(cut)]).transpose()
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

    fig.suptitle(f"Principle Component {i+1}", fontsize=50)

    # Remove unnecessary space around the image and colorbar
    plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0.11)

    plt.savefig(outfolder / f"invert_pc_{i+1}_5x6.png", bbox_inches="tight", pad_inches=0)
    plt.close()















####################################PC SLICES##########################################

for i in range(0, 18):
    path = outfolder / f"zscore_pc_{i+1}.nii"
    nifti_img = nibabel.load(path)
    img = nifti_img.get_fdata()

    data = img

    # used to get signum of PC correct
    if i == 1:
        factor = -1
    else:
        factor = 1

    a, b, c = data.shape
    data = factor * data.reshape(-1)

    mask = nibabel.load(outfolder / "mask_resampled.nii").get_fdata()
    mask = mask.reshape(-1)
    data, unmask = mask_vector(data, mask)

    global_min = np.nanmin(data)  # Ignore NaN values
    global_max = np.nanmax(data)  # Ignore NaN values
    
    # shift data to [0, max - min]
    #data = data - global_min

    data = unmask(data, padding=np.nan)
    data= data.reshape(a, b, c)

    img = data

    min_img = jnp.min(img)
    max_img = jnp.max(img)

    # Create a figure and an array of subplots (1 row, 3 columns)
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    

    # Plot on the first subplot
    c0 = axs[0, 0].imshow(
        #ndimage.rotate(img[45, :, :].transpose(), 180),
        rotate_180(img[45, :, :]).transpose(),
        cmap=cmap,
        vmin=global_min,
        vmax=global_max,
    )
    axs[0, 0].set_title("Saggital Slice at [45, :, :]")
    fig.colorbar(c0, ax=axs[0, 0], orientation="vertical", shrink=0.63)

    # Plot on the second subplot
    c1 = axs[0, 1].imshow(
        #ndimage.rotate(img[:, 48, :].transpose(), 180),
        rotate_180(img[:, 48, :]).transpose(),
        cmap=cmap,
        vmin=global_min,
        vmax=global_max,
    )
    axs[0, 1].set_title("Coronal Slice at [:, 48, :]")
    fig.colorbar(c1, ax=axs[0, 1], orientation="vertical", shrink=0.73)

    # Plot on the third subplot
    c2 = axs[1, 0].imshow(
        img[:, :, 39], 
        cmap=cmap, 
        vmin=global_min,
        vmax=global_max,
        )
    axs[1, 0].set_title("Horizontal Slice at [:, :, 39]")
    fig.colorbar(c2, ax=axs[1, 0], orientation="vertical", shrink=0.63)

    # get info on the PC from the json file
    with open(outfolder / f"pc_{i+1}.json", "r") as file:
        pc = json.load(file)

    data = [
        ["PC Number", "Singular Value", "vaf value", "accuracy"],
        [
            f"{pc['pc_number']+1:.1f}",
            f"{pc['sv']:.1f}",
            f"{pc['vaf']:.2f}",
            f"{pc['accuracy']:.2f}",
        ],
    ]

    # Plot table data
    axs[1, 1].axis("off")  # Turn off the axis for the table subplot
    table = axs[1, 1].table(cellText=data, cellLoc="center", loc="center")

    # Adjust table appearance
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width([0, 1, 2])

    # Add a main title for the figure
    fig.suptitle(f"Principle Component {i+1}")

    plt.savefig(outfolder / f"zscore_pc_{i+1}.png")
    plt.close()
