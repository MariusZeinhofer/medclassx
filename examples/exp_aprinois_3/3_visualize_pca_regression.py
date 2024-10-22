"""Visualizes the computed principle components."""

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import nibabel
from matplotlib import pyplot as plt
from scipy import ndimage


parser = argparse.ArgumentParser(description="Provide input and ouput data locations.")

parser.add_argument(
    "--out_folder",
    type=str,
    help="Folder to store output",
    default="examples/exp_aprinois_3/out",
)
args = parser.parse_args()

####################################LOAD DATA##########################################

print("Loading Data.")
pcs = []
for i in range(0, 20):
    with open(Path(args.out_folder) / f"pc_{i+1}.json", "r") as file:
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
            f"{vafs[i] * 100:.1f}%",
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
plt.savefig(Path(args.out_folder) / "screeplot.png")
plt.clf()

####################################PC SLICES##########################################

for i in range(0, 20):
    path = Path(args.out_folder) / f"zscore_pc_{i+1}.nii"
    nifti_img = nibabel.load(path)
    img = nifti_img.get_fdata()

    min_img = jnp.min(img)
    max_img = jnp.max(img)

    # Create a figure and an array of subplots (1 row, 3 columns)
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # Plot on the first subplot
    c0 = axs[0, 0].imshow(
        ndimage.rotate(img[64, :, :].transpose(), 180),
        cmap="viridis",
        vmin=min_img,
        vmax=max_img,
    )
    axs[0, 0].set_title("Saggital Slice at [64, :, :]")
    fig.colorbar(c0, ax=axs[0, 0], orientation="vertical", shrink=0.63)

    # Plot on the second subplot
    c1 = axs[0, 1].imshow(
        ndimage.rotate(img[:, 72, :].transpose(), 180),
        cmap="viridis",
        vmin=min_img,
        vmax=max_img,
    )
    axs[0, 1].set_title("Coronal Slice at [:, 72, :]")
    fig.colorbar(c1, ax=axs[0, 1], orientation="vertical", shrink=0.73)

    # Plot on the third subplot
    c2 = axs[1, 0].imshow(img[:, :, 60], cmap="viridis", vmin=min_img, vmax=max_img)
    axs[1, 0].set_title("Horizontal Slice at [:, :, 60]")
    fig.colorbar(c2, ax=axs[1, 0], orientation="vertical", shrink=0.63)

    # get info on the PC from the json file
    with open(Path(args.out_folder) / f"pc_{i+1}.json", "r") as file:
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

    plt.savefig(Path(args.out_folder) / f"zscore_pc_{i+1}.png")
    plt.close()
