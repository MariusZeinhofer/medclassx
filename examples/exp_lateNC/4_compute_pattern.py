"""Based on the compute_pca.py and visualize_pca.py derives the PCA pattern."""

from pathlib import Path

import jax.numpy as jnp
import nibabel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from medclassx.mask_trafo import mask_vector

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

outfolder = Path("examples/exp_lateNC/out")

# choose pcs
for pcs_idx in [[2,3], [2,4], [2,6], [2, 8]]:


    # the latent data, remember 0 based counting
    T = jnp.load(outfolder / "T.npy")[:, [p-1 for p in pcs_idx]]

    # load the principle components
    pcs = [nibabel.load(outfolder / f"pc_{pc}.nii") for pc in pcs_idx]

    # convert to a list of numpy arrays
    pcs = [pc.get_fdata() for pc in pcs]
    print(pcs[0].shape)

    # the first 9 patients are healthy controls, the last 9 are LATE diagnosed
    y_1 = jnp.zeros(shape=(9,), dtype=int)
    y_2 = jnp.ones(shape=(10,), dtype=int)
    y = jnp.concatenate((y_1, y_2), axis=0)

    # Set up logistic regression
    model = LogisticRegression(max_iter=200)

    # uses L-BFGS per default
    model.fit(T, y)

    # Access the trained parameters
    weights = model.coef_
    intercept = model.intercept_

    # Print the weights and intercept
    print("Weights (coefficients):")
    print(weights)
    print(weights.shape)
    print("Intercept:")
    print(intercept)

    # Make predictions
    y_pred = model.predict(T)

    # Evaluate the model
    accuracy = accuracy_score(y, y_pred)
    conf_matrix = confusion_matrix(y, y_pred)

    # Print the evaluation results
    print(f"Accuracy: {accuracy}")
    print("Confusion Matrix:")
    print(conf_matrix)

    # derive pattern using logistic weights
    pattern = sum([weights[0, i] * pcs[i] for i in range(0, len(pcs))])
    jnp.save(outfolder / f"pattern{str(pcs_idx)}.npy", pattern)



    ##########################################################################


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
    pattern = -pattern.reshape(-1)

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

    fig.suptitle(f"Pattern{str(pcs_idx)}", fontsize=50)

    # Remove unnecessary space around the image and colorbar
    plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0.11)

    plt.savefig(outfolder / f"pattern{str(pcs_idx)}inverted.png", bbox_inches="tight", pad_inches=0)
    plt.close()

















###############################################################
from itertools import combinations

# pcs allowed to choose from
pc_pool = set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
combi = list(combinations(pc_pool, 1))


# maximal number of PCs allowed
N_max = 2

# all combinations to be tested
pcs_idxs = [c for n in range(1, N_max+1) for c in list(combinations(pc_pool, n))]

print(f"NB: We will test {len(pcs_idxs)} combinations.")


Ts = jnp.load(outfolder / "T.npy")

accuracies = []
for pcs_idx in pcs_idxs:
    
    # the latent data, remember 0 based counting
    T = Ts[:, [p-1 for p in pcs_idx]]

    # load the principle components
    pcs = [nibabel.load(outfolder / f"pc_{pc}.nii") for pc in pcs_idx]

    # convert to a list of numpy arrays
    pcs = [pc.get_fdata() for pc in pcs]

    # the first 9 patients are healthy controls, the last 10 are LATE diagnosed
    y_1 = jnp.zeros(shape=(9,), dtype=int)
    y_2 = jnp.ones(shape=(10,), dtype=int)
    y = jnp.concatenate((y_1, y_2), axis=0)

    # Set up logistic regression
    model = LogisticRegression(max_iter=200)

    # uses L-BFGS per default
    model.fit(T, y)

    # Access the trained parameters
    weights = model.coef_
    intercept = model.intercept_

    # Print the weights and intercept
    #print("Weights (coefficients):")
    #print(weights)
    #print(weights.shape)
    #print("Intercept:")
    #print(intercept)

    # Make predictions
    y_pred = model.predict(T)

    if pcs_idx == (2, 4):
        print("y", y)
        print("y pred", y_pred)
        exit()

    # Evaluate the model
    accuracy = accuracy_score(y, y_pred)
    #conf_matrix = confusion_matrix(y, y_pred)

    # Print the evaluation results
    #print(f"{pcs_idx}: Accuracy: {accuracy}")
    accuracies.append((pcs_idx, accuracy))
    #print("Confusion Matrix:")
    #print(conf_matrix)

accuracies.sort(key=lambda x: x[1], reverse=True)
print(accuracies[0:25])