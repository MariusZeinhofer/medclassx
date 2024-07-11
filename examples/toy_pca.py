"""Extracts some principal components from a toy dataset."""

from pathlib import Path

import jax.numpy as jnp
import nibabel
from medclassx.pca import pca

# collect data paths in a list
paths = [p for p in Path(r"data\MCI39\nifti_preproc").iterdir()]

# load the images with nibabel
nifti_imgs = [nibabel.load(p) for p in paths]

# extract the numerical data as float64
img_data = [n.get_fdata() for n in nifti_imgs]

# convert to a single 4-d jax tensor of shape (batch, h, w, d)
img_data = jnp.array(img_data)

print(f"Data group shape: {jnp.shape(img_data)} and dtype {img_data.dtype}.")

# data shapes
n, a, b, c = jnp.shape(img_data)

# Reshape into matrix form
X = jnp.reshape(img_data, shape=(n, a * b * c))

# column center
X -= jnp.mean(X, axis=0, keepdims=True)

# row center
X -= jnp.mean(X, axis=1, keepdims=True)

# shift data to the range [1, upper_bound]
X = X - jnp.min(X) + 1

# log transform
X = jnp.log(X)

# perform pca
transform, recover, singular_values, W = pca(X)


pcs = [jnp.reshape(w, shape=(a, b, c)) for w in jnp.transpose(W)]


for i, pc in enumerate(pcs):
    # totally unclear to me if passing these headers makes any sense!
    pc_to_nifti = nibabel.Nifti1Image(
        pc,
        affine=nifti_imgs[0].affine,
        header=nifti_imgs[0].header,
    )

    nibabel.save(pc_to_nifti, "out\pc" + f"{i+1}" + ".nii")
