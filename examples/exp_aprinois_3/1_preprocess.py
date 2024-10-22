"""Preprocesses the PSP-RS and healthy control dataset.

This script log-transforms, masks and normalizes the dataset. It saves:

- The dataset X in masked and matrix form, i.e., X is of shape (n, p) where n denotes
    the number of subjects and p denotes the number of voxels that are not removed
    by the mask.

- The mask as a flattened vector in npy format.

- The group mean profile (GMP) of the healthy control group. This row-centers the
    healthy controls and takes the mean over the healthy control.
"""

import argparse
from pathlib import Path

import jax.numpy as jnp
import nibabel
from medclassx.mask_trafo import mask_vector

#########################PREPARE DATA ACCESS###########################################
# assemble a list that contains all paths to the data

parser = argparse.ArgumentParser(description="Provide input and ouput data locations.")

parser.add_argument(
    "--hc_folder",
    type=str,
    help="Path to the healthy control folder",
    default="data/HC",
)
parser.add_argument(
    "--ad_folder",
    type=str,
    help="Path to the AD folder",
    default="data/AD",
)
parser.add_argument(
    "--psp_folder",
    type=str,
    help="Path to the PSP folder",
    default="data/PSP",
)
parser.add_argument(
    "--mask_path",
    type=str,
    help="Path to the mask",
    default="data/aprinois_nuk_data/mask_for_scanvp.nii",
)
parser.add_argument(
    "--out_folder",
    type=str,
    help="Folder to store output",
    default="examples/exp_aprinois_3/out",
)

args = parser.parse_args()

# this comes from command line arguments
hc_dir = Path(args.hc_folder)
ad_dir = Path(args.ad_folder)
psp_dir = Path(args.psp_folder)
mask_path = Path(args.mask_path)
out_dir = Path(args.out_folder)

# get the actual paths from the directory info
hcs_paths = [p for p in hc_dir.iterdir()]
ad_paths = [p for p in ad_dir.iterdir()]
psp_paths = [p for p in psp_dir.iterdir()]

# collect all data paths in a list
paths = hcs_paths + psp_paths

#############################PREPARE MASKING###########################################

print("Load mask.")

# load the mask with nibabel
nifti_mask = nibabel.load(mask_path)

# extract the numerical data
mask = nifti_mask.get_fdata()

# make it into a vector mask
mask = mask.reshape(-1)

# save it for later use
jnp.save(out_dir / "mask.npy", mask)

#############################PREPARE DATASET###########################################

print("Mask, shift, log tranform and double center data.")

# load the images with nibabel
nifti_imgs = [nibabel.load(p) for p in paths]

# extract the numerical data as float64
img_data = [n.get_fdata() for n in nifti_imgs]

# convert to a single 4-d jax tensor of shape (batch, a, b, c)
img_data = jnp.array(img_data)

# data shapes
n, a, b, c = jnp.shape(img_data)

# Reshape into matrix form
X = jnp.reshape(img_data, shape=(n, a * b * c))

# prepare unmasking
unmask = mask_vector(X[0], mask)[1]

# mask the data
X = jnp.array([mask_vector(x, mask)[0] for x in X])

# shift data to the range [1, upper_bound]
X = X - jnp.min(X) + 1

# log transform
X = jnp.log(X)

# row center
X -= jnp.mean(X, axis=1, keepdims=True)

# save group mean profile of healthy group
jnp.save(out_dir / "GMP.npy", jnp.mean(X[0:30, :], axis=0))

# column center
X -= jnp.mean(X, axis=0, keepdims=True)

# save X
jnp.save(out_dir / "X.npy", X)

print(f"Done, data of shape: {X.shape}.")
