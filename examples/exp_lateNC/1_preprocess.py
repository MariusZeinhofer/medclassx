from pathlib import Path
import nibabel as nb
from nilearn.image import resample_to_img
from medclassx.mask_trafo import mask_vector

import jax.numpy as jnp
import jax

jax.config.update("jax_enable_x64", True)

out_dir = Path("examples/exp_lateNC/out")

# data HC
paths_hc = [p for p in Path("examples/exp_lateNC/data/HC/").iterdir()]
data_nifti_hc = [nb.load(p) for p in paths_hc]
data_numpy_hc = [n.get_fdata() for n in data_nifti_hc]

# data LATE
paths_late = [p for p in Path("examples/exp_lateNC/data/LATE/").iterdir()]
data_nifti_late = [nb.load(p) for p in paths_late]
data_numpy_late = [n.get_fdata() for n in data_nifti_late]

data_numpy = data_numpy_hc + data_numpy_late

# mask
mask_nifti = nb.load(Path("data/aprinois_nuk_data/mask_for_scanvp.nii"))
mask = resample_to_img(mask_nifti, data_nifti_hc[0], interpolation="nearest")

# make it into a vector mask
mask = mask.get_fdata().reshape(-1)

# save it for later use
jnp.save(out_dir / "mask.npy", mask)

#############################PREPARE DATASET###########################################

print("Mask, shift, log tranform and double center data.")

# convert to a single 4-d jax tensor of shape (batch, a, b, c)
img_data = jnp.array(data_numpy)

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
jnp.save(out_dir / "GMP.npy", jnp.mean(X[0:9, :], axis=0))

# column center
X -= jnp.mean(X, axis=0, keepdims=True)

# save X
jnp.save(out_dir / "X.npy", X)

print(f"Done, data of shape: {X.shape}.")