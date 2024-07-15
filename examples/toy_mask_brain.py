"""Masks a brain with the standard mask and saves as nifti again."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import nibabel
from medclassx.pca import pca

# path to mask
path = Path(r"data\Mask_brain_wo_CSF.nii")
path = Path(r"data\aprinois_nuk_data\mask_for_scanvp.nii")

# load the mask with nibabel
nifti_mask = nibabel.load(path)

# extract the numerical data as uint16 (for the brain mask above)
mask = np.asanyarray(nifti_mask.dataobj)
print(mask.dtype)

print(mask.shape, mask[50:53, 50:53, 50])