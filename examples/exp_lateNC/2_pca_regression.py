import json
from pathlib import Path

import jax.numpy as jnp
import nibabel
from medclassx.binary_pca_regression import binary_pca_regression
from medclassx.mask_trafo import mask_vector
import jax

jax.config.update("jax_enable_x64", True)

outfolder = Path("examples/exp_lateNC/out")

print("Compute PCA regression.")

X = jnp.load(outfolder / "X.npy")

# the first 9 are healthy controls
X_con = X[0:9]

# the remaining 9 are LATE patients
X_pat = X[9:]

# compute the pca regression
results, T = binary_pca_regression(X_control=X_con, X_patient=X_pat, cut_off=18)

# save the transformed dataset
jnp.save(outfolder / "T.npy", T)

print("Computed and saved PCA.")

##################################SAVE RESULTS#########################################

# load one image to retrieve the following data (how to do that smarter?)
# path to mask
_mask_path = outfolder / "mask_resampled.nii"
_nifti_mask = nibabel.load(_mask_path)
_mask = _nifti_mask.get_fdata()
affine = _nifti_mask.affine  # retrieving from mask using for pcs...ok?
header = _nifti_mask.header  # retrieving from mask using for pcs...ok?

mask = jnp.load(outfolder / "mask.npy")
unmask = mask_vector(mask, mask)[1]
a, b, c = _mask.shape

# retrieve the principle components from the results list
pcs = [result["pc"] for result in results]

# get the principle components back to unmasked space
pcs = [unmask(pc) for pc in pcs]

# reshape back to 3d space
pcs = [jnp.reshape(pc, shape=(a, b, c)) for pc in pcs]

# save as niftis for visual inspection
for i, pc in enumerate(pcs):
    pc_to_nifti = nibabel.Nifti1Image(
        pc,
        affine=affine,
        header=header,
    )

    nibabel.save(
        pc_to_nifti,
        outfolder / f"pc_{i+1}.nii",
    )

# repeat to save the z-score normalized principle components
# retrieve the normalized principle components from the results list
pcs_zscore = [d["pcz"] for d in results]

# get the principle components back to unmasked space
pcs_zscore = [unmask(pcz) for pcz in pcs_zscore]

# reshape back to 3d space
pcs_zscore = [jnp.reshape(pcz, shape=(a, b, c)) for pcz in pcs_zscore]

# save as niftis for visual inspection
for i, pcz in enumerate(pcs_zscore):
    # totally unclear to me if passing these headers makes any sense!
    pcz_to_nifti = nibabel.Nifti1Image(
        pcz,
        affine=affine,
        header=header,
    )

    nibabel.save(
        pcz_to_nifti,
        outfolder / f"zscore_pc_{i+1}.nii",
    )

# save principle components and metadata as json file
for i, d in enumerate(results):
    # save only the metadata in the json files
    d_ser = {}
    for k, v in d.items():
        if not (k == "pc" or k == "pcz"):
            if hasattr(v, "tolist"):
                d_ser[k] = v.tolist()
            else:
                d_ser[k] = v

    with open(outfolder / f"pc_{i+1}.json", "w") as j_file:
        json.dump(d_ser, j_file)
