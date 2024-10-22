"""Performs PCA-regression on the PSP-RS and healthy control dataset.

This script produces and saves the following files:

- the first `cut_off` principle components in image space. That means
    it unmasks the files and transforms them back to volumetric format and saves
    them in the nifti format. This is done both for the z-score normalized PCs and
    for the unnormalized ones. These files are for visual inspection mainly.

- For every principle component it saves a json file containing the "pc_number",
    the singular value "sv" the variance accounted for "vaf" (singular_value/sum of
    singular values) and the "accuracy".
    Keys for the json: {"pc_number", "sv", "vaf", "accuracy"}.
"""

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import nibabel
from medclassx.binary_pca_regression import binary_pca_regression
from medclassx.mask_trafo import mask_vector

parser = argparse.ArgumentParser(description="Provide input and ouput data locations.")

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


print("Compute PCA regression.")

X = jnp.load(Path(args.out_folder) / "X.npy")

# the first 30 are healthy controls
X_con = X[0:30]

# the remaining 30 are PSP patients
X_pat = X[30:]

# compute the pca regression
results, T = binary_pca_regression(X_control=X_con, X_patient=X_pat, cut_off=20)

print("Done, save results.")

##################################SAVE RESULTS#########################################

# load one image to retrieve the following data (how to do that smarter?)
# path to mask
_mask_path = Path(args.mask_path)
_nifti_mask = nibabel.load(_mask_path)
_mask = _nifti_mask.get_fdata()
affine = _nifti_mask.affine  # retrieving from mask using for pcs...ok?
header = _nifti_mask.header  # retrieving from mask using for pcs...ok?

mask = jnp.load(Path(args.out_folder) / "mask.npy")
unmask = mask_vector(mask, mask)[1]
a, b, c = _mask.shape

# save the transformed dataset
jnp.save(Path(args.out_folder) / "T.npy", T)

# retrieve the principle components from the results list
pcs = [result["pc"] for result in results]

# get the principle components back to unmasked space
pcs = [unmask(pc) for pc in pcs]

# reshape back to 3d space
pcs = [jnp.reshape(pc, shape=(a, b, c)) for pc in pcs]

# save as niftis for visual inspection
for i, pc in enumerate(pcs):
    # totally unclear to me if passing these headers makes any sense!
    pc_to_nifti = nibabel.Nifti1Image(
        pc,
        affine=affine,
        header=header,
    )

    nibabel.save(
        pc_to_nifti,
        Path(args.out_folder) / f"pc_{i+1}.nii",
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
        Path(args.out_folder) / f"zscore_pc_{i+1}.nii",
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

    with open(Path(args.out_folder) / f"pc_{i+1}.json", "w") as j_file:
        json.dump(d_ser, j_file)
