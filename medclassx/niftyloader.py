"""Conversion from DICOM to numpy ndarray."""

import nibabel
import warnings
from pathlib import Path

import numpy as np
import pydicom as dicom





if __name__ == "__main__":
    path = Path(r"data\MCI39\nifti_preproc")
    files = []
    for p in path.iterdir():
        files.append(p)


    nifti_img = nibabel.load(files[2])
    header = nifti_img.header.copy()
    img_data = np.asanyarray(nifti_img.dataobj)
    print(f"data type of loaded image: {img_data.dtype}")


    _nifti_img = nibabel.Nifti1Image(
        img_data,
        affine=nifti_img.affine,
        header=nifti_img.header,
    )

    _img_data = np.asanyarray(_nifti_img.dataobj)

    print(f"max abs diff between img_data and _img_data: {np.max(np.abs(img_data - _img_data))}")
    
    nibabel.save(_nifti_img, r"out\modified_example.nii")
    _nifti_img = nibabel.load(r"out\modified_example.nii")
    #_nifti_img.header = header
    for key in header.keys():
        _nifti_img.header[key] = header[key]
    _header = _nifti_img.header.copy()

    #header_diff = np.any([header[key] != _header[key] for key in header.keys()])
    #print(f"Headers are identical: {not header_diff}")

    _img_data = np.asanyarray(_nifti_img.dataobj)

    print(f"max abs diff between img_data and _img_data: {np.max(np.abs(img_data - _img_data))}")
    exit()

    # Save the modified NIfTI image to a new file
    nibabel.save(modified_nifti_img, r"out\modified_example.nii")

    new_nifti_image = nibabel.load(r"out\modified_example.nii")
    new_img_data = np.asanyarray(new_nifti_image.dataobj)
    diff = new_img_data - img_data
    print(np.max(img_data), np.min(img_data))
    print(np.max(np.abs(diff)))
    print((np.mean(diff**2))**0.5)