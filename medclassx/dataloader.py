"""Conversion from DICOM to numpy ndarray."""

import warnings
from pathlib import Path

import numpy as np
import pydicom as dicom


def get_3d_nparray_from_dicoms(files):
    """Convert DICOM files to numpy array.

    Args:
        files: An iterable of paths to the DICOMS representing the slices.

    Returns:
        A tuple consisting of the 3d numpy array and a list of sorted DICOM slices.
    """
    # traverse directory
    files_dicom = [dicom.dcmread(f) for f in files]
    print(files_dicom[0].SeriesDescription)

    # check for available position in dicom files
    slices = [f for f in files_dicom if hasattr(f, "ImagePositionPatient")]

    # sort the slices
    slices = sorted(slices, key=lambda s: int(s.ImagePositionPatient[2]))

    # warn in cases of missing SliceLocation
    if len(files_dicom) - len(slices) > 0:
        warnings.warn(
            f"Encountered {len(files_dicom) - len(slices)} slices without "
            f"SliceLocation.",
            UserWarning,
        )

    # assemble 3d arry
    img3d = np.zeros(shape=(len(slices), *slices[0].pixel_array.shape))
    for i, s in enumerate(slices):
        img3d[i, :, :] = (s.pixel_array * s.RescaleSlope) + s.RescaleIntercept

    return img3d, slices


if __name__ == "__main__":
    path = Path(r"data\10000002")
    files = []
    for p in path.iterdir():
        files.append(p)

    tensor, slices = get_3d_nparray_from_dicoms(files)
    print(tensor.shape)
    print(type(slices))
