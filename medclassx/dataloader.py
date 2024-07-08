import pydicom as dicom
import numpy as np

def get_3d_nparray_from_dicoms(files):
    # skip files with no SliceLocation (eg scout views)
    files_dicom = []
    for i, f in enumerate(files):
        files_dicom.append(dicom.dcmread(f))
        if i == 0:
            print(files_dicom[i].SeriesDescription)


    slices = []
    skipcount = 0
    for f in files_dicom:
        
        # if hasattr(f, 'SliceLocation'): # some exports dont have field SliceLocation
        if hasattr(f, 'ImagePositionPatient'):
            slices.append(f)
        else:
            skipcount = skipcount + 1
    
    print("skipped, no SliceLocation: {}".format(skipcount))
    # ensure they are in the correct order
    # slices = sorted(slices, key=lambda s: s.SliceLocation) # some exports dont have field SliceLocation
    slices = sorted(slices, key=lambda s: int(s.ImagePositionPatient[2]))

    # pixel aspects, assuming all slices are the same
    ps = slices[0].PixelSpacing
    ss = slices[0].SliceThickness
    ax_aspect = ps[1]/ps[0]
    sag_aspect = ps[1]/ss
    cor_aspect = ss/ps[0]
    # create 3D array
    img_shape = list(slices[0].pixel_array.shape)
    img_shape.append(len(slices))
    # img_shape.reverse()
    img_shape = [img_shape[2],img_shape[0],img_shape[1]]
    img3d = np.zeros(img_shape)
    # img3d = np.zeros(img_shape,dtype='ushort')
    
    # fill 3D array with the images from the files
    for i, s in enumerate(slices):
        array2d = s.pixel_array
        if sum(sum(array2d<0)) > 0:
            print('sum of negative elements in array:',sum(sum(array2d<0)))
        # print(s.pixel_array.max,s.RescaleSlope)
        img2d = (array2d * s.RescaleSlope) + s.RescaleIntercept
        
        img3d[i, :, :] = img2d

    # img4d = [img3d] # required to simulate a color channel for the CBS scripts, they shrink the array by 1 dimension
    # img4d = np.array([img3d,img3d,img3d])
    return img3d, slices