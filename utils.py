import numpy as np
import SimpleITK as itk


def prob2binary(prob, thres=0.5):
    res = np.zeros_like(prob)
    res[prob > 0.5] = 1
    return res


def get_itk_array(filenameOrImage):
    reader = itk.ImageFileReader()
    reader.SetFileName(filenameOrImage)
    image = reader.Execute()
    imageArray = itk.GetArrayFromImage(image)  # (slices, length, height)
    spacing = image.GetSpacing()
    return imageArray, spacing
