# -*- coding: utf-8 -*-
"""
Remove false positive regions in lung segmentation.
"""
from utils_mri import *
from glob import glob
from skimage import measure


def remove_small_objects(input_path='/*', output_path=None, return_nii=True):
    """
    Remove small objects after lung prediction.
    :param input_path: The folder contains nii format files.
    :param output_path: If you only need to save postprocessed results, keep output_path = None
    :param return_nii: save to nii format.
    :return: None
    """

    niilabel_path = glob(input_path + '/*')
    niilabel_path.sort()

    if output_path == None:
        output_path = input_path

    tag = 1
    for name in niilabel_path:
        nii = get_itk_array(name, False)
        print('matrix shape:\t', nii.shape)  # (301, 512, 512)

        label_matrix = measure.label(nii, 8, return_num=True)


        connected_regions = []
        for k in range(label_matrix[1]):
            connected_regions.append(np.array(label_matrix[0] == [k]).sum())

        connected_regions_sorted = sorted(range(len(connected_regions)),
                                          key=lambda k: connected_regions[k])
        connected_regions_sorted.reverse()

        for _tag, rank in enumerate(connected_regions_sorted):
            if _tag == 2 and connected_regions[rank] < connected_regions[
                connected_regions_sorted.index(1)] * 0.3:
                nii = nii * (1 - np.array(label_matrix[0] == [rank]))
            if _tag > 2:
                nii = nii * (1 - np.array(label_matrix[0] == [rank]))

        if return_nii:
            ref_nii = get_itk_image(name)
            outname = output_path + '/' + name.split('\\')[-1]
            nii = nii.astype(float)
            write_itk_imageArray(nii, outname, ref_nii)
            tag = tag + 1

        tag = tag + 1


if __name__ == '__main__':
    remove_small_objects(input_path='I:\gmycode\Lung_ui\output', output_path='I:\gmycode\Lung_ui\output')
