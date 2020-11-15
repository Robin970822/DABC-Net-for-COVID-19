# -*- coding: utf-8 -*-
"""
去除肺分割中假阳性

"""
from utils_mri import *
from glob import glob
from skimage import measure, morphology


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

    if output_path is None:
        output_path = input_path

    tag = 1
    # lung =[]
    for name in niilabel_path:
        nii = get_itk_array(name, False)  # 标签不能标准化！！！
        # nii_raw = get_itk_image(name)
        print('matrix shape:\t', nii.shape)  # (301, 512, 512)
        # nii = np.flip(nii,axis=0)  # 前10个似乎是反的

        # slices = nii.shape[0]-20-20  # 301-20-20=261 去掉首未层 ,等价于下面两行
        # nii = nii[20:,:,:]
        # nii = nii[:-20,:,:]

        # nii[nii>=1] = 1  # whole lung
        # nii[nii!=3] = 0  # left lung
        # nii[nii>0] = 1

        # # 左右肺中病灶
        # width = nii.shape[-1]
        # nii = nii[:, :, int(width/2):]  # rigth lung
        # # nii = nii[:, :, 0:int(width/2)]  # left lung

        """
        method 1
        """
        # # 连通域
        # nii = nii.astype(bool)
        # slices = nii.shape[0]  # 注意这里并非slice，而是z轴
        # for i in range(slices):
        #     morphology.remove_small_holes(nii[i,:,:], area_threshold=np.sin(i/slices*3.14)*4000, in_place=True)
        #     morphology.remove_small_objects(nii[i,:,:], min_size=int(np.sin(i/slices*3.14)*1000), connectivity=1, in_place=True)  # 2:8临接,1:4临接
        # # dst = sm.closing(nii, sm.disk(9))  # 在3D图像上闭运算未找到合适的核

        """
        method 2
        """
        label_matrix = measure.label(nii, 8, return_num=True)
        # 2d,3d 都行 对各个连通域用数字进行标记，返回一个元组：标记矩阵和连通域个数
        # label2 = np.array(label[0])  # 最大值即为连通域个数

        connected_regions = []  # connected_regions
        for k in range(label_matrix[1]):
            # connected_regions.append(label_matrix[0]==[k])  # 把不同连通域取出来，都是二值图
            connected_regions.append(np.array(label_matrix[0] == [k]).sum())  # 把不同连通域 大小算出来，放在列表里

        # connected_regions.sort()
        connected_regions_sorted = sorted(range(len(connected_regions)),
                                          key=lambda k: connected_regions[k])  # 返回排序列表的索引（升序）
        connected_regions_sorted.reverse()  # 降序

        for _tag, rank in enumerate(connected_regions_sorted):  # 误区第一/最大 连通域不是肺，是背景！故_tag要用低三大的比较第二大的！
            # print(num)
            # 比较第二大*0.3
            if _tag == 2 and connected_regions[rank] < connected_regions[connected_regions_sorted.index(1)] * 0.3:
                nii = nii * (1 - np.array(label_matrix[0] == [rank]))
            if _tag > 2:
                nii = nii * (1 - np.array(label_matrix[0] == [rank]))

        if return_nii:
            ref_nii = get_itk_image(name)
            output_name = output_path + '/' + name.split('\\')[-1]
            nii = nii.astype(float)
            write_itk_imageArray(nii, output_name, ref_nii)
            tag = tag + 1
            # continue

        tag = tag + 1


remove_small_objects(input_path='I:\gmycode\Lung_ui\output', output_path='I:\gmycode\Lung_ui\output')
