import numpy as np
import SimpleITK as itk

from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.stats import ttest_ind, f

label_font = {'family': 'Times New Roman', 'size': 24}


def crop_volume(volume, crop):
    volume_ = volume.copy()
    volume_[:crop[0]] = 0
    volume_[-crop[1]:] = 0
    return volume_


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


def make_itk_image(imageArray, protoImage=None):
    image = itk.GetImageFromArray(imageArray)
    if protoImage is not None:
        image.CopyInformation(protoImage)

    return image


def write_itk_imageArray(imageArray, filename, src_nii=None):
    img = make_itk_image(imageArray, src_nii)
    write_itk_image(img, filename)


def write_itk_image(image, filename):
    writer = itk.ImageFileWriter()
    writer.SetFileName(filename)

    if filename.endswith('.nii'):
        Warning('You are converting nii, be careful with type conversions')

    writer.Execute(image)
    return


def ftest(s1, s2):
    '''F检验样本总体方差是否相等'''
    F = np.var(s1) / np.var(s2)
    v1 = len(s1) - 1
    v2 = len(s2) - 1
    p_val = 1 - 2 * abs(0.5 - f.cdf(F, v1, v2))
    print(p_val)
    if p_val < 0.05:
        print('Reject the Null Hypothesis.')
        equal_var = False
    else:
        print('Accept the Null Hypothesis.')
        equal_var = True
    return equal_var


def ttest_ind_func(s1, s2):
    '''t检验独立样本所需的两个总体均值是否存在差异'''
    equal_var = ftest(s1, s2)
    print('Null Hypothesis: mean(s1) = mean(s2), α=0.5')
    ttest, pval = ttest_ind(s1, s2, equal_var=equal_var)
    if pval < 0.05:
        print("Reject the Null Hypothesis.")
    else:
        print("Accept the Null Hypothesis.")
    return ttest, pval


def formatnum(x, pos):
    if x == 0:
        return 0
    return '$%.1f$x$10^{5}$' % (x / 1e5)


formatter = FuncFormatter(formatnum)


def box_plot_dict(plot_data, ylabel, formatter=None):
    fig, ax = plt.subplots(figsize=(16, 9))
    p = 0
    for (k, v) in plot_data.items():
        ax.boxplot(v, positions=[p], labels=[k], showfliers=False)
        ax.scatter(p * np.ones_like(v) +
                   (np.random.random(len(v)) - 0.5) * 0.1, v, alpha=0.2)
        p = p + 1
    ax.set_ylabel(ylabel, label_font)
    plt.tick_params(labelsize=16)
    if formatter:
        ax = plt.gca()
        ax.yaxis.set_major_formatter(formatter)
