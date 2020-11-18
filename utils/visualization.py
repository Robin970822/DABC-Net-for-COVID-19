import cv2
import matplotlib
import numpy as np
import pandas as pd
import nibabel as nib
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from glob import glob
from .calculate_feature import get_days
from .utils_mri import get_itk_array


def transp_imshow(data, tvmin=None, tvmax=None, tmax=1.,
                  gam=1., cmap='Blues', **kwargs):
    """
    Displays the 2d array `data` with pixel-dependent transparency.
    Parameters
    ----------
    data: 2d numpy array of floats or ints
        Contains the data to be plotted as a 2d map
    tvmin, tvmax: floats or None, optional
        The values (for the elements of `data`) that will be plotted
        with minimum opacity and maximum opacity, respectively.
        If no value is provided, this uses by default the arguments
        `vmin` and `vmax` of `imshow`, or the min and max of `data`.
    tmax: float, optional
        Value between 0 and 1. Maximum opacity, which is reached
        for pixel that have a value greater or equal to `tvmax`.
        Default: 1.
    gam: float, optional
        Distortion of the opacity with pixel-value.
        For `gam` = 1, the opacity varies linearly with pixel-value
        For `gam` < 1, low values have higher-than-linear opacity
        For `gam` > 1, low values have lower-than-linear opacity
    cmap: a string or a maplotlib.colors.Colormap object
        Colormap to be used
    kwargs: dict
        Optional arguments, which are passed to matplotlib's `imshow`.
    """
    # Determine the values between which the transparency will be scaled
    if 'vmax' in kwargs:
        vmax = kwargs['vmax']
    else:
        vmax = data.max()
    if 'vmin' in kwargs:
        vmin = kwargs['vmin']
    else:
        vmin = data.min()
    if tvmax is None:
        tvmax = vmax
    if tvmin is None:
        tvmin = vmin

    # Rescale the data to get the transparency and color
    color = (data - vmin) / (vmax - vmin)
    color[color > 1.] = 1.
    color[color < 0.] = 0.
    transparency = tmax * (data - tvmin) / (tvmax - tvmin)
    transparency[transparency > 1.] = 1
    transparency[transparency < 0.] = 0.
    # Application of a gamma distortion
    transparency = tmax * transparency ** gam

    # Get the colormap
    if isinstance(cmap, matplotlib.colors.Colormap):
        colormap = cmap
    elif type(cmap) == str:
        colormap = getattr(plt.cm, cmap)
    else:
        raise ValueError('Invalid type for argument `cmap`.')

    # Create an rgba stack of the data, using the colormap
    rgba_data = colormap(color)
    # Modify the transparency
    rgba_data[:, :, 3] = transparency

    sc = plt.imshow(rgba_data, **kwargs)

    # test
    # plt.colorbar(sc)
    return sc


def data_disease_slice(patientID, slice_id):
    """
    Load slice at slice_id of each CT scan and segmentation and compute lesion-lung ratio
    Parameters
    ----------
    patientID:
    slice_id:
    """

    raw_list = glob(r'{}/*nii*'.format(patientID))
    lung_list = glob(r'{}_output/lung/*'.format(patientID))
    covid_list = glob(r'{}_output/covid/*'.format(patientID))

    raw_list.sort()
    lung_list.sort()
    covid_list.sort()

    timepoint_count = len(raw_list)
    raw = np.zeros((timepoint_count, 512, 512))
    lesion = np.zeros((timepoint_count, 512, 512))
    lung = np.zeros((timepoint_count, 512, 512))

    lesion_volume = []
    lung_volume = []

    for i, name in enumerate(raw_list):
        raw[i] = np.flip(get_itk_array(name).astype('float32')[slice_id[i]], axis=0)
    raw[raw < -1024] = -1024
    raw[raw > 512] = 512

    for i, name in enumerate(covid_list):
        nii = get_itk_array(name).astype('float32')
        lesion_volume.append(nii.sum())
        lesion[i] = np.flip(nii[slice_id[i]], axis=0)

    for i, name in enumerate(lung_list):
        nii = get_itk_array(name).astype('float32')
        lung_volume.append(nii.sum())
        lung[i] = np.flip(nii[slice_id[i]], axis=0)

    return raw, lung, lesion, np.array(lesion_volume) / np.array(lung_volume)


def plot_segmentation_montage(raw, lung, lesion, state, patient_num=1, color_map='Reds'):
    """
    Displays the segmentation results (montage).
    Parameters
    ----------
    raw: ndarray: shape like:(8, 512, 512) or list: [(8, 512, 512),(6, 512, 512)]
    lung: segmentation results. Shape like:(8, 512, 512)
    lesion:
    state:
    patient_num: int. 1 or 2.
    color_map:
    """

    raw = raw - np.min(raw)
    raw = raw * 1.0 / np.max(raw)

    row1 = np.column_stack(raw)
    # plt.imshow(row1,cmap='gray')
    row2 = np.column_stack(lesion)
    canvas = np.vstack((row1,row2))
    plt.imshow(canvas, cmap='gray')


def plot_segmentation(raw, lung, lesion, color_map, state, hspace=-0.6):
    """
    Displays the segmentation results.
    Parameters
    ----------
    raw:
    lung:
    lesion:
    color_map:
    state:
    hspace: float, optional. The height of the padding between subplots, as a fraction of the average axes height.
    """
    fig = plt.figure(figsize=(16, 9))

    timepoint_count = raw.shape[0]

    for i in range(timepoint_count):
        plt.subplot(3, timepoint_count, i + 1)
        plt.imshow(raw[i], cmap='gray')
        plt.title('No.{} scan\n'.format(i + 1), fontsize=16)
        plt.xticks([]), plt.yticks([])

    for i in range(timepoint_count):
        plt.subplot(3, timepoint_count, timepoint_count + i + 1)
        plt.imshow(raw[i], cmap='gray')
        transp_imshow(lung[i], cmap='Greens', alpha=0.7)
        # plt.title('No.{} scan lung\n'.format(i + 1), fontsize=16)
        plt.xticks([]), plt.yticks([])

    for i in range(timepoint_count):
        plt.subplot(3, timepoint_count, timepoint_count * 2 + i + 1)
        plt.imshow(raw[i], cmap='gray')
        transp_imshow(lesion[i], cmap=color_map, alpha=0.7)
        # plt.title('No.{} scan lesion\n'.format(i + 1), fontsize=16)
        plt.xticks([]), plt.yticks([])

    plt.subplots_adjust(hspace=hspace,wspace=0.0)
    # plt.tight_layout()
    fig.suptitle('Progress of {} patient in longitudinal study'.format(state), fontsize=26)
    plt.show()


def plot_uncertainty(name_id='2020035365_0204_3050_20200204184413_4.nii.gz', patientID='2020035365', slice_id=175,
                     sform_code=1):
    """
    Displays the uncertainty results.
    Parameters
    ----------
    name_id:
    patientID:
    slice_id:
    sform_code:
    """

    rawimg = nib.load(r'{}/'.format(patientID) + name_id).get_fdata()
    aleatoric = nib.load(r'{}_output/uncertainty/'.format(patientID) + 'aleatoric_' + name_id).get_fdata()
    epistemic = nib.load(r'{}_output/uncertainty/'.format(patientID) + 'epistemic_' + name_id).get_fdata()

    slices_num = rawimg.shape[-1]

    our = nib.load(r'{}_output/covid/'.format(patientID) + name_id).get_fdata()

    our = our[:, :, slice_id]

    rawimg = rawimg[:, :, slice_id]
    rawimg[rawimg < -1024] = -1024
    rawimg[rawimg > 255] = 255
    # gt = gt[:,:,slice_id]
    aleatoric = aleatoric[:, :, slice_id]
    epistemic = epistemic[:, :, slice_id]

    # sform_code==1:rot90,1. else:rot90,-1
    if sform_code == 1:
        rotate = 1
        aleatoric = np.rot90(aleatoric, rotate)
        epistemic = np.rot90(epistemic, rotate)
        rawimg = np.rot90(rawimg, rotate)
        our = np.rot90(our, rotate)
    else:
        rotate = -1
        aleatoric = np.rot90(aleatoric, rotate)
        aleatoric = cv2.flip(aleatoric, 1)
        epistemic = np.rot90(epistemic, rotate)
        epistemic = cv2.flip(epistemic, 1)
        rawimg = np.rot90(rawimg, rotate)
        rawimg = cv2.flip(rawimg, 1)
        our = np.rot90(our, rotate)
        our = cv2.flip(our, 1)

    def overlay(_src, _pred, _gt, need_crop=True, need_save=False,
                need_overlay=True, need_overlay_alea=None, need_overlay_epis=False, aleatoric=None, epistemic=None,
                need_overlay_alea_scale=False):
        # need_save:'img_File_name'
        rawimg = _src
        prediction = _pred
        gt = _gt

        if need_crop:  # (row1,row2,c1,c2)
            if 'cor' in str(need_save):
                need_crop = (30, 350, 20, -20)
                rawimg = rawimg[need_crop[0]:need_crop[1], need_crop[2]:need_crop[3]]
                prediction = prediction[need_crop[0]:need_crop[1], need_crop[2]:need_crop[3]]
                gt = gt[need_crop[0]:need_crop[1], need_crop[2]:need_crop[3]]
            if 'radio' in str(need_save):
                need_crop = (90, 570, 0, -1)
                rawimg = rawimg[need_crop[0]:need_crop[1], need_crop[2]:need_crop[3]]
                prediction = prediction[need_crop[0]:need_crop[1], need_crop[2]:need_crop[3]]
                gt = gt[need_crop[0]:need_crop[1], need_crop[2]:need_crop[3]]

        plt.imshow(rawimg, cmap='gray', )
        if need_overlay:
            TP = prediction * gt
            FP = prediction * (np.ones_like(gt) - gt)
            FN = (1 - prediction) * gt
            transp_imshow(TP, cmap='RdYlGn', alpha=0.7)
            transp_imshow(FP, cmap='cool', alpha=0.7)  #
            transp_imshow(FN, cmap='Wistia', alpha=0.7)
        if need_overlay_epis:
            if need_crop:
                epistemic = epistemic[need_crop[0]:need_crop[1], need_crop[2]:need_crop[3]]
            plt.imshow(rawimg, cmap='gray', )
            transp_imshow(epistemic, cmap='autumn_r', alpha=1.0)
        if need_overlay_alea:
            if need_crop:
                aleatoric = aleatoric[need_crop[0]:need_crop[1], need_crop[2]:need_crop[3]]
            plt.imshow(rawimg, cmap='gray', )

            #         print(need_overlay_alea)
            #         print('vmin: ',0.5)
            transp_imshow(aleatoric, cmap='winter',

                          )

        plt.axis('off')
        # plt.xticks([])
        # plt.yticks([])
        if need_save:  # False or file string
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())

            plt.savefig(need_save,
                        bbox_inches='tight',
                        dpi=300, pad_inches=0.0)
        # plt.show()

    print('Slice: {0}/{1}'.format(slice_id, slices_num))
    plt.subplots(figsize=(16, 9))
    plt.subplot(1, 4, 1)
    plt.title('Raw image:')
    plt.imshow(rawimg, cmap='gray')
    plt.xticks([]), plt.yticks([])

    plt.subplot(1, 4, 2)
    plt.title('Lesion segmentation:')
    plt.imshow(rawimg, cmap='gray')
    transp_imshow(our, cmap='Reds', alpha=0.7)
    plt.xticks([]), plt.yticks([])

    plt.subplot(1, 4, 3)
    plt.title('Aleatoric uncertainty:')
    overlay(rawimg, np.zeros_like(rawimg), np.zeros_like(rawimg), need_crop=False, need_overlay=False,
            aleatoric=aleatoric, need_overlay_alea=True,
            need_save='{}_output/'.format(patientID) + name_id + str(slice_id) + '_Uc_alea_.png')
    plt.subplot(1, 4, 4)
    plt.title('Epistemic uncertainty:')
    overlay(rawimg, np.zeros_like(rawimg), np.zeros_like(rawimg), need_crop=False, need_overlay=False,
            epistemic=epistemic, need_overlay_epis=True,
            need_save='{}_output/'.format(patientID) + name_id + str(slice_id) + '_Uc_epis_.png')


def plot_progress_curve(all_info, patientID, line_color=sns.color_palette('Reds')[5], label='Severe'):
    """
    Displays the progress curve for patients.
    Parameters
    ----------
    all_info:
    patientID:
    line_color:
    label:
    """
    all_info['date'] = (pd.to_datetime(all_info['StudyDate']) - pd.to_datetime(all_info['StudyDate']).iloc[0]).map(
        get_days)

    colors = [sns.color_palette('Greens')[2], sns.color_palette('Reds')[4]]

    plt.plot(all_info['date'], all_info['ratio'], color=line_color, linestyle='-',
             label='{} {}: Lesion Ratio'.format(label, patientID),
             alpha=0.4)
    mild_info = all_info[all_info['Severe'] == 0]
    plt.scatter(mild_info['date'], mild_info['ratio'], color=colors[0], marker='o', s=100, alpha=1.0)
    severe_info = all_info[all_info['Severe'] > 0]
    plt.scatter(severe_info['date'], severe_info['ratio'], color=colors[1], marker='^', s=100, alpha=1.0)


def data_disease_progress_slice(all_info, patientID, slice_id, timepoint_count):
    """
    Load slice at slice_id of each CT scan and segmentation.
    Parameters
    ----------
    all_info:
    patientID:
    slice_id:
    timepoint_count:
    """
    gt = np.array(all_info['Severe'])

    raw_list = glob(r'{}/*nii*'.format(patientID))
    covid_list = glob(r'{}_output/covid/*'.format(patientID))

    raw_list.sort()
    covid_list.sort()

    raw = np.zeros((timepoint_count, 512, 512))
    lesion = np.zeros((timepoint_count, 512, 512))

    for i, name in enumerate(raw_list):
        raw[i] = np.flip(get_itk_array(name).astype('float32')[slice_id[i]], axis=0)
    raw[raw < -1024] = -1024
    raw[raw > 512] = 512

    for i, name in enumerate(covid_list):
        lesion[i] = np.flip(get_itk_array(name).astype('float32')[slice_id[i]], axis=0)

    return raw, lesion, gt


def plot_progress(raw, lesion, p, gt, color_map='Reds', state='severe', timepoint_count=8):
    """
    Display the disease progression.
    Parameters
    ----------
    raw:
    lesion:
    p:
    gt:
    color_map:
    state:
    timepoint_count:
    """
    fig = plt.figure(figsize=(30, 9))

    for i in range(timepoint_count):
        plt.subplot(2, timepoint_count, i + 1)
        plt.imshow(raw[i], cmap='gray')
        plt.title('No.{} scan\n'.format(i + 1), fontsize=16)
        plt.xticks([]), plt.yticks([])

    for i in range(timepoint_count):
        plt.subplot(2, timepoint_count, timepoint_count + i + 1)
        plt.imshow(raw[i], cmap='gray')
        # plt.imshow(lesion[i], alpha=0.5, cmap=color_map)
        transp_imshow(lesion[i], cmap=color_map, alpha=0.7)
        plt.title('Prediction:{}\nGround Truth:{}'.format(round(p[i], 3), round(gt[i], 3)), fontsize=16)
        plt.xticks([]), plt.yticks([])

    fig.suptitle('Progress of {} patient in longitudinal study'.format(state), fontsize=26)
    plt.show()

'''

'''
def plot_fetures(all_info_severe, all_info_mild, save_to_html=False):
    """
    Display the features.
    Parameters
    ----------
    all_info_severe:
    all_info_mild:
    save_to_html: e.g. results.html
    """
    # all_info = pd.read_csv('all_info.csv')
    all_info_severe['date'] = (pd.to_datetime(all_info_severe['StudyDate']) - pd.to_datetime(all_info_severe['StudyDate']).iloc[0]).map(
        get_days)
    # all_info_mild = pd.read_csv('all_info_mild.csv')
    all_info_mild['date'] = (pd.to_datetime(all_info_mild['StudyDate']) - pd.to_datetime(all_info_mild['StudyDate']).iloc[0]).map(
        get_days)

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        specs=[[{"type": "table"}],
               [{"type": "table"}],
               [{"type": "scatter"}],
               [{"type": "scatter"}]]
    )

    # line plot - mild
    fig.add_trace(
        go.Scatter(
            x=all_info_mild["date"],
            y=all_info_mild["ratio"],
            mode="lines",
            name="Mild-ratio",
    #         fillcolor='Green',
    #         marker={mark.color:'Green'},
            marker={'color':'green','opacity':0.3,'size':30}
        ),
        row=3, col=1
    )

    # line plot - severe
    fig.add_trace(
        go.Scatter(
            x=all_info_severe["date"],
            y=all_info_severe["ratio"],
            mode="lines",
            name="Severe-ratio",  # mouse hover event trace
            marker={'color':'red','opacity':0.3,'size':30}

        ),
        row=3, col=1
    )

    # tabel - mild
    fig.add_trace(
        go.Table(
            header=dict(
                values=['patient<br>ID', 'slice', 'spacing',
                        'shape', 'Patient<br>Sex', 'Study<br>Date', 'Inst.<br>Name',  # Inst.
                        'Age', 'Severe', 'lung', 'lesion', 'ratio', 'lung<br>lesion', 'left<br>lung',
                        'right<br>lung', 'left<br>lesion', 'right<br>lesion', 'left<br>ratio',
                        'right<br>ratio', 'weighted<br>lesion', 'weighted<br>lung<br>lesion',
                        'left<br>weighted<br>lesion', 'right<br>weighted<br>lesion', 'consolidation',
                        'lesion<br>consolidation', 'left<br>consolidation', 'right<br>consolidation',
                        'z', 'left<br>z', 'right<br>z'],
                font=dict(size=10),
                align="left"
            ),
            cells=dict(
                values=[all_info_mild[k].tolist() for k in all_info_mild.columns[1:] if k not in ['index', 'filename', 'Manufacturer']],
                align="left")
        ),
        row=2, col=1
    )

    # tabel - severe
    fig.add_trace(
        go.Table(
            header=dict(
                values=['patient<br>ID', 'slice', 'spacing',
                        'shape', 'Patient<br>Sex', 'Study<br>Date', 'Inst.<br>Name',  # Inst.
                        'Age', 'Severe', 'lung', 'lesion', 'ratio', 'lung<br>lesion', 'left<br>lung',
                        'right<br>lung', 'left<br>lesion', 'right<br>lesion', 'left<br>ratio',
                        'right<br>ratio', 'weighted<br>lesion', 'weighted<br>lung<br>lesion',
                        'left<br>weighted<br>lesion', 'right<br>weighted<br>lesion', 'consolidation',
                        'lesion<br>consolidation', 'left<br>consolidation', 'right<br>consolidation',
                        'z', 'left<br>z', 'right<br>z'],
                font=dict(size=10),
                align="left"
            ),
            cells=dict(
                values=[all_info_severe[k].tolist() for k in all_info_severe.columns[1:] if k not in ['index', 'filename', 'Manufacturer']],
                align="left")
        ),
        row=1, col=1
    )

    fig.update_layout(
        height=800,
        showlegend=False,
        title_text="Severe(red) and mild(green) patient progress curve",
    )

    fig.show()

    if save_to_html:
        fig.write_html('Progress curve.html')


def plot_animation_curve(all_info, save_to_html=False):
    """
    Display the animation of curve.
    Parameters
    ----------
    all_info: feature of severe or mild patient
    save_to_html: e.g. results.html
    """
    x = all_info["date"]
    y = all_info["ratio"]
    fig = go.Figure(
        data=[go.Scatter(x=[0, 0], y=[0, 0])],  # Start point
        layout=go.Layout(
            xaxis=dict(range=[0, 50], autorange=False),
            yaxis=dict(range=[0, 0.15], autorange=False),
            title="Click button to display animation of progress curve",
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="Display",
                              method="animate",
                              args=[None])])]
        ),

        frames=[go.Frame(data=[
            go.Scatter(x=x[:1], y=y[:1], marker={'color': 'green', 'opacity': 0.3, 'size': 10})]),
                go.Frame(data=[go.Scatter(x=x[:2], y=y[:2],
                                          marker={'color': 'green', 'opacity': 0.3, 'size': 10})]),
                go.Frame(data=[
                    go.Scatter(x=np.array(x[:3]), y=(y[:3]), marker={'color': 'red', 'opacity': 0.3, 'size': 10})]),
                go.Frame(data=[go.Scatter(x=x[:4], y=y[:4])]),
                go.Frame(data=[go.Scatter(x=x[:5], y=y[:5])]),
                go.Frame(data=[go.Scatter(x=x[:6], y=y[:6])]),
                go.Frame(data=[go.Scatter(x=x[:7], y=y[:7], marker={'color': 'green', 'opacity': 0.3, 'size': 10})]),
                go.Frame(data=[go.Scatter(x=x[:8], y=y[:8])],
                         layout=go.Layout(title_text="Animation of severe patient progress curve"))]
    )

    fig.show()

    if save_to_html:
        fig.write_html('Animation of severe patient progress curve.html')

