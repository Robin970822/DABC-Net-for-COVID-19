# -*- coding: utf-8 -*-
import pickle
import json
import tensorflow as tf

from keras.backend.tensorflow_backend import set_session
from infer_colab import infer_colab
from infer_uncertainty_colab import infer_uncertainty
from utils.read_all_data_from_nii_pipe import read_from_nii
from utils.postprocess_lung import remove_small_objects
from utils.visualization import *
from utils.calculate_feature import *


def predict_base_learners(base_learners, x):
    P = np.zeros((x.shape[0], len(base_learners)))
    print('Generating base learner predictions.')
    for i, (name, m) in enumerate(base_learners.items()):
        print('%s...' % name, end='', flush=False)
        p = m.predict_proba(x)
        P[:, i] = p[:, 1]
    print('done.')
    return P


"""Infer_demo.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1bgqFuXF6li5HNCKtmtXDUGx1ZzJgOxJn
"""

# from google.colab import drive
# drive.mount('/content/drive')
# cd drive/My\ Drive/Colab\ Notebooks/DABC-Net(colab)/
# pip install SimpleITK scipy==1.1 tensorflow-gpu==1.15 keras==2.2.4 xgboost==1.1.0 scikit-learn==0.21.3

"""# Run segmentation"""

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

# Segment lesions
# input='input'
# output='output'
input_path = '2020035365'
output_path = '2020035365_output/covid'
infer_colab(input_path, output_path)

# Segment lung
# output='output_lung'
output_path = '2020035365_output/lung'
infer_colab(input_path, output_path, usage='lung')

# remove_small_objects('output_lung')  # (This step is optional.)
remove_small_objects('2020035365_output/lung')

infer_uncertainty('2020035365/2020035365_0204_3050_20200204184413_4.nii.gz', '2020035365_output/uncertainty',
                  sample_value=5, uc_chosen='Both')

"""
## Visualisation

## Segmentation results
"""

raw, lung, lesion, ratio = data_disease_slice(patientID='2020035365', slice_id=[175, 162, 195, 195, 195, 195, 195, 195])
meta_path = 'meta/2020035365.csv'
meta = pd.read_csv(meta_path, index_col=[0])
_meta = meta[meta['slice'] > 100]
_meta['ratio'] = ratio

plot_segmentation(raw, lung, lesion, color_map='Reds', state='Severe')
print('\nProgress curve of severe patient')

# plot_segmentation(raw, lung, lesion, color_map='Reds', state='Severe')

plot_progress_curve(_meta, '2020035365')
plot_uncertainty(name_id='2020035365_0204_3050_20200204184413_4.nii.gz', slice_id=175)

"""#Early triage of critically ill

## Load data
  In this section, we present one mild patient and one severe patient with multi-scans to show progress of the disease and illustrate our model performance.
"""

# pip install SimpleITK xgboost==1.1.0 scikit-learn==0.21.3 scipy==1.1

# severe patient
meta_path = 'meta/2020035365.csv'
meta = pd.read_csv(meta_path, index_col=[0])

raw_data = read_from_nii(r'2020035365/*').astype('float32')  # (3201, 256, 256)  0-1
lung = read_from_nii(r'2020035365_output/lung/*').astype('float32')  # (3201, 256, 256)
lesion = read_from_nii(r'2020035365_output/covid/*').astype('float32')  # (3201, 256, 256)

# mild patient
meta_path_mild = 'meta/2020035021.csv'
meta_mild = pd.read_csv(meta_path_mild, index_col=[0])

"""## Calculate"""
res_list, all_info = calculate(raw_data, lung, lesion, meta)

del raw_data, lung, lesion

# mild patient
raw_data_mild = read_from_nii(r'2020035021/*').astype('float32')
lung_mild = read_from_nii(r'2020035021_output/lung/*').astype('float32')
lesion_mild = read_from_nii(r'2020035021_output/covid/*').astype('float32')

res_list_mild, all_info_mild = calculate(raw_data_mild, lung_mild, lesion_mild, meta_mild)
del raw_data_mild, lung_mild, lesion_mild

"""## Multi-time Visualization"""

print('Severe pateint vs Mild pateint')
plt.figure(figsize=(16, 9))
plot_progress_curve(all_info, patientID=2020035365, line_color=sns.color_palette('Reds')[5], label='Severe patient')
plot_progress_curve(all_info_mild, patientID=2020035021, line_color=sns.color_palette('Greens')[3],
                    label='Mild patient')
plt.legend(loc='upper right')

"""## Prediction
### Load model
"""

with open('model/prediction.pkl', 'rb') as j:
    base_pred = pickle.load(j)

with open('model/min_max_prediction.json', 'r') as j:
    min_max_dict_pred = json.load(j)

with open('model/classification.pkl', 'rb') as j:
    base_cls = pickle.load(j)

with open('model/min_max_classification.json', 'r') as j:
    min_max_dict_cls = json.load(j)

feature = [
    'left_ratio', 'right_ratio',
    'left_lung', 'right_lung',
    'left_lesion', 'right_lesion',

    'left_weighted_lesion', 'right_weighted_lesion',

    'left_consolidation', 'right_consolidation',

    'left_z', 'right_z',
    'Age', 'sex',
]

"""## Preprocessing"""

# Preprocessing
X = preprocessing(all_info, feature)
X_mild = preprocessing(all_info_mild, feature)

"""## Per Scan Classification"""


def Per_Scan_Classification(X):
    x = min_max_scalar(np.array(X), np.array(min_max_dict_cls['min']), np.array(min_max_dict_cls['max']))
    P_pred = predict_base_learners(base_cls, np.array(x))
    p = P_pred.mean(axis=1)
    return p


p = Per_Scan_Classification(X)
print('Prediction of severe patient(per scan):\n{}\n'.format(p))
p_mild = Per_Scan_Classification(X_mild)
print('Prediction of mild patient(per scan):\n{}\n'.format(p_mild))

print('\n' + '*' * 10 + '\tSevere patient\t' + '*' * 10)
print('pred\t{} \ngt\t{} \nprob {}'.format((p > 0.5).astype('int'), np.array(all_info['Severe']), p))
print('\n' + '*' * 10 + '\tMild patient\t' + '*' * 10)
print('pred\t{} \ngt\t{} \nprob {}'.format((p_mild > 0.5).astype('int'), np.array(all_info_mild['Severe']), p_mild))

"""## First Two Scans"""


def First_Two_Scans(X):
    # first two scan
    x_list = X.iloc[1].tolist()[:-2] + X.iloc[0].tolist()
    # min max scale
    x = min_max_scalar(np.array(x_list), np.array(min_max_dict_pred['min']), np.array(min_max_dict_pred['max']))

    # Predition
    P_pred = predict_base_learners(base_pred, np.array([x]))
    return P_pred.mean()


print('\n' + '*' * 10 + '\tSevere patient\t' + '*' * 10)
print(First_Two_Scans(X))
print('\n' + '*' * 10 + '\tMild patient\t' + '*' * 10)
print(First_Two_Scans(X_mild))

"""## Progress of disease"""

slice_id = [175, 162, 195, 195, 195, 195, 195, 195]
raw, lesion, gt = data_disease_progress_slice(all_info, patientID=2020035365, slice_id=slice_id, timepoint_count=8)
plot_progress(raw, lesion, p, gt, state='severe', color_map='Reds', timepoint_count=8)

print('\n\n')
slice_id = [200, 200, 200, 200, 200, 200]
raw, lesion, gt = data_disease_progress_slice(all_info_mild, patientID=2020035021, slice_id=slice_id, timepoint_count=6)
plot_progress(raw, lesion, p_mild, gt, state='mild', color_map='Reds', timepoint_count=6)
