import numpy as np
from sklearn.metrics import roc_auc_score


def local_inference(test_vol, model, mode=3, threshold_after_infer=0.):
    """
    Load data
    threshold_after_infer: float 0-1.
    """
    if np.max(test_vol) > 1:
        test_vol = test_vol / 255.0

    print('Dataset loaded')
    te_data2 = test_vol

    def get_infer_data(te_data2):
        te_data3 = []
        tag = 0
        for i in range(int(te_data2.shape[0] / 4)):
            te_data3.append(te_data2[tag:tag + 4])
            tag += 4

        te_data4 = np.array(te_data3)
        return te_data4

    def get_evaluate_data(data, k=None, output_folder=None):

        """
        stack patches to slices
        (slices,patches,H,W,1) -> (slices*patches,H,W)
        :param data: None, as a placeholder.
        :param k: if k=1, save images.
        :param output_folder:
        :return: stacked matrix.
        """
        pred = []
        for i in range(int(te_data2.shape[0] / 4)):
            for j in range(4):
                if k:
                    from matplotlib import pyplot as plt
                    plt.imsave(output_folder + "{:0>4}".format(str(k)) + '_pred.png', predictions[i, j, :, :, 0], cmap='gray')
                    k += 1
                pred.append(predictions[i, j, :, :, 0])
        return np.array(pred)

    if mode == 3:
        te_data4 = get_infer_data(te_data2)
    else:
        te_data4 = te_data2
    predictions = model.predict(te_data4, batch_size=4, verbose=1)

    if mode == 3:
        predictions2 = get_evaluate_data(predictions)
    else:
        predictions2 = predictions
    predictions3 = predictions2

    if threshold_after_infer:
        assert predictions3.max() <= 1.0
        predictions3[predictions3 < threshold_after_infer] = 0
        predictions3[predictions3 >= threshold_after_infer] = 1
    return predictions3


def local_evaluate(test_vol, test_mask, model=None, mode=3, only_infer=False,
                   _slice_count=4, threshold_after_infer=0.):
    '''
    Load data
    threshold_after_infer: float 0-1.
    '''
    test_vol = test_vol
    test_mask = test_mask
    print('test_mask.shape:', test_mask.shape)
    if np.max(test_vol) > 1:
        test_vol = test_vol / 255.0
    if np.max(test_mask) > 1:
        test_mask = test_mask / 255.0

    print('Dataset loaded')
    te_data2 = test_vol

    def get_infer_data(te_data2):
        te_data3 = []
        tag = 0
        for i in range(int(te_data2.shape[0] / _slice_count)):
            te_data3.append(te_data2[tag:tag + _slice_count])
            tag += _slice_count

        te_data4 = np.array(te_data3)
        return te_data4

    def get_evaluate_data(data, k=None, outfloder=None):

        '''
        stack patches to slices
        (slices,patches,H,W,1) -> (slices*patches,H,W)
        :param data: None, as a placeholder.
        :param k: if k=1, save images.
        :return: stacked matrix.
        '''
        pred = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if k:
                    plt.imsave(outfloder + "{:0>4}".format(str(k)) + '_pred.png', predictions[i, j, :, :, 0],
                               cmap='gray');
                    k += 1
                pred.append(data[i, j, :, :, 0])
        return np.array(pred)

    if mode == 3:
        te_data4 = get_infer_data(te_data2)
    else:
        te_data4 = te_data2

    predictions = model.predict(te_data4, batch_size=4, verbose=1)

    if mode == 3:
        predictions2 = get_evaluate_data(predictions)
    else:
        predictions2 = predictions
    predictions3 = predictions2

    if threshold_after_infer:
        assert predictions3.max() <= 1.0
        predictions3[predictions3 < threshold_after_infer] = 0
        predictions3[predictions3 >= threshold_after_infer] = 1

    if only_infer:
        return predictions3


    '''
    evaluate
    '''
    y_scores = predictions3.flatten()
    y_true = test_mask.flatten()

    assert np.max(y_scores) <= 1
    assert np.max(y_true) <= 1

    y_scores[y_scores <= 0.5] = 0
    y_scores[y_scores > 0.5] = 1
    y_true[y_true <= 0.5] = 0
    y_true[y_true > 0.5] = 1

    # Metric
    dice_coef = (2. * np.sum(y_true * y_scores) + 0.001) / (np.sum(y_true) + np.sum(y_scores) + 0.001)
    print("Dice_coef: " + str(dice_coef))
    # Volumetric Similarity
    VS = 1 - (abs((np.sum(y_scores) - np.sum(y_true))) + 0.0001) / (np.sum(y_true) + np.sum(y_scores) + 0.0001)
    print("VS: " + str(VS))
    # Relative volume difference
    RVD = ((np.sum(y_scores) - np.sum(y_true)) + 0.0001) / (np.sum(y_true) + 0.0001)
    print("RVD: " + str(RVD))
    VOE = (2 * (1 - dice_coef) + 0.0001) / ((2 - dice_coef) + 0.0001)
    print("VOE: " + str(VOE))

    # ROC
    AUC_ROC = roc_auc_score(y_true, y_scores)
    print("ROC: " + str(AUC_ROC))

    # save result to txt
    txt_file = open('result.txt', 'w')
    txt_file.write("Area under the ROC curve: " + str(AUC_ROC)
                   + "\nDice score: " + str(dice_coef)
                   + "\nVolumetric Similarity: " + str(VS)
                   + "\nRelative volume difference: " + str(RVD)
                   + "\nVOE: " + str(VOE)
                   )
    txt_file.close()

    return None
