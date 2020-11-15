import numpy as np


def local_evaluate(test_vol, test_mask, model, mode=3, only_infer=False, threshold_after_infer=0.):
    """
    Load data
    threshold_after_infer: float 0-1.
    """
    test_vol = test_vol
    test_mask = test_mask
    if np.max(test_vol) > 1:
        test_vol = test_vol / 255.0
    if np.max(test_mask) > 1:
        test_mask = test_mask / 255.0

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
        te_mask2 = get_infer_data(test_mask)
    else:
        te_data4 = te_data2
        te_mask2 = test_mask
    predictions = model.predict(te_data4, batch_size=4, verbose=1)

    if mode == 3:
        predictions2 = get_evaluate_data(predictions)
        te_mask3 = get_evaluate_data(te_mask2)
    else:
        predictions2 = predictions
        te_mask3 = te_mask2
    predictions3 = predictions2

    if only_infer:
        if threshold_after_infer:
            assert predictions3.max() <= 1.0
            predictions3[predictions3 < threshold_after_infer] = 0
            predictions3[predictions3 >= threshold_after_infer] = 1
        return predictions3

    '''
    evaluate
    '''
    return None
