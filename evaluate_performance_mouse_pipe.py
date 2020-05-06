from my_email import *
from scipy.ndimage.morphology import binary_erosion
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import scipy
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # "1"
# import models4 as M


####################################  Load Data #####################################
# folder    = './processed_data/'
# te_data   = np.load(folder+'data_test.npy')
# FOV       = np.load(folder+'FOV_te.npy')
# te_mask   = np.load(folder+'mask_test.npy')  #

# te_data  = np.expand_dims(te_data, axis=3)  # (307, 512, 512, 1)


def my_evaluate1(test_vol, test_mask, model, model_name_id, mode=3, evaluatation=None):
    '''
    Load mouse data
    '''
    # # temp=np.load('matlab_adjust_data.npy',allow_pickle=True)  # for brain
    # te_data = temp[1]  # = = test_vol  (84, 160, 160, 1)
    # te_mask = temp[3] /255.  # = = test_mask # (84, 160, 160, 1)  # 记得除以255 归一化
    te_data = test_vol
    te_mask = test_mask
    if np.max(te_data) > 1:
        te_data = te_data / 255.0
    if np.max(te_mask) > 1:
        te_mask = te_mask / 255.0

    # FOV=np.ones((84,160,160,1))  # FOV是mask扩充像素的区域吗? 如果FOV取mask得到的指标都是满分 # 需要降到 3维
    FOV = np.ones_like(te_data)
    FOV = FOV.reshape((FOV.shape[0], FOV.shape[1], FOV.shape[2]))

    print('Dataset loaded')
    # te_data2  = dataset_normalized(te_data)
    te_data2 = te_data  # /255.  # matlab老鼠的数据已经归一化

    def get_infer_data(te_data2):
        # make dataset match the model input. 3维到5维
        # from te_data2  (84, 160, 160, 1) to te_data4  (21, 4, 160, 160, 1)
        # 被4整除，多的余数/slices 舍弃！
        te_data3 = []
        tag = 0
        for i in range(int(te_data2.shape[0] / 4)):  # 相当于整除 84//4 = 21
            te_data3.append(te_data2[tag:tag + 4])
            tag += 4

        te_data4 = np.array(te_data3)
        return te_data4

    def get_evaluate_data(data, k=None, outflod=None):
        # 将model输出5维的数据转为 3维供后续验证 01/07 顺序不一致，呈1234，2345，3456，故修改。后发现是上面的bug...
        # pred = []
        # batchs = data.shape[0]  # 21
        # slice_counts = data.shape[1]  # 4 slices_count
        # for i in range(batchs):
        #     for j in range(slice_counts):
        #         pred.append(data[i,j,:,:,:])
        # return np.array(pred)
        '''

        :param data: None
        :param k: 是否输出图像，是则传入 k=1
        :return: 40,4,256,256,1) -> (160,256,256)
        '''
        pred = []
        # k = None  # 保存计数器。若= None 则不保存
        for i in range(int(te_data2.shape[0] / 4)):
            for j in range(4):
                if k:
                    # plt.imsave('./test2/' + "{:0>4}".format(str(k)) + '.png', predictions[i, j, :, :, 0], cmap='gray');k+=1
                    plt.imsave(outflod + "{:0>4}".format(str(k)) + '_pred.png', predictions[i, j, :, :, 0], cmap='gray')
                    k += 1
                pred.append(predictions[i, j, :, :, 0])
        return np.array(pred)

    if mode == 3:
        te_data4 = get_infer_data(te_data2)  # checked right
        te_mask2 = get_infer_data(te_mask)  # checked right
    else:
        te_data4 = te_data2
        te_mask2 = te_mask
    # model = M.BCDU_net_D3(input_size = (4,160,160,1))  # 原文(512,512,1) # 注意import的 models要更新，不然报错
    # model.summary()
    # model.load_weights('TimeDist_weight_lung_kaggle_0107')
    # (None,slices_count,512,512,1) (21,4,160,160,1)
    predictions = model.predict(te_data4, batch_size=4, verbose=1)

    if mode == 3:
        # 得到 (all test slices=84，160，160，1)
        predictions2 = get_evaluate_data(predictions)
        te_data3 = get_evaluate_data(te_mask2)
    else:
        predictions2 = predictions
        te_data3 = te_mask2
    # 用完记得形状还原
    predictions3 = predictions2  # 模型输出全为0. 01/07 不为0
    # mask 为了保持顺序一致也做一遍

    if not evaluatation:  # 如果没有标签，则只返回预测值
        return predictions3

    # Post-processing
    predictions3 = np.squeeze(predictions3)  # (307,512,512)
    predictions3 = np.where(predictions3 > 0.5, 1, 0)
    # Estimated_lung = np.where((FOV - predictions)>0.5, 1, 0)  # mouse do not have,so ignore
    Estimated_lung = predictions3

    '''
    # 保存图片. 似乎图片的顺序不一致导致分低
    # 用 3维 保存到图片
    predictions3.shape  Out[33]: (84, 160, 160)
    for i in range(84):
        plt.imsave('./test/'+str(i)+'.png',predictions3[i,:,:],cmap='gray')

    # 直接用模型的输入输出（5维）保存到图片
    k=1
    for i in range(21):
        for j in range(4):
            plt.imsave('./test2/' + str(k) + '_train.png', te_data4[i, j, :, :, 0], cmap='gray');k+=1
    '''

    # Performance checking

    y_scores = Estimated_lung.reshape(
        Estimated_lung.shape[0] * Estimated_lung.shape[1] * Estimated_lung.shape[2], 1)
    print(y_scores.shape)

    y_true = te_mask.reshape(
        te_mask.shape[0] * te_mask.shape[1] * te_mask.shape[2], 1)

    y_scores = np.where(y_scores > 0.5, 1, 0)
    y_true = np.where(y_true > 0.5, 1, 0)
    '''
    最终铺平成一维向量的预测值和真实值。下面计算指标
    '''
    import time
    time_id = np.int64(time.strftime(
        '%Y%m%d%H%M', time.localtime(time.time())))
    time_id = str(time_id)[-8:]  # '02131414' 去掉年份
    # output_folder = 'output_mouse/'
    output_folder = 'output_mouse/' + model_name_id + '_' + time_id + '/'
    # 'output_mouse/models3_0212_v2_02131414/'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Area under the ROC curve
    fpr, tpr, thresholds = roc_curve((y_true), y_scores)
    AUC_ROC = roc_auc_score(y_true, y_scores)
    print("\nArea under the ROC curve: " + str(AUC_ROC))
    my_roc_curve = plt.figure()  # 无需定义名称。图片保存到文件。
    plt.plot(fpr, tpr, '-', label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
    plt.title('ROC curve')
    plt.xlabel("FPR (False Positive Rate)")
    plt.ylabel("TPR (True Positive Rate)")
    plt.legend(loc="lower right")
    plt.savefig(output_folder + "ROC.png")

    # dice_coef 自加：
    dice_coef = (2. * np.sum(y_true * y_scores) + 0.001) / \
        (np.sum(y_true) + np.sum(y_scores) + 0.001)
    print("dice_coef: " + str(dice_coef))
    # Volumetric Similarity
    VS = 1 - (abs((np.sum(y_scores) - np.sum(y_true))) + 0.0001) / \
        (np.sum(y_true) + np.sum(y_scores) + 0.0001)
    print("RVD: " + str(VS))
    # relative volume difference
    RVD = ((np.sum(y_scores) - np.sum(y_true)) + 0.0001) / \
        (np.sum(y_true) + 0.0001)  # *100%
    print("RVD: " + str(RVD))
    VOE = (2 * (1 - dice_coef) + 0.0001) / ((2 - dice_coef) + 0.0001)
    print("VOE: " + str(VOE))

    # Precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    precision = np.fliplr([precision])[0]
    recall = np.fliplr([recall])[0]
    AUC_prec_rec = np.trapz(precision, recall)
    print("\nArea under Precision-Recall curve: " + str(AUC_prec_rec))
    prec_rec_curve = plt.figure()
    plt.plot(recall, precision, '-',
             label='Area Under the Curve (AUC = %0.4f)' % AUC_prec_rec)
    plt.title('Precision - Recall curve')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower right")
    plt.savefig(output_folder + "Precision_recall.png")

    # Confusion matrix
    threshold_confusion = 0.5
    print("\nConfusion matrix:  Custom threshold (for positive) of " +
          str(threshold_confusion))
    y_pred = np.empty((y_scores.shape[0]))
    for i in range(y_scores.shape[0]):
        if y_scores[i] >= threshold_confusion:
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    confusion = confusion_matrix(y_true, y_pred)
    print(confusion)
    accuracy = 0
    if float(np.sum(confusion)) != 0:
        accuracy = float(confusion[0, 0] + confusion[1, 1]
                         ) / float(np.sum(confusion))
    print("Global Accuracy: " + str(accuracy))
    specificity = 0
    if float(confusion[0, 0] + confusion[0, 1]) != 0:
        specificity = float(confusion[0, 0]) / \
            float(confusion[0, 0] + confusion[0, 1])
    print("Specificity: " + str(specificity))
    sensitivity = 0
    if float(confusion[1, 1] + confusion[1, 0]) != 0:
        sensitivity = float(confusion[1, 1]) / \
            float(confusion[1, 1] + confusion[1, 0])
    print("Sensitivity: " + str(sensitivity))
    precision = 0
    if float(confusion[1, 1] + confusion[0, 1]) != 0:
        precision = float(confusion[1, 1]) / \
            float(confusion[1, 1] + confusion[0, 1])
    print("Precision: " + str(precision))

    # Jaccard similarity index
    jaccard_index = jaccard_similarity_score(y_true, y_pred, normalize=True)
    print("\nJaccard similarity score: " + str(jaccard_index))

    # F1 score
    F1_score = f1_score(y_true, y_pred, labels=None,
                        average='binary', sample_weight=None)
    print("\nF1 score (F-measure): " + str(F1_score))

    # Save the results
    file_perf = open(output_folder + 'performances.txt', 'w')
    file_perf.write("Area under the ROC curve: " + str(AUC_ROC)
                    + "\nArea under Precision-Recall curve: " +
                    str(AUC_prec_rec)
                    + "\nJaccard similarity score: " + str(jaccard_index)
                    + "\nDice score: " + str(dice_coef)  # dice
                    + "\nF1 score (F-measure): " + str(F1_score)
                    + "\n\nConfusion matrix:"
                    + str(confusion)
                    + "\nACCURACY: " + str(accuracy)
                    + "\nSENSITIVITY: " + str(sensitivity)
                    + "\nSPECIFICITY: " + str(specificity)
                    + "\nPRECISION: " + str(precision)
                    )
    file_perf.close()

    # # Sample results  画图展示
    # fig,ax = plt.subplots(5, 3, figsize=[15,15])
    # # all_ind = [1, 100, 200, 253, 193] # random samples
    # all_ind = [1, 10, 20, 25, 19] # random samples  # 如果样本没那么多
    # all_ind = np.array(all_ind)
    # for idx in range(5):
    #     # ax[idx, 0].imshow(np.uint8(np.squeeze(te_data[all_ind[idx]])))  # (160,160)
    #     ax[idx, 0].imshow(np.squeeze(te_data[all_ind[idx]]))  # (160,160)
    #     ax[idx, 1].imshow(np.squeeze(te_mask[all_ind[idx]]), cmap='gray')
    #     ax[idx, 2].imshow(np.squeeze(Estimated_lung[all_ind[idx]]), cmap='gray')
    #
    # plt.savefig('sample_results.png')  # 保存到主目录
    '''
    send results to mail
    '''
    mail_txt = (""+str(AUC_ROC)
                + "\n" + str(AUC_prec_rec)
                + "\n" + str(jaccard_index)
                + "\n" + str(dice_coef)  # dice
                + "\n" + str(F1_score)
                + "\n\nConfusion matrix:"
                + str(confusion)
                + "\n" + str(accuracy)
                + "\n" + str(sensitivity)
                + "\n" + str(specificity)
                + "\n" + str(precision)
                + "\nVS,RVD,VOE\n" + str(VS)
                + "\n" + str(RVD)
                + "\n" + str(VOE)
                )
    # # 保存测试的原图 test_vol 2D、3D网络都适用
    # print(test_vol.shape)
    # for k in range(int(test_vol.shape[0])):
    #     plt.imsave(output_folder + "{:0>4}".format(str(k+1)) + '.png', test_vol[k, :, :, 0], cmap='gray')
    #     k += 1
    # # 保存预测图片
    # if mode ==3:
    #     print(predictions.shape)
    #     predictions2 = get_evaluate_data(predictions, k=1, outflod=output_folder)  # 预测图
    # else:
    #     print(predictions.shape)
    #     for k in range(int(test_vol.shape[0])):
    #         plt.imsave(output_folder + "{:0>4}".format(str(k + 1)) + '_pred.png', predictions[k, :, :, 0], cmap='gray')
    #         k += 1

    # sendEmail(content='我用Python3333', title='人生苦短9', txt_name='log.txt')  # 对于当前路径下的txt文件
    try:
        sendEmail(content=mail_txt, title=output_folder,
                  txt_name=output_folder+'performances.txt')  # 对于当前路径下的txt文件
    except:
        print('mail failed to seed, may not connect to Internet\n')

    return None


if __name__ == '__main__':
    pass
