import os
import numpy as np
import torch
import torch.nn.functional as F
import sklearn as skr
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_multi_hot(test_y, classes, assumes_starts_zero=True):
    bs = test_y.shape[0]
    label_cnt = 0

    # TODO ranking labels: (-1,-1,4,5,3,7)->(4,4,1,2,0,3)
    if not assumes_starts_zero:
        for label_val in torch.unique(test_y):
            if label_val >= 0:
                test_y[test_y == label_val] = label_cnt
                label_cnt += 1

    gt = torch.zeros(bs, classes + 1)  # TODO(yue) +1 for -1 in multi-label case
    for i in range(test_y.shape[1]):
        gt[torch.LongTensor(range(bs)), test_y[:, i]] = 1  # TODO(yue) see?

    return gt[:, :classes]


def cal_map(output, old_test_y, is_soft=False):
    batch_size = output.size(0)
    num_classes = output.size(1)
    ap = torch.zeros(num_classes)
    test_y = old_test_y.clone()

    if is_soft:
        gt = test_y
    else:
        gt = get_multi_hot(test_y, num_classes)

    probs = F.softmax(output, dim=1)

    rg = torch.arange(1, batch_size+1).float()
    for k in range(num_classes):
        scores = probs[:, k]
        targets = gt[:, k]
        _, sortind = torch.sort(scores, 0, True)
        truth = targets[sortind]
        tp = truth.float().cumsum(0)
        precision = tp.div(rg)
        # ap[k] = precision[truth.byte()].sum() / max(float(truth.sum()), 1)
        ap[k] = precision[truth.byte().bool()].sum() / max(float(truth.sum()), 1)
    return ap.mean() * 100, ap * 100


# def multi_map(logits, test_y, topk=None, have_softmax=False):
#     from torchnet import meter
#     num_classes = logits.shape[1]
#     topk = [1, min(5, num_classes)] if topk is None else topk
#     single_label = True if len(test_y.shape) == 1 else False
#     probs = F.softmax(logits, dim=1) if not have_softmax else logits
#     if single_label:
#         acc_meter = meter.ClassErrorMeter

def evaluate_map(all_trn_scores, all_trn_labels, rm_bg=False, logger=None):
    '''
    compute mAP for multiple classes
    '''
    ap_dict = {}
    start = 0 if not rm_bg else 1
    n_class = all_trn_scores.shape[1]

    for i in range(start, n_class):
        # label = id_2_class[i]
        gt_label = []
        pred_scores = all_trn_scores[:, i]
        for now_label in all_trn_labels:
            if now_label == i:
                gt_label.append(1)
            else:
                gt_label.append(0)

        gt_label = np.asarray(gt_label)
        ap = average_precision_score(gt_label, pred_scores)
        # print(label, ap) if logger is None else logger.log(str(label) + ': ' + str(ap))
        ap_dict[label] = ap
    map = np.nanmean(list(ap_dict.values()))
    # ap_dict['mAP'] = map
    # print("mAP:%s" % map) if logger is None else logger.log("mAP:%s" % map)
    # return ap_dict
    return map


def show_confusion_matrix(
        score_data, label_data, id2cls, thresh=0.5, percentage=True,
        size=(10.8, 9.6), is_save=False, save_path=None):
    '''
    Compute confusion matrix for multiple classes and show it.
    :param score_data: path to scores data, should be a [num_samples, num_classes] numpy array.
    :param label_data: path to labels data, should be a [num_samples, 1] numpy array.
    :param id2cls: class map dictionary.
    :param thresh: threshold to choose color for text.
    :param percentage: show the percentage or number
    :param size: size of figure window, 1 equal 100 pixel
    :param is_save: if save the image
    :param save_path: image will save to this path.

    Example:
    to show confusion matrix by percenge form as follow.
    show_confusion_matrix('path/to/your/scores.npy', 'path/to/your/labels.npy', {0:'background',1:'class1'}})
    or
    show_confusion_matrix(scores, labels, {0:'background',1:'class1'}})

    '''
    import itertools
    if isinstance(score_data, str):
        # load numpy array
        scores = np.load(score_data)
        labels = np.load(label_data)
    elif isinstance(score_data, np.ndarray):
        scores = score_data
        labels = label_data
    else:
        raise (ValueError, 'Input data should be a path for score and label data  or numpy data.')
    if not isinstance(id2cls, dict):
        raise (ValueError, 'id2cls should be a dictionary.')

    n_class = scores.shape[1]
    if n_class != len(id2cls.keys()):
        raise (IOError, 'The length n_class ')
    confusion_matrix = np.zeros((n_class, n_class), dtype=np.uint)
    pred = np.argmax(scores, axis=1)
    for i in range(pred.size):
        confusion_matrix[int(labels[i]), int(pred[i])] += 1
    confusionMatrixColor = np.apply_along_axis(lambda a: a / np.sum(a), axis=1, arr=confusion_matrix)
    plt.figure(figsize=size)
    plt.imshow(confusionMatrixColor, interpolation='nearest', cmap=plt.cm.Blues)

    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        if confusionMatrixColor[i, j] == 0:
            show_str = '0'
        else:
            if percentage:
                show_str = '{:.2f}'.format(confusionMatrixColor[i, j])
            else:
                show_str = '{:d}'.format(confusion_matrix[i, j])

        plt.text(j, i, show_str,
                 fontsize=10,
                 horizontalalignment="center",
                 color="white" if confusionMatrixColor[i, j] > thresh else "black")
        if i == j:
            r = plt.Rectangle((i - 0.5, j - 0.5), 1, 1, facecolor='none', edgecolor='k', linewidth=2)
            plt.gca().add_patch(r)
    # plt.tick_params(labeltop=True)
    Xticks = [id2cls[x] for x in range(n_class)]
    plt.xticks(range(n_class), Xticks, rotation=45, fontsize=8)
    plt.yticks(range(n_class), Xticks, fontsize=10)
    fig = plt.gcf()
    fig.autofmt_xdate()

    plt.title('Confusion Matrix')
    plt.xlabel('Prediction')
    plt.ylabel('Groundtruth')
    if is_save:
        plt.savefig(save_path)
    else:
        plt.show()


