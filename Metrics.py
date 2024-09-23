from sklearn import metrics
import math
def MetricsCalculate(y_true_label, y_predict_label, y_predict_pro=None):
    metrics_value = []
    confusion = []
    tn, fp, fn, tp = metrics.confusion_matrix(y_true_label, y_predict_label).ravel()
    sn = round(tp / (tp + fn) * 100, 3) if (tp + fn) != 0 else 0
    sp = round(tn / (tn + fp) * 100, 3) if (tn + fp) != 0 else 0
    pre = round(tp / (tp + fp) * 100, 3) if (tp + fp) != 0 else 0
    acc = round((tp + tn) / (tp + fn + tn + fp) * 100, 3)
    mcc = round((tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)), 3) if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) != 0 else 0
    f1 = round(2 * tp / (2 * tp + fp + fn), 3) if (2 * tp + fp + fn) != 0 else 0

    if y_predict_pro is not None:
        fpr, tpr, thresholds = metrics.roc_curve(y_true_label, y_predict_pro)
        precision, recall, thresholds = metrics.precision_recall_curve(y_true_label, y_predict_pro)

        auroc = metrics.auc(fpr, tpr)
        auprc = metrics.auc(recall, precision)

    if y_predict_pro is None:
        auroc = 0
        auprc = 0

    metrics_value.append(sn)
    metrics_value.append(sp)
    metrics_value.append(pre)
    metrics_value.append(acc)
    metrics_value.append(mcc)
    metrics_value.append(f1)
    metrics_value.append(auroc)
    metrics_value.append(auprc)

    confusion.append(tp)
    confusion.append(fn)
    confusion.append(tn)
    confusion.append(fp)

    return metrics_value, confusion