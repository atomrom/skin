# -*- coding: utf-8 -*-

import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.utils.multiclass import unique_labels
import json


def save_classification_report(file_path, truths, preds, class_names):
    y_true = []
    for truth_array in truths:
        y_true.append(np.argmax(truth_array))

    y_pred = []
    for pred_array in preds:
        y_pred.append(np.argmax(pred_array))

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    with open(file_path, 'w') as json_file:
        json.dump(report, json_file)


def plot_history(file_path, history):
    # Plot training & validation accuracy values
    plt.clf()
    plt.figure()

    training_loss = history.history['loss']
    test_loss = history.history['val_loss']

    epoch_count = range(1, len(training_loss) + 1)
    
    plt.subplot(211)
    plt.plot(epoch_count, history.history['acc'], label='Train Accuracy')
    plt.plot(epoch_count, history.history['val_acc'], label='Validation Accuracy')
    plt.title('Training History')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')
    plt.grid(True)

    plt.subplot(212)
    plt.plot(epoch_count, training_loss, label='Train Loss')
    plt.plot(epoch_count, test_loss, label='Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.grid(True)

    plt.savefig(file_path)


def plot_roc(file_path, class_names, y_test, y_pred, title):
    # Plot linewidth.
    lw = 2

    n_classes = len(class_names)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(1)
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['red', 'yellow', 'green', 'blue', 'magenta', 'black', 'gray'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(class_names[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(str(title))
    plt.legend(loc="lower right")

    plt.savefig(file_path)


def plot_confusion_matrix(file_path, class_names, y_test, y_pred,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    def to_class_index_array(y):
        class_index_array = []
        for i in range(len(y)):
            y_array = y[i]
            class_index = np.argmax(y_array)

            class_index_array.append(class_index)

        return class_index_array

    y_test = to_class_index_array(y_test)
    y_pred = to_class_index_array(y_pred)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    # Only use the labels that appear in the data
    classes = class_names#[unique_labels(y_test, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Histology',
           xlabel='Predicted')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    np.set_printoptions(precision=2)

    plt.savefig(file_path)

