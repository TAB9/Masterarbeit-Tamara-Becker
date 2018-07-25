print(__doc__)

import itertools
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix

# function to plot the confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=80)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt) if cm[i, j] > 0 else "",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    # lines for a better overwiew in colums and rows
    plt.hlines(y=np.arange(16+1)- 0.5, xmin=-0.5, xmax=16-0.5, colors = 'silver')
    plt.vlines(x=np.arange(16+1)- 0.5, ymin=-0.5, ymax=16-0.5, colors = 'silver')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
