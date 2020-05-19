"""
Using MNIST, compare classification performance of:
1) logistic regression by itself,
2) logistic regression on outputs of an RBM, and
3) logistic regression on outputs of a stacks of RBMs / a DBN.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import BernoulliRBM
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


def norm(arr):
    arr = arr.astype(np.float)
    arr -= arr.min()
    arr /= arr.max()
    return arr


if __name__ == '__main__':

    # load MNIST data set
    filename = 'bugzilla'
    data: pd.DataFrame = pd.read_csv('input/' + filename + '.csv')
    date = data.pop('commitdate').values
    Y: np.ndarray = data.pop('bug').values
    X: np.ndarray = data.values

    # normalize inputs to 0-1 range
    X = norm(X)

    # split into train, validation, and test data sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, random_state=0)

    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, train_size=0.9, random_state=0)

    # --------------------------------------------------------------------------------
    # set hyperparameters

    learning_rate = 0.02  # from Erhan et el. (2010): median value in grid-search
    total_units = 14  # from Erhan et el. (2010): optimal for MNIST / only slightly worse than 1200 units when using InfiniteMNIST
    total_epochs = 100  # from Erhan et el. (2010): optimal for MNIST
    batch_size = 25  # seems like a representative sample; backprop literature often uses 256 or 512 samples

    C = 10.  # optimum for benchmark model according to sklearn docs: https://scikit-learn.org/stable/auto_examples/neural_networks/plot_rbm_logistic_classification.html#sphx-glr-auto-examples-neural-networks-plot-rbm-logistic-classification-py)

    # TODO optimize using grid search, etc

    # --------------------------------------------------------------------------------
    # construct models

    # RBM
    rbm = BernoulliRBM(n_components=total_units, learning_rate=learning_rate, batch_size=batch_size,
                       n_iter=total_epochs, verbose=1)
    rbm2 = BernoulliRBM(n_components=20, learning_rate=learning_rate, batch_size=batch_size,
                        n_iter=total_epochs, verbose=1)
    rbm3 = BernoulliRBM(n_components=12, learning_rate=learning_rate, batch_size=batch_size,
                        n_iter=total_epochs, verbose=1)
    rbm4 = BernoulliRBM(n_components=total_units, learning_rate=learning_rate, batch_size=batch_size,
                        n_iter=total_epochs, verbose=1)

    # "output layer"
    logistic = LogisticRegression(C=C, solver='lbfgs', multi_class='multinomial', max_iter=200, verbose=1)

    models = []
    models.append(Pipeline(steps=[('logistic', clone(logistic))]))  # base model / benchmark
    models.append(
        Pipeline(steps=[('rbm1', clone(rbm)), ('rbm2', clone(rbm2)),('rbm3', clone(rbm3)), ('logistic', clone(logistic))]))  # RBM stack / DBN

    # --------------------------------------------------------------------------------
    # train and evaluate models

    for model in models:
        # train
        model.fit(X_train, Y_train)

        # evaluate using validation set
        print("Model performance:\n%s\n" % (
            classification_report(Y_val, model.predict(X_val))))
