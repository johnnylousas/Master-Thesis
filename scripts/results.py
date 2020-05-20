import random

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

np.random.seed()
data_path = '/Users/joaolousada/Documents/5ºAno/Master-Thesis/main/data/'


train: pd.DataFrame = pd.read_csv(data_path + 'tempest-full/tf_train.csv')
test: pd.DataFrame = pd.read_csv(data_path + 'tempest-full/tf_test.csv')
dev: pd.DataFrame = pd.read_csv(data_path + 'tempest-full/tf_dev.csv')

data = pd.concat([train, dev, test], ignore_index=False)
data.rename(columns={'Unnamed: 0': 'test_id'}, inplace=True)
data = data.drop(columns=['test_id'])

y: np.ndarray = data.pop('class').values
X: np.ndarray = data.values

"""
ClosureCompiler = pd.read_csv(data_path + 'palma_data/Joda-Time Metrics Raw_Data.csv')
ClosureCompiler = ClosureCompiler.drop(columns=['TestID', 'TestName', 'Status'])

y: np.ndarray = ClosureCompiler.pop('Result').values
X: np.ndarray = ClosureCompiler.values
"""

labels = pd.unique(y)


def apfd(test_arr: np.array):
    n = len(test_arr)
    m = sum(test_arr)
    pos = 0

    for i in range(n):
        if test_arr[i] == 1:
            pos += i
    return 1 - pos / (n * m) + 1 / (2 * n)


def rank(clf):
    # Manual KFold
    kf = KFold(n_splits=10, random_state=42, shuffle=True)

    accuracies = []
    apfd_sort = []
    apfd_rand = []

    for train_index, test_index in kf.split(X):
        trnX = X[train_index]
        trnY = y[train_index]

        tstX = X[test_index]
        tstY = y[test_index]

        # preprocessing
        scaler = preprocessing.MinMaxScaler()
        scaler.fit(trnX)
        trnX = scaler.transform(trnX)
        tstX = scaler.transform(tstX)

        # train model
        clf.fit(trnX, trnY)
        preds = clf.predict(tstX)
        preds = (preds > 0.5)

        # accuracy for the current fold only
        accuracy = accuracy_score(tstY, preds)
        accuracies.append(accuracy)

        # probability of failure for every test case
        probs = clf.predict_proba(tstX)[:, 1]
        probs = np.round(probs * 100, 1)

        # Order tests from more to less likely to fail
        test_order = [x for _, x in sorted(zip(probs, tstY), reverse=True)]

        # Order tests randomly
        nr_rand = random.sample(range(len(tstY)), len(tstY))
        test_random = [x for _, x in sorted(zip(nr_rand, tstY), reverse=True)]

        # Calculate APFD
        apfd_sort.append(apfd(test_order))
        apfd_rand.append(apfd(test_random))

    return apfd_sort, apfd_rand


def boxplots(apfds: list, dataset: str, save: bool):
    plt.figure()
    plt.boxplot(apfds, labels=['LR', 'DT', 'RF', 'XGB', 'ANN', 'Random'])
    plt.title('Test Case Ranking Performance' + ' ' + dataset)
    plt.ylabel('APFD')
    plt.ylim((0, 1))
    if save:
        plt.savefig('/Users/joaolousada/Documents/5ºAno/Master-Thesis/main/metrics/APFD/APFD_' + dataset + '.png',
                    bbox_inches='tight')
    plt.show()


def tune_ann():
    from sklearn.model_selection import GridSearchCV
    param_grid = [
        {
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
            'solver': ['lbfgs', 'sgd', 'adam'],
            'hidden_layer_sizes': [
                (1,), (2,), (3,), (4,), (5,), (5, 2), (4, 2), (3, 3), (5, 4), (4, 4),
            ]
        }
    ]
    clf = GridSearchCV(MLPClassifier(), param_grid, cv=3,
                       scoring='accuracy')
    clf.fit(X, y)

    print("Best parameters set found on development set:")
    print(clf.best_params_)


# =========

models = [LogisticRegression(solver='liblinear'), DecisionTreeClassifier(), RandomForestClassifier(), XGBClassifier(),
          MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(5,4))]
apfds = []
i = 0
for clf in models:
    apfd_sort = rank(clf)[0]
    apfds.append(apfd_sort)
    i += 1
    if i == len(models):
        apfds.append(rank(clf)[1])

boxplots(apfds, dataset='tempest-full', save=False)
